from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
import optuna

from .evaluation import CostConfig, PolicyEvaluation, evaluate_policy
from .model import SimulationConfig, StaffingPolicy

SamplerName = Literal["tpe", "random"]


@dataclass(frozen=True)
class SearchBounds:
    lower: tuple[int, ...]
    upper: tuple[int, ...]

    def validate_for(self, simulation: SimulationConfig) -> None:
        if len(self.lower) != simulation.n_blocks or len(self.upper) != simulation.n_blocks:
            raise ValueError("Search bounds must match the number of staffing blocks.")
        for lower, upper in zip(self.lower, self.upper, strict=True):
            if lower <= 0:
                raise ValueError("Lower staffing bounds must be positive.")
            if upper < lower:
                raise ValueError("Every upper bound must be at least its lower bound.")

    @classmethod
    def uniform(cls, n_blocks: int, lower: int, upper: int) -> "SearchBounds":
        return cls(lower=(lower,) * n_blocks, upper=(upper,) * n_blocks)


@dataclass(frozen=True)
class OptimizationResult:
    best_policy: StaffingPolicy
    training_evaluation: PolicyEvaluation
    training_seeds: tuple[int, ...]
    study: optuna.Study


def make_replication_seeds(seed: int, n_replications: int) -> tuple[int, ...]:
    if n_replications <= 0:
        raise ValueError("n_replications must be positive.")
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 2**32 - 1, size=n_replications, dtype=np.uint32)
    return tuple(int(value) for value in values)


def _sampler(name: SamplerName, seed: int) -> optuna.samplers.BaseSampler:
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unsupported sampler: {name}")


def optimize_staffing(
    simulation: SimulationConfig | None = None,
    costs: CostConfig | None = None,
    bounds: SearchBounds | None = None,
    *,
    n_trials: int = 50,
    n_replications: int = 12,
    seed: int = 2026,
    sampler: SamplerName = "tpe",
) -> OptimizationResult:
    """Optimize staffing with a fixed CRN seed set for all candidate policies."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    simulation = simulation or SimulationConfig()
    costs = costs or CostConfig()
    bounds = bounds or SearchBounds.uniform(simulation.n_blocks, 2, 12)
    bounds.validate_for(simulation)
    training_seeds = make_replication_seeds(seed, n_replications)

    study = optuna.create_study(direction="minimize", sampler=_sampler(sampler, seed))

    def objective(trial: optuna.Trial) -> float:
        policy = StaffingPolicy(
            tuple(
                trial.suggest_int(f"staff_block_{index + 1}", lower, upper)
                for index, (lower, upper) in enumerate(
                    zip(bounds.lower, bounds.upper, strict=True)
                )
            )
        )
        evaluation = evaluate_policy(
            policy,
            simulation,
            costs,
            seeds=training_seeds,
        )
        trial.set_user_attr("mean_wait_minutes", evaluation.mean_wait.mean)
        trial.set_user_attr("p90_wait_minutes", evaluation.p90_wait.mean)
        return evaluation.objective.mean

    study.optimize(objective, n_trials=n_trials)
    best_policy = StaffingPolicy(
        tuple(int(study.best_params[f"staff_block_{index + 1}"]) for index in range(simulation.n_blocks))
    )
    training_evaluation = evaluate_policy(
        best_policy,
        simulation,
        costs,
        seeds=training_seeds,
    )
    return OptimizationResult(
        best_policy=best_policy,
        training_evaluation=training_evaluation,
        training_seeds=training_seeds,
        study=study,
    )


def exhaustive_search(
    simulation: SimulationConfig,
    costs: CostConfig,
    bounds: SearchBounds,
    *,
    seeds: tuple[int, ...],
) -> PolicyEvaluation:
    """Enumerate a finite integer search space; intended only for small benchmarks."""

    bounds.validate_for(simulation)
    if not seeds:
        raise ValueError("At least one replication seed is required.")

    ranges = [range(lower, upper + 1) for lower, upper in zip(bounds.lower, bounds.upper, strict=True)]
    best: PolicyEvaluation | None = None
    for staffing in product(*ranges):
        evaluation = evaluate_policy(
            StaffingPolicy(tuple(int(value) for value in staffing)),
            simulation,
            costs,
            seeds=seeds,
        )
        if best is None or evaluation.objective.mean < best.objective.mean:
            best = evaluation

    if best is None:
        raise RuntimeError("The exhaustive search space was unexpectedly empty.")
    return best

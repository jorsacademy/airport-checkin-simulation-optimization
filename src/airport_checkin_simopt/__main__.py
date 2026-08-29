from __future__ import annotations

import argparse
import json

from .evaluation import CostConfig, evaluate_policy
from .model import SimulationConfig
from .optimization import SearchBounds, make_replication_seeds, optimize_staffing


def _summary(evaluation):
    return {
        "policy": list(evaluation.policy.staff_by_block),
        "objective_mean": evaluation.objective.mean,
        "objective_ci95": [evaluation.objective.ci_low, evaluation.objective.ci_high],
        "mean_wait_minutes": evaluation.mean_wait.mean,
        "p90_wait_minutes": evaluation.p90_wait.mean,
        "mean_overtime_minutes": evaluation.overtime.mean,
        "mean_passenger_count": evaluation.passenger_count.mean,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize synthetic airport check-in staffing with simulation optimization."
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--replications", type=int, default=12)
    parser.add_argument("--validation-replications", type=int, default=100)
    parser.add_argument("--min-staff", type=int, default=2)
    parser.add_argument("--max-staff", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sampler", choices=("tpe", "random"), default="tpe")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    simulation = SimulationConfig()
    costs = CostConfig()
    bounds = SearchBounds.uniform(simulation.n_blocks, args.min_staff, args.max_staff)

    result = optimize_staffing(
        simulation,
        costs,
        bounds,
        n_trials=args.trials,
        n_replications=args.replications,
        seed=args.seed,
        sampler=args.sampler,
    )

    validation_seeds = make_replication_seeds(args.seed + 1_000_003, args.validation_replications)
    validation = evaluate_policy(
        result.best_policy,
        simulation,
        costs,
        seeds=validation_seeds,
    )

    print(
        json.dumps(
            {
                "sampler": args.sampler,
                "n_trials": args.trials,
                "training": _summary(result.training_evaluation),
                "validation": _summary(validation),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

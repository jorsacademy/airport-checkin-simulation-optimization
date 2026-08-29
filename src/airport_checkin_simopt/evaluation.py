from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np
from scipy.stats import t

from .model import DayMetrics, SimulationConfig, StaffingPolicy, simulate_day


@dataclass(frozen=True)
class CostConfig:
    """Economic weights used to turn simulation outputs into one objective value."""

    regular_staff_cost_per_hour: float = 32.0
    overtime_staff_cost_per_hour: float = 48.0
    waiting_cost_per_passenger_minute: float = 0.75
    p90_wait_target_minutes: float = 20.0
    p90_excess_penalty_per_minute: float = 40.0

    def __post_init__(self) -> None:
        values = (
            self.regular_staff_cost_per_hour,
            self.overtime_staff_cost_per_hour,
            self.waiting_cost_per_passenger_minute,
            self.p90_excess_penalty_per_minute,
        )
        if any(value < 0 for value in values):
            raise ValueError("Cost parameters cannot be negative.")
        if self.p90_wait_target_minutes < 0:
            raise ValueError("The p90 wait target cannot be negative.")


@dataclass(frozen=True)
class SummaryStats:
    mean: float
    sample_std: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True)
class DayCost:
    total: float
    regular_staff: float
    waiting: float
    sla_penalty: float
    overtime: float


@dataclass(frozen=True)
class PolicyEvaluation:
    policy: StaffingPolicy
    objective: SummaryStats
    mean_wait: SummaryStats
    p90_wait: SummaryStats
    overtime: SummaryStats
    passenger_count: SummaryStats
    replication_costs: tuple[float, ...]


def summarize(values: Iterable[float], confidence: float = 0.95) -> SummaryStats:
    data = np.asarray(tuple(values), dtype=float)
    if data.size == 0:
        raise ValueError("At least one observation is required.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    mean = float(np.mean(data))
    if data.size == 1:
        return SummaryStats(mean=mean, sample_std=0.0, ci_low=mean, ci_high=mean, n=1)

    sample_std = float(np.std(data, ddof=1))
    critical = float(t.ppf((1.0 + confidence) / 2.0, df=data.size - 1))
    half_width = critical * sample_std / sqrt(data.size)
    return SummaryStats(
        mean=mean,
        sample_std=sample_std,
        ci_low=mean - half_width,
        ci_high=mean + half_width,
        n=int(data.size),
    )


def day_cost(
    metrics: DayMetrics,
    policy: StaffingPolicy,
    costs: CostConfig,
) -> DayCost:
    regular_staff = metrics.staff_minutes / 60.0 * costs.regular_staff_cost_per_hour
    waiting = metrics.total_wait_minutes * costs.waiting_cost_per_passenger_minute
    sla_penalty = (
        max(0.0, metrics.p90_wait_minutes - costs.p90_wait_target_minutes)
        * costs.p90_excess_penalty_per_minute
    )
    overtime = (
        metrics.overtime_minutes
        / 60.0
        * policy.staff_by_block[-1]
        * costs.overtime_staff_cost_per_hour
    )
    total = regular_staff + waiting + sla_penalty + overtime
    return DayCost(
        total=float(total),
        regular_staff=float(regular_staff),
        waiting=float(waiting),
        sla_penalty=float(sla_penalty),
        overtime=float(overtime),
    )


def evaluate_policy(
    policy: StaffingPolicy,
    simulation: SimulationConfig | None = None,
    costs: CostConfig | None = None,
    *,
    seeds: Iterable[int],
    confidence: float = 0.95,
) -> PolicyEvaluation:
    """Estimate expected performance using independent simulation replications."""

    simulation = simulation or SimulationConfig()
    costs = costs or CostConfig()
    policy.validate_for(simulation)
    seed_tuple = tuple(int(seed) for seed in seeds)
    if not seed_tuple:
        raise ValueError("At least one replication seed is required.")

    totals: list[float] = []
    mean_waits: list[float] = []
    p90_waits: list[float] = []
    overtimes: list[float] = []
    passenger_counts: list[float] = []

    for seed in seed_tuple:
        metrics = simulate_day(policy, simulation, seed=seed)
        cost = day_cost(metrics, policy, costs)
        totals.append(cost.total)
        mean_waits.append(metrics.mean_wait_minutes)
        p90_waits.append(metrics.p90_wait_minutes)
        overtimes.append(metrics.overtime_minutes)
        passenger_counts.append(float(metrics.passenger_count))

    return PolicyEvaluation(
        policy=policy,
        objective=summarize(totals, confidence),
        mean_wait=summarize(mean_waits, confidence),
        p90_wait=summarize(p90_waits, confidence),
        overtime=summarize(overtimes, confidence),
        passenger_count=summarize(passenger_counts, confidence),
        replication_costs=tuple(float(value) for value in totals),
    )

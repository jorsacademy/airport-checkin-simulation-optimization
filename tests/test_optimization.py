import math

from airport_checkin_simopt.evaluation import CostConfig
from airport_checkin_simopt.model import SimulationConfig
from airport_checkin_simopt.optimization import (
    SearchBounds,
    exhaustive_search,
    make_replication_seeds,
    optimize_staffing,
)


def test_replication_seeds_are_reproducible_and_nonempty():
    assert make_replication_seeds(7, 5) == make_replication_seeds(7, 5)
    assert len(make_replication_seeds(7, 5)) == 5


def test_exhaustive_search_finds_minimum_staff_when_demand_is_zero():
    simulation = SimulationConfig(
        block_durations_minutes=(30.0, 30.0),
        arrival_rates_per_minute=(0.0, 0.0),
    )
    bounds = SearchBounds(lower=(1, 1), upper=(2, 2))
    result = exhaustive_search(
        simulation,
        CostConfig(),
        bounds,
        seeds=(1, 2),
    )
    assert result.policy.staff_by_block == (1, 1)


def test_optuna_smoke_test_returns_feasible_policy():
    simulation = SimulationConfig(
        block_durations_minutes=(30.0, 30.0),
        arrival_rates_per_minute=(0.25, 0.35),
    )
    bounds = SearchBounds(lower=(1, 1), upper=(3, 3))
    result = optimize_staffing(
        simulation,
        CostConfig(),
        bounds,
        n_trials=5,
        n_replications=3,
        seed=1234,
        sampler="tpe",
    )
    assert all(1 <= value <= 3 for value in result.best_policy.staff_by_block)
    assert math.isfinite(result.training_evaluation.objective.mean)
    assert len(result.study.trials) == 5

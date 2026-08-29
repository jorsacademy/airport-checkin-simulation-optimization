from airport_checkin_simopt.evaluation import CostConfig, evaluate_policy, summarize
from airport_checkin_simopt.model import SimulationConfig, StaffingPolicy


def test_summary_single_observation_has_degenerate_interval():
    result = summarize([3.5])
    assert result.mean == 3.5
    assert result.ci_low == 3.5
    assert result.ci_high == 3.5


def test_policy_evaluation_is_reproducible():
    simulation = SimulationConfig(
        block_durations_minutes=(45.0, 45.0),
        arrival_rates_per_minute=(0.25, 0.30),
    )
    policy = StaffingPolicy((2, 3))
    seeds = (10, 20, 30, 40)
    first = evaluate_policy(policy, simulation, CostConfig(), seeds=seeds)
    second = evaluate_policy(policy, simulation, CostConfig(), seeds=seeds)
    assert first == second


def test_staff_cost_dominates_when_there_is_no_demand():
    simulation = SimulationConfig(
        block_durations_minutes=(60.0,),
        arrival_rates_per_minute=(0.0,),
    )
    costs = CostConfig()
    seeds = (1, 2, 3)
    low = evaluate_policy(StaffingPolicy((1,)), simulation, costs, seeds=seeds)
    high = evaluate_policy(StaffingPolicy((3,)), simulation, costs, seeds=seeds)
    assert low.objective.mean < high.objective.mean

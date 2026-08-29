import pytest

from airport_checkin_simopt.model import (
    SimulationConfig,
    StaffingPolicy,
    generate_day_inputs,
    simulate_day,
)


def test_config_requires_matching_blocks_and_rates():
    with pytest.raises(ValueError):
        SimulationConfig(
            block_durations_minutes=(30.0, 30.0),
            arrival_rates_per_minute=(0.2,),
        )


def test_day_inputs_are_reproducible():
    config = SimulationConfig(
        block_durations_minutes=(60.0, 60.0),
        arrival_rates_per_minute=(0.25, 0.35),
    )
    first = generate_day_inputs(config, seed=17)
    second = generate_day_inputs(config, seed=17)
    assert first == second


def test_simulation_is_reproducible_for_same_seed():
    config = SimulationConfig(
        block_durations_minutes=(60.0, 60.0),
        arrival_rates_per_minute=(0.30, 0.40),
    )
    policy = StaffingPolicy((3, 4))
    assert simulate_day(policy, config, seed=123) == simulate_day(policy, config, seed=123)


def test_more_staff_does_not_increase_total_wait_under_fifo_crn():
    config = SimulationConfig(
        block_durations_minutes=(90.0,),
        arrival_rates_per_minute=(0.90,),
        business_probability=0.0,
        economy_service_mean_minutes=6.0,
        service_time_cv=0.20,
    )
    low = simulate_day(StaffingPolicy((2,)), config, seed=99)
    high = simulate_day(StaffingPolicy((5,)), config, seed=99)
    assert high.total_wait_minutes <= low.total_wait_minutes + 1e-9


def test_zero_demand_is_supported():
    config = SimulationConfig(
        block_durations_minutes=(30.0,),
        arrival_rates_per_minute=(0.0,),
    )
    metrics = simulate_day(StaffingPolicy((2,)), config, seed=1)
    assert metrics.passenger_count == 0
    assert metrics.total_wait_minutes == 0.0

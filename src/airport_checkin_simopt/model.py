from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Literal

import numpy as np
import simpy

PassengerClass = Literal["business", "economy"]


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for one synthetic airport check-in operating day."""

    block_durations_minutes: tuple[float, ...] = (240.0, 240.0, 240.0)
    arrival_rates_per_minute: tuple[float, ...] = (0.45, 0.75, 0.50)
    business_probability: float = 0.18
    business_service_mean_minutes: float = 4.0
    economy_service_mean_minutes: float = 6.0
    service_time_cv: float = 0.35

    def __post_init__(self) -> None:
        if not self.block_durations_minutes:
            raise ValueError("At least one staffing block is required.")
        if len(self.block_durations_minutes) != len(self.arrival_rates_per_minute):
            raise ValueError("Block durations and arrival rates must have the same length.")
        if any(duration <= 0 for duration in self.block_durations_minutes):
            raise ValueError("All block durations must be positive.")
        if any(rate < 0 for rate in self.arrival_rates_per_minute):
            raise ValueError("Arrival rates cannot be negative.")
        if not 0.0 <= self.business_probability <= 1.0:
            raise ValueError("business_probability must be between 0 and 1.")
        if self.business_service_mean_minutes <= 0 or self.economy_service_mean_minutes <= 0:
            raise ValueError("Service-time means must be positive.")
        if self.service_time_cv < 0:
            raise ValueError("service_time_cv cannot be negative.")

    @property
    def n_blocks(self) -> int:
        return len(self.block_durations_minutes)

    @property
    def operating_minutes(self) -> float:
        return float(sum(self.block_durations_minutes))


@dataclass(frozen=True)
class StaffingPolicy:
    """Number of parallel check-in agents assigned to each staffing block."""

    staff_by_block: tuple[int, ...]

    def validate_for(self, config: SimulationConfig) -> None:
        if len(self.staff_by_block) != config.n_blocks:
            raise ValueError("Staffing policy length must match the number of staffing blocks.")
        if any(staff <= 0 for staff in self.staff_by_block):
            raise ValueError("Every staffing block must have at least one agent.")


@dataclass(frozen=True)
class PassengerInput:
    passenger_id: int
    arrival_time: float
    passenger_class: PassengerClass
    service_time: float
    block_index: int


@dataclass(frozen=True)
class DayMetrics:
    passenger_count: int
    mean_wait_minutes: float
    p90_wait_minutes: float
    p95_wait_minutes: float
    total_wait_minutes: float
    mean_time_in_system_minutes: float
    overtime_minutes: float
    staff_minutes: float
    service_minutes_by_block: tuple[float, ...]
    nominal_load_ratio_by_block: tuple[float, ...]


def _lognormal_parameters(mean: float, cv: float) -> tuple[float, float]:
    if cv == 0:
        return log(mean), 0.0
    sigma_squared = log(1.0 + cv * cv)
    sigma = sqrt(sigma_squared)
    mu = log(mean) - 0.5 * sigma_squared
    return mu, sigma


def generate_day_inputs(config: SimulationConfig, seed: int) -> tuple[PassengerInput, ...]:
    """Generate exogenous arrivals and service requirements for one replication.

    All stochastic inputs are generated before the simulation. Reusing the same seed
    across candidate staffing policies therefore implements common random numbers.
    """

    rng = np.random.default_rng(seed)
    business_mu, business_sigma = _lognormal_parameters(
        config.business_service_mean_minutes, config.service_time_cv
    )
    economy_mu, economy_sigma = _lognormal_parameters(
        config.economy_service_mean_minutes, config.service_time_cv
    )

    passengers: list[PassengerInput] = []
    block_start = 0.0
    passenger_id = 0

    for block_index, (duration, rate) in enumerate(
        zip(config.block_durations_minutes, config.arrival_rates_per_minute, strict=True)
    ):
        count = int(rng.poisson(rate * duration))
        arrivals = np.sort(block_start + rng.uniform(0.0, duration, size=count))
        is_business = rng.random(count) < config.business_probability

        for arrival, business in zip(arrivals, is_business, strict=True):
            if business:
                service_time = (
                    config.business_service_mean_minutes
                    if business_sigma == 0
                    else float(rng.lognormal(business_mu, business_sigma))
                )
                passenger_class: PassengerClass = "business"
            else:
                service_time = (
                    config.economy_service_mean_minutes
                    if economy_sigma == 0
                    else float(rng.lognormal(economy_mu, economy_sigma))
                )
                passenger_class = "economy"

            passengers.append(
                PassengerInput(
                    passenger_id=passenger_id,
                    arrival_time=float(arrival),
                    passenger_class=passenger_class,
                    service_time=float(service_time),
                    block_index=block_index,
                )
            )
            passenger_id += 1

        block_start += duration

    passengers.sort(key=lambda passenger: (passenger.arrival_time, passenger.passenger_id))
    return tuple(passengers)


def simulate_day(
    policy: StaffingPolicy,
    config: SimulationConfig | None = None,
    *,
    seed: int = 0,
) -> DayMetrics:
    """Run one SimPy replication for a fixed staffing policy."""

    config = config or SimulationConfig()
    policy.validate_for(config)
    inputs = generate_day_inputs(config, seed)

    env = simpy.Environment()
    resources = [
        simpy.PriorityResource(env, capacity=staff) for staff in policy.staff_by_block
    ]

    waits: list[float] = []
    times_in_system: list[float] = []
    completion_times: list[float] = []
    service_by_block = np.zeros(config.n_blocks, dtype=float)

    def passenger_process(passenger: PassengerInput):
        yield env.timeout(passenger.arrival_time)
        resource = resources[passenger.block_index]
        priority = 0 if passenger.passenger_class == "business" else 1

        with resource.request(priority=priority) as request:
            yield request
            service_start = env.now
            wait = service_start - passenger.arrival_time
            yield env.timeout(passenger.service_time)
            completion = env.now

        waits.append(float(wait))
        times_in_system.append(float(completion - passenger.arrival_time))
        completion_times.append(float(completion))
        service_by_block[passenger.block_index] += passenger.service_time

    for passenger in inputs:
        env.process(passenger_process(passenger))

    env.run()

    staff_minutes = float(
        sum(
            staff * duration
            for staff, duration in zip(
                policy.staff_by_block, config.block_durations_minutes, strict=True
            )
        )
    )

    load_ratios = tuple(
        float(service / (staff * duration))
        for service, staff, duration in zip(
            service_by_block,
            policy.staff_by_block,
            config.block_durations_minutes,
            strict=True,
        )
    )

    if not waits:
        return DayMetrics(
            passenger_count=0,
            mean_wait_minutes=0.0,
            p90_wait_minutes=0.0,
            p95_wait_minutes=0.0,
            total_wait_minutes=0.0,
            mean_time_in_system_minutes=0.0,
            overtime_minutes=0.0,
            staff_minutes=staff_minutes,
            service_minutes_by_block=tuple(float(x) for x in service_by_block),
            nominal_load_ratio_by_block=load_ratios,
        )

    wait_array = np.asarray(waits, dtype=float)
    overtime = max(0.0, max(completion_times) - config.operating_minutes)

    return DayMetrics(
        passenger_count=len(waits),
        mean_wait_minutes=float(np.mean(wait_array)),
        p90_wait_minutes=float(np.quantile(wait_array, 0.90)),
        p95_wait_minutes=float(np.quantile(wait_array, 0.95)),
        total_wait_minutes=float(np.sum(wait_array)),
        mean_time_in_system_minutes=float(np.mean(times_in_system)),
        overtime_minutes=float(overtime),
        staff_minutes=staff_minutes,
        service_minutes_by_block=tuple(float(x) for x in service_by_block),
        nominal_load_ratio_by_block=load_ratios,
    )

from airport_checkin_simopt import CostConfig, SearchBounds, SimulationConfig, evaluate_policy, optimize_staffing
from airport_checkin_simopt.optimization import make_replication_seeds


simulation = SimulationConfig()
costs = CostConfig()
bounds = SearchBounds.uniform(simulation.n_blocks, 2, 12)

result = optimize_staffing(
    simulation,
    costs,
    bounds,
    n_trials=40,
    n_replications=10,
    seed=2026,
    sampler="tpe",
)

validation_seeds = make_replication_seeds(9001, 100)
validation = evaluate_policy(
    result.best_policy,
    simulation,
    costs,
    seeds=validation_seeds,
)

print("Best staffing policy:", result.best_policy.staff_by_block)
print("Training objective:", round(result.training_evaluation.objective.mean, 2))
print("Validation objective:", round(validation.objective.mean, 2))
print("Validation mean wait:", round(validation.mean_wait.mean, 2), "minutes")
print("Validation p90 wait:", round(validation.p90_wait.mean, 2), "minutes")

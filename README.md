# Airport Check-in Simulation Optimization

A reproducible simulation-optimization case study for airport check-in staffing. The project couples a discrete-event simulation built with SimPy to black-box optimization with Optuna. It is intended as an Operations Research / Industrial Engineering example of optimizing decisions whose performance can only be estimated through stochastic simulation.

## Problem

Let

- `x = (x1, x2, ..., xB)` be the number of check-in agents assigned to each staffing block,
- `xi` denote a random operating day containing passenger arrivals, classes, and service times,
- `C(x, xi)` denote the simulated operating cost for policy `x` under day `xi`.

The computational problem is

```text
minimize    E[C(x, xi)]
subject to  L_b <= x_b <= U_b
            x_b integer
```

The expectation is not available analytically. It is estimated with independent simulation replications.

## What the model contains

The synthetic airport model includes:

- a finite operating day split into staffing blocks,
- piecewise-Poisson passenger arrivals,
- economy and business passenger classes,
- lognormal service times,
- non-preemptive priority for business passengers,
- parallel check-in agents represented by `simpy.PriorityResource`,
- staffing cost, passenger waiting cost, an SLA penalty on the 90th percentile of waiting time, and overtime cost.

The parameters are intentionally synthetic. This repository demonstrates methodology and software structure; it is not a calibrated model of a specific airport.

## Simulation-optimization design

The important feature is not simply that simulation and optimization both appear in the repository. The optimizer repeatedly proposes a staffing policy, the simulator estimates its performance, and that estimate is returned to the optimizer.

Two experimental-design details are implemented explicitly:

1. **Common random numbers (CRN).** During optimization, every candidate policy is evaluated on the same fixed replication seeds. Exogenous passenger arrivals, classes, and service requirements are generated before the SimPy run. This produces paired comparisons across candidate policies and typically reduces comparison noise.
2. **Independent out-of-sample validation.** The policy selected by the optimizer should be reevaluated on a new seed set that was not used during search. The CLI does this automatically.

The default Optuna sampler is `TPESampler`. TPE is a model-based black-box search method; it does not provide a proof of global optimality. For small integer search spaces, `exhaustive_search` is included as a benchmark.

## Installation

```bash
python -m pip install -e ".[dev]"
```

The project targets Python 3.10+.

## Run the optimizer

```bash
python -m airport_checkin_simopt \
  --trials 50 \
  --replications 12 \
  --validation-replications 100 \
  --min-staff 2 \
  --max-staff 12 \
  --sampler tpe \
  --seed 2026
```

The command prints JSON containing the selected staffing policy, the in-sample estimate used during search, and an independent validation estimate with a 95% t confidence interval.

A smaller script is also available:

```bash
python examples/run_experiment.py
```

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── examples/run_experiment.py
├── src/airport_checkin_simopt/
│   ├── __init__.py
│   ├── __main__.py
│   ├── evaluation.py
│   ├── model.py
│   └── optimization.py
├── tests/
│   ├── test_evaluation.py
│   ├── test_model.py
│   └── test_optimization.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Testing

```bash
python -m pytest
```

The tests cover reproducible random inputs, simulation reproducibility, a monotonic FIFO staffing sanity check, zero-demand behavior, confidence-interval handling, deterministic evaluation under a fixed seed set, exhaustive-search correctness on a tiny case, and an Optuna integration smoke test.

GitHub Actions runs the suite on Python 3.10 and 3.12.

## Methodological notes

- Repeating a simulation several times and averaging does not make the resulting policy "robust" in the formal robust-optimization sense.
- A noisy optimizer result should not be reported without a fresh validation experiment.
- TPE is useful when simulation evaluations are expensive, but it is not universally superior to random search, stochastic approximation, ranking-and-selection procedures, or problem-specific simulation-optimization algorithms.
- The current confidence interval summarizes replication-level sampling error for a fixed policy. It does not quantify all selection bias introduced by choosing the policy after optimization.
- The staffing blocks use independent resource pools. Passengers are assigned to the pool corresponding to their arrival block; this is a deliberately transparent shift model rather than a detailed labor-rostering model.

## References

- N. Jian and S. G. Henderson, *An Introduction to Simulation Optimization*, Winter Simulation Conference, 2015: https://people.orie.cornell.edu/shane/pubs/WSC2015Tut.pdf
- M. C. Fu, F. W. Glover, and J. April, *Simulation Optimization: A Review, New Developments, and Applications*, Winter Simulation Conference, 2005: https://informs-sim.org/wsc05papers/010.pdf
- D. J. Eckman, S. G. Henderson, and S. Shashaani, *SimOpt: A Testbed for Simulation-Optimization Experiments*, INFORMS Journal on Computing, 2023: https://doi.org/10.1287/ijoc.2023.1273
- SimPy documentation: https://simpy.readthedocs.io/
- Optuna documentation: https://optuna.readthedocs.io/

## License

MIT

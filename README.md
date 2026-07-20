# actgpr

**Active GPR (Gaussian Process Regression) Optimisation** — a Python package that finds the minimum of an expensive-to-evaluate scalar function by iteratively fitting a Gaussian Process surrogate and using Expected Improvement to pick the most informative next evaluation point.

## How it works

1. Evaluate the Objective at the initial input points.
2. Repeat:
   - Fit the Surrogate to all training data collected so far.
   - Maximise the Acquisition function (Expected Improvement) → choose the next input point.
   - Evaluate the Objective at that point.
3. Stop when the maximum EI score falls below `ei_threshold` (nothing left to gain) **or** the number of optimisation iterations reaches `max_evaluations` (budget cap) — whichever fires first.
4. Optionally, every run writes a complete reproducibility record (MRR — see below).

## Installation

Requires Python ≥ 3.13 and [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/LxdrScr/actgpr.git
cd actgpr
poetry install
```

## Quick start

```python
from actgpr import ObjectiveFn, OptimisationRun, GPyTorchSurrogate

run = OptimisationRun.with_training(
    objective=ObjectiveFn(lambda x: (x - 1) ** 2),
    surrogate=GPyTorchSurrogate(),
    search_bounds=(-3.0, 5.0),
    initial_train_x=[-2.0, 4.0],
    max_evaluations=20,
    ei_threshold=0.001,
    run_dir="results",           # optional: write the MRR record
)
result = run.run()
print(result["best_x"], result["best_y"])
```

Expected output: `best_x` close to `1.0` and `best_y` close to `0.0` (the minimum of `(x − 1)²`). The result dict also contains `train_x`, `train_y`, `n_iterations`, and `stop_reason`.

**Fit modes** — the two constructors select how GP hyperparameters are handled:

- `OptimisationRun.with_training(...)` — lengthscale, outputscale, and noise are optimised each iteration (Adam on the marginal log likelihood, `training_iter` steps).
- `OptimisationRun.without_training(...)` — hyperparameters are fixed to user-supplied values; no optimisation.

Set `store_snapshots=True` to browse the GP and EI state of every iteration afterwards with `run.plot_iterations()` (interactive slider).

## Run outputs (MRR)

When `run_dir` is given, each run creates a timestamped **run directory** (named from timestamp + key parameters) containing the five **MRR artifacts**:

| Artifact | Contents |
|---|---|
| `config.json` | All run parameters (written at start — survives crashes) |
| `manifest.json` | SHA-256 checksum of the inputs |
| `meta.json` | Environment: git commit, Python/library versions, platform, timestamps, output summary |
| `run.log` | Per-iteration audit trail |
| `results.h5` | Self-describing HDF5 with all numerical results |

`results.h5` layout:

```
/            attrs: run configuration
├── history/     per-iteration scalar series (iteration, next_point, new_y,
│                current_best, max_ei, prediction_error, improvement)
├── iterations/  iter_NNN/ GP snapshot arrays (only with store_snapshots=True)
└── final/       best_x, best_y, stop_reason, n_iterations + final train_x/train_y
```

## Vocabulary

### The optimisation problem

| Term | Meaning |
|---|---|
| **Objective** | The real-valued scalar function being minimised. Wrapped by `ObjectiveFn`; defaults to `f(x) = x²`. |
| **Analytic objective** | An Objective computed by a mathematical formula (e.g. `x²`) — used for development and testing. |
| **Experiment objective** | An Objective whose output comes from a real-world measurement or instrument (planned). |
| **`train_x`** (or `x`) | The input points passed to the Objective. |
| **`train_y`** (or `y`) | The Objective outputs at those input points. |
| **`test_x`** | Input points where the surrogate predicts without evaluating the Objective. |
| **Training data** | The set of `(train_x, train_y)` pairs the GP model is fitted to. |
| **Search bounds** | The closed interval `[lo, hi]` within which input points are considered. |
| **`initial_train_x`** | The input points that seed the optimisation loop. |

### The surrogate (GP model)

| Term | Meaning |
|---|---|
| **Surrogate** | A Gaussian Process model fitted to all training data so far, used to predict the Objective cheaply at unevaluated points. |
| **`GPyTorchSurrogate`** | The surrogate backend wrapper (fitting + prediction) built on [GPyTorch](https://gpytorch.ai/); hides GPyTorch API details. |
| **`ExactGPModel`** | The GP model definition inside the wrapper: constant mean + scaled RBF kernel. |
| **Prior / posterior** | The GP distribution before / after conditioning on the training data. |
| **Likelihood** | The Gaussian noise model mapping latent function values to observed targets. |
| **Kernel (RBF)** | The covariance function: a radial-basis-function kernel wrapped in a scale kernel. |
| **`lengthscale`** | RBF kernel hyperparameter — how far correlations reach (smoothness). |
| **`outputscale`** | Kernel signal variance. |
| **`noise`** | Observation noise variance of the likelihood. |
| **MLL** | Marginal log likelihood — the training objective maximised when fitting hyperparameters. |
| **Cholesky jitter** | Small value (`1e-4`) added to the covariance diagonal to keep it numerically positive definite; all computations use float64. |
| **`f_mean`** | Predicted posterior mean at given input points. |
| **`f_var`** | Predicted posterior variance (per-point uncertainty), shape `(m,)`. |
| **`f_covar`** | Full posterior covariance matrix, shape `(m, m)`. |
| **`f_preds`** | Predictive distribution of the latent function `f(test_x)`. |
| **`observed_pred`** | Predictive distribution of observed targets `y = f(x) + noise`. |
| **`f_samples`** | Samples drawn from the predictive posterior (only computed when `n_samples > 0`). |
| **`f_std`** | `sqrt(f_var)` — used inside EI and for the ±2σ (≈95 % CI) plot band. |

### The acquisition function

| Term | Meaning |
|---|---|
| **Acquisition function** | Scores candidate input points and selects the next input point to evaluate. |
| **Expected Improvement (EI)** | The closed-form acquisition score (Jones et al., 1998) balancing exploitation (confidently better mean) and exploration (high uncertainty). |
| **Candidates / `n_candidates`** | The evenly spaced grid of points within the search bounds that EI scores (default 500). "Candidates" refers only to this acquisition grid — never to training data. |
| **`ei_scores`** | The EI value of every candidate. |
| **`max_ei`** | The largest EI score in an iteration; compared against `ei_threshold` for convergence. |
| **`next_point`** | The candidate with the highest EI — the next input point to evaluate. |
| **Current best** | The smallest Objective value observed so far. |

### The optimisation loop

| Term | Meaning |
|---|---|
| **`OptimisationRun`** | Top-level orchestrator: owns the loop and all MRR writes. |
| **Fit mode** | `with_training` (hyperparameters optimised each iteration) vs. `without_training` (fixed); recorded as `"training"` / `"notraining"` in `config.json`. |
| **`max_evaluations`** | Budget cap: the maximum number of active optimisation iterations (GPR fit cycles) — not individual Objective calls. |
| **`ei_threshold`** | Convergence threshold: the loop stops when `max_ei` falls below it. |
| **Convergence criterion** | EI below threshold **or** budget reached — whichever fires first. |
| **`stop_reason`** | Which criterion fired: `"ei_threshold"` or `"max_evaluations"`. |
| **`new_y`** | The Objective output at the newly evaluated `next_point`. |
| **`best_x` / `best_y`** | The input point with the lowest Objective output, and that output — the final result. |
| **`store_snapshots`** | If `True`, each iteration's GP + EI state is kept for interactive browsing via `plot_iterations()`. |
| **Deferred-write accumulator** | Per-iteration results are collected in memory during the run and written to `results.h5` once at the end. |

### Validation metrics

Computed every iteration and recorded in `run.log`, `results.h5` (`/history`), and the snapshot plot titles:

| Term | Meaning |
|---|---|
| **`prediction_error`** | `predicted_y − new_y`: the surrogate's signed error at the chosen point. |
| **`improvement`** | `max(0, current_best − new_y)`: the gain of this iteration's evaluation over the previous best. |

### Reproducibility (MRR)

| Term | Meaning |
|---|---|
| **MRR** | Minimal Reproducible Run — a pattern requiring every run to record: what was run, with what inputs, in which environment, what happened, and what came out. |
| **Run directory** | The timestamped folder under `run_dir` holding all MRR artifacts of a single run. |
| **Self-describing HDF5** | Configuration is stored as HDF5 attributes alongside the data, so `results.h5` can be understood without any other file. |

## Development

```bash
poetry run pytest tests/            # all tiers: unit, integration, regression
poetry run black src/ tests/        # format
poetry run ruff check src/ tests/   # lint
poetry run sphinx-build -W docs docs/build/html   # API docs (warnings = errors)
```

The regression tier compares a fixed-seed run against `tests/regression/data/quadratic_baseline.csv`; the test module documents how to regenerate the baseline after an intentional behaviour change.

## License

MIT — see [LICENSE](LICENSE).

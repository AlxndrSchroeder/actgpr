# actgpr

**Active GPR (Gaussian Process Regression) Optimisation** — a Python package that finds the minimum of a scalar blackbox function by iteratively fitting a Gaussian Process surrogate and using Expected Improvement to pick the most informative next evaluation point.

The Gaussian Process surrogate is built on [GPyTorch](https://gpytorch.ai/); the Expected Improvement acquisition function follows [Jones, Schonlau & Welch (1998), *Efficient Global Optimization of Expensive Black-Box Functions*](https://doi.org/10.1023/A:1008306431147).

**Documentation:** [alxndrschroeder.github.io/actgpr](https://alxndrschroeder.github.io/actgpr/) — full API reference and a step-by-step tutorial, built from this repository's docstrings and reST sources with Sphinx.

## How it works

1. Evaluate the Objective at the initial input points.
2. Repeat:
   - Fit the Surrogate to all training data collected so far.
   - Maximise the Acquisition function (Expected Improvement) → choose the next input point.
   - Evaluate the Objective at that point.
3. Stop when the maximum EI score falls below `ei_threshold` (nothing left to gain) **or** the number of optimisation iterations reaches `max_iterations` (budget cap) — whichever fires first.
4. Optionally, every run writes a complete reproducibility record (MRR — see below).

## Installation

Requires Python ≥ 3.13 and [Poetry](https://python-poetry.org/) ≥ 2.0 (the project uses the PEP 621 `pyproject.toml` format, which Poetry 1.x cannot read). All dependency versions are pinned in `poetry.lock`, so `poetry install` reproduces the exact environment.

```bash
git clone https://github.com/AlxndrSchroeder/actgpr.git
cd actgpr
poetry install
```

## Quick start

The usage pattern:

1. **Write an Objective wrapper for your blackbox function** — `ObjectiveFn` turns any `Callable[[float], float]` into an Objective with `objective.evaluate(*x) -> tuple[float, ...]`.
2. **Choose the search interval** — `search_bounds` is the closed interval `[lo, hi]` in which the algorithm searches for the minimum.
3. **Hand both to an `OptimisationRun`** and call `run()`.

```python
from actgpr import ObjectiveFn, OptimisationRun, GPyTorchSurrogate


# 1. Your blackbox function. Here an analytic stand-in for the tutorial —
#    in practice it might run a simulation or trigger an experiment.
def my_blackbox(x: float) -> float:
    return (x - 1) ** 2


# 2. Wrap it in an Objective. Conceptually, the wrapper is as small as:
#
#        class ObjectiveFn:
#            def __init__(self, func):
#                self.func = func
#
#            def evaluate(self, *x: float) -> tuple[float, ...]:
#                return tuple(float(self.func(v)) for v in x)
#
objective = ObjectiveFn(my_blackbox)

# 3. Configure and execute the optimisation run
run = OptimisationRun.with_training(
    objective=objective,            # the wrapped blackbox to minimise
    surrogate=GPyTorchSurrogate(),  # the GP model that approximates it
    search_bounds=(-3.0, 5.0),      # interval in which the minimum is searched
    initial_train_x=[-3.0, 5.0],    # points where we start looking for the minimum
    max_iterations=20,              # budget: max optimisation iterations
    ei_threshold=0.001,             # stop early once max EI drops below this
    noise=1e-4,                     # starting observation-noise variance
                                    # (tuned further during training)
    run_dir="results",              # optional: write the MRR record
)
result = run.run()
print(result["best_x"], result["best_y"])
```

Expected output: `best_x` close to `1.0` and `best_y` close to `0.0` (the minimum of `(x − 1)²`). The result dict also contains `train_x`, `train_y`, `n_iterations`, and `stop_reason`.

**Fit modes** — the two constructors select how GP hyperparameters are handled:

- `OptimisationRun.with_training(...)` — lengthscale, outputscale, and noise are re-tuned at every iteration using [Adam](https://arxiv.org/abs/1412.6980) (`torch.optim.Adam`), a gradient-descent variant with momentum and per-parameter step sizes: over `training_iter` steps it adjusts the hyperparameters to maximise the marginal log likelihood — how plausible the observed training data is under a GP with those hyperparameters. Adam only fits the surrogate; it never evaluates the blackbox. Use this mode when you do not know good hyperparameters — the usual case.
- `OptimisationRun.without_training(...)` — hyperparameters stay fixed at exactly the values you pass; nothing is tuned. Use this for controlled comparisons or when good values are already known:

```python
run = OptimisationRun.without_training(
    objective=objective,            # the wrapped blackbox to minimise
    surrogate=GPyTorchSurrogate(),  # the GP model that approximates it
    search_bounds=(-3.0, 5.0),      # interval in which the minimum is searched
    initial_train_x=[-3.0, 5.0],    # points where we start looking for the minimum
    max_iterations=20,              # budget: max optimisation iterations
    ei_threshold=0.001,             # stop early once max EI drops below this
    lengthscale=1.0,                # RBF kernel lengthscale (fixed)
    outputscale=1.0,                # kernel signal variance (fixed)
    noise=1e-4,                     # observation-noise variance (fixed)
)
result = run.run()
```

Set `store_snapshots=True` to browse the GP and EI state of every iteration afterwards with `run.plot_iterations()` (interactive slider). The `prediction_error`/`improvement` history used by `plot_run_history()` is recorded either way, regardless of this flag.

EI often shrinks by orders of magnitude as a run converges — enough to look like a flat line at zero on a linear axis. Pass `log_scale=True` to `plot_iterations()` to keep that shrinkage visible; `ei_threshold` is drawn as a reference line so you can see the EI curve cross into converged territory.

## Run outputs (MRR)

When `run_dir` is given, each run creates a timestamped **run directory** (named from timestamp + key parameters) containing the five **MRR artifacts**:

| Artifact | Contents |
|---|---|
| `config.json` | All run parameters (written at start — survives crashes) |
| `manifest.json` | SHA-256 checksum of the inputs |
| `meta.json` | Environment: package name/version, repository, git commit, Python/library versions, platform, timestamps, output summary |
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

To visualise a past run, `plot_run_history(run_dir)` builds a plot of `prediction_error` and `improvement` vs. iteration straight from a run directory's `results.h5` — no `OptimisationRun` object needed, so it works on any run you (or someone else) have on disk:

```python
from actgpr.plotting import plot_run_history

plot_run_history("results/2026-07-20_212046_training50iter_ei0.001_maxiter20_n0.0002")
```

## Vocabulary

### The optimisation problem

| Term | Meaning |
|---|---|
| **Objective** | The real-valued scalar function being minimised — your blackbox function, wrapped by `ObjectiveFn`. Defaults to `f(x) = x²` (handy for tutorials and tests). |
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
| **`max_iterations`** | Budget cap: the maximum number of active optimisation iterations (GPR fit cycles) — not individual Objective calls. |
| **`ei_threshold`** | Convergence threshold: the loop stops when `max_ei` falls below it. |
| **Convergence criterion** | EI below threshold **or** budget reached — whichever fires first. |
| **`stop_reason`** | Which criterion fired: `"ei_threshold"` or `"max_iterations"`. |
| **`new_y`** | The Objective output at the newly evaluated `next_point`. |
| **`best_x` / `best_y`** | The input point with the lowest Objective output, and that output — the final result. |
| **`store_snapshots`** | If `True`, each iteration's full GP + EI state is also kept for interactive browsing via `plot_iterations()`. The `prediction_error`/`improvement` history used by `plot_run_history()` is recorded regardless of this flag. |
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
| **`plot_run_history()`** | Builds the `prediction_error`/`improvement` plot from a run directory's `results.h5` alone — no `OptimisationRun` object needed. |

## Development

```bash
poetry run pytest tests/            # all tiers: unit, integration, regression
poetry run black src/ tests/        # format
poetry run ruff check src/ tests/   # lint
poetry run sphinx-build -W docs docs/build/html   # API docs (warnings = errors)
```

The regression tier compares a fixed-seed run against `tests/regression/data/quadratic_baseline.csv`; the test module documents how to regenerate the baseline after an intentional behaviour change.

Pushing to `main` rebuilds and republishes the docs above via GitHub Pages (see `.github/workflows/ci.yml`), so the local `sphinx-build` command is for previewing changes before they merge.

## Security

Found a vulnerability? See [SECURITY.md](SECURITY.md) for how to report it privately.

## License

MIT — see [LICENSE](LICENSE).

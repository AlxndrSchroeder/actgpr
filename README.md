# actgpr

[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/13772/badge)](https://bestpractices.coreinfrastructure.org/projects/13772)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B%20%20%E2%97%8F%20%20%E2%97%8F-yellow)](https://fair-software.eu)

**Active GPR (Gaussian Process Regression) Optimisation** is a Python package that finds the minimum of a blackbox function with one or more inputs and a single output by iteratively fitting a Gaussian Process surrogate and using Expected Improvement to pick the most informative next evaluation point.

The Gaussian Process surrogate is built on [GPyTorch](https://gpytorch.ai/); the Expected Improvement acquisition function follows [Jones, Schonlau & Welch (1998), *Efficient Global Optimization of Expensive Black-Box Functions*](https://doi.org/10.1023/A:1008306431147).

**Documentation:** [alxndrschroeder.github.io/actgpr](https://alxndrschroeder.github.io/actgpr/) gives the full API reference and a step by step tutorial, built from this repository's docstrings and reST sources with Sphinx.

## How it works

1. Evaluate the Objective at the initial input points.
2. Repeat:
   - Fit the Surrogate to all training data collected so far.
   - Maximise the Acquisition function (Expected Improvement) → choose the next input point.
   - Evaluate the Objective at that point.
3. Stop when the maximum EI score falls below `ei_threshold` (nothing left to gain) **or** the number of optimisation iterations reaches `max_iterations` (budget cap), whichever fires first.
4. Optionally, every run writes a complete reproducibility record (MRR, see below).

## Installation

Requires Python ≥ 3.13. Pick whichever ecosystem you already use.

> **Planned:** publishing `actgpr` to PyPI. Until then, use one of the two paths below.

**With [Poetry](https://python-poetry.org/)** (≥ 2.0). Versions are pinned in `poetry.lock`:

```bash
git clone https://github.com/AlxndrSchroeder/actgpr.git
cd actgpr
poetry install
```

**With [conda-lock](https://github.com/conda/conda-lock)**. Versions are pinned in `conda-lock.yml`, solved for `linux-64`, `osx-64`, `osx-arm64`, and `win-64`:

```bash
git clone https://github.com/AlxndrSchroeder/actgpr.git
cd actgpr
conda-lock install --name actgpr conda-lock.yml
conda activate actgpr
pip install -e . --no-deps          # the package itself; deps come from the lock file
```

`environment.yml` and `pyproject.toml` hold version *ranges*; `conda-lock.yml` and `poetry.lock` hold the *resolved* versions. Install from the lock files.

## Quick start

The usage pattern:

1. **Give it an Objective.** Anything exposing `.evaluate(x1, ..., xd) -> float`: it receives the coordinates of **one** input point and returns that point's **one** output. There are two routes, described under [Writing your Objective](#writing-your-objective) below.
2. **Configure the run.** Beyond the Objective and Surrogate you set `search_bounds` (one `(lo, hi)` interval per input; their number decides how many inputs the problem has), `initial_train_x` (the points that seed the loop, one row per point), `max_iterations` (budget cap), `ei_threshold` (early stopping threshold), `noise` (surrogate observation noise variance), and optionally `run_dir` (where to write the MRR record). See the parameter tables in the [tutorial](https://alxndrschroeder.github.io/actgpr/tutorial.html) for the full list.
3. **Execute** with `run()`, which returns `best_x` (a tuple with one coordinate per input) and `best_y` along with the full training data.
4. **Inspect** the run with `plot_metrics()` (any number of inputs) or `plot_iterations()` (one input).

```python
from actgpr import ObjectiveFn, OptimisationRun, GPyTorchSurrogate


# 1. Your blackbox function. Here an analytic stand-in.
#    In practice it might run a simulation.
def my_blackbox(x: float) -> float:
    return (x - 1) ** 2


# 2. Wrap it in an Objective. ObjectiveFn(func) is a convenience for plain
#    functions like this one; a real simulation would instead be its own
#    class with an evaluate() method.
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

# 4. Browse the surrogate iteration by iteration (EI axis is log-scaled
#    by default; pass log_scale=False for a linear one). This figure draws
#    the surrogate as a curve, so it is for problems with one input.
run.plot_iterations()
```

Expected output: `best_x` close to `(1.0,)` and `best_y` close to `0.0` (the minimum of `(x − 1)²`). `best_x` is a tuple even with one input, so the same code reads it for any number of inputs. The result dict also contains `train_x` (shape `(n, d)`, one row per evaluated point), `train_y`, `n_iterations`, and `stop_reason`.

### More than one input

Give one `(lo, hi)` pair per input, and one row per starting point. Your function takes one argument per input:

```python
from actgpr import ObjectiveFn, OptimisationRun, GPyTorchSurrogate


def my_blackbox(x1: float, x2: float, x3: float) -> float:
    return (x1 - 0.5) ** 2 + (x2 + 1.0) ** 2 + 0.2 * (x3 - 2.0) ** 2


run = OptimisationRun.with_training(
    objective=ObjectiveFn(my_blackbox),
    surrogate=GPyTorchSurrogate(),
    search_bounds=[(-2.0, 2.0), (-3.0, 1.0), (0.0, 5.0)],  # x1, x2, x3
    initial_train_x=[[-1.0, 0.0, 1.0], [1.5, -2.5, 4.0]],  # two 3D points
    max_iterations=20,
    ei_threshold=1e-4,
    run_dir="results",
)
result = run.run()

x1, x2, x3 = result["best_x"]  # close to (0.5, -1.0, 2.0)
run.plot_metrics()
```

The one rule to remember: **the number of `search_bounds` pairs is the number of inputs**, and everything else has to agree with it. Mismatches fail immediately with a message saying what was expected:

| You pass | What happens |
|---|---|
| `search_bounds=(-3.0, 5.0)` | A single pair is shorthand for one input |
| `initial_train_x=[-3.0, 5.0]` with one input | Accepted: a flat list is two 1D points |
| `initial_train_x=[0.0, 1.0]` with two inputs | `ValueError`: pass one row per point, `[[0.0, 1.0]]` |
| a point with 3 coordinates for a 2-input run | `ValueError`: expected 2 coordinates per point |
| `lengthscale=2.0` (`without_training`) | Used for every input |
| `lengthscale=[0.5, 3.0]` | One value per input; the count must match |

With several inputs the candidates are drawn from a seeded Sobol sequence instead of an evenly spaced grid, since a grid of 500 points per input would be 125 million points in 3D. `n_candidates` is then the number of points spread across all inputs, so raise it for problems with many inputs, but not without limit: prediction cost grows faster than linearly with it (on a laptop about 0.04 s per stage at 2048 candidates and 4 s at 16384, and there are two stages per iteration), so a few thousand is a sensible ceiling.

### Example output

`run.plot_iterations()` opens an interactive slider over a finished run: the GP fit on top, the EI landscape that picked each next point below. Here on `sin(x) + x²/40` over `[-16, 16]`, a harder objective with several local minima:

<img src="assets/plot_iterations_demo.gif" width="500" alt="Per-iteration GP fit and EI landscape for sin(x) + x^2/40, converging on the minimum">

The blue band is the GP's 95% confidence interval, wide where the objective has not been evaluated and pinched shut at each training point. Watching it narrow around `x = -1.5` is watching the surrogate become certain. It converges after 17 iterations via `ei_threshold` at `best_x = -1.4965`, within `5.5e-4` of the true minimum, after 18 evaluations.

`run.plot_metrics()` summarises the same run as four metrics against iteration:

<img src="assets/plot_metrics_demo.png" width="700" alt="current_best, improvement, max_ei and prediction_error against iteration for the same run">

`current_best` steps down then flattens as the minimum is found and confirmed; `max_ei` falls three orders of magnitude until it crosses `ei_threshold` and stops the run. The [tutorial](https://alxndrschroeder.github.io/actgpr/tutorial.html) gives the exact configuration behind both figures and reads each panel in detail.

### Writing your Objective

`actgpr` never inspects the type of your Objective. It only ever calls `.evaluate(...)`, so there is no base class to inherit and nothing to register. That is what lets you plug in a simulation of your own without adapting it to `actgpr`.

- **A plain function?** Wrap it in `ObjectiveFn`, as in the quick start above. Pass `jitter=` to simulate a noisy instrument.
- **Your own simulation,** with setup, configuration, or state? Write a class with an `evaluate` method taking one argument per input and returning one float, and pass it directly. Do *not* wrap it in `ObjectiveFn`:

```python
class MySimulation:
    def evaluate(self, pressure: float, temperature: float) -> float:
        return ...  # run the simulation at this one input point
```

`evaluate` is called once per input point. It never receives several points at once, and it must return a single number.

`OptimisationRun` treats both identically, and `config.json` records whichever you used via its `repr()`. The [tutorial](https://alxndrschroeder.github.io/actgpr/tutorial.html) has a worked example of each, plus the two fit modes (`with_training` retunes the GP hyperparameters every iteration; `without_training` holds them fixed) and the full parameter reference.

### Coming from actgpr 0.3

One-input code that uses `OptimisationRun`, `ObjectiveFn` and the four plotting functions needs at most these changes:

| 0.3 | Now |
|---|---|
| `evaluate(x_a, x_b)` evaluated two points and returned `(y_a, y_b)`; `evaluate(x)` returned `(y,)` | `evaluate(x1, x2)` evaluates **one** 2D point, and `evaluate(x)` one 1D point; both return a plain `float`. A class of your own that still returns a tuple fails with a `TypeError` naming this change |
| `result["best_x"]` was a `float` | It is a tuple: use `result["best_x"][0]`, or unpack it, `(best_x,) = result["best_x"]` |
| `run.search_bounds` was `(lo, hi)` | It is one pair per input, `((lo, hi),)` |

The constructor calls themselves, including `search_bounds=(lo, hi)` and a flat `initial_train_x`, are unchanged. Run directories written by 0.3 still load with `load_metrics` and `load_iterations`. Changes to the lower-level surrogate and acquisition classes and to the `results.h5` shapes are listed in the [CHANGELOG](CHANGELOG.md).

## Run outputs (MRR)

When `run_dir` is given, each run creates a timestamped **run directory** (named from timestamp, number of inputs, and key parameters, e.g. `2026-09-16_110436_3d_training50iter_...`) containing the five **MRR artifacts**:

| Artifact | Contents |
|---|---|
| `config.json` | All run parameters (written at start, so it survives crashes) |
| `manifest.json` | SHA-256 checksum of the inputs |
| `meta.json` | Environment: package name/version, repository, git commit, Python/library versions, platform, timestamps, output summary |
| `run.log` | Per-iteration audit trail, ending with a summary line giving `best_x`/`best_y` |
| `results.h5` | Self-describing HDF5 with all numerical results |

If the run raises partway through, `meta.json` and `results.h5` are still written as a best-effort checkpoint covering every iteration completed before the failure (`stop_reason="crashed"`), and `run.log` ends with an error line instead of the summary line.

`results.h5` layout:

```
/            attrs: run configuration, including n_dims (number of inputs)
├── history/     one row per iteration: iteration, new_y, current_best, max_ei,
│                prediction_error, improvement; next_point with one column
│                per input; plus lengthscale (one column per input) and
│                outputscale/noise when the surrogate reports them, so a
│                with_training run's retuning is visible per iteration
│                rather than collapsed to its final value;
│                plus point_and_output, the evaluated input point and its
│                output in one array (columns x1..xd, y), each column
│                scaled to [0, 1] so a viewer such as H5Web can show them
│                in a single heatmap under one colour scale. The raw values
│                stay in next_point/new_y, and column_min/column_max on the
│                dataset recover them
├── iterations/  iter_NNN/ GP snapshot arrays (omitted if store_snapshots=False);
│                candidates and train_x have one column per input
└── final/       best_x (one value per input), best_y, stop_reason,
                 n_iterations + final train_x/train_y
                 + converged_max_ei/converged_next_point/converged_candidates/
                 converged_f_mean/converged_f_var/converged_ei_scores when the
                 run stopped via ei_threshold and snapshots were kept. This is
                 the GP/EI state of the fit that triggered convergence, whose
                 next_point was scored but never evaluated (so it has no
                 place in history/ or iterations/)
                 + fitted_lengthscale/fitted_outputscale/fitted_noise, the
                 surrogate's hyperparameters as the run left them. For a
                 with_training run these are the values Adam converged to,
                 which config.json cannot hold: it is written before the
                 loop starts and records only the starting point
```

To browse `results.h5` without writing code, the [H5Web](https://marketplace.visualstudio.com/items?itemName=h5web.vscode-h5web) VS Code extension opens HDF5 files directly in the editor, with the groups, attributes, and plots of any dataset.

Both figures below can also be rebuilt from this directory alone, with no `OptimisationRun` object, so they work on any run you (or someone else) have on disk. Run directories written by actgpr 0.3 still load.

## Plotting

Two figures, each reachable two ways. That is the whole plotting API:

| | The surrogate itself | Validation metrics |
|---|---|---|
| **From the run** | `run.plot_iterations()` | `run.plot_metrics()` |
| **From the logs** | `load_iterations(run_dir)` | `load_metrics(run_dir)` |

The prefix says where the data comes from. `plot_` draws from the run object you are still holding, so those two are methods on `OptimisationRun`. `load_` takes the path to a run's log directory, reads its `results.h5`, and draws the same figure, so those two are functions imported from `actgpr.plotting`.

- **`run.plot_iterations()`** opens an interactive slider over every iteration: the GP fit on top, the EI landscape below. Watching the confidence band narrow around the minimum is the clearest picture of what the algorithm did. Shown in the GIF above.
- **`run.plot_metrics()`** draws the whole run as four panels, `current_best`, `improvement`, `max_ei` and `prediction_error` against iteration, to judge convergence at a glance. Shown in the image above.
- **`load_iterations(run_dir)`** is the same slider, for a run read back from its logs. Keep the returned `Slider` in a variable, or matplotlib collects it and it stops responding to drags.
- **`load_metrics(run_dir)`** is the same four panels, for a run read back from its logs.

**The slider needs a problem with one input**, because it draws the surrogate as a curve over that input. For a run with more inputs, `plot_iterations()` and `load_iterations()` raise a `ValueError` saying so; `plot_metrics()` and `load_metrics()` work for any number of inputs.

Each pair is the identical figure, so which one to reach for depends solely on what you still have to hand. `run.run_dir` holds the directory a run wrote to, so the `load_` pair works on a run you just finished as well as on one from last month. Both `load_` functions need only that path:

```python
import matplotlib.pyplot as plt
from actgpr.plotting import load_iterations, load_metrics

run_dir = "results/2026-07-20_212046_training50iter_ei0.001_maxiter20_n0.0002"

slider = load_iterations(run_dir, show=False)
load_metrics(run_dir, show=False)

plt.show()  # once, for both
```

All four open their window by calling `plt.show()` for you. That is fine for one figure, but `plt.show()` opens *every* figure that exists, not just the one you last built, so two calls in a row open the first window twice. Pass `show=False` to build a figure without opening it, then call `plt.show()` once at the end to open everything together.

`load_iterations` needs the run to have kept snapshots (the default); `load_metrics` works either way, since the validation metrics are always recorded. The [tutorial](https://alxndrschroeder.github.io/actgpr/tutorial.html#plotting-reference) walks through both figures panel by panel.

## Vocabulary

### The optimisation problem

| Term | Meaning |
|---|---|
| **Objective** | The function being minimised: one or more inputs, one real-valued output. Your blackbox, wrapped as anything exposing `.evaluate(x1, ..., xd) -> float`, called once per input point (e.g. `ObjectiveFn`, or your own class). Defaults to the sum of squared inputs, `f(x) = x²` with one input (handy for tutorials and tests). |
| **Analytic objective** | An Objective computed by a mathematical formula (e.g. `x²`), used for development and testing. |
| **Experiment objective** | An Objective whose output comes from a real-world measurement or instrument. `ObjectiveFn(func, jitter=...)` simulates this on an analytic function by adding Gaussian sensor/measurement noise to each evaluation. |
| **`jitter`** | Standard deviation of the Gaussian noise `ObjectiveFn` optionally adds per evaluation, by default `0.0` (off). Not to be confused with **Cholesky jitter** below: same word, unrelated purpose, since this simulates experimental noise in the Objective while Cholesky jitter stabilises the GP's covariance matrix. If used, set the surrogate's `noise` to `jitter**2` (noise is a variance), otherwise the GP starts out assuming the data is far cleaner than it is. |
| **Input dimension** / **`n_dims`** | The number of inputs of the problem, `d`. Set by how many `(lo, hi)` pairs `search_bounds` has. |
| **Input point** | One setting of all inputs: a tuple with one coordinate per input dimension, e.g. `(x1, x2, x3)`. |
| **`train_x`** (or `x`) | The input points passed to the Objective, as a tensor of shape `(n, d)`: one row per point. |
| **`train_y`** (or `y`) | The Objective outputs at those input points. |
| **`test_x`** | Input points where the surrogate predicts without evaluating the Objective. |
| **Training data** | The set of `(train_x, train_y)` pairs the GP model is fitted to. |
| **Search bounds** | One closed interval `(lo, hi)` per input dimension, within which input points are considered, e.g. `[(-2, 2), (0, 5)]`. A single `(lo, hi)` means one input. |
| **`initial_train_x`** | The input points that seed the optimisation loop, one row per point, e.g. `[[0.0, 1.0], [2.0, 3.0]]`. A flat list is accepted with one input. |

### The surrogate (GP model)

| Term | Meaning |
|---|---|
| **Surrogate** | A Gaussian Process model fitted to all training data so far, used to predict the Objective cheaply at unevaluated points. |
| **`GPyTorchSurrogate`** | The surrogate backend wrapper (fitting + prediction) built on [GPyTorch](https://gpytorch.ai/); hides GPyTorch API details. |
| **`ExactGPModel`** | The GP model definition inside the wrapper: constant mean + scaled RBF kernel with one lengthscale per input dimension. |
| **Prior / posterior** | The GP distribution before / after conditioning on the training data. |
| **Likelihood** | The Gaussian noise model mapping latent function values to observed targets. |
| **Kernel (RBF)** | The covariance function: a radial-basis-function kernel wrapped in a scale kernel. |
| **`lengthscale`** | RBF kernel hyperparameter controlling how far correlations reach (smoothness), one value per input dimension, since inputs generally vary on different scales. `without_training` accepts one value for all inputs or a list with one per input. |
| **`outputscale`** | Kernel signal variance. |
| **`noise`** | Observation noise variance of the likelihood. Should match `jitter**2` if the Objective adds jitter (see above), since it is a variance, not a standard deviation. |
| **MLL** | Marginal log likelihood, the training objective maximised when fitting hyperparameters. |
| **Cholesky jitter** | Small value (`1e-4`) added to the covariance diagonal to keep it numerically positive definite; all computations use float64. |
| **`f_mean`** | Predicted posterior mean at given input points. |
| **`f_var`** | Predicted posterior variance (per-point uncertainty), shape `(m,)`. |
| **`f_preds`** | Predictive distribution of the latent function `f(test_x)`. Its full covariance matrix is available on demand as `f_preds.covariance_matrix`, shape `(m, m)`; it is not computed during the loop, since it grows with the square of the candidate count. |
| **`observed_pred`** | Predictive distribution of observed targets `y = f(x) + noise`. |
| **`f_samples`** | Samples drawn from the predictive posterior (only computed when `n_samples > 0`). |
| **`f_std`** | `sqrt(f_var)`, used inside EI and for the ±2σ (≈95 % CI) plot band. |

### The acquisition function

| Term | Meaning |
|---|---|
| **Acquisition function** | Scores candidate input points and selects the next input point to evaluate. |
| **Expected Improvement (EI)** | The closed-form acquisition score (Jones et al., 1998) balancing exploitation (confidently better mean) and exploration (high uncertainty). |
| **Candidates / `n_candidates`** | The input points within the search bounds that EI scores (default 500 per stage). With one input they are an evenly spaced grid; with more, a seeded scrambled Sobol sample, since a grid grows exponentially with the number of inputs. "Candidates" refers only to these acquisition points, never to training data. |
| **`candidate_seed`** | The seed of the Sobol sampler (25), recorded in `config.json` so a multi-input run is reproducible. |
| **`ei_scores`** | The EI value of every candidate. |
| **`max_ei`** | The largest EI score in an iteration; compared against `ei_threshold` for convergence. |
| **`next_point`** | The next input point to evaluate, one coordinate per input. Found by taking the highest-EI candidate, then zoom-refining with a second, much finer set of candidates confined to a small box around it, so `next_point` is not limited to the coarse candidates' spacing. |
| **Current best** | The smallest Objective value observed so far. |

### The optimisation loop

| Term | Meaning |
|---|---|
| **`OptimisationRun`** | Top-level orchestrator: owns the loop and all MRR writes. |
| **Fit mode** | `with_training` (hyperparameters optimised each iteration) vs. `without_training` (fixed); recorded as `"training"` / `"notraining"` in `config.json`. |
| **`max_iterations`** | Budget cap: the maximum number of active optimisation iterations (GPR fit cycles), not individual Objective calls. |
| **`ei_threshold`** | Convergence threshold: the loop stops when `max_ei` falls below it. |
| **Convergence criterion** | EI below threshold **or** budget reached, whichever fires first. |
| **`stop_reason`** | Which criterion fired: `"ei_threshold"` or `"max_iterations"`. |
| **`new_y`** | The Objective output at the newly evaluated `next_point`. |
| **`best_x` / `best_y`** | The input point with the lowest Objective output (a tuple with one coordinate per input, also for a single input) and that output, which together are the final result. |
| **`store_snapshots`** | If `True` (the default), each iteration's full GP + EI state is also kept for interactive browsing via `plot_iterations()`. Set `False` to omit them and keep `results.h5` small. The validation metrics shown by `load_metrics()` are recorded regardless of this flag. |
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
| **MRR** | Minimal Reproducible Run, a pattern requiring every run to record: what was run, with what inputs, in which environment, what happened, and what came out. |
| **Run directory** | The timestamped folder under `run_dir` holding all MRR artifacts of a single run. |
| **Self-describing HDF5** | Configuration is stored as HDF5 attributes alongside the data, so `results.h5` can be understood without any other file. |
| **`load_metrics()`** | Builds the four-panel validation-metrics figure from a run directory's `results.h5` alone, with no `OptimisationRun` object needed. |
| **`load_iterations()`** | Opens the per-iteration slider from a run directory's `results.h5` alone, the same figure as `run.plot_iterations()`. One input only. |

## Development

```bash
poetry run pytest tests/            # all tiers: unit, integration, regression
poetry run black src/ tests/        # format
poetry run ruff check src/ tests/   # lint
poetry run sphinx-build -W docs docs/build/html   # API docs (warnings = errors)
```

The regression tier compares a fixed-seed run against `tests/regression/data/quadratic_baseline.csv`; the test module documents how to regenerate the baseline after an intentional behaviour change.

Pushing to `main` rebuilds and republishes the docs above via GitHub Pages (see `.github/workflows/ci.yml`), so the local `sphinx-build` command is for previewing changes before they merge.

## Contributing

Bug reports, enhancement requests, and pull requests are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to report issues, the branch/PR
workflow, and the coding standards CI enforces.

## Security

Found a vulnerability? See [SECURITY.md](SECURITY.md) for how to report it privately.

## License

MIT, see [LICENSE](LICENSE).

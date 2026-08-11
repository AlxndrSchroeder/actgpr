# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The plotting API is now four entry points and nothing else, named so the
  prefix says where the data comes from. `plot_` draws from the run object
  you are holding: `OptimisationRun.plot_iterations()` and
  `OptimisationRun.plot_metrics()`. `load_` takes the path to a run's log
  directory and draws the same figure from its `results.h5`:
  `plotting.load_iterations(run_dir)` and `plotting.load_metrics(run_dir)`.
  Two figures, each reachable two ways. This renames
  `plot_run_iterations` to `load_iterations`, `plot_run_history` to
  `load_metrics`, and `OptimisationRun.plot_history()` to `plot_metrics()`:
  the old names said neither what they drew nor where they read it from,
  and `plot_iterations` sitting next to `plot_run_iterations` gave no hint
  that one was in-memory and the other on-disk. `plot_gp`, `plot_acquisition`,
  `plot_iteration_snapshot` and `load_iteration_snapshots` are the helpers
  those four are built from and are now private (`_`-prefixed): they were
  public only because they existed, and a user choosing between eight
  plotting functions has to understand the internals to pick one.
  `plot_surrogate` is removed outright, being a three-line wrapper around
  `plot_gp` that nothing in the package called
- `load_metrics()` and `plot_metrics()` now draw one panel per metric
  in a 2x2 grid (`current_best`, `improvement`, `max_ei` on a log axis,
  `prediction_error`) instead of overlaying `prediction_error` and
  `improvement` on one axes with `max_ei` on a log twin axis. The four
  series have unrelated units and ranges, so sharing an axes flattened all
  but the largest into a line along zero, and `current_best`, the series
  that actually shows the run finding the minimum, was not plotted at all.
  Both now return the figure and its 2x2 array of axes; the `ax=`
  parameter is gone, since a grid cannot be drawn onto a single axes

- The Objective interface is now declared as a `typing.Protocol`
  (`actgpr.objective_fn.Objective`) and `OptimisationRun` is typed against
  it. The docs always said any object with `evaluate()` works and the
  runtime never checked the type, but the signatures declared
  `objective: ObjectiveFn`, so a type checker rejected a user's own
  simulation class. The duck typing is now expressed in the code
- `OptimisationRun.run_dir` exposes the timestamped directory a run wrote
  to. `run_dir` passed in is only a base path, and the resolved directory
  was previously a local variable, so a caller could not locate its own MRR
  record without globbing `results/`. This also makes
  `load_metrics(run.run_dir)` work on a run you just finished, which
  had no equivalent before

- `store_snapshots` now defaults to `True`. Browsing a run with
  `plot_iterations()` previously required setting the flag before the run
  started, so anyone who had not anticipated it got a `RuntimeError` and had
  to re-run. Pass `store_snapshots=False` to opt out, since the snapshot arrays
  are the bulk of `results.h5`'s size
- `OptimisationRun.plot_iterations()` now defaults to `log_scale=True`. EI
  shrinks by orders of magnitude as a run converges, which a linear axis
  compresses into an invisible flat line at zero; pass `log_scale=False`
  for a linear EI axis
- Plot titles now report `best_x` alongside `best_y`, in both figures. They previously showed only
  `best:` followed by a number, which was the lowest Objective *output* so
  far but gave no hint of that, and never showed the input point that
  achieved it; the validation figure omitted `best_x` entirely
- Plot titles carry a second line with the surrogate's hyperparameters:
  that iteration's fit in the slider, the run's final values in the
  validation figure. Omitted when the surrogate does not report them

### Added

- Both figures now title their window (`actgpr: iterations (GP fit and EI)`
  and `actgpr: validation metrics`). Matplotlib names windows `Figure 1`
  and `Figure 2` and opens them at the same default position, so with both
  on screen the second covers the first and neither the title bar nor the
  window switcher says which is which

- `OptimisationRun.plot_iterations()` gained `show=False`, which the other
  three entry points already had, so all four can now defer `plt.show()`.
  Without it, opening the slider and the metrics figure from one script
  meant two `plt.show()` calls, and `plt.show()` displays *every* open
  figure rather than only the newest, so the slider window was re-displayed
  by the second call and flickered back into view as the metrics window was
  closed. Building both with `show=False` and calling `plt.show()` once
  opens them side by side instead

- `plotting.load_iterations(run_dir)` opens the interactive
  per-iteration slider for a run read back from disk, the counterpart to
  `OptimisationRun.plot_iterations()`. Browsing a saved run previously
  meant rebuilding the slider by hand from `load_iteration_snapshots` and
  `plot_iteration_snapshot`, which drew a single frame onto axes you
  supplied, so the slider was the one thing the on-disk route could not
  reach. Both routes now delegate to the same private helper, so the
  figure has a single definition

- `OptimisationRun.plot_metrics()` plots a run's validation metrics from
  the object you still hold, the in-memory counterpart to
  `load_metrics()`. The series were previously reachable only by
  reading `results.h5`, so a run without `run_dir` had no route to them at
  all. Both delegate to one private helper, so the figure has a single
  definition

- Conda is now a supported install path alongside Poetry: `environment.yml`
  declares the conda dependency spec and `conda-lock.yml` pins it, solved
  separately for `linux-64`, `osx-64`, `osx-arm64`, and `win-64`. A new
  `conda` CI job installs from the lock file and runs the full test suite,
  and fails the build if `conda-lock.yml` has drifted out of sync with
  `environment.yml`, so the alternative path cannot rot unnoticed.
  `environment.yml` deliberately narrows Python and PyTorch below the ranges
  `pyproject.toml` permits, so conda resolves the same versions the Poetry
  path is tested on; conda-forge and PyPI still build independently, so the
  binaries are not guaranteed identical
- `ObjectiveFn(func, jitter=...)` optionally adds independent Gaussian
  noise to each evaluation, simulating the sensor/measurement noise of a
  real experiment on an otherwise-analytic objective. Defaults to `0.0`
  (off, no behaviour change). Pairs with the surrogate's existing `noise`
  hyperparameter, which models exactly this observation noise in the GP
  likelihood. The noise comes from a generator the `ObjectiveFn` owns,
  seeded with 25 by default (override with `seed=`), so a jittered run is
  reproducible without the caller seeding anything and jitter does not
  disturb the global `torch` RNG
- `results.h5`'s `history/` group now records `lengthscale`, `outputscale`,
  and `noise` per iteration, alongside the existing scalar series. In
  `with_training` the surrogate is refitted every iteration and the
  hyperparameters move substantially (lengthscale swung from 1.95 to 0.65
  within two iterations in testing), so a single final value cannot
  describe the run. Present only when the surrogate reports them
- `results.h5`'s `final/` group and `run.log` now record the surrogate's
  final hyperparameters (`fitted_lengthscale`, `fitted_outputscale`,
  `fitted_noise`). For `with_training` runs these are the values Adam
  converged to, and they were previously absent from the MRR record
  entirely: `config.json` is written before the loop starts, so it holds
  `None` for lengthscale/outputscale and only the *starting* noise. Read
  via an optional `hyperparameters()` method on the surrogate, so a
  backend without one is unaffected
- `config.json` now records `repr(objective)` under `"objective"`, so two
  runs with identical search parameters but different Objectives (or
  different `ObjectiveFn` jitter) are distinguishable from their MRR record
  alone. Uses `repr()` rather than a specific field since the Objective is
  duck-typed and `OptimisationRun` has no generic way to know what a
  particular Objective considers worth recording; `repr()` is the one thing
  every object provides

### Fixed

- `OptimisationRun.plot_iterations()` on a run that had not been executed
  yet reported "Set store_snapshots=True before calling run()", blaming a
  flag that was never the problem and sending the caller after the wrong
  cause. It now says to call `run()` first, matching `plot_metrics()`. The
  store_snapshots message is unchanged for the case it does describe

## [0.2.0] - 2026-07-28

### Added

- `plot_acquisition()` now marks the highest EI score at `next_point` with
  a labelled point, so the chosen point's EI value is visible directly on
  the graph rather than only in the subplot title
- `plot_acquisition()`, `plot_iteration_snapshot()`, and
  `OptimisationRun.plot_iterations(log_scale=True)` can now render the EI
  y-axis on a log scale, with `ei_threshold` drawn as a reference line one
  order of magnitude above the axis floor. This makes the EI shrinkage across
  a converging run visible instead of compressed into an invisible flat
  line on a linear axis
- Sphinx API docs and tutorial are now published to GitHub Pages
  ([alxndrschroeder.github.io/actgpr](https://alxndrschroeder.github.io/actgpr/)) via a
  `deploy-docs` CI stage that runs after `docs` passes on `main`
- `meta.json` now records `package_name` and `repository`, fetched from
  installed package metadata, so a run's provenance file identifies the
  software that produced it even if shared on its own
- `plotting.load_metrics(run_dir)` builds the `prediction_error` /
  `improvement` vs. iteration plot directly from a saved run's `results.h5`
  with no `OptimisationRun` object needed
- `run.log` now ends with a summary line giving `best_x`/`best_y`/
  `stop_reason`, so the identified minimum can be read straight off the log
  without opening `results.h5` or holding onto `run.run()`'s return value
- `CONTRIBUTING.md` documents how to report bugs/enhancements, the branch/PR
  workflow, and the coding standards CI enforces; linked from the README
- OpenSSF Best Practices and fair-software.eu compliance badges added to
  the README

### Changed

- Clarified in `run.py` docstrings and the README that `store_snapshots`
  only gates the per-iteration GP/EI arrays used by `plot_iterations()`;
  the `prediction_error`/`improvement` history used by `load_metrics()`
  is always recorded, regardless of this flag
- Renamed `max_evaluations` to `max_iterations` everywhere: the constructor
  parameter, the `config.json`/`results.h5` keys, the `stop_reason` value,
  and the run-folder naming (`eval20` → `maxiter20`)

- README quickstart and new docs tutorial reframed around wrapping a
  blackbox function in an `ObjectiveFn`

### Fixed

- `OptimisationRun.run()` wrote `results.h5`/`meta.json` only after the loop
  finished, so a crash mid-run (e.g. iteration 19 of 20) discarded every
  completed iteration, leaving only `config.json`/`manifest.json`/`run.log`
  behind. On an unhandled exception, `run()` now writes a best-effort
  `results.h5`/`meta.json` checkpoint (`stop_reason="crashed"`) covering
  every iteration completed before the failure, then re-raises
- `prediction_error` compared the objective's actual output against the
  surrogate's predicted mean at the coarse candidate grid's EI-argmax,
  not at `next_point` itself, which the EI zoom-refinement fix above can
  now shift away from that exact grid point. `Acquisition` now tracks
  `next_point_mean`, the predicted mean at the refined `next_point`, and
  `prediction_error` is computed against that instead
- `Acquisition.find_next_input_point()` was limited to the resolution of its
  coarse candidate grid: once the grid point nearest the true EI maximum had
  been evaluated, its posterior variance stopped shrinking enough to let any
  neighbouring grid point win, so the run kept re-selecting the same point
  and stalled short of the true optimum. A second, much finer grid confined
  to a small window around the coarse best point is now scored and used to
  refine `next_point`, recovering precision well beyond the coarse grid's
  spacing
- The GP/EI fit that triggers `ei_threshold` convergence was computed but
  never recorded: `plot_iterations()`'s slider silently stopped one frame
  short, showing the second-to-last state (still above `ei_threshold`) as
  if the run had stopped prematurely. The converging fit's state is now
  captured in `OptimisationRun._convergence_snapshot`, shown as the
  slider's final frame (titled "converged, not evaluated", since its
  candidate was scored but never evaluated), and written to `results.h5`
  under `final/converged_*` when `store_snapshots=True`

## [0.1.0] - 2026-07-20

### Added

- Active GPR optimisation loop (`OptimisationRun`) with `with_training` and
  `without_training` fit modes
- GPyTorch surrogate backend (`GPyTorchSurrogate`, `ExactGPModel`) with
  float64 precision and Cholesky jitter for numerical stability
- Expected Improvement acquisition function (`Acquisition`, Jones et al. 1998)
- `ObjectiveFn` wrapper for arbitrary scalar objectives; errors from the
  Objective propagate with their original exception type
- MRR reproducibility record per run: `config.json`, `manifest.json`,
  `meta.json`, `run.log`, `results.h5`
- Self-describing `results.h5` layout: `/history` per-iteration series,
  `/iterations` GP snapshots, `/final` summary
- Per-iteration validation metrics: `prediction_error` and `improvement`,
  recorded in `run.log`, `results.h5`, and plot titles
- Interactive per-iteration snapshot browser (`plot_iterations`)
- Test tiers: unit, integration, and regression (stored seeded baseline);
  warnings treated as errors with documented exceptions
- Sphinx API documentation built from NumPy-style docstrings
- GitHub Actions CI pipeline: lint (black, ruff) → test (pytest) → docs (sphinx)

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `store_snapshots` now defaults to `True`. Browsing a run with
  `plot_iterations()` previously required setting the flag before the run
  started, so anyone who had not anticipated it got a `RuntimeError` and had
  to re-run. Pass `store_snapshots=False` to opt out, since the snapshot arrays
  are the bulk of `results.h5`'s size
- `OptimisationRun.plot_iterations()` now defaults to `log_scale=True`. EI
  shrinks by orders of magnitude as a run converges, which a linear axis
  compresses into an invisible flat line at zero; pass `log_scale=False`
  for a linear EI axis
- `plot_run_history()` now also plots `max_ei`, on a log-scaled second
  y-axis, so both plotting entry points show EI on a log scale by default.
  `prediction_error` and `improvement` stay on the linear primary axis:
  the former is signed and the latter is frequently exactly zero, and a log
  axis can render neither. Pass `log_scale=False` to omit the `max_ei` axis
- Per-iteration plot titles now report `best_x` alongside `best_y`. The
  title previously showed only `best:` followed by a number, which was the
  lowest Objective *output* so far but gave no hint of that, and never
  showed the input point that achieved it

### Added

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
- `plotting.plot_run_history(run_dir)` builds the `prediction_error` /
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
  the `prediction_error`/`improvement` history used by `plot_run_history()`
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

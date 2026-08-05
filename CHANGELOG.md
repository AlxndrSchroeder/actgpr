# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Conda is now a supported install path alongside Poetry: `environment.yml`
  declares the conda dependency ranges and `conda-lock.yml` pins them,
  solved separately for `linux-64`, `osx-64`, `osx-arm64`, and `win-64`.
  A new `conda` CI job installs from the lock file and runs the full test
  suite, and fails the build if `conda-lock.yml` has drifted out of sync
  with `environment.yml` — so the alternative path cannot rot unnoticed.
  Note that conda-forge and PyPI build their packages independently, so the
  two paths pin the same versions but not necessarily identical binaries

## [0.2.0] - 2026-07-28

### Added

- `plot_acquisition()` now marks the highest EI score at `next_point` with
  a labelled point, so the chosen point's EI value is visible directly on
  the graph rather than only in the subplot title
- `plot_acquisition()`, `plot_iteration_snapshot()`, and
  `OptimisationRun.plot_iterations(log_scale=True)` can now render the EI
  y-axis on a log scale, with `ei_threshold` drawn as a reference line one
  order of magnitude above the axis floor — makes the EI shrinkage across
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
  — no `OptimisationRun` object needed
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
  surrogate's predicted mean at the coarse candidate grid's EI-argmax —
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
  slider's final frame (titled "converged — not evaluated", since its
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

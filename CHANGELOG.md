# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Sphinx API docs and tutorial are now published to GitHub Pages
  ([alxndrschroeder.github.io/actgpr](https://alxndrschroeder.github.io/actgpr/)) via a
  `deploy-docs` CI stage that runs after `docs` passes on `main`
- `meta.json` now records `package_name` and `repository`, fetched from
  installed package metadata, so a run's provenance file identifies the
  software that produced it even if shared on its own
- `plotting.plot_run_history(run_dir)` builds the `prediction_error` /
  `improvement` vs. iteration plot directly from a saved run's `results.h5`
  — no `OptimisationRun` object needed

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

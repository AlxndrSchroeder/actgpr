# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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

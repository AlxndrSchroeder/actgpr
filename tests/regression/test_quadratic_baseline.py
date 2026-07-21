"""Regression test: seeded quadratic run against a stored baseline.

Compares the full training data of a fixed-seed, fixed-hyperparameter run on
the x² Objective against tests/regression/data/quadratic_baseline.csv. Any
change to the loop, surrogate, or acquisition that alters numerical output
trips this test.

Regenerating the baseline (only after an intentional behaviour change):

    poetry run python -c "from tests.regression.test_quadratic_baseline \\
        import write_baseline; write_baseline()"
"""

from pathlib import Path

import numpy as np
import torch

from actgpr.objective_fn import ObjectiveFn
from actgpr.run import OptimisationRun
from actgpr.surrogate import GPyTorchSurrogate

SEED = 42
BASELINE_PATH = Path(__file__).parent / "data" / "quadratic_baseline.csv"

# Frozen run configuration — do not change without regenerating the baseline.
BASELINE_CONFIG = {
    "search_bounds": (-4.0, 4.0),
    "initial_train_x": [-3.0, 3.0],
    "max_iterations": 10,
    "ei_threshold": 1e-9,
    "n_candidates": 200,
    "lengthscale": 1.0,
    "outputscale": 1.0,
    "noise": 1e-4,
}


def execute_baseline_run() -> np.ndarray:
    """Execute the frozen baseline configuration and return (n, 2) train data."""
    torch.manual_seed(SEED)
    run = OptimisationRun.without_training(
        objective=ObjectiveFn(),
        surrogate=GPyTorchSurrogate(),
        **BASELINE_CONFIG,
    )
    result = run.run()
    return np.column_stack([result["train_x"].numpy(), result["train_y"].numpy()])


def write_baseline() -> None:
    """Regenerate the stored baseline CSV from the frozen configuration."""
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        BASELINE_PATH,
        execute_baseline_run(),
        delimiter=",",
        header=f"train_x,train_y (seed={SEED}, fixed hyperparameters, x^2)",
    )


def test_quadratic_run_matches_baseline() -> None:
    """Test that the seeded run reproduces the stored baseline exactly."""
    baseline = np.genfromtxt(BASELINE_PATH, delimiter=",")
    actual = execute_baseline_run()

    assert actual.shape == baseline.shape, (
        f"Run produced {actual.shape[0]} points, baseline has "
        f"{baseline.shape[0]} — the loop behaviour changed"
    )
    np.testing.assert_allclose(actual, baseline, rtol=1e-6, atol=1e-12)

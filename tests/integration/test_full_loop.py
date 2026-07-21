"""Integration tests: the full optimisation loop with MRR artifacts.

Runs OptimisationRun end-to-end on the analytic x² Objective and verifies
that the components work together: the loop converges, the run() result is
consistent with the accumulated history, and all 5 MRR artifacts are written
with the documented results.h5 layout.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from actgpr.objective_fn import ObjectiveFn
from actgpr.run import OptimisationRun
from actgpr.surrogate import GPyTorchSurrogate

SEED = 42

MRR_ARTIFACTS = ("config.json", "manifest.json", "meta.json", "run.log", "results.h5")

HISTORY_FIELDS = (
    "iteration",
    "next_point",
    "new_y",
    "current_best",
    "max_ei",
    "prediction_error",
    "improvement",
)


def make_quadratic_run(run_dir: Path | None) -> OptimisationRun:
    """Return a seeded fixed-hyperparameter run on the x² Objective."""
    torch.manual_seed(SEED)
    return OptimisationRun.without_training(
        objective=ObjectiveFn(),
        surrogate=GPyTorchSurrogate(),
        search_bounds=(-4.0, 4.0),
        initial_train_x=[-3.0, 3.0],
        max_iterations=8,
        ei_threshold=1e-9,
        n_candidates=200,
        lengthscale=1.0,
        outputscale=1.0,
        noise=1e-4,
        store_snapshots=True,
        run_dir=run_dir,
    )


class TestFullLoop:
    """End-to-end behaviour of the optimisation loop."""

    def test_finds_minimum_of_quadratic(self, tmp_path: Path) -> None:
        """Test that the loop closes in on the x² minimum at x=0."""
        result = make_quadratic_run(tmp_path).run()

        assert abs(result["best_x"]) < 0.5
        assert result["best_y"] < 0.5

    def test_result_consistent_with_training_data(self, tmp_path: Path) -> None:
        """Test that best_x/best_y match the argmin of the returned data."""
        result = make_quadratic_run(tmp_path).run()

        best_idx = torch.argmin(result["train_y"])
        assert result["best_x"] == result["train_x"][best_idx].item()
        assert result["best_y"] == result["train_y"][best_idx].item()
        assert result["train_x"].numel() == result["train_y"].numel()

    def test_run_is_deterministic(self, tmp_path: Path) -> None:
        """Test that two seeded runs produce identical training data."""
        result_a = make_quadratic_run(tmp_path / "a").run()
        result_b = make_quadratic_run(tmp_path / "b").run()

        assert torch.equal(result_a["train_x"], result_b["train_x"])
        assert torch.equal(result_a["train_y"], result_b["train_y"])

    def test_training_mode_runs_end_to_end(self, tmp_path: Path) -> None:
        """Test that the hyperparameter-training mode also completes a run."""
        torch.manual_seed(SEED)
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-4.0, 4.0),
            initial_train_x=[-3.0, 3.0],
            max_iterations=3,
            ei_threshold=1e-9,
            n_candidates=100,
            training_iter=10,
            run_dir=tmp_path,
        )
        result = run.run()

        assert result["n_iterations"] >= 1
        assert result["stop_reason"] in ("ei_threshold", "max_iterations")


class TestMrrArtifacts:
    """End-to-end verification of the 5 MRR artifacts."""

    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> tuple[Path, dict]:
        """Execute a run and return its run directory and result."""
        result = make_quadratic_run(tmp_path).run()
        (run_dir,) = list(tmp_path.iterdir())
        return run_dir, result

    def test_writes_all_five_artifacts(self, run_dir: tuple[Path, dict]) -> None:
        """Test that every MRR artifact exists after a run."""
        directory, _ = run_dir
        for artifact in MRR_ARTIFACTS:
            assert (directory / artifact).exists(), f"missing artifact: {artifact}"

    def test_config_records_run_parameters(self, run_dir: tuple[Path, dict]) -> None:
        """Test that config.json holds the parameters actually used."""
        directory, _ = run_dir
        config = json.loads((directory / "config.json").read_text())

        assert config["fit_mode"] == "notraining"
        assert config["search_bounds"] == [-4.0, 4.0]
        assert config["max_iterations"] == 8
        assert config["lengthscale"] == 1.0

    def test_meta_summary_matches_result(self, run_dir: tuple[Path, dict]) -> None:
        """Test that meta.json's output summary matches the returned result."""
        directory, result = run_dir
        meta = json.loads((directory / "meta.json").read_text())

        summary = meta["output_summary"]
        assert summary["best_x"] == pytest.approx(result["best_x"])
        assert summary["best_y"] == pytest.approx(result["best_y"])
        assert summary["n_iterations"] == result["n_iterations"]
        assert summary["stop_reason"] == result["stop_reason"]

    def test_history_matches_result(self, run_dir: tuple[Path, dict]) -> None:
        """Test that the results.h5 history is aligned and consistent."""
        directory, result = run_dir

        with h5py.File(directory / "results.h5", "r") as f:
            history = f["history"]
            n_recorded = len(history["iteration"])

            for field in HISTORY_FIELDS:
                assert len(history[field]) == n_recorded

            # improvement Δᵢ = gain of this iteration's own new_y over
            # current_best (0 when the new point is not an improvement)
            current_best = history["current_best"][:]
            new_y = history["new_y"][:]
            expected = np.maximum(0.0, current_best - new_y)
            np.testing.assert_allclose(history["improvement"][:], expected)

            assert f["final"].attrs["best_x"] == pytest.approx(result["best_x"])
            assert f["final"].attrs["n_iterations"] == result["n_iterations"]
            assert len(f["final/train_x"]) == result["train_x"].numel()

    def test_snapshots_written_per_iteration(self, run_dir: tuple[Path, dict]) -> None:
        """Test that iterations/iter_NNN snapshot groups hold the GP arrays."""
        directory, _ = run_dir

        with h5py.File(directory / "results.h5", "r") as f:
            n_recorded = len(f["history/iteration"])
            assert len(f["iterations"]) == n_recorded

            first = f["iterations/iter_001"]
            for name in ("candidates", "f_mean", "f_var", "ei_scores"):
                assert first[name].shape == (200,)

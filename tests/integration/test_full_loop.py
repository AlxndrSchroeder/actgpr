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

SEED = 25

MRR_ARTIFACTS = ("config.json", "manifest.json", "meta.json", "run.log", "results.h5")

METRIC_FIELDS = (
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
        run_dir=run_dir,
    )


class TestFullLoop:
    """End-to-end behaviour of the optimisation loop."""

    def test_finds_minimum_of_quadratic(self, tmp_path: Path) -> None:
        """Test that the loop closes in on the x² minimum at x=0."""
        result = make_quadratic_run(tmp_path).run()

        assert abs(result["best_x"][0]) < 0.5
        assert result["best_y"] < 0.5

    def test_result_consistent_with_training_data(self, tmp_path: Path) -> None:
        """Test that best_x/best_y match the argmin of the returned data."""
        result = make_quadratic_run(tmp_path).run()

        best_idx = torch.argmin(result["train_y"])
        assert result["best_x"] == tuple(result["train_x"][best_idx].tolist())
        assert result["best_y"] == result["train_y"][best_idx].item()
        assert result["train_x"].shape[0] == result["train_y"].shape[0]

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
        assert config["search_bounds"] == [[-4.0, 4.0]]
        assert config["n_dims"] == 1
        assert config["candidate_seed"] == 25
        assert config["max_iterations"] == 8
        assert config["lengthscale"] == 1.0
        assert config["objective"] == "ObjectiveFn(function=sum(x_i^2))"

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

            for field in METRIC_FIELDS:
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


class TestMrrArtifactsOnCrash:
    """End-to-end verification that the MRR record survives a mid-run crash.

    Complements TestMrrArtifacts (the success path) by checking that all 5
    artifacts — not just the pieces exercised individually in the run.py
    unit tests — still form a coherent, readable record together when the
    Objective raises partway through.
    """

    @pytest.fixture()
    def crashed_run_dir(self, tmp_path: Path) -> Path:
        """Execute a run whose Objective fails after 2 initial + 2 loop evaluations."""
        calls = {"count": 0}

        def flaky(x: float) -> float:
            """Succeed like x² for 4 calls (2 initial points, 2 iterations), then fail."""
            calls["count"] += 1
            if calls["count"] > 4:
                raise RuntimeError("objective backend failure")
            return x**2

        torch.manual_seed(SEED)
        run = OptimisationRun.without_training(
            objective=ObjectiveFn(flaky),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-4.0, 4.0),
            initial_train_x=[-3.0, 3.0],
            max_iterations=8,
            ei_threshold=1e-9,
            n_candidates=200,
            lengthscale=1.0,
            outputscale=1.0,
            noise=1e-4,
            run_dir=tmp_path,
        )

        with pytest.raises(RuntimeError, match="objective backend failure"):
            run.run()

        (run_dir,) = list(tmp_path.iterdir())
        return run_dir

    def test_writes_all_five_artifacts(self, crashed_run_dir: Path) -> None:
        """Test that every MRR artifact still exists after a crash."""
        for artifact in MRR_ARTIFACTS:
            assert (
                crashed_run_dir / artifact
            ).exists(), f"missing artifact: {artifact}"

    def test_results_h5_and_meta_agree_on_partial_progress(
        self, crashed_run_dir: Path
    ) -> None:
        """Test that results.h5 and meta.json consistently report the crash.

        Both files are written independently by _write_mrr_record(); this
        checks they stay in sync rather than one lagging the other.
        """
        meta = json.loads((crashed_run_dir / "meta.json").read_text())
        summary = meta["output_summary"]

        with h5py.File(crashed_run_dir / "results.h5", "r") as f:
            history = f["history"]
            final = f["final"]

            # 2 loop iterations completed before the 5th evaluate() call failed
            assert len(history["iteration"]) == 2
            assert final.attrs["n_iterations"] == 2
            assert summary["n_iterations"] == 2

            assert final.attrs["stop_reason"] == "crashed"
            assert summary["stop_reason"] == "crashed"

            assert final.attrs["best_x"] == pytest.approx(summary["best_x"])
            assert final.attrs["best_y"] == pytest.approx(summary["best_y"])
            # final/train_x holds only the 2 initial points + 2 completed
            # iterations — the point that triggered the crash was never
            # appended to training data.
            assert len(f["final/train_x"]) == 4


def _bowl_3d(x1: float, x2: float, x3: float) -> float:
    """Return a 3D bowl with its minimum at (0.5, -1.0, 0.0)."""
    return (x1 - 0.5) ** 2 + (x2 + 1.0) ** 2 + x3**2


BOUNDS_3D = [(-2.0, 2.0), (-3.0, 1.0), (-1.0, 1.0)]


@pytest.fixture(scope="module")
def finished_3d(tmp_path_factory: pytest.TempPathFactory):
    """Run a seeded 3D optimisation once and share it across the tests."""
    torch.manual_seed(SEED)
    run = OptimisationRun.with_training(
        objective=ObjectiveFn(_bowl_3d),
        surrogate=GPyTorchSurrogate(),
        search_bounds=BOUNDS_3D,
        initial_train_x=[[-1.5, 0.5, 0.8], [1.5, -2.5, -0.8], [0.0, 0.0, 0.0]],
        max_iterations=12,
        ei_threshold=1e-6,
        n_candidates=256,
        training_iter=30,
        run_dir=tmp_path_factory.mktemp("results"),
    )
    return run, run.run()


class TestThreeDimensionalRun:
    """A 3-input run end to end, through the MRR record and back out."""

    def test_converges_near_the_known_minimum(self, finished_3d) -> None:
        """Test that the run gets close to (0.5, -1.0, 0.0)."""
        _, result = finished_3d

        assert len(result["best_x"]) == 3
        assert result["best_y"] < 0.2

    def test_every_evaluated_point_is_inside_its_bounds(self, finished_3d) -> None:
        """Test that no coordinate ever left its own interval."""
        _, result = finished_3d

        for dim, (lo, hi) in enumerate(BOUNDS_3D):
            assert torch.all(result["train_x"][:, dim] >= lo)
            assert torch.all(result["train_x"][:, dim] <= hi)

    def test_writes_all_five_artifacts_into_a_3d_named_directory(
        self, finished_3d
    ) -> None:
        """Test the MRR record exists and the folder name states the dimension."""
        run, _ = finished_3d

        assert "_3d_" in run.run_dir.name
        for artifact in MRR_ARTIFACTS:
            assert (run.run_dir / artifact).exists()

    def test_config_and_meta_describe_three_inputs(self, finished_3d) -> None:
        """Test that the JSON artifacts carry per-dimension values."""
        run, result = finished_3d
        config = json.loads((run.run_dir / "config.json").read_text())
        meta = json.loads((run.run_dir / "meta.json").read_text())

        assert config["n_dims"] == 3
        assert config["search_bounds"] == [list(pair) for pair in BOUNDS_3D]
        assert np.array(config["initial_train_x"]).shape == (3, 3)
        assert meta["output_summary"]["best_x"] == pytest.approx(list(result["best_x"]))

    def test_results_h5_has_one_column_per_input(self, finished_3d) -> None:
        """Test that every input-shaped dataset is (rows, 3)."""
        run, result = finished_3d
        n_iter = result["n_iterations"]

        with h5py.File(run.run_dir / "results.h5", "r") as f:
            assert f.attrs["n_dims"] == 3
            evaluated = f["history/iteration"].shape[0]
            assert evaluated in (n_iter, n_iter - 1)
            assert f["history/next_point"].shape == (evaluated, 3)
            assert f["history/lengthscale"].shape == (evaluated, 3)
            assert f["history/outputscale"].shape == (evaluated, 1)
            assert f["final/train_x"].shape == (result["train_x"].shape[0], 3)
            assert list(f["final"].attrs["best_x"]) == pytest.approx(
                list(result["best_x"])
            )
            assert "iterations" not in f

    def test_surrogate_rebuilds_from_the_record(self, finished_3d) -> None:
        """Test that any iteration's surrogate survives without stored arrays.

        results.h5 no longer keeps the per-iteration GP arrays. It does not
        need to: the training data up to that iteration plus that row's
        hyperparameters, the mean constant included, reproduce the fit.
        """
        from actgpr.surrogate import GPyTorchSurrogate

        run, result = finished_3d
        iteration = 3
        n_initial = 3

        with h5py.File(run.run_dir / "results.h5", "r") as f:
            assert "iterations" not in f  # the arrays really are gone
            points = n_initial + iteration - 1
            rebuilt = GPyTorchSurrogate()
            rebuilt.fit_no_training(
                torch.from_numpy(f["final/train_x"][:points]),
                torch.from_numpy(f["final/train_y"][:points]),
                lengthscale=f["history/lengthscale"][iteration - 1].tolist(),
                outputscale=float(f["history/outputscale"][iteration - 1][0]),
                noise=float(f["history/noise"][iteration - 1][0]),
                mean_constant=float(f["history/mean_constant"][iteration - 1][0]),
            )

        predicted = rebuilt.predict(torch.zeros(1, 3, dtype=torch.float64))["f_mean"]

        assert rebuilt.train_x.shape == (points, 3)
        assert torch.all(torch.isfinite(predicted))

    def test_metrics_figure_rebuilds_from_the_record(self, finished_3d) -> None:
        """Test that load_metrics works on a 3D run directory."""
        from actgpr.plotting import load_metrics

        run, result = finished_3d
        fig, axes = load_metrics(run.run_dir, show=False)

        assert axes.shape == (2, 2)
        assert "best_x: (" in fig._suptitle.get_text()

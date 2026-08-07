"""Unit tests for the OptimisationRun class."""

import json
import logging
from pathlib import Path
from typing import Callable

import h5py
import pytest
import torch

from actgpr.objective_fn import ObjectiveFn
from actgpr.run import OptimisationRun
from actgpr.surrogate import GPyTorchSurrogate

SEED = 25


@pytest.fixture()
def simple_run() -> OptimisationRun:
    """Return an OptimisationRun configured for a simple x^2 optimisation."""
    torch.manual_seed(SEED)
    return OptimisationRun.with_training(
        objective=ObjectiveFn(),
        surrogate=GPyTorchSurrogate(),
        search_bounds=(-3.0, 3.0),
        initial_train_x=torch.tensor([-2.0, -1.0, 1.0, 2.0]),
        max_iterations=10,
        ei_threshold=0.01,
        n_candidates=100,
        training_iter=20,
    )


class TestOptimisationRunInit:
    """Tests for OptimisationRun.__init__."""

    def test_stores_objective(self, simple_run: OptimisationRun) -> None:
        """Test that the objective is stored correctly."""
        assert isinstance(simple_run.objective, ObjectiveFn)

    def test_stores_surrogate(self, simple_run: OptimisationRun) -> None:
        """Test that the surrogate is stored correctly."""
        assert isinstance(simple_run.surrogate, GPyTorchSurrogate)

    def test_stores_search_bounds(self, simple_run: OptimisationRun) -> None:
        """Test that search bounds are stored correctly."""
        assert simple_run.search_bounds == (-3.0, 3.0)

    def test_stores_max_iterations(self, simple_run: OptimisationRun) -> None:
        """Test that max_iterations is stored correctly."""
        assert simple_run.max_iterations == 10

    def test_stores_ei_threshold(self, simple_run: OptimisationRun) -> None:
        """Test that ei_threshold is stored correctly."""
        assert simple_run.ei_threshold == 0.01

    def test_evaluates_initial_points(self, simple_run: OptimisationRun) -> None:
        """Test that train_y is computed from initial_train_x on construction."""
        # x^2 at [-2, -1, 1, 2] = [4, 1, 1, 4]
        expected = torch.tensor([4.0, 1.0, 1.0, 4.0], dtype=torch.float64)
        assert torch.allclose(simple_run.train_y, expected)

    def test_train_x_is_float64(self, simple_run: OptimisationRun) -> None:
        """Test that train_x is cast to float64 regardless of input dtype."""
        assert simple_run.train_x.dtype == torch.float64

    def test_accepts_list_input(self) -> None:
        """Test that initial_train_x can be a plain Python list."""
        run = OptimisationRun(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=[-2.0, 0.0, 2.0],
            max_iterations=10,
            ei_threshold=0.01,
        )
        assert run.train_x.numel() == 3

    def test_raises_on_empty_train_x(self) -> None:
        """Test that ValueError is raised when initial_train_x is empty."""
        with pytest.raises(ValueError, match="at least one point"):
            OptimisationRun(
                objective=ObjectiveFn(),
                surrogate=GPyTorchSurrogate(),
                search_bounds=(-3.0, 3.0),
                initial_train_x=torch.tensor([]),
                max_iterations=10,
                ei_threshold=0.01,
            )

    def test_raises_on_max_iterations_too_small(self) -> None:
        """Test that ValueError is raised when max_iterations <= 0."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            OptimisationRun(
                objective=ObjectiveFn(),
                surrogate=GPyTorchSurrogate(),
                search_bounds=(-3.0, 3.0),
                initial_train_x=torch.tensor([-1.0, 0.0, 1.0]),
                max_iterations=0,
                ei_threshold=0.01,
            )

    def test_raises_on_inverted_search_bounds(self) -> None:
        """Test that ValueError is raised when search_bounds are not increasing."""
        with pytest.raises(ValueError, match="must be <"):
            OptimisationRun(
                objective=ObjectiveFn(),
                surrogate=GPyTorchSurrogate(),
                search_bounds=(3.0, -3.0),
                initial_train_x=torch.tensor([-1.0, 0.0, 1.0]),
                max_iterations=10,
                ei_threshold=0.01,
            )

    def test_raises_on_non_positive_ei_threshold(self) -> None:
        """Test that ValueError is raised when ei_threshold <= 0."""
        with pytest.raises(ValueError, match="ei_threshold must be positive"):
            OptimisationRun(
                objective=ObjectiveFn(),
                surrogate=GPyTorchSurrogate(),
                search_bounds=(-3.0, 3.0),
                initial_train_x=torch.tensor([-1.0, 0.0, 1.0]),
                max_iterations=10,
                ei_threshold=0.0,
            )

    def test_results_accumulator_starts_empty(
        self, simple_run: OptimisationRun
    ) -> None:
        """Test that the deferred-write accumulator is empty before run()."""
        assert simple_run._results == []

    def test_repr_shows_fit_mode(self, simple_run: OptimisationRun) -> None:
        """Test the string representation includes the fit mode."""
        r = repr(simple_run)
        assert "OptimisationRun" in r
        assert "fit=training" in r
        assert "bounds=(-3.0, 3.0)" in r
        assert "max_iter=10" in r

    def test_config_dict_records_objective_repr(
        self, simple_run: OptimisationRun
    ) -> None:
        """Test that _config_dict() identifies the objective via repr().

        repr() is used (rather than a specific field) because the Objective
        is duck-typed: OptimisationRun has no generic way to know what a
        particular Objective considers worth recording.
        """
        assert simple_run._config_dict()["objective"] == repr(simple_run.objective)

    def test_config_dict_objective_repr_surfaces_jitter(self) -> None:
        """Test that a jittered ObjectiveFn's config.json entry shows it.

        This is the concrete case the duck-typed repr() approach exists
        for: jitter is an ObjectiveFn-specific detail OptimisationRun has
        no other way to record generically.
        """
        run = OptimisationRun.without_training(
            objective=ObjectiveFn(jitter=0.1),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=[-2.0, 2.0],
            max_iterations=5,
            ei_threshold=0.01,
        )
        assert (
            run._config_dict()["objective"] == "ObjectiveFn(function=x^2, jitter=0.1)"
        )


class TestOptimisationRunRun:
    """Tests for OptimisationRun.run()."""

    def test_returns_expected_keys(self, simple_run: OptimisationRun) -> None:
        """Test that run() returns a dict with all expected keys."""
        result = simple_run.run()
        expected_keys = {
            "best_x",
            "best_y",
            "train_x",
            "train_y",
            "n_iterations",
            "stop_reason",
        }
        assert set(result.keys()) == expected_keys

    def test_run_log_ends_with_result_summary(self, tmp_path: Path) -> None:
        """Test that run.log's final line reports best_x/best_y, not just per-iteration data.

        So the identified minimum can be read straight off run.log without
        opening results.h5 or keeping the run() return value around.
        """
        torch.manual_seed(SEED)
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, -1.0, 1.0, 2.0]),
            max_iterations=10,
            ei_threshold=0.01,
            n_candidates=100,
            training_iter=20,
            run_dir=tmp_path,
        )
        result = run.run()

        (run_dir,) = list(tmp_path.iterdir())
        log_text = (run_dir / "run.log").read_text()

        assert "Finished" in log_text
        assert f"best_x={result['best_x']:.6f}" in log_text
        assert f"best_y={result['best_y']:.6f}" in log_text
        assert result["stop_reason"] in log_text

    def test_best_y_is_float(self, simple_run: OptimisationRun) -> None:
        """Test that best_y is a Python float."""
        result = simple_run.run()
        assert isinstance(result["best_y"], float)

    def test_best_x_is_float(self, simple_run: OptimisationRun) -> None:
        """Test that best_x is a Python float."""
        result = simple_run.run()
        assert isinstance(result["best_x"], float)

    def test_best_y_is_non_negative(self, simple_run: OptimisationRun) -> None:
        """Test scientific invariant: x² is always non-negative."""
        result = simple_run.run()
        assert result["best_y"] >= 0

    def test_n_iterations_is_positive(self, simple_run: OptimisationRun) -> None:
        """Test that at least one iteration was executed."""
        result = simple_run.run()
        assert result["n_iterations"] >= 1

    def test_train_data_grows(self, simple_run: OptimisationRun) -> None:
        """Test that training data grows during the run (unless stopped by EI on first iteration)."""
        initial_count = 4
        result = simple_run.run()
        # If stopped by EI on first iteration, no new points are added
        if result["stop_reason"] == "ei_threshold" and result["n_iterations"] == 1:
            assert result["train_x"].numel() == initial_count
        else:
            assert result["train_x"].numel() > initial_count

    def test_train_x_train_y_same_length(self, simple_run: OptimisationRun) -> None:
        """Test that train_x and train_y always have the same length."""
        result = simple_run.run()
        assert result["train_x"].numel() == result["train_y"].numel()

    def test_respects_max_iterations(self) -> None:
        """Test that the loop stops at max_iterations even without convergence."""
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, 2.0]),
            max_iterations=5,
            ei_threshold=1e-20,  # impossibly low — forces max_iterations stop
            n_candidates=50,
            training_iter=10,
        )
        result = run.run()
        assert result["n_iterations"] == 5
        assert result["stop_reason"] == "max_iterations"

    def test_stop_reason_ei_threshold(self) -> None:
        """Test that stop_reason is 'ei_threshold' when EI drops below threshold."""
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
            max_iterations=50,
            ei_threshold=1.0,  # very high — should stop immediately
            n_candidates=50,
            training_iter=10,
        )
        result = run.run()
        assert result["stop_reason"] == "ei_threshold"

    def test_convergence_snapshot_populated_with_store_snapshots(self) -> None:
        """Test that the converging fit's GP/EI state is captured.

        Regression test for the bug where the fit that triggers
        ei_threshold convergence was computed but never recorded anywhere,
        making the last visible snapshot look like it stopped one
        iteration early with EI still above threshold.
        """
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
            max_iterations=50,
            ei_threshold=1.0,  # very high — should stop immediately
            n_candidates=50,
            training_iter=10,
            store_snapshots=True,
        )
        result = run.run()

        assert result["stop_reason"] == "ei_threshold"
        assert run._convergence_snapshot is not None
        snapshot = run._convergence_snapshot
        assert snapshot["max_ei"] < run.ei_threshold
        for key in ("candidates", "f_mean", "f_var", "ei_scores", "train_x", "train_y"):
            assert key in snapshot
        # The converging fit was never evaluated, so it must not carry
        # evaluation-only fields.
        assert "prediction_error" not in snapshot
        assert "improvement" not in snapshot

    def test_convergence_snapshot_absent_without_store_snapshots(self) -> None:
        """Test that no convergence snapshot is captured when store_snapshots=False."""
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
            max_iterations=50,
            ei_threshold=1.0,
            n_candidates=50,
            training_iter=10,
            store_snapshots=False,
        )
        result = run.run()

        assert result["stop_reason"] == "ei_threshold"
        assert run._convergence_snapshot is None

    def test_convergence_snapshot_absent_when_max_iterations_stops(self) -> None:
        """Test that no convergence snapshot is captured for a max_iterations stop."""
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, 2.0]),
            max_iterations=5,
            ei_threshold=1e-20,  # impossibly low — forces max_iterations stop
            n_candidates=50,
            training_iter=10,
            store_snapshots=True,
        )
        result = run.run()

        assert result["stop_reason"] == "max_iterations"
        assert run._convergence_snapshot is None

    def test_results_accumulator_populated(self, simple_run: OptimisationRun) -> None:
        """Test that _results accumulator is populated after run()."""
        simple_run.run()
        # Each iteration that evaluates a new point adds one entry
        for entry in simple_run._results:
            assert "iteration" in entry
            assert "next_point" in entry
            assert "new_y" in entry
            assert "current_best" in entry
            assert "max_ei" in entry

    def test_improvement_reflects_current_iteration(
        self, simple_run: OptimisationRun
    ) -> None:
        """Test that improvement is the gain from this iteration's own new_y."""
        simple_run.run()
        for entry in simple_run._results:
            expected = max(0.0, entry["current_best"] - entry["new_y"])
            assert entry["improvement"] == pytest.approx(expected)

    def test_file_logger_detached_after_crash(
        self, tmp_path: Path, flaky_objective: Callable[[int], ObjectiveFn]
    ) -> None:
        """Test that the run.log handler is removed when the loop raises."""
        run = OptimisationRun.without_training(
            objective=flaky_objective(2),  # succeeds for the 2 initial points only
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=[-2.0, 2.0],
            max_iterations=5,
            ei_threshold=1e-12,
            n_candidates=50,
            run_dir=tmp_path,
        )

        logger = logging.getLogger("actgpr")
        n_handlers_before = len(logger.handlers)

        # Errors from the Objective propagate with their original type
        with pytest.raises(RuntimeError, match="objective backend failure"):
            run.run()

        assert len(logger.handlers) == n_handlers_before

        # All 5 MRR artifacts still exist as the crash trace, including a
        # best-effort results.h5/meta.json checkpoint (see test_checkpoint_*
        # below for their content) even though zero iterations completed.
        (run_dir,) = list(tmp_path.iterdir())
        assert (run_dir / "config.json").exists()
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "run.log").exists()
        assert (run_dir / "results.h5").exists()
        assert (run_dir / "meta.json").exists()

    def test_checkpoint_written_on_crash_with_partial_results(
        self, tmp_path: Path, flaky_objective: Callable[[int], ObjectiveFn]
    ) -> None:
        """Test that a crash checkpoints every iteration completed so far."""
        run = OptimisationRun.without_training(
            objective=flaky_objective(4),  # 2 initial points + 2 loop iterations
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=[-2.0, 2.0],
            max_iterations=5,
            ei_threshold=1e-12,
            n_candidates=50,
            run_dir=tmp_path,
        )

        with pytest.raises(RuntimeError, match="objective backend failure"):
            run.run()

        assert len(run._results) == 2

        (run_dir,) = list(tmp_path.iterdir())
        with h5py.File(run_dir / "results.h5", "r") as f:
            assert len(f["history/iteration"]) == 2
            assert f["final"].attrs["stop_reason"] == "crashed"
            assert f["final"].attrs["n_iterations"] == 2
            # final/train_x holds only the points evaluated before the
            # crash: the 2 initial points plus the 2 completed iterations.
            assert len(f["final/train_x"]) == 4

        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["output_summary"]["stop_reason"] == "crashed"
        assert meta["output_summary"]["n_iterations"] == 2

    def test_checkpoint_written_on_crash_before_any_iteration_completes(
        self, tmp_path: Path, flaky_objective: Callable[[int], ObjectiveFn]
    ) -> None:
        """Test the degenerate case: the crash happens before iteration 1 finishes."""
        run = OptimisationRun.without_training(
            objective=flaky_objective(2),  # succeeds for the 2 initial points only
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=[-2.0, 2.0],
            max_iterations=5,
            ei_threshold=1e-12,
            n_candidates=50,
            run_dir=tmp_path,
        )

        with pytest.raises(RuntimeError, match="objective backend failure"):
            run.run()

        assert run._results == []

        (run_dir,) = list(tmp_path.iterdir())
        with h5py.File(run_dir / "results.h5", "r") as f:
            assert len(f["history/iteration"]) == 0
            assert f["final"].attrs["stop_reason"] == "crashed"
            assert f["final"].attrs["n_iterations"] == 0
            # best_x/best_y fall back to the 2 initial points, the only
            # data that exists yet.
            assert f["final"].attrs["best_y"] == pytest.approx(4.0)
            # The loop fits the surrogate before it calls the objective, so
            # even a crash in iteration 1 leaves a fitted surrogate whose
            # hyperparameters are worth recording for debugging.
            assert "fitted_lengthscale" in f["final"].attrs


class TestFittedHyperparameters:
    """Tests for recording the surrogate's final hyperparameters in the MRR record."""

    def test_recorded_for_a_training_run(self, tmp_path: Path) -> None:
        """Test that the values Adam tuned to reach results.h5.

        config.json is written before the loop starts and holds `None` for
        lengthscale/outputscale in training mode, so without this the fitted
        surrogate is absent from the record entirely.
        """
        torch.manual_seed(SEED)
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=[-2.0, 2.0],
            max_iterations=3,
            ei_threshold=1e-9,
            n_candidates=50,
            training_iter=10,
            run_dir=tmp_path,
        )
        run.run()

        expected = run.surrogate.hyperparameters()
        (run_dir,) = list(tmp_path.iterdir())

        with h5py.File(run_dir / "results.h5", "r") as f:
            attrs = f["final"].attrs
            assert attrs["fitted_lengthscale"] == pytest.approx(expected["lengthscale"])
            assert attrs["fitted_outputscale"] == pytest.approx(expected["outputscale"])
            assert attrs["fitted_noise"] == pytest.approx(expected["noise"])

        config = json.loads((run_dir / "config.json").read_text())
        assert config["lengthscale"] is None  # the gap this closes

    def test_logged_to_run_log(self, tmp_path: Path) -> None:
        """Test that the values are readable without opening results.h5."""
        torch.manual_seed(SEED)
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=[-2.0, 2.0],
            max_iterations=3,
            ei_threshold=1e-9,
            n_candidates=50,
            training_iter=10,
            run_dir=tmp_path,
        )
        run.run()

        (run_dir,) = list(tmp_path.iterdir())
        log_text = (run_dir / "run.log").read_text()

        assert "Final surrogate hyperparameters" in log_text
        assert "lengthscale=" in log_text

    def test_surrogate_without_the_method_is_tolerated(
        self, simple_run: OptimisationRun
    ) -> None:
        """Test that a surrogate not exposing hyperparameters() still works.

        The surrogate is duck-typed, so OptimisationRun must not require the
        method; a backend without one simply contributes nothing here.
        """

        class MinimalSurrogate:
            """A surrogate exposing no hyperparameters() method."""

        simple_run.surrogate = MinimalSurrogate()

        assert simple_run._fitted_hyperparameters() is None

    def test_unfitted_surrogate_is_tolerated(self) -> None:
        """Test that an unfitted surrogate yields None rather than raising."""
        torch.manual_seed(SEED)
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=[-2.0, 2.0],
            max_iterations=3,
            ei_threshold=1e-9,
            n_candidates=50,
            training_iter=10,
        )

        assert run._fitted_hyperparameters() is None

    def test_custom_objective_converges(self) -> None:
        """Test that the loop works with a custom objective function."""
        torch.manual_seed(SEED)
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(lambda x: (x - 1) ** 2),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 5.0),
            initial_train_x=torch.tensor([-2.0, 0.0, 2.0, 4.0]),
            max_iterations=20,
            ei_threshold=0.01,
            n_candidates=100,
            training_iter=20,
        )
        result = run.run()
        # The minimum of (x-1)^2 is at x=1, y=0
        assert result["best_y"] < 1.0


class TestOptimisationRunSnapshots:
    """Tests for snapshot storage and plot_iterations."""

    @pytest.fixture()
    def snapshot_run(self) -> OptimisationRun:
        """Return an OptimisationRun with store_snapshots=True."""
        torch.manual_seed(SEED)
        return OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, -1.0, 1.0, 2.0]),
            max_iterations=8,
            ei_threshold=0.01,
            n_candidates=50,
            training_iter=10,
            store_snapshots=True,
        )

    @pytest.fixture()
    def no_snapshot_run(self) -> OptimisationRun:
        """Return an OptimisationRun with store_snapshots explicitly disabled."""
        torch.manual_seed(SEED)
        return OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, -1.0, 1.0, 2.0]),
            max_iterations=8,
            ei_threshold=0.01,
            n_candidates=50,
            training_iter=10,
            store_snapshots=False,
        )

    def test_snapshots_enabled_by_default(self, simple_run: OptimisationRun) -> None:
        """Test that snapshots are kept without asking, so plot_iterations() works.

        Opting out (store_snapshots=False) keeps results.h5 small; opting in
        used to be required, which made the plotting entry point fail for
        anyone who had not set the flag up front.
        """
        assert simple_run.store_snapshots is True

        simple_run.run()

        assert all("candidates" in entry for entry in simple_run._results)

    def test_snapshots_stored_when_enabled(self, snapshot_run: OptimisationRun) -> None:
        """Test that snapshots are present in _results when store_snapshots=True."""
        snapshot_run.run()
        snapshot_keys = {
            "candidates",
            "f_mean",
            "f_var",
            "ei_scores",
            "train_x",
            "train_y",
        }
        for entry in snapshot_run._results:
            assert snapshot_keys.issubset(entry.keys())

    def test_snapshots_absent_when_disabled(
        self, no_snapshot_run: OptimisationRun
    ) -> None:
        """Test that no snapshot tensors are stored when store_snapshots=False."""
        no_snapshot_run.run()
        for entry in no_snapshot_run._results:
            assert "candidates" not in entry

    def test_snapshot_tensors_have_correct_shapes(
        self, snapshot_run: OptimisationRun
    ) -> None:
        """Test that snapshot tensors have consistent shapes."""
        snapshot_run.run()
        for entry in snapshot_run._results:
            n_candidates = entry["candidates"].numel()
            assert entry["f_mean"].shape == (n_candidates,)
            assert entry["f_var"].shape == (n_candidates,)
            assert entry["ei_scores"].shape == (n_candidates,)
            assert entry["train_x"].numel() == entry["train_y"].numel()

    def test_snapshot_train_data_grows(self, snapshot_run: OptimisationRun) -> None:
        """Test that snapshot train_x grows across iterations."""
        snapshot_run.run()
        results = snapshot_run._results
        if len(results) >= 2:
            assert results[1]["train_x"].numel() > results[0]["train_x"].numel()

    def test_plot_iterations_raises_without_snapshots(
        self, no_snapshot_run: OptimisationRun
    ) -> None:
        """Test that plot_iterations raises RuntimeError without snapshots."""
        no_snapshot_run.run()
        with pytest.raises(RuntimeError, match="No snapshots available"):
            no_snapshot_run.plot_iterations()

    def test_plot_iterations_ei_axis_has_fixed_range(
        self, snapshot_run: OptimisationRun, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the EI axis uses one fixed range shared across iterations.

        Without a fixed range, matplotlib would autoscale the EI subplot to
        each iteration's own scores, hiding the shrinking EI maximum that
        signals convergence. Checked on the linear axis, hence
        log_scale=False.
        """
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)

        snapshot_run.run()
        snapshot_run.plot_iterations(log_scale=False)

        _, ei_ax = plt.gcf().axes[:2]
        expected_max = max(r["ei_scores"].max().item() for r in snapshot_run._results)
        assert ei_ax.get_ylim() == (0.0, pytest.approx(expected_max * 1.05))

    def test_plot_iterations_log_scale_is_the_default(
        self, snapshot_run: OptimisationRun, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the EI axis is log-scaled without asking for it.

        EI shrinks by orders of magnitude as a run converges, which a linear
        axis flattens into an invisible line at zero — so log is the useful
        default and linear is the opt-out.
        """
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)

        snapshot_run.run()
        snapshot_run.plot_iterations()

        _, ei_ax = plt.gcf().axes[:2]
        assert ei_ax.get_yscale() == "log"

    def test_plot_iterations_log_scale(
        self, snapshot_run: OptimisationRun, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that log_scale=True sets a log axis with an ei_threshold-based floor."""
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)

        snapshot_run.run()
        snapshot_run.plot_iterations(log_scale=True)

        _, ei_ax = plt.gcf().axes[:2]
        assert ei_ax.get_yscale() == "log"

        lo, _ = ei_ax.get_ylim()
        assert lo == pytest.approx(snapshot_run.ei_threshold * 0.1)

        labels = [line.get_label() for line in ei_ax.get_lines()]
        assert any("ei_threshold" in label for label in labels)

    def test_plot_iterations_includes_convergence_snapshot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the slider's last frame is the converging fit's own state.

        Regression test: before the fix, the slider silently stopped one
        frame short of the actual convergence-triggering EI value.
        """
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)

        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
            max_iterations=50,
            ei_threshold=0.05,  # converges after a couple of iterations
            n_candidates=50,
            training_iter=10,
            store_snapshots=True,
        )
        run.run()
        run.plot_iterations()

        evaluated_count = len(run._results)
        slider = run._active_slider
        assert slider.valmax == evaluated_count + 1

        slider.set_val(slider.valmax)
        gp_ax, _ = plt.gcf().axes[:2]
        assert "converged" in gp_ax.get_title()
        assert float(f"{run._convergence_snapshot['max_ei']:.6f}") < run.ei_threshold

    def test_slider_kept_alive_and_responsive(
        self, snapshot_run: OptimisationRun, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the slider survives after plot_iterations() returns.

        Matplotlib widgets stop responding to input if their only reference
        is garbage collected. This forces a collection pass — which would
        previously have destroyed a locally-scoped Slider — and then drags
        the slider programmatically to confirm it still redraws the figure.
        """
        import gc

        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)

        snapshot_run.run()
        snapshot_run.plot_iterations()

        assert snapshot_run._active_slider is not None

        gc.collect()  # would destroy an unreferenced Slider

        if len(snapshot_run._results) < 2:
            return  # converged after one iteration — nothing to scrub to

        gp_ax, _ = plt.gcf().axes[:2]
        title_before = gp_ax.get_title()

        snapshot_run._active_slider.set_val(2)  # simulate dragging to iteration 2

        assert gp_ax.get_title() != title_before


class TestOptimisationRunWithoutTraining:
    """Tests for OptimisationRun.without_training() classmethod."""

    @pytest.fixture()
    def fixed_run(self) -> OptimisationRun:
        """Return an OptimisationRun with fixed hyperparameters."""
        torch.manual_seed(SEED)
        return OptimisationRun.without_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, -1.0, 1.0, 2.0]),
            max_iterations=8,
            ei_threshold=0.01,
            n_candidates=50,
            lengthscale=1.0,
            outputscale=1.0,
            noise=1e-4,
        )

    def test_internal_flag_is_false(self, fixed_run: OptimisationRun) -> None:
        """Test that the internal train flag is False."""
        assert fixed_run._train_hyperparameters is False

    def test_stores_fixed_hyperparameters(self, fixed_run: OptimisationRun) -> None:
        """Test that lengthscale and outputscale are stored."""
        assert fixed_run._lengthscale == 1.0
        assert fixed_run._outputscale == 1.0

    def test_repr_shows_fixed_mode(self, fixed_run: OptimisationRun) -> None:
        """Test that __repr__ shows fit=fixed."""
        assert "fit=fixed" in repr(fixed_run)

    def test_run_returns_expected_keys(self, fixed_run: OptimisationRun) -> None:
        """Test that run() returns all expected keys in fixed mode."""
        result = fixed_run.run()
        expected_keys = {
            "best_x",
            "best_y",
            "train_x",
            "train_y",
            "n_iterations",
            "stop_reason",
        }
        assert set(result.keys()) == expected_keys

    def test_run_produces_results(self, fixed_run: OptimisationRun) -> None:
        """Test that the fixed-mode loop runs and produces iterations."""
        result = fixed_run.run()
        assert result["n_iterations"] > 0

    def test_train_data_grows(self, fixed_run: OptimisationRun) -> None:
        """Test that training data grows in fixed mode."""
        initial_n = fixed_run.train_x.numel()
        fixed_run.run()
        assert fixed_run.train_x.numel() > initial_n


class TestOptimisationRunWithTraining:
    """Tests for OptimisationRun.with_training() classmethod."""

    def test_internal_flag_is_true(self, simple_run: OptimisationRun) -> None:
        """Test that the internal train flag is True."""
        assert simple_run._train_hyperparameters is True

    def test_stores_training_iter(self, simple_run: OptimisationRun) -> None:
        """Test that training_iter is stored."""
        assert simple_run._training_iter == 20

    def test_repr_shows_training_mode(self, simple_run: OptimisationRun) -> None:
        """Test that __repr__ shows fit=training."""
        assert "fit=training" in repr(simple_run)

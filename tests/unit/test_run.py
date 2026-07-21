"""Unit tests for the OptimisationRun class."""

import logging
from pathlib import Path

import pytest
import torch

from actgpr.objective_fn import ObjectiveFn
from actgpr.run import OptimisationRun
from actgpr.surrogate import GPyTorchSurrogate

SEED = 42


@pytest.fixture()
def simple_run() -> OptimisationRun:
    """Return an OptimisationRun configured for a simple x^2 optimisation."""
    torch.manual_seed(SEED)
    return OptimisationRun.with_training(
        objective=ObjectiveFn(),
        surrogate=GPyTorchSurrogate(),
        search_bounds=(-3.0, 3.0),
        initial_train_x=torch.tensor([-2.0, -1.0, 1.0, 2.0]),
        max_evaluations=10,
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

    def test_stores_max_evaluations(self, simple_run: OptimisationRun) -> None:
        """Test that max_evaluations is stored correctly."""
        assert simple_run.max_evaluations == 10

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
            max_evaluations=10,
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
                max_evaluations=10,
                ei_threshold=0.01,
            )

    def test_raises_on_max_eval_too_small(self) -> None:
        """Test that ValueError is raised when max_evaluations <= 0."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            OptimisationRun(
                objective=ObjectiveFn(),
                surrogate=GPyTorchSurrogate(),
                search_bounds=(-3.0, 3.0),
                initial_train_x=torch.tensor([-1.0, 0.0, 1.0]),
                max_evaluations=0,
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
                max_evaluations=10,
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
                max_evaluations=10,
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
        assert "max_eval=10" in r


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

    def test_respects_max_evaluations(self) -> None:
        """Test that the loop stops at max_evaluations even without convergence."""
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, 2.0]),
            max_evaluations=5,
            ei_threshold=1e-20,  # impossibly low — forces max_evaluations stop
            n_candidates=50,
            training_iter=10,
        )
        result = run.run()
        assert result["n_iterations"] == 5
        assert result["stop_reason"] == "max_evaluations"

    def test_stop_reason_ei_threshold(self) -> None:
        """Test that stop_reason is 'ei_threshold' when EI drops below threshold."""
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
            max_evaluations=50,
            ei_threshold=1.0,  # very high — should stop immediately
            n_candidates=50,
            training_iter=10,
        )
        result = run.run()
        assert result["stop_reason"] == "ei_threshold"

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

    def test_file_logger_detached_after_crash(self, tmp_path: Path) -> None:
        """Test that the run.log handler is removed when the loop raises."""
        calls = {"count": 0}

        def flaky(x: float) -> float:
            """Evaluate x² for the initial points, then fail inside the loop."""
            calls["count"] += 1
            if calls["count"] > 2:
                raise RuntimeError("objective backend failure")
            return x**2

        run = OptimisationRun.without_training(
            objective=ObjectiveFn(flaky),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=[-2.0, 2.0],
            max_evaluations=5,
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

        # config.json and manifest.json still exist as the crash trace
        (run_dir,) = list(tmp_path.iterdir())
        assert (run_dir / "config.json").exists()
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "run.log").exists()

    def test_custom_objective_converges(self) -> None:
        """Test that the loop works with a custom objective function."""
        torch.manual_seed(SEED)
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(lambda x: (x - 1) ** 2),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 5.0),
            initial_train_x=torch.tensor([-2.0, 0.0, 2.0, 4.0]),
            max_evaluations=20,
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
            max_evaluations=8,
            ei_threshold=0.01,
            n_candidates=50,
            training_iter=10,
            store_snapshots=True,
        )

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

    def test_snapshots_absent_when_disabled(self, simple_run: OptimisationRun) -> None:
        """Test that no snapshot tensors are stored when store_snapshots=False."""
        simple_run.run()
        for entry in simple_run._results:
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
        self, simple_run: OptimisationRun
    ) -> None:
        """Test that plot_iterations raises RuntimeError without snapshots."""
        simple_run.run()
        with pytest.raises(RuntimeError, match="No snapshots available"):
            simple_run.plot_iterations()

    def test_plot_iterations_ei_axis_has_fixed_range(
        self, snapshot_run: OptimisationRun, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the EI axis uses one fixed range shared across iterations.

        Without a fixed range, matplotlib would autoscale the EI subplot to
        each iteration's own scores, hiding the shrinking EI maximum that
        signals convergence.
        """
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)

        snapshot_run.run()
        snapshot_run.plot_iterations()

        _, ei_ax = plt.gcf().axes[:2]
        expected_max = max(r["ei_scores"].max().item() for r in snapshot_run._results)
        assert ei_ax.get_ylim() == (0.0, pytest.approx(expected_max * 1.05))


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
            max_evaluations=8,
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

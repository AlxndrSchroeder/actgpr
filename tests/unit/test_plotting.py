"""Unit tests for plotting utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

from actgpr import mrr
from actgpr.objective_fn import ObjectiveFn
from actgpr.run import OptimisationRun
from actgpr.surrogate import GPyTorchSurrogate
from actgpr.plotting import (
    load_snapshots,
    plot_acquisition,
    plot_iteration_snapshot,
    plot_run_history,
)

SEED = 25


def _make_snapshot(iteration: int, ei_scores: torch.Tensor) -> dict:
    """Build a minimal snapshot dict for plot_iteration_snapshot."""
    candidates = torch.linspace(-1.0, 1.0, ei_scores.numel())
    return {
        "iteration": iteration,
        "candidates": candidates,
        "f_mean": torch.zeros_like(candidates),
        "f_var": torch.ones_like(candidates),
        "train_x": torch.tensor([-1.0, 1.0]),
        "train_y": torch.tensor([1.0, 1.0]),
        "ei_scores": ei_scores,
        "next_point": 0.0,
        "current_best": 1.0,
        "max_ei": ei_scores.max().item(),
        "prediction_error": 0.0,
        "improvement": 0.0,
    }


class TestPlotAcquisition:
    """Tests for plot_acquisition's y-axis scaling."""

    def test_autoscales_by_default(self) -> None:
        """Test that the y-axis autoscales to the data when ylim is not given."""
        candidates = torch.linspace(-1.0, 1.0, 10)
        ei_scores = torch.linspace(0.0, 0.1, 10)

        _, ax = plot_acquisition(candidates, ei_scores, show=False)

        lo, hi = ax.get_ylim()
        assert hi < 1.0

    def test_respects_fixed_ylim(self) -> None:
        """Test that a fixed ylim is applied instead of autoscaling."""
        candidates = torch.linspace(-1.0, 1.0, 10)
        ei_scores = torch.linspace(0.0, 0.1, 10)

        _, ax = plot_acquisition(candidates, ei_scores, show=False, ylim=(0.0, 5.0))

        assert ax.get_ylim() == (0.0, 5.0)

    def test_fixed_ylim_stays_constant_across_calls(self) -> None:
        """Test that the same ylim applies regardless of that call's own EI scores.

        This is what lets a viewer see the EI maximum shrink across
        iterations instead of every plot being autoscaled to fill the axes.
        """
        fig, ax = plt.subplots()
        shared_ylim = (0.0, 2.0)

        big_scores = torch.linspace(0.0, 1.8, 10)
        small_scores = torch.linspace(0.0, 0.01, 10)

        plot_acquisition(
            torch.linspace(-1.0, 1.0, 10),
            big_scores,
            ax=ax,
            show=False,
            ylim=shared_ylim,
        )
        first_ylim = ax.get_ylim()

        ax.cla()
        plot_acquisition(
            torch.linspace(-1.0, 1.0, 10),
            small_scores,
            ax=ax,
            show=False,
            ylim=shared_ylim,
        )
        second_ylim = ax.get_ylim()

        assert first_ylim == second_ylim == shared_ylim

    def test_log_scale_sets_axis_scale(self) -> None:
        """Test that log_scale=True switches the y-axis to a log scale."""
        candidates = torch.linspace(-1.0, 1.0, 10)
        ei_scores = torch.linspace(0.0, 0.1, 10)

        _, ax = plot_acquisition(candidates, ei_scores, show=False, log_scale=True)

        assert ax.get_yscale() == "log"

    def test_log_scale_clamps_zero_scores(self) -> None:
        """Test that exact-zero EI scores are clamped to a positive floor.

        A log axis cannot represent zero; EI is exactly 0 at training
        points, so plotting must clamp rather than error or silently drop
        those points.
        """
        candidates = torch.linspace(-1.0, 1.0, 5)
        ei_scores = torch.tensor([0.0, 0.1, 0.0, 0.2, 0.0])

        _, ax = plot_acquisition(
            candidates, ei_scores, show=False, log_scale=True, ei_threshold=0.01
        )

        ei_line = next(
            line
            for line in ax.get_lines()
            if line.get_label() == "Expected Improvement"
        )
        plotted = ei_line.get_ydata()
        assert all(
            v > 0 for v in plotted
        ), "log scale must never receive zero/negative values"

    def test_log_scale_floor_defaults_below_ei_threshold(self) -> None:
        """Test that the default log floor sits one order of magnitude below ei_threshold."""
        candidates = torch.linspace(-1.0, 1.0, 5)
        ei_scores = torch.linspace(0.0, 0.05, 5)

        _, ax = plot_acquisition(
            candidates, ei_scores, show=False, log_scale=True, ei_threshold=0.01
        )

        lo, _ = ax.get_ylim()
        assert lo == pytest.approx(0.001)

    def test_ei_threshold_drawn_as_reference_line(self) -> None:
        """Test that ei_threshold is drawn as a horizontal reference line."""
        candidates = torch.linspace(-1.0, 1.0, 5)
        ei_scores = torch.linspace(0.0, 0.05, 5)

        _, ax = plot_acquisition(
            candidates, ei_scores, show=False, log_scale=True, ei_threshold=0.01
        )

        labels = [line.get_label() for line in ax.get_lines()]
        assert any("ei_threshold" in label for label in labels)

    def test_no_reference_line_without_ei_threshold(self) -> None:
        """Test that no threshold line is drawn when ei_threshold is not given."""
        candidates = torch.linspace(-1.0, 1.0, 5)
        ei_scores = torch.linspace(0.0, 0.05, 5)

        _, ax = plot_acquisition(candidates, ei_scores, show=False)

        labels = [line.get_label() for line in ax.get_lines()]
        assert not any("ei_threshold" in label for label in labels)

    def test_marks_max_ei_value_at_next_point(self) -> None:
        """Test that the highest EI score is marked at next_point with its value."""
        candidates = torch.linspace(-1.0, 1.0, 5)
        ei_scores = torch.tensor([0.0, 0.01, 0.05, 0.02, 0.0])
        next_point = candidates[2].item()  # matches argmax(ei_scores)

        _, ax = plot_acquisition(
            candidates, ei_scores, next_point=next_point, show=False
        )

        labels = [line.get_label() for line in ax.get_lines()]
        assert any("Max EI" in label and "5.00e-02" in label for label in labels)

        marker = next(
            line for line in ax.get_lines() if line.get_label().startswith("Max EI")
        )
        assert marker.get_xdata()[0] == pytest.approx(next_point)
        assert marker.get_ydata()[0] == pytest.approx(0.05)

    def test_no_max_ei_marker_without_next_point(self) -> None:
        """Test that no peak marker is drawn when next_point is not given."""
        candidates = torch.linspace(-1.0, 1.0, 5)
        ei_scores = torch.linspace(0.0, 0.05, 5)

        _, ax = plot_acquisition(candidates, ei_scores, show=False)

        labels = [line.get_label() for line in ax.get_lines()]
        assert not any("Max EI" in label for label in labels)


class TestPlotIterationSnapshot:
    """Tests for plot_iteration_snapshot's ei_ylim passthrough."""

    def test_ei_ylim_none_autoscales(self) -> None:
        """Test that omitting ei_ylim autoscales the EI axis as before."""
        fig, (gp_ax, ei_ax) = plt.subplots(2, 1)
        snapshot = _make_snapshot(1, torch.linspace(0.0, 0.05, 20))

        plot_iteration_snapshot(snapshot, (gp_ax, ei_ax))

        lo, hi = ei_ax.get_ylim()
        assert hi < 1.0

    def test_ei_ylim_applied_to_ei_axis_only(self) -> None:
        """Test that a fixed ei_ylim is applied to the EI axis, not the GP axis."""
        fig, (gp_ax, ei_ax) = plt.subplots(2, 1)
        snapshot = _make_snapshot(1, torch.linspace(0.0, 0.05, 20))

        plot_iteration_snapshot(snapshot, (gp_ax, ei_ax), ei_ylim=(0.0, 3.0))

        assert ei_ax.get_ylim() == (0.0, 3.0)
        assert gp_ax.get_ylim() != (0.0, 3.0)

    def test_convergence_snapshot_gets_distinct_title(self) -> None:
        """Test that a snapshot without prediction_error/improvement is
        labelled as a convergence snapshot rather than a normal iteration.
        """
        fig, (gp_ax, ei_ax) = plt.subplots(2, 1)
        candidates = torch.linspace(-1.0, 1.0, 10)
        convergence_snapshot = {
            "iteration": 18,
            "candidates": candidates,
            "f_mean": torch.zeros_like(candidates),
            "f_var": torch.ones_like(candidates),
            "train_x": torch.tensor([-1.0, 1.0]),
            "train_y": torch.tensor([1.0, 1.0]),
            "ei_scores": torch.linspace(0.0, 0.001, 10),
            "next_point": 0.0,
            "current_best": -0.94,
            "max_ei": 0.001,
        }

        plot_iteration_snapshot(convergence_snapshot, (gp_ax, ei_ax))

        assert "converged" in gp_ax.get_title()
        assert "pred_error" not in gp_ax.get_title()

    def test_title_reports_this_iterations_hyperparameters(self) -> None:
        """Test that a snapshot's own hyperparameters reach the title.

        In with_training they are retuned every iteration, so the title
        must show the fit that produced this frame, not the run's final one.
        """
        fig, (gp_ax, ei_ax) = plt.subplots(2, 1)
        candidates = torch.linspace(-1.0, 1.0, 10)
        snapshot = {
            "iteration": 3,
            "candidates": candidates,
            "f_mean": torch.zeros_like(candidates),
            "f_var": torch.ones_like(candidates),
            "train_x": torch.tensor([-1.0, 1.0]),
            "train_y": torch.tensor([1.0, 0.5]),
            "ei_scores": torch.linspace(0.0, 0.5, 10),
            "next_point": 0.0,
            "current_best": 0.5,
            "max_ei": 0.5,
            "prediction_error": 0.1,
            "improvement": 0.2,
            "lengthscale": 0.75,
            "outputscale": 1.5,
            "noise": 1e-4,
        }

        plot_iteration_snapshot(snapshot, (gp_ax, ei_ax))
        title = gp_ax.get_title()

        assert "lengthscale: 0.75" in title
        assert "outputscale: 1.5" in title

    def test_title_omits_hyperparameters_when_absent(self) -> None:
        """Test that a snapshot lacking them still produces a valid title.

        Snapshots from a surrogate that does not report hyperparameters, or
        rebuilt from an older results.h5, must still plot.
        """
        fig, (gp_ax, ei_ax) = plt.subplots(2, 1)
        candidates = torch.linspace(-1.0, 1.0, 10)
        snapshot = {
            "iteration": 3,
            "candidates": candidates,
            "f_mean": torch.zeros_like(candidates),
            "f_var": torch.ones_like(candidates),
            "train_x": torch.tensor([-1.0, 1.0]),
            "train_y": torch.tensor([1.0, 0.5]),
            "ei_scores": torch.linspace(0.0, 0.5, 10),
            "next_point": 0.0,
            "current_best": 0.5,
            "max_ei": 0.5,
            "prediction_error": 0.1,
            "improvement": 0.2,
        }

        plot_iteration_snapshot(snapshot, (gp_ax, ei_ax))
        title = gp_ax.get_title()

        assert "lengthscale" not in title
        assert "best_x" in title


class TestPlotRunHistory:
    """Tests for plot_run_history — plotting a saved run from its path alone."""

    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        """Write a minimal results.h5 into tmp_path and return the directory."""
        results = [
            {
                "iteration": i,
                "next_point": float(i),
                "new_y": 1.0 / i,
                "current_best": 1.0 / i,
                "max_ei": 1.0 / i,
                "prediction_error": 0.5 / i,
                "improvement": 0.1 / i,
            }
            for i in range(1, 6)
        ]
        mrr.save_hdf5(
            tmp_path,
            results=results,
            config={"noise": 1e-4},
            store_snapshots=False,
            final_train_x=torch.tensor([0.0, 1.0]),
            final_train_y=torch.tensor([1.0, 0.5]),
            best_x=1.0,
            best_y=0.2,
            stop_reason="max_iterations",
            n_iterations=5,
        )
        return tmp_path

    def test_raises_when_no_results_h5(self, tmp_path: Path) -> None:
        """Test that a clear error is raised for a directory without results.h5."""
        with pytest.raises(FileNotFoundError, match="results.h5"):
            plot_run_history(tmp_path, show=False)

    def test_accepts_only_the_run_directory(self, run_dir: Path) -> None:
        """Test that the run directory alone is enough to build the plot."""
        fig, ax = plot_run_history(run_dir, show=False)

        assert fig is not None
        assert ax is not None

    def test_plots_prediction_error_and_improvement(self, run_dir: Path) -> None:
        """Test that both validation metric series are drawn."""
        _, ax = plot_run_history(run_dir, show=False)

        labels = [line.get_label() for line in ax.get_lines()]
        assert "prediction_error" in labels
        assert "improvement" in labels

        # Each plotted line has one point per iteration.
        pred_error_line = next(
            line for line in ax.get_lines() if line.get_label() == "prediction_error"
        )
        assert len(pred_error_line.get_xdata()) == 5

    def test_title_reports_best_x_best_y_and_stop_reason(self, run_dir: Path) -> None:
        """Test that the title surfaces the run's final outcome.

        Labels match plot_iteration_snapshot's, so a run reconstructed from
        results.h5 reports its outcome the same way as one plotted straight
        from memory. `best:` alone is ambiguous about which of x or y it is.
        """
        _, ax = plot_run_history(run_dir, show=False)
        title = ax.get_title()

        assert "best_x: 1.0000" in title
        assert "best_y: 0.2000" in title
        assert "max_iterations" in title

    def test_accepts_string_path(self, run_dir: Path) -> None:
        """Test that a plain string path works, not just a Path object."""
        fig, ax = plot_run_history(str(run_dir), show=False)
        assert ax is not None

    def test_max_ei_drawn_on_a_log_twin_axis_by_default(self, run_dir: Path) -> None:
        """Test that max_ei gets its own log-scaled axis without being asked.

        Matches plot_iterations()'s log-scaled EI default. max_ei needs a
        separate axis because it spans orders of magnitude, while
        prediction_error and improvement are linear and signed.
        """
        fig, ax = plot_run_history(run_dir, show=False)

        (ei_ax,) = [other for other in fig.axes if other is not ax]
        assert ei_ax.get_yscale() == "log"

        ei_labels = [line.get_label() for line in ei_ax.get_lines()]
        assert "max_ei" in ei_labels

    def test_primary_axis_stays_linear(self, run_dir: Path) -> None:
        """Test that the signed/zero-valued metrics keep a linear axis.

        prediction_error is signed and improvement is frequently exactly 0,
        neither of which a log axis can render — so only max_ei goes log.
        """
        _, ax = plot_run_history(run_dir, show=False)

        assert ax.get_yscale() == "linear"

    def test_log_scale_false_omits_the_ei_axis(self, run_dir: Path) -> None:
        """Test that opting out leaves only the original linear plot."""
        fig, ax = plot_run_history(run_dir, show=False, log_scale=False)

        assert fig.axes == [ax]
        labels = [line.get_label() for line in ax.get_lines()]
        assert "max_ei" not in labels

    def test_title_reports_final_hyperparameters(self, tmp_path: Path) -> None:
        """Test that the run's final hyperparameters reach the title.

        Mirrors plot_iteration_snapshot, so the two plotting entry points
        surface the same information about the surrogate.
        """
        mrr.save_hdf5(
            tmp_path,
            results=[
                {
                    "iteration": 1,
                    "next_point": 0.5,
                    "new_y": 0.25,
                    "current_best": 0.25,
                    "max_ei": 0.1,
                    "prediction_error": 0.01,
                    "improvement": 0.0,
                }
            ],
            config={"noise": 1e-4},
            store_snapshots=False,
            final_train_x=torch.tensor([0.0, 1.0]),
            final_train_y=torch.tensor([1.0, 0.5]),
            best_x=1.0,
            best_y=0.2,
            stop_reason="max_iterations",
            n_iterations=1,
            fitted_hyperparameters={
                "lengthscale": 1.25,
                "outputscale": 2.5,
                "noise": 1e-4,
            },
        )

        _, ax = plot_run_history(tmp_path, show=False)
        title = ax.get_title()

        assert "lengthscale: 1.25" in title
        assert "outputscale: 2.5" in title

    def test_title_omits_hyperparameters_when_absent(self, run_dir: Path) -> None:
        """Test that a record without them still produces a valid title."""
        _, ax = plot_run_history(run_dir, show=False)

        assert "lengthscale" not in ax.get_title()
        assert "best_x" in ax.get_title()

    def test_legend_covers_both_axes(self, run_dir: Path) -> None:
        """Test that max_ei appears in the legend despite being on a twin axis.

        Twin axes each own their lines, so a plain ax.legend() would silently
        drop max_ei from the legend.
        """
        _, ax = plot_run_history(run_dir, show=False)

        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert set(legend_labels) == {"prediction_error", "improvement", "max_ei"}


class TestLoadSnapshots:
    """Tests for rebuilding a saved run's snapshots from results.h5."""

    def test_raises_when_no_results_h5(self, tmp_path: Path) -> None:
        """Test the same clear error plot_run_history gives."""
        with pytest.raises(FileNotFoundError, match="results.h5"):
            load_snapshots(tmp_path)

    def test_raises_when_run_stored_no_snapshots(self, tmp_path: Path) -> None:
        """Test that a run without store_snapshots says so, rather than KeyError."""
        mrr.save_hdf5(
            tmp_path,
            results=[
                {
                    "iteration": 1,
                    "next_point": 0.5,
                    "new_y": 0.25,
                    "current_best": 0.25,
                    "max_ei": 0.1,
                    "prediction_error": 0.01,
                    "improvement": 0.0,
                }
            ],
            config={"noise": 1e-4},
            store_snapshots=False,
            final_train_x=torch.tensor([0.0, 1.0]),
            final_train_y=torch.tensor([1.0, 0.5]),
            best_x=1.0,
            best_y=0.2,
            stop_reason="max_iterations",
            n_iterations=1,
        )

        with pytest.raises(RuntimeError, match="store_snapshots"):
            load_snapshots(tmp_path)

    def test_round_trips_the_in_memory_snapshots(self, tmp_path: Path) -> None:
        """Test that what comes back matches what the run held in memory.

        This is the contract anything replotting a saved run depends on:
        reassembling snapshots by hand loses whichever fields the caller
        forgets, which is how the demo GIF silently lost its hyperparameter
        line.
        """
        torch.manual_seed(SEED)
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=[-2.0, 2.0],
            max_iterations=4,
            ei_threshold=1e-9,
            n_candidates=40,
            training_iter=5,
            run_dir=tmp_path,
        )
        run.run()

        (run_dir,) = list(tmp_path.iterdir())
        loaded = load_snapshots(run_dir)
        in_memory = [r for r in run._results if "candidates" in r]

        assert len(loaded) == len(in_memory)
        for from_disk, from_memory in zip(loaded, in_memory):
            assert set(from_disk) == set(from_memory)
            for key, value in from_memory.items():
                if isinstance(value, torch.Tensor):
                    assert torch.allclose(from_disk[key], value)
                else:
                    assert from_disk[key] == pytest.approx(value)

    def test_includes_hyperparameters_and_convergence_frame(
        self, tmp_path: Path
    ) -> None:
        """Test that the converged fit comes back with hyperparameters too."""
        torch.manual_seed(SEED)
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=[-2.0, 2.0],
            max_iterations=20,
            ei_threshold=0.05,
            n_candidates=40,
            training_iter=5,
            run_dir=tmp_path,
        )
        result = run.run()
        assert result["stop_reason"] == "ei_threshold"

        (run_dir,) = list(tmp_path.iterdir())
        loaded = load_snapshots(run_dir)

        # The converged frame is last and carries no evaluation metrics.
        converged = loaded[-1]
        assert "prediction_error" not in converged
        assert "lengthscale" in converged
        assert all("lengthscale" in snapshot for snapshot in loaded)

    def test_snapshots_are_plottable(self, tmp_path: Path) -> None:
        """Test that a loaded snapshot feeds straight into plot_iteration_snapshot."""
        torch.manual_seed(SEED)
        run = OptimisationRun.with_training(
            objective=ObjectiveFn(),
            surrogate=GPyTorchSurrogate(),
            search_bounds=(-3.0, 3.0),
            initial_train_x=[-2.0, 2.0],
            max_iterations=3,
            ei_threshold=1e-9,
            n_candidates=40,
            training_iter=5,
            run_dir=tmp_path,
        )
        run.run()

        (run_dir,) = list(tmp_path.iterdir())
        loaded = load_snapshots(run_dir)

        _, (gp_ax, ei_ax) = plt.subplots(2, 1)
        plot_iteration_snapshot(loaded[-1], (gp_ax, ei_ax))

        assert "best_x" in gp_ax.get_title()
        assert "lengthscale" in gp_ax.get_title()

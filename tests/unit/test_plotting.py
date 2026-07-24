"""Unit tests for plotting utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

from actgpr import mrr
from actgpr.plotting import plot_acquisition, plot_iteration_snapshot, plot_run_history


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

    def test_title_reports_best_y_and_stop_reason(self, run_dir: Path) -> None:
        """Test that the title surfaces the run's final outcome."""
        _, ax = plot_run_history(run_dir, show=False)

        assert "0.2" in ax.get_title()
        assert "max_iterations" in ax.get_title()

    def test_accepts_string_path(self, run_dir: Path) -> None:
        """Test that a plain string path works, not just a Path object."""
        fig, ax = plot_run_history(str(run_dir), show=False)
        assert ax is not None

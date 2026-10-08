"""Unit tests for plotting utilities."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from actgpr import mrr
from actgpr.plotting import METRIC_FIELDS, load_metrics

SEED = 25


class TestPlotRunHistory:
    """Tests for load_metrics — plotting a saved run from its path alone."""

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
            load_metrics(tmp_path, show=False)

    def test_accepts_only_the_run_directory(self, run_dir: Path) -> None:
        """Test that the run directory alone is enough to build the plot."""
        fig, axes = load_metrics(run_dir, show=False)

        assert fig is not None
        assert axes.shape == (2, 2)

    def test_draws_one_panel_per_metric(self, run_dir: Path) -> None:
        """Test that every validation metric gets its own panel.

        The four series have unrelated units and ranges, so one shared axes
        would flatten all but the largest into a line along zero.
        """
        _, axes = load_metrics(run_dir, show=False)

        titles = [ax.get_title() for ax in axes.flatten()]
        assert titles == list(METRIC_FIELDS)

        # Each panel plots one point per iteration, plus the zero reference
        # line the metric is read against.
        for ax in axes.flatten():
            metric_line = ax.get_lines()[0]
            assert len(metric_line.get_xdata()) == 5

    def test_title_reports_best_x_best_y_and_stop_reason(self, run_dir: Path) -> None:
        """Test that the figure title surfaces the run's final outcome.

        Labels match _plot_iteration_snapshot's, so a run reconstructed from
        results.h5 reports its outcome the same way as one plotted straight
        from memory. `best:` alone is ambiguous about which of x or y it is.
        """
        fig, _ = load_metrics(run_dir, show=False)
        title = fig._suptitle.get_text()

        assert "best_x: 1.0000" in title
        assert "best_y: 0.2000" in title
        assert "max_iterations" in title

    def test_accepts_string_path(self, run_dir: Path) -> None:
        """Test that a plain string path works, not just a Path object."""
        _, axes = load_metrics(str(run_dir), show=False)
        assert axes.shape == (2, 2)

    def test_max_ei_panel_is_log_scaled_by_default(self, run_dir: Path) -> None:
        """Test that max_ei gets a log axis without being asked.

        Matches plot_iterations()'s log-scaled EI default: max_ei spans
        orders of magnitude as a run converges, and a linear axis compresses
        that shrinkage into an invisible flat line at zero.
        """
        _, axes = load_metrics(run_dir, show=False)

        panels = dict(zip(METRIC_FIELDS, axes.flatten()))
        assert panels["max_ei"].get_yscale() == "log"

    def test_other_panels_stay_linear(self, run_dir: Path) -> None:
        """Test that the signed and zero-valued metrics keep a linear axis.

        prediction_error is signed and improvement is frequently exactly 0,
        neither of which a log axis can render, so only max_ei goes log.
        """
        _, axes = load_metrics(run_dir, show=False)

        panels = dict(zip(METRIC_FIELDS, axes.flatten()))
        for field in ("current_best", "improvement", "prediction_error"):
            assert panels[field].get_yscale() == "linear"

    def test_log_scale_false_leaves_every_panel_linear(self, run_dir: Path) -> None:
        """Test that opting out drops the log axis on the max_ei panel too."""
        _, axes = load_metrics(run_dir, show=False, log_scale=False)

        assert all(ax.get_yscale() == "linear" for ax in axes.flatten())

    def test_title_reports_final_hyperparameters(self, tmp_path: Path) -> None:
        """Test that the run's final hyperparameters reach the title.

        Mirrors _plot_iteration_snapshot, so the two figures surface the
        same information about the surrogate.
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

        fig, _ = load_metrics(tmp_path, show=False)
        title = fig._suptitle.get_text()

        assert "lengthscale: 1.25" in title
        assert "outputscale: 2.5" in title

    def test_title_omits_hyperparameters_when_absent(self, run_dir: Path) -> None:
        """Test that a record without them still produces a valid title."""
        fig, _ = load_metrics(run_dir, show=False)
        title = fig._suptitle.get_text()

        assert "lengthscale" not in title
        assert "best_x" in title


def _write_v030_record(run_dir: Path) -> None:
    """Write a results.h5 in the layout actgpr 0.3.0 produced.

    Input points were flat then: next_point, candidates and train_x were
    one-dimensional, and best_x and the fitted hyperparameters were scalar
    attributes.
    """
    candidates = np.linspace(-1.0, 1.0, 20)
    with h5py.File(run_dir / "results.h5", "w") as f:
        f.attrs["ei_threshold"] = 0.01
        history = f.create_group("history")
        history["iteration"] = np.array([1, 2])
        history["next_point"] = np.array([0.25, -0.5])
        for field in ("new_y", "current_best", "max_ei"):
            history[field] = np.array([0.5, 0.25])
        history["prediction_error"] = np.array([0.1, -0.1])
        history["improvement"] = np.array([0.0, 0.25])
        history["lengthscale"] = np.array([1.5, 1.25])
        history["outputscale"] = np.array([1.0, 1.0])
        history["noise"] = np.array([1e-4, 1e-4])
        for iteration in (1, 2):
            group = f.create_group(f"iterations/iter_{iteration:03d}")
            group["candidates"] = candidates
            group["f_mean"] = np.zeros(20)
            group["f_var"] = np.ones(20)
            group["ei_scores"] = np.linspace(0.0, 0.05, 20)
            group["train_x"] = np.array([-1.0, 1.0])
            group["train_y"] = np.array([1.0, 0.5])
        final = f.create_group("final")
        final.attrs["best_x"] = 0.25
        final.attrs["best_y"] = 0.25
        final.attrs["stop_reason"] = "max_iterations"
        final.attrs["n_iterations"] = 2
        final.attrs["fitted_lengthscale"] = 1.25
        final.attrs["fitted_outputscale"] = 1.0
        final.attrs["fitted_noise"] = 1e-4
        final["train_x"] = np.array([-1.0, 1.0, 0.25])
        final["train_y"] = np.array([1.0, 0.5, 0.25])


class TestReadingVersion030Records:
    """Tests that run directories written by actgpr 0.3.0 still load.

    0.4 stores input points with one column per dimension, but an existing
    archive of 1D runs must stay readable without rewriting it.
    """

    def test_metrics_figure_loads(self, tmp_path: Path) -> None:
        """Test that load_metrics reads the scalar best_x and hyperparameters."""
        _write_v030_record(tmp_path)

        fig, _ = load_metrics(tmp_path, show=False)
        title = fig._suptitle.get_text()

        assert "best_x: 0.2500" in title
        assert "lengthscale: 1.25" in title

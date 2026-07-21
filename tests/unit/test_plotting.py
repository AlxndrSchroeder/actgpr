"""Unit tests for plotting utilities."""

import matplotlib.pyplot as plt
import torch

from actgpr.plotting import plot_acquisition, plot_iteration_snapshot


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

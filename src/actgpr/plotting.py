"""Plotting utilities for active GPR optimisation.

Functions
---------
plot_gp
    Core plotting function: draws GP mean, 95% CI, and training data from raw tensors.
plot_surrogate
    Convenience wrapper: extracts tensors from a fitted surrogate, then calls plot_gp.
plot_acquisition
    Plots the Expected Improvement acquisition landscape.
plot_iteration_snapshot
    Draws one iteration's GP + EI side by side from a snapshot dict.
plot_run_history
    Plots validation metrics vs. iteration from a saved run's results.h5.
"""

from pathlib import Path

import h5py
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from actgpr.surrogate import GPyTorchSurrogate

# Half-width of the shaded confidence band in standard deviations;
# ±2σ covers ≈95% of a Gaussian posterior.
CI_STD_FACTOR = 2.0


def plot_gp(
    candidates: torch.Tensor,
    f_mean: torch.Tensor,
    f_var: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    next_point: float | None = None,
    ax: Axes | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot GP predictions from raw tensors.

    This is the core GP plotting function. All other GP plot functions
    delegate to this one.

    Parameters
    ----------
    candidates : torch.Tensor of shape (m,)
        The x-axis grid of input points.
    f_mean : torch.Tensor of shape (m,)
        The GP posterior mean at each candidate point.
    f_var : torch.Tensor of shape (m,)
        The GP posterior variance at each candidate point.
    train_x : torch.Tensor of shape (n,)
        The training input points.
    train_y : torch.Tensor of shape (n,)
        The training output values.
    next_point : float or None, optional
        The selected next input point. If provided, a vertical line is drawn.
    ax : matplotlib.axes.Axes or None, optional
        An existing axes to draw on. If None, a new figure and axes are created.
    show : bool, optional
        Whether to call plt.show() immediately, by default True.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes used for the plot.
    """
    assert f_mean.shape == f_var.shape == candidates.shape, (
        f"Shape mismatch: candidates={candidates.shape}, "
        f"f_mean={f_mean.shape}, f_var={f_var.shape}"
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    with torch.no_grad():
        f_std = torch.sqrt(f_var)
        lower = f_mean - CI_STD_FACTOR * f_std
        upper = f_mean + CI_STD_FACTOR * f_std

        ax.plot(
            train_x.numpy(),
            train_y.numpy(),
            "k*",
            markersize=10,
            label="Training data",
        )
        ax.plot(candidates.numpy(), f_mean.numpy(), "b", label="Mean prediction")
        ax.fill_between(
            candidates.numpy(),
            lower.numpy(),
            upper.numpy(),
            alpha=0.3,
            label="95% CI",
        )

        if next_point is not None:
            ax.axvline(
                next_point,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Next point (x={next_point:.2f})",
            )

        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()

    if show:
        plt.show()

    return fig, ax


def plot_surrogate(
    surrogate: GPyTorchSurrogate,
    test_x: torch.Tensor,
    ax: Axes | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot surrogate predictions against training data.

    Convenience wrapper that extracts prediction tensors from a fitted
    surrogate model and delegates to plot_gp.

    Parameters
    ----------
    surrogate : GPyTorchSurrogate
        A fitted surrogate model.
    test_x : torch.Tensor of shape (m,)
        The test input points used to compute predictions for plotting.
    ax : matplotlib.axes.Axes or None, optional
        An existing axes to draw on. If None, a new figure and axes are created.
    show : bool, optional
        Whether to call plt.show() immediately, by default True.
        Set to False when composing multiple plots.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes used for the plot.

    Raises
    ------
    RuntimeError
        If the surrogate has not been fitted or has no training data.
    """
    if (
        surrogate.model is None
        or surrogate.likelihood is None
        or surrogate.train_x is None
        or surrogate.train_y is None
    ):
        raise RuntimeError("The surrogate must be fitted before plotting.")

    preds = surrogate.predict(test_x)

    return plot_gp(
        candidates=test_x,
        f_mean=preds["f_mean"],
        f_var=preds["f_var"],
        train_x=surrogate.train_x,
        train_y=surrogate.train_y,
        next_point=None,
        ax=ax,
        show=show,
    )


def plot_acquisition(
    candidates: torch.Tensor,
    ei_scores: torch.Tensor,
    next_point: float | None = None,
    ax: Axes | None = None,
    show: bool = True,
    ylim: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Plot the Expected Improvement acquisition landscape.

    Parameters
    ----------
    candidates : torch.Tensor of shape (m,)
        The candidate input points that were scored.
    ei_scores : torch.Tensor of shape (m,)
        The EI score for each candidate point.
    next_point : float or None, optional
        The selected next input point. If provided, a vertical line is drawn.
    ax : matplotlib.axes.Axes or None, optional
        An existing axes to draw on. If None, a new figure and axes are created.
    show : bool, optional
        Whether to call plt.show() immediately, by default True.
        Set to False when composing multiple plots.
    ylim : tuple[float, float] or None, optional
        Fixed (min, max) for the y-axis. If None (default), matplotlib
        autoscales to this call's own EI scores. Pass a fixed range — e.g.
        the maximum EI score across an entire run — when comparing EI
        landscapes across iterations, so a shrinking maximum is visible
        rather than being autoscaled to fill the axes every time.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes used for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.plot(candidates.numpy(), ei_scores.numpy(), "g", label="Expected Improvement")
    ax.fill_between(candidates.numpy(), 0, ei_scores.numpy(), alpha=0.2, color="g")

    if next_point is not None:
        ax.axvline(
            next_point,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Next point (x={next_point:.2f})",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("EI score")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()

    if show:
        plt.show()

    return fig, ax


def plot_iteration_snapshot(
    snapshot: dict,
    axes: tuple[Axes, Axes],
    ei_ylim: tuple[float, float] | None = None,
) -> None:
    """Draw one iteration's GP and EI plots onto the given axes pair.

    Parameters
    ----------
    snapshot : dict
        A snapshot dictionary containing keys: ``candidates``, ``f_mean``,
        ``f_var``, ``train_x``, ``train_y``, ``ei_scores``, ``next_point``,
        ``iteration``, ``current_best``, ``max_ei``, ``prediction_error``,
        ``improvement``.
    axes : tuple[Axes, Axes]
        A pair of axes (gp_ax, ei_ax) to draw on.
    ei_ylim : tuple[float, float] or None, optional
        Fixed (min, max) for the EI subplot's y-axis, shared across all
        iterations being browsed. If None (default), the EI axis autoscales
        to this iteration's own scores.
    """
    gp_ax, ei_ax = axes

    plot_gp(
        candidates=snapshot["candidates"],
        f_mean=snapshot["f_mean"],
        f_var=snapshot["f_var"],
        train_x=snapshot["train_x"],
        train_y=snapshot["train_y"],
        next_point=snapshot["next_point"],
        ax=gp_ax,
        show=False,
    )
    gp_ax.set_title(
        f"Iteration {snapshot['iteration']} | "
        f"best: {snapshot['current_best']:.4f} | "
        f"pred_error: {snapshot['prediction_error']:.4f} | "
        f"improvement: {snapshot['improvement']:.4f}"
    )

    plot_acquisition(
        candidates=snapshot["candidates"],
        ei_scores=snapshot["ei_scores"],
        next_point=snapshot["next_point"],
        ax=ei_ax,
        show=False,
        ylim=ei_ylim,
    )
    ei_ax.set_title(f"EI | max: {snapshot['max_ei']:.6f}")


def plot_run_history(
    run_dir: Path | str,
    ax: Axes | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot validation metrics vs. iteration from a saved run's results.h5.

    Reads the ``/history`` series directly from ``results.h5`` — no
    OptimisationRun object is needed, so a past run can be visualised from
    its run directory alone at any later time.

    Parameters
    ----------
    run_dir : Path or str
        The run directory written by OptimisationRun.run() (the folder
        containing ``results.h5``, not the file itself).
    ax : matplotlib.axes.Axes or None, optional
        An existing axes to draw on. If None, a new figure and axes are created.
    show : bool, optional
        Whether to call plt.show() immediately, by default True.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes used for the plot.

    Raises
    ------
    FileNotFoundError
        If run_dir does not contain a results.h5 file.
    """
    h5_path = Path(run_dir) / "results.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"No results.h5 found in {run_dir} — is this a run directory "
            "written by OptimisationRun.run()?"
        )

    with h5py.File(h5_path, "r") as f:
        history = f["history"]
        iteration = history["iteration"][:]
        prediction_error = history["prediction_error"][:]
        improvement = history["improvement"][:]
        best_y = f["final"].attrs["best_y"]
        stop_reason = f["final"].attrs["stop_reason"]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.plot(iteration, prediction_error, "o-", label="prediction_error")
    ax.plot(iteration, improvement, "o-", label="improvement")
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.set_title(f"Run history | best_y: {best_y:.4f} | stop: {stop_reason}")
    ax.legend()

    if show:
        plt.show()

    return fig, ax

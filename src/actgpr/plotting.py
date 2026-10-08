"""Plotting utilities for active GPR optimisation.

One figure, reachable two ways: ``OptimisationRun.plot_metrics()`` draws it
from a run object you still hold, and ``load_metrics(run_dir)`` reads a run
back from its log directory and draws the same thing. Both work for any
number of inputs.

Functions
---------
load_metrics
    Reads a run directory and plots its validation metrics against
    iteration, one panel per metric.
"""

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from actgpr._points import format_values
from actgpr.surrogate import HYPERPARAMETER_KEYS


def _as_values(stored: object) -> tuple[float, ...]:
    """Read stored per-dimension values as a tuple, whether scalar or array."""
    # np.atleast_1d also accepts the scalars v0.3.0 wrote for best_x and the
    # hyperparameters, so older results.h5 files still load.
    return tuple(float(value) for value in np.atleast_1d(stored))


def _name_window(fig: Figure, title: str) -> None:
    """Title a figure's window so several open at once stay tellable apart.

    Matplotlib names windows "Figure 1", "Figure 2", and opens them all at
    the same default position, so the two actgpr figures land on top of one
    another and neither the title bar nor the window switcher says which is
    which.
    """
    manager = fig.canvas.manager
    if manager is not None:
        manager.set_window_title(title)


METRIC_FIELDS = ("current_best", "improvement", "max_ei", "prediction_error")


def _draw_metrics(
    iteration: Sequence[float],
    series: dict[str, Sequence[float]],
    best_x: Sequence[float],
    best_y: float,
    stop_reason: str,
    fitted_hyperparameters: dict[str, tuple[float, ...]] | None = None,
    show: bool = True,
    log_scale: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Draw the validation-metrics figure from already-loaded series.

    The single definition of this figure, shared by
    ``OptimisationRun.plot_metrics`` (which passes the series it holds in
    memory) and ``load_metrics`` (which reads them from a saved
    ``results.h5``), so the two cannot drift apart.

    One panel per metric rather than one shared axes: the four series have
    unrelated units and ranges (``max_ei`` spans orders of magnitude,
    ``improvement`` is frequently exactly zero, ``prediction_error`` is
    signed), so overlaying them flattens all but the largest into a line
    along zero.

    Parameters
    ----------
    iteration : sequence of float
        The iteration numbers, the shared x-axis of every panel.
    series : dict
        The per-iteration values, keyed by the names in METRIC_FIELDS.
        All entries must be the same length as iteration.
    best_x : sequence of float
        The input point with the lowest output, one value per input
        dimension, reported in the figure title.
    best_y : float
        The lowest output, reported in the figure title.
    stop_reason : str
        Which convergence criterion fired, reported in the figure title.
    fitted_hyperparameters : dict or None, optional
        The surrogate's final hyperparameters, added as a second title
        line when given.
    show : bool, optional
        Whether to call plt.show() immediately, by default True.
    log_scale : bool, optional
        If True (the default), the ``max_ei`` panel is log-scaled. EI
        shrinks by orders of magnitude as a run converges, which a linear
        axis compresses into an invisible flat line at zero.

    Returns
    -------
    tuple[Figure, numpy.ndarray]
        The figure and its 2x2 array of axes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _name_window(fig, "actgpr: validation metrics")

    for ax, field in zip(axes.flatten(), METRIC_FIELDS):
        ax.plot(iteration, series[field], "o-", color="tab:blue")
        ax.axhline(0, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel("iteration")
        ax.set_ylabel(field)
        ax.set_title(field)

    if log_scale:
        axes.flatten()[METRIC_FIELDS.index("max_ei")].set_yscale("log")

    # Same best_x/best_y labelling as the iteration snapshots, so the two
    # figures report the run's outcome identically whether it comes from
    # memory or from results.h5.
    title = (
        f"Validation metrics | best_x: {format_values(best_x)} | "
        f"best_y: {best_y:.4f} | stop: {stop_reason}"
    )
    if fitted_hyperparameters:
        title += "\nfinal " + " | ".join(
            f"{key}: {format_values(value, '.4g')}"
            for key, value in fitted_hyperparameters.items()
        )
    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes


def load_metrics(
    run_dir: Path | str,
    show: bool = True,
    log_scale: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot validation metrics vs. iteration from a saved run's results.h5.

    The from-the-logs counterpart to ``OptimisationRun.plot_metrics()``: it
    reads the ``/history`` series straight from ``results.h5``, with no
    OptimisationRun object needed, so a past run can be visualised from its
    run directory alone at any later time. Both open the identical figure.

    Parameters
    ----------
    run_dir : Path or str
        The run directory written by OptimisationRun.run() (the folder
        containing ``results.h5``, not the file itself).
    show : bool, optional
        Whether to call plt.show() immediately, by default True.
    log_scale : bool, optional
        If True (the default), the ``max_ei`` panel is log-scaled. EI
        shrinks by orders of magnitude as a run converges, so a linear axis
        hides the shrinkage that signals convergence.

    Returns
    -------
    tuple[Figure, numpy.ndarray]
        The figure and its 2x2 array of axes, one panel per metric.

    Raises
    ------
    FileNotFoundError
        If run_dir does not contain a results.h5 file.
    """
    h5_path = Path(run_dir) / "results.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"No results.h5 found in {run_dir}. Is this a run directory "
            "written by OptimisationRun.run()?"
        )

    with h5py.File(h5_path, "r") as f:
        history = f["history"]
        iteration = history["iteration"][:]
        series = {field: history[field][:] for field in METRIC_FIELDS}
        best_x = _as_values(f["final"].attrs["best_x"])
        best_y = f["final"].attrs["best_y"]
        stop_reason = f["final"].attrs["stop_reason"]
        # The hyperparameters the run finished with, written by
        # mrr.save_hdf5 when the surrogate reports them.
        fitted = {
            key: _as_values(f["final"].attrs[f"fitted_{key}"])
            for key in HYPERPARAMETER_KEYS
            if f"fitted_{key}" in f["final"].attrs
        }

    return _draw_metrics(
        iteration=iteration,
        series=series,
        best_x=best_x,
        best_y=best_y,
        stop_reason=stop_reason,
        fitted_hyperparameters=fitted,
        show=show,
        log_scale=log_scale,
    )

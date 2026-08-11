"""Plotting utilities for active GPR optimisation.

Two figures, each reachable two ways. The ``plot_`` prefix draws from a run
object you still hold and lives on ``OptimisationRun``; the ``load_``
prefix reads a run back from its log directory and lives here:

- the surrogate itself: ``run.plot_iterations()`` / ``load_iterations(dir)``
- validation metrics: ``run.plot_metrics()`` / ``load_metrics(dir)``

Each pair draws the identical figure, so which one to reach for depends
solely on what you still have to hand. Everything else in this module is a
private helper that these four are built from.

Functions
---------
load_iterations
    Reads a run directory and browses its iterations with an interactive
    slider over the GP fit and the EI landscape, one frame per iteration.
load_metrics
    Reads a run directory and plots its validation metrics against
    iteration, one panel per metric.
"""

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

# Half-width of the shaded confidence band in standard deviations;
# ±2σ covers ≈95% of a Gaussian posterior.
CI_STD_FACTOR = 2.0

# Default log-scale EI y-axis floor, one order of magnitude below
# ei_threshold, which keeps the threshold line inside the plot rather than at
# its bottom edge, with room to see the curve dip below it.
EI_LOG_FLOOR_MARGIN = 0.1
# Fallback log-scale floor when no ei_threshold is available.
EI_LOG_FLOOR_DEFAULT = 1e-8

# Surrogate hyperparameters reported in plot titles, when available. Named
# once so the iteration and metric figures stay in step.
HYPERPARAMETER_KEYS = ("lengthscale", "outputscale", "noise")


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


def _plot_gp(
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


def _plot_acquisition(
    candidates: torch.Tensor,
    ei_scores: torch.Tensor,
    next_point: float | None = None,
    ax: Axes | None = None,
    show: bool = True,
    ylim: tuple[float, float] | None = None,
    log_scale: bool = False,
    ei_threshold: float | None = None,
) -> tuple[Figure, Axes]:
    """Plot the Expected Improvement acquisition landscape.

    Parameters
    ----------
    candidates : torch.Tensor of shape (m,)
        The candidate input points that were scored.
    ei_scores : torch.Tensor of shape (m,)
        The EI score for each candidate point.
    next_point : float or None, optional
        The selected next input point. If provided, a vertical line is drawn,
        along with a marker at (next_point, max EI score) labelled with its
        value.
    ax : matplotlib.axes.Axes or None, optional
        An existing axes to draw on. If None, a new figure and axes are created.
    show : bool, optional
        Whether to call plt.show() immediately, by default True.
        Set to False when composing multiple plots.
    ylim : tuple[float, float] or None, optional
        Fixed (min, max) for the y-axis. If None (default), the range is
        either autoscaled (linear) or derived from ei_threshold (log_scale).
        Pass a fixed range, e.g. the maximum EI score across an entire run,
        when comparing EI landscapes across iterations, so a shrinking
        maximum is visible rather than being autoscaled to fill the axes
        every time.
    log_scale : bool, optional
        If True, draws the y-axis on a log scale, by default False. EI often
        shrinks by orders of magnitude as a run converges, which a linear
        axis compresses into an invisible flat line, so log scale keeps that
        shrinkage visible. EI is exactly 0 at training points (no
        uncertainty); since a log axis cannot show zero, scores are clamped
        to the y-axis floor before plotting.
    ei_threshold : float or None, optional
        The run's convergence threshold. If given, drawn as a horizontal
        reference line. When log_scale is True and ylim is not given, the
        y-axis floor defaults to one order of magnitude below this value,
        so the threshold line sits inside the plot rather than at its edge.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes used for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    if log_scale:
        ax.set_yscale("log")
        if ylim is not None:
            floor = ylim[0]
        elif ei_threshold is not None:
            floor = ei_threshold * EI_LOG_FLOOR_MARGIN
        else:
            floor = EI_LOG_FLOOR_DEFAULT
        assert floor > 0, f"log_scale requires a positive y-axis floor, got {floor}"
        plotted_scores = torch.clamp(ei_scores, min=floor)
    else:
        floor = 0.0
        plotted_scores = ei_scores

    ax.plot(
        candidates.numpy(), plotted_scores.numpy(), "g", label="Expected Improvement"
    )
    ax.fill_between(
        candidates.numpy(), floor, plotted_scores.numpy(), alpha=0.2, color="g"
    )

    if next_point is not None:
        ax.axvline(
            next_point,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Next point (x={next_point:.2f})",
        )
        # Mark the EI score at next_point (its highest value, by construction
        # of find_next_input_point) directly on the curve, not just in the
        # subplot title.
        true_max_ei = ei_scores.max().item()
        marker_y = plotted_scores.max().item()  # lands on the (possibly clamped) curve
        ax.plot(
            [next_point],
            [marker_y],
            "ro",
            markersize=7,
            label=f"Max EI = {true_max_ei:.2e}",
        )

    if ei_threshold is not None:
        ax.axhline(
            ei_threshold,
            color="grey",
            linestyle=":",
            linewidth=1.5,
            label=f"ei_threshold={ei_threshold:.2e}",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("EI score")
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif log_scale:
        # Fix the floor even without an explicit ylim, so it always matches
        # what the data was clamped to rather than whatever matplotlib
        # autoscales the bottom to.
        ax.set_ylim(bottom=floor)
    ax.legend()

    if show:
        plt.show()

    return fig, ax


def _plot_iteration_snapshot(
    snapshot: dict,
    axes: tuple[Axes, Axes],
    ei_ylim: tuple[float, float] | None = None,
    ei_log_scale: bool = False,
    ei_threshold: float | None = None,
) -> None:
    """Draw one iteration's GP and EI plots onto the given axes pair.

    Parameters
    ----------
    snapshot : dict
        A snapshot dictionary containing keys: ``candidates``, ``f_mean``,
        ``f_var``, ``train_x``, ``train_y``, ``ei_scores``, ``next_point``,
        ``iteration``, ``current_best``, ``max_ei``. ``prediction_error``
        and ``improvement`` are optional and absent for a convergence
        snapshot, whose ``next_point`` was scored but never evaluated.
    axes : tuple[Axes, Axes]
        A pair of axes (gp_ax, ei_ax) to draw on.
    ei_ylim : tuple[float, float] or None, optional
        Fixed (min, max) for the EI subplot's y-axis, shared across all
        iterations being browsed. If None (default), the EI axis autoscales
        to this iteration's own scores.
    ei_log_scale : bool, optional
        If True, draws the EI subplot's y-axis on a log scale. See
        _plot_acquisition for why this matters as EI shrinks during a run,
        by default False.
    ei_threshold : float or None, optional
        The run's convergence threshold, drawn as a horizontal reference
        line on the EI subplot. See _plot_acquisition.
    """
    gp_ax, ei_ax = axes

    _plot_gp(
        candidates=snapshot["candidates"],
        f_mean=snapshot["f_mean"],
        f_var=snapshot["f_var"],
        train_x=snapshot["train_x"],
        train_y=snapshot["train_y"],
        next_point=snapshot["next_point"],
        ax=gp_ax,
        show=False,
    )
    # current_best is the lowest Objective output so far, i.e. best_y. The
    # input point that achieved it is the argmin of this snapshot's own
    # train_y, which is the same tensor current_best was taken from, so the
    # reported pair always belongs together.
    best_index = int(torch.argmin(snapshot["train_y"]))
    best_x = snapshot["train_x"][best_index].item()

    if "prediction_error" in snapshot:
        title = (
            f"Iteration {snapshot['iteration']} | "
            f"best_x: {best_x:.4f} | "
            f"best_y: {snapshot['current_best']:.4f} | "
            f"pred_error: {snapshot['prediction_error']:.4f} | "
            f"improvement: {snapshot['improvement']:.4f}"
        )
    else:
        # The fit that triggered ei_threshold convergence: next_point was
        # scored but never evaluated, so there is no prediction_error or
        # improvement to report for it.
        title = (
            f"Iteration {snapshot['iteration']} (converged, not evaluated) | "
            f"best_x: {best_x:.4f} | "
            f"best_y: {snapshot['current_best']:.4f}"
        )

    # The hyperparameters of this iteration's fit, on a second line so the
    # first stays readable. Absent for snapshots from a surrogate that does
    # not report them, so they are optional here too.
    if all(key in snapshot for key in HYPERPARAMETER_KEYS):
        title += "\n" + " | ".join(
            f"{key}: {snapshot[key]:.4g}" for key in HYPERPARAMETER_KEYS
        )

    gp_ax.set_title(title)

    _plot_acquisition(
        candidates=snapshot["candidates"],
        ei_scores=snapshot["ei_scores"],
        next_point=snapshot["next_point"],
        ax=ei_ax,
        show=False,
        ylim=ei_ylim,
        log_scale=ei_log_scale,
        ei_threshold=ei_threshold,
    )
    ei_ax.set_title(f"EI | max: {snapshot['max_ei']:.6f}")


def _load_iteration_snapshots(run_dir: Path | str) -> list[dict]:
    """Rebuild a saved run's per-iteration snapshots from its results.h5.

    The per-iteration counterpart to ``load_metrics``: it returns the
    same snapshot dictionaries ``OptimisationRun`` holds in memory, so a
    finished run's iterations can be replotted without an OptimisationRun
    object. Pass the whole list to ``_draw_iteration_slider`` for the
    interactive slider, or one entry to ``_plot_iteration_snapshot`` to draw
    a single frame onto your own axes. Without this, anything replotting a
    saved run has to reassemble the snapshots field by field and silently
    loses whichever fields it forgets.

    The fit that triggered ``ei_threshold`` convergence is appended as the
    final snapshot when the run recorded one. It carries no
    ``prediction_error``/``improvement``, since its candidate was scored
    but never evaluated.

    Parameters
    ----------
    run_dir : Path or str
        The run directory written by OptimisationRun.run() (the folder
        containing ``results.h5``, not the file itself).

    Returns
    -------
    list[dict]
        One snapshot per iteration, in iteration order.

    Raises
    ------
    FileNotFoundError
        If run_dir does not contain a results.h5 file.
    RuntimeError
        If the run was executed without ``store_snapshots``, so the file
        holds no per-iteration GP arrays to rebuild from.
    """
    h5_path = Path(run_dir) / "results.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"No results.h5 found in {run_dir}. Is this a run directory "
            "written by OptimisationRun.run()?"
        )

    snapshots: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        if "iterations" not in f:
            raise RuntimeError(
                f"{h5_path} holds no per-iteration snapshots. Re-run with "
                "store_snapshots=True to record them."
            )

        history = f["history"]
        iterations = history["iteration"][:]

        for row, iteration in enumerate(iterations):
            group = f[f"iterations/iter_{int(iteration):03d}"]
            snapshot: dict = {"iteration": int(iteration)}
            for field in (
                "next_point",
                "new_y",
                "current_best",
                "max_ei",
                "prediction_error",
                "improvement",
            ):
                snapshot[field] = float(history[field][row])
            # Recorded only when the surrogate reports them.
            for field in HYPERPARAMETER_KEYS:
                if field in history:
                    snapshot[field] = float(history[field][row])
            for field in ("candidates", "f_mean", "f_var", "ei_scores"):
                snapshot[field] = torch.from_numpy(group[field][:])
            snapshot["train_x"] = torch.from_numpy(group["train_x"][:])
            snapshot["train_y"] = torch.from_numpy(group["train_y"][:])
            snapshots.append(snapshot)

        final = f["final"]
        if "converged_max_ei" in final.attrs:
            converged: dict = {
                "iteration": int(final.attrs["n_iterations"]),
                "next_point": float(final.attrs["converged_next_point"]),
                "current_best": float(final["train_y"][:].min()),
                "max_ei": float(final.attrs["converged_max_ei"]),
                "train_x": torch.from_numpy(final["train_x"][:]),
                "train_y": torch.from_numpy(final["train_y"][:]),
            }
            for field in ("candidates", "f_mean", "f_var", "ei_scores"):
                converged[field] = torch.from_numpy(final[f"converged_{field}"][:])
            for field in HYPERPARAMETER_KEYS:
                if f"fitted_{field}" in final.attrs:
                    converged[field] = float(final.attrs[f"fitted_{field}"])
            snapshots.append(converged)

    return snapshots


def _draw_iteration_slider(
    snapshots: list[dict],
    ei_threshold: float,
    log_scale: bool = True,
    show: bool = True,
) -> Slider:
    """Browse a list of snapshots with the interactive iteration slider.

    The plural counterpart to ``_plot_iteration_snapshot``, which draws a
    single frame onto axes you supply: this one owns the figure and adds
    the slider that scrubs through every frame. Pairs with
    ``_load_iteration_snapshots``, so a saved run can be browsed in two
    calls without an OptimisationRun object.

    This is the single definition of the slider figure. Both
    ``OptimisationRun.plot_iterations`` (passing the snapshots it holds in
    memory) and ``load_iterations`` (loading them from a saved
    ``results.h5``) delegate here, so the two cannot drift apart.

    Parameters
    ----------
    snapshots : list[dict]
        Per-iteration snapshots, as held by OptimisationRun or returned by
        ``_load_iteration_snapshots``.
    ei_threshold : float
        The run's convergence threshold, used to place the EI axis floor
        and draw the reference line.
    log_scale : bool, optional
        If True (the default), the EI axis is log-scaled.
    show : bool, optional
        If True (the default), calls ``plt.show()``. Pass False to keep the
        figure open for further composition, or to store the returned
        Slider somewhere durable before the window appears.

    Returns
    -------
    matplotlib.widgets.Slider
        The slider. **Keep a reference to it**: matplotlib holds only a weak
        one, so a slider that goes out of scope is garbage collected and
        silently stops responding to drags while still being drawn.

    Raises
    ------
    RuntimeError
        If snapshots is empty.
    """
    if not snapshots:
        raise RuntimeError(
            "No snapshots available. Set store_snapshots=True before calling run()."
        )

    # One fixed EI range across every frame, so the shrinking EI maximum
    # stays visible instead of each frame autoscaling to its own scores.
    max_ei_overall = max(s["ei_scores"].max().item() for s in snapshots)
    if log_scale:
        ei_ylim = (ei_threshold * EI_LOG_FLOOR_MARGIN, max_ei_overall * 2)
        threshold_line = ei_threshold
    else:
        ei_ylim = (0.0, max_ei_overall * 1.05)
        threshold_line = None

    fig, (gp_ax, ei_ax) = plt.subplots(2, 1, figsize=(10, 8))
    _name_window(fig, "actgpr: iterations (GP fit and EI)")
    plt.subplots_adjust(bottom=0.18, hspace=0.35)

    def _draw(index: int) -> None:
        """Render one frame onto the shared axes."""
        gp_ax.cla()
        ei_ax.cla()
        _plot_iteration_snapshot(
            snapshots[index],
            (gp_ax, ei_ax),
            ei_ylim=ei_ylim,
            ei_log_scale=log_scale,
            ei_threshold=threshold_line,
        )

    _draw(0)

    slider_ax = fig.add_axes([0.15, 0.04, 0.7, 0.04])
    slider = Slider(
        slider_ax,
        "Iteration",
        valmin=1,
        valmax=len(snapshots),
        valinit=1,
        valstep=1,
    )

    def _update(value: float) -> None:
        """Redraw both subplots for the selected iteration."""
        _draw(int(value) - 1)
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    if show:
        plt.show()

    return slider


def load_iterations(
    run_dir: Path | str,
    log_scale: bool = True,
    show: bool = True,
) -> Slider:
    """Browse a saved run's iterations with the same slider as a live run.

    The from-the-logs counterpart to ``OptimisationRun.plot_iterations()``:
    it reads a run directory's ``results.h5`` and opens the identical
    interactive figure, so a run finished last month can be stepped through
    exactly like one still in memory.

    Parameters
    ----------
    run_dir : Path or str
        The run directory written by OptimisationRun.run().
    log_scale : bool, optional
        If True (the default), the EI axis is log-scaled.
    show : bool, optional
        If True (the default), calls ``plt.show()``.

    Returns
    -------
    matplotlib.widgets.Slider
        The slider. **Assign it to a variable that outlives the call**, or
        matplotlib will garbage collect it and it will stop responding.

    Raises
    ------
    FileNotFoundError
        If run_dir does not contain a results.h5 file.
    RuntimeError
        If the run was executed without ``store_snapshots``.
    """
    snapshots = _load_iteration_snapshots(run_dir)

    with h5py.File(Path(run_dir) / "results.h5", "r") as f:
        ei_threshold = float(f.attrs["ei_threshold"])

    return _draw_iteration_slider(
        snapshots, ei_threshold, log_scale=log_scale, show=show
    )


# The validation metrics, one panel each, in reading order. Named once so
# plot_metrics and load_metrics cannot drift apart, and so a panel is added
# by editing this tuple alone.
METRIC_FIELDS = ("current_best", "improvement", "max_ei", "prediction_error")


def _draw_metrics(
    iteration: Sequence[float],
    series: dict[str, Sequence[float]],
    best_x: float,
    best_y: float,
    stop_reason: str,
    fitted_hyperparameters: dict[str, float] | None = None,
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
    best_x, best_y : float
        The run's outcome, reported in the figure title.
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
        f"Validation metrics | best_x: {best_x:.4f} | "
        f"best_y: {best_y:.4f} | stop: {stop_reason}"
    )
    if fitted_hyperparameters:
        title += "\nfinal " + " | ".join(
            f"{key}: {value:.4g}" for key, value in fitted_hyperparameters.items()
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
        best_x = f["final"].attrs["best_x"]
        best_y = f["final"].attrs["best_y"]
        stop_reason = f["final"].attrs["stop_reason"]
        # The hyperparameters the run finished with, written by
        # mrr.save_hdf5 when the surrogate reports them.
        fitted = {
            key: float(f["final"].attrs[f"fitted_{key}"])
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

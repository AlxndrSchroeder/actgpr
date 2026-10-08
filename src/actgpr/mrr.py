"""Minimal Reproducible Run (MRR) file I/O operations."""

import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

from actgpr.surrogate import HYPERPARAMETER_KEYS


def create_run_dir(
    base_path: Path,
    fit_mode: str,
    training_iter: int | None,
    ei_threshold: float,
    max_iterations: int,
    noise: float,
    lengthscale: float | Sequence[float] | None,
    outputscale: float | None,
    n_dims: int = 1,
) -> Path:
    """Create a timestamped run directory with parameters in the name.

    Parameters
    ----------
    base_path : Path
        The root directory where runs are stored (e.g., "results").
    fit_mode : str
        "training" or "notraining".
    training_iter : int | None
        Number of training iterations (if fit_mode is "training").
    ei_threshold : float
        Expected improvement threshold for convergence.
    max_iterations : int
        Maximum number of evaluations.
    noise : float
        Noise level for the surrogate.
    lengthscale : float, sequence of float, or None
        Lengthscale (if fit_mode is "notraining"). A per-dimension sequence
        is joined with hyphens in the folder name.
    outputscale : float | None
        Outputscale (if fit_mode is "notraining").
    n_dims : int, optional
        Number of input dimensions of the run, by default 1. Leads the
        parameter part of the name, since it is the first thing that
        distinguishes one run from another.

    Returns
    -------
    Path
        The created run directory path.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if fit_mode == "training":
        folder_name = (
            f"{timestamp}_{n_dims}d_training{training_iter}iter_"
            f"ei{ei_threshold}_maxiter{max_iterations}_n{noise}"
        )
    else:
        # A per-dimension lengthscale would otherwise render as "[1.0, 2.0]",
        # putting brackets, commas and spaces into a directory name.
        values = np.atleast_1d(np.asarray(lengthscale, dtype=float))
        if values.size > 1:
            lengthscale = "-".join(str(value) for value in values.tolist())
        folder_name = (
            f"{timestamp}_{n_dims}d_notraining_ei{ei_threshold}_"
            f"maxiter{max_iterations}_ls{lengthscale}_os{outputscale}_n{noise}"
        )

    run_dir = base_path / folder_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_config(run_dir: Path, config: dict[str, object]) -> None:
    """Write all run parameters to config.json."""
    config_path = run_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)


def write_manifest(run_dir: Path) -> None:
    """Compute SHA-256 of config.json and write manifest.json."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return

    with config_path.open("rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()

    manifest = {"config.json": f"sha256:{checksum}"}

    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)


def write_meta(
    run_dir: Path,
    run_start: datetime,
    run_end: datetime,
    best_x: Sequence[float],
    best_y: float,
    n_iterations: int,
    stop_reason: str,
) -> None:
    """Write environment and output summary to meta.json.

    Includes the package name, version, and repository URL so a meta.json
    file remains identifiable on its own, e.g. if shared or archived
    separately from this repository.
    """
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    try:
        package_metadata = importlib.metadata.metadata("actgpr")
        package_name = package_metadata["Name"]
        actgpr_version = package_metadata["Version"]
        # Project-URL entries look like "Repository, https://github.com/...";
        # pull the URL out of whichever one is labelled "Repository".
        repository = next(
            (
                url.split(", ", 1)[1]
                for url in package_metadata.get_all("Project-URL") or []
                if url.startswith("Repository,")
            ),
            "unknown",
        )
    except importlib.metadata.PackageNotFoundError:
        package_name = "unknown"
        actgpr_version = "unknown"
        repository = "unknown"

    libraries = {}
    for pkg in ["torch", "gpytorch", "h5py"]:
        try:
            libraries[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            libraries[pkg] = "unknown"

    meta = {
        "timestamp_utc": run_start.isoformat(),
        "duration_seconds": round((run_end - run_start).total_seconds(), 4),
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_name": package_name,
        "actgpr_version": actgpr_version,
        "repository": repository,
        "libraries": libraries,
        "output_summary": {
            "best_x": [float(value) for value in best_x],
            "best_y": float(best_y),
            "n_iterations": n_iterations,
            "stop_reason": stop_reason,
        },
    }

    meta_path = run_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


def save_hdf5(
    run_dir: Path,
    results: list[dict[str, object]],
    config: dict[str, object],
    final_train_x: torch.Tensor,
    final_train_y: torch.Tensor,
    best_x: Sequence[float],
    best_y: float,
    stop_reason: str,
    n_iterations: int,
    convergence_snapshot: dict[str, object] | None = None,
    fitted_hyperparameters: dict[str, tuple[float, ...]] | None = None,
) -> None:
    """Write a self-describing HDF5 file with the run history and results.

    Layout
    ------
    ``/`` (root)
        Attributes holding the run configuration (``n_dims``, bounds,
        thresholds, ...).
    ``history/``
        Per-iteration series, each a dataset with one row per iteration,
        aligned by the ``iteration`` index dataset: ``new_y``,
        ``current_best``, ``max_ei``, ``prediction_error``, ``improvement``
        (one value per row), ``next_point`` (one column per input
        dimension), plus ``lengthscale``/``outputscale``/``noise`` when the
        surrogate reports them, giving the hyperparameters behind each
        iteration's fit (lengthscale has one column per input dimension).
        Also ``point_and_output``, a heatmap-friendly view putting the
        evaluated input point and its output in one array, each column
        scaled to [0, 1] (see its own attributes).
        This is the single authoritative record of the run's history.
        Covers only *evaluated* iterations. See ``convergence_snapshot``
        below for the one fit that never reached evaluation. Any iteration's
        surrogate can be rebuilt from the training data up to that point
        plus that row's hyperparameters, so the fitted arrays themselves are
        not stored.
    ``final/``
        Attributes ``best_x`` (one value per input dimension), ``best_y``,
        ``stop_reason``, ``n_iterations``
        and the final ``train_x``/``train_y`` datasets. When ``stop_reason``
        is ``"ei_threshold"`` and ``convergence_snapshot`` is given, also
        holds ``converged_max_ei`` and ``converged_next_point``, the EI
        value that fell below the threshold and the point it pointed at.
        That fit's candidate was never evaluated, so it has no place in
        ``history/``, so this is the only place it is recorded.
        When ``fitted_hyperparameters`` is given, also holds
        ``fitted_lengthscale``/``fitted_outputscale``/``fitted_noise``,
        the surrogate's hyperparameters as the run left them.

    Parameters
    ----------
    convergence_snapshot : dict or None, optional
        The ``max_ei`` and ``next_point`` of the fit that triggered
        ei_threshold convergence, or None if the run stopped for another
        reason. The surrogate behind it is reconstructible from the training
        data and the recorded hyperparameters, so no arrays are stored.
    fitted_hyperparameters : dict or None, optional
        The surrogate's final hyperparameters, as returned by its
        ``hyperparameters()`` method. None when the surrogate does not
        expose one, or was never fitted (a crash before iteration 1).
        Recording these matters most for ``with_training`` runs, where
        ``config.json`` holds only the starting values Adam began from.
    """
    h5_path = run_dir / "results.h5"
    with h5py.File(h5_path, "w") as f:
        # Root attributes: the run configuration.
        for key, value in config.items():
            if value is not None:
                f.attrs[key] = value

        # History: per-iteration series sharing one iteration index.
        # Single authoritative record of the run's history.
        history = f.create_group("history")
        history.attrs["description"] = (
            "Per-iteration series, one row per iteration; align by the "
            "'iteration' dataset."
        )
        history.create_dataset(
            "iteration",
            data=np.array([res["iteration"] for res in results], dtype=np.int64),
        )
        # next_point is an input point, so it gets one column per input
        # dimension: shape (n_iterations, d).
        history.create_dataset(
            "next_point",
            data=np.array([res["next_point"] for res in results], dtype=np.float64),
        ).attrs["description"] = "One row per iteration, one column per input."
        for field in (
            "new_y",
            "current_best",
            "max_ei",
            "prediction_error",
            "improvement",
        ):
            history.create_dataset(
                field,
                data=np.array([res[field] for res in results], dtype=np.float64),
            )

        # A single heatmap-friendly view of the run: the evaluated input
        # point and its output side by side, one row per iteration. The raw
        # values are already in next_point/new_y; a viewer draws a heatmap
        # with one colour scale for every column, and the inputs and the
        # output generally span different ranges, so each column is scaled
        # to [0, 1] here. column_min/column_max make the raw values
        # recoverable, so this view loses nothing.
        if results:
            combined = np.column_stack(
                [
                    np.array([res["next_point"] for res in results], dtype=np.float64),
                    np.array([res["new_y"] for res in results], dtype=np.float64),
                ]
            )
            column_min = combined.min(axis=0)
            column_max = combined.max(axis=0)
            span = np.where(column_max == column_min, 1.0, column_max - column_min)
            scaled = history.create_dataset(
                "point_and_output", data=(combined - column_min) / span
            )
            n_dims = combined.shape[1] - 1
            scaled.attrs["description"] = (
                "One row per iteration: the evaluated input point and its "
                "output, each column scaled to [0, 1] so they share one "
                "colour scale. Recover a raw value with "
                "value * (column_max - column_min) + column_min, or read "
                "next_point and new_y directly. A column whose values are "
                "all equal is written as 0."
            )
            scaled.attrs["columns"] = [f"x{i + 1}" for i in range(n_dims)] + ["y"]
            scaled.attrs["column_min"] = column_min
            scaled.attrs["column_max"] = column_max

        # The surrogate hyperparameters behind each iteration's fit, present
        # only when the surrogate exposes them (see OptimisationRun.
        # _fitted_hyperparameters). Written as their own series so a
        # with_training run's retuning is visible per iteration rather than
        # collapsed to the final value. Each is shape (n_iterations, k), with
        # k the number of values: one per input dimension for lengthscale,
        # one for outputscale and noise.
        for field in HYPERPARAMETER_KEYS:
            if results and all(field in res for res in results):
                dataset = history.create_dataset(
                    field,
                    data=np.array([res[field] for res in results], dtype=np.float64),
                )
                dataset.attrs["description"] = (
                    "One row per iteration; lengthscale has one column per "
                    "input dimension."
                )

        # Final: run summary and final state.
        final_group = f.create_group("final")
        final_group.attrs["best_x"] = np.asarray(best_x, dtype=np.float64)
        final_group.attrs["best_y"] = float(best_y)
        final_group.attrs["stop_reason"] = stop_reason
        final_group.attrs["n_iterations"] = n_iterations
        final_group.create_dataset("train_x", data=final_train_x.numpy())
        final_group.create_dataset("train_y", data=final_train_y.numpy())

        # The surrogate's hyperparameters as the run left them. For
        # with_training these are what Adam converged to and are recorded
        # nowhere else: config.json is written before the loop starts, so it
        # can only hold the starting values.
        if fitted_hyperparameters is not None:
            for name, value in fitted_hyperparameters.items():
                final_group.attrs[f"fitted_{name}"] = np.asarray(
                    value, dtype=np.float64
                )

        if convergence_snapshot is not None:
            final_group.attrs["converged_max_ei"] = float(
                convergence_snapshot["max_ei"]
            )
            final_group.attrs["converged_next_point"] = np.asarray(
                convergence_snapshot["next_point"], dtype=np.float64
            )


def setup_file_logger(run_dir: Path) -> logging.FileHandler:
    """Add a FileHandler to the actgpr logger and return it."""
    logger = logging.getLogger("actgpr")
    # Make sure logger processes INFO level messages. NOTSET is 0.
    if logger.level not in (logging.DEBUG, logging.INFO):
        logger.setLevel(logging.INFO)

    log_path = run_dir / "run.log"
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return handler

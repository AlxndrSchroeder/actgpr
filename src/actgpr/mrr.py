"""Minimal Reproducible Run (MRR) file I/O operations."""

import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch


def create_run_dir(
    base_path: Path,
    fit_mode: str,
    training_iter: int | None,
    ei_threshold: float,
    max_iterations: int,
    noise: float,
    lengthscale: float | None,
    outputscale: float | None,
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
    lengthscale : float | None
        Lengthscale (if fit_mode is "notraining").
    outputscale : float | None
        Outputscale (if fit_mode is "notraining").

    Returns
    -------
    Path
        The created run directory path.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if fit_mode == "training":
        folder_name = (
            f"{timestamp}_training{training_iter}iter_"
            f"ei{ei_threshold}_maxiter{max_iterations}_n{noise}"
        )
    else:
        folder_name = (
            f"{timestamp}_notraining_ei{ei_threshold}_"
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
    best_x: float,
    best_y: float,
    n_iterations: int,
    stop_reason: str,
) -> None:
    """Write environment and output summary to meta.json.

    Includes the package name, version, and repository URL so a meta.json
    file remains identifiable on its own — e.g. if shared or archived
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
            "best_x": float(best_x),
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
    store_snapshots: bool,
    final_train_x: torch.Tensor,
    final_train_y: torch.Tensor,
    best_x: float,
    best_y: float,
    stop_reason: str,
    n_iterations: int,
    convergence_snapshot: dict[str, object] | None = None,
) -> None:
    """Write a self-describing HDF5 file with the run history and results.

    Layout
    ------
    ``/`` (root)
        Attributes holding the run configuration (bounds, thresholds, ...).
    ``history/``
        Per-iteration scalar series, each a dataset of length ``n_iterations``
        aligned by the ``iteration`` index dataset: ``next_point``, ``new_y``,
        ``current_best``, ``max_ei``, ``prediction_error``, ``improvement``.
        This is the single authoritative record of the run's scalar history.
        Covers only *evaluated* iterations — see ``convergence_snapshot``
        below for the one fit that never reached evaluation.
    ``iterations/iter_NNN/``
        Written only when ``store_snapshots`` is True: the GP snapshot arrays
        ``candidates``, ``f_mean``, ``f_var``, ``ei_scores``, ``train_x``,
        ``train_y`` for that iteration.
    ``final/``
        Attributes ``best_x``, ``best_y``, ``stop_reason``, ``n_iterations``
        and the final ``train_x``/``train_y`` datasets. When ``stop_reason``
        is ``"ei_threshold"`` and ``convergence_snapshot`` is given, also
        holds the GP/EI state of the fit that triggered convergence —
        attributes ``converged_max_ei``/``converged_next_point`` and
        datasets ``converged_candidates``/``converged_f_mean``/
        ``converged_f_var``/``converged_ei_scores``. That fit's candidate
        was never evaluated, so it has no place in ``history/`` or
        ``iterations/`` — this is the only place it is recorded.

    Parameters
    ----------
    convergence_snapshot : dict or None, optional
        The GP/EI snapshot of the fit that triggered ei_threshold
        convergence (``OptimisationRun._convergence_snapshot``), or None if
        the run stopped via max_iterations or store_snapshots was False.
    """
    h5_path = run_dir / "results.h5"
    with h5py.File(h5_path, "w") as f:
        # Root attributes: the run configuration.
        for key, value in config.items():
            if value is not None:
                f.attrs[key] = value

        # History: per-iteration scalar series sharing one iteration index.
        # Single authoritative record of the run's scalar history.
        history = f.create_group("history")
        history.attrs["description"] = (
            "Per-iteration scalar series; align by the 'iteration' dataset."
        )
        history.create_dataset(
            "iteration",
            data=np.array([res["iteration"] for res in results], dtype=np.int64),
        )
        for field in (
            "next_point",
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

        # Snapshot arrays, only when captured — one group per iteration.
        if store_snapshots:
            iter_group = f.create_group("iterations")
            for res in results:
                grp = iter_group.create_group(f"iter_{res['iteration']:03d}")
                grp.create_dataset("candidates", data=res["candidates"].numpy())
                grp.create_dataset("f_mean", data=res["f_mean"].numpy())
                grp.create_dataset("f_var", data=res["f_var"].numpy())
                grp.create_dataset("ei_scores", data=res["ei_scores"].numpy())
                grp.create_dataset("train_x", data=res["train_x"].numpy())
                grp.create_dataset("train_y", data=res["train_y"].numpy())

        # Final: run summary and final state.
        final_group = f.create_group("final")
        final_group.attrs["best_x"] = float(best_x)
        final_group.attrs["best_y"] = float(best_y)
        final_group.attrs["stop_reason"] = stop_reason
        final_group.attrs["n_iterations"] = n_iterations
        final_group.create_dataset("train_x", data=final_train_x.numpy())
        final_group.create_dataset("train_y", data=final_train_y.numpy())

        if convergence_snapshot is not None:
            final_group.attrs["converged_max_ei"] = float(
                convergence_snapshot["max_ei"]
            )
            final_group.attrs["converged_next_point"] = float(
                convergence_snapshot["next_point"]
            )
            for field in ("candidates", "f_mean", "f_var", "ei_scores"):
                final_group.create_dataset(
                    f"converged_{field}", data=convergence_snapshot[field].numpy()
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

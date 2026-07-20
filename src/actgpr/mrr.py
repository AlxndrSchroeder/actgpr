"""Minimal Reproducible Run (MRR) file I/O operations."""

import hashlib
import importlib.metadata
import json
import logging
import platform
from datetime import datetime
from pathlib import Path

import h5py
import torch


def create_run_dir(
    base_path: Path,
    fit_mode: str,
    training_iter: int | None,
    ei_threshold: float,
    max_evaluations: int,
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
    max_evaluations : int
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
            f"ei{ei_threshold}_eval{max_evaluations}_n{noise}"
        )
    else:
        folder_name = (
            f"{timestamp}_notraining_ei{ei_threshold}_"
            f"eval{max_evaluations}_ls{lengthscale}_os{outputscale}_n{noise}"
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
    """Write environment and output summary to meta.json."""
    try:
        import subprocess

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
        actgpr_version = importlib.metadata.version("actgpr")
    except importlib.metadata.PackageNotFoundError:
        actgpr_version = "unknown"

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
        "actgpr_version": actgpr_version,
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
) -> None:
    """Write self-describing HDF5 with iteration data and final results."""
    h5_path = run_dir / "results.h5"
    with h5py.File(h5_path, "w") as f:
        # Root attributes (config)
        for key, value in config.items():
            if value is not None:
                if isinstance(value, list):
                    f.attrs[key] = value
                else:
                    f.attrs[key] = value

        # Iterations group
        iter_group = f.create_group("iterations")
        for res in results:
            i = res["iteration"]
            grp = iter_group.create_group(f"iter_{i:03d}")

            # Scalar attributes
            grp.attrs["next_point"] = float(res["next_point"])
            grp.attrs["new_y"] = float(res["new_y"])
            grp.attrs["current_best"] = float(res["current_best"])
            grp.attrs["max_ei"] = float(res["max_ei"])
            grp.attrs["prediction_error"] = float(res["prediction_error"])
            grp.attrs["improvement"] = float(res["improvement"])

            # Tensor datasets (only if stored)
            if store_snapshots:
                grp.create_dataset("candidates", data=res["candidates"].numpy())
                grp.create_dataset("f_mean", data=res["f_mean"].numpy())
                grp.create_dataset("f_var", data=res["f_var"].numpy())
                grp.create_dataset("ei_scores", data=res["ei_scores"].numpy())
                grp.create_dataset("train_x", data=res["train_x"].numpy())
                grp.create_dataset("train_y", data=res["train_y"].numpy())

        # Final group
        final_group = f.create_group("final")
        final_group.attrs["best_x"] = float(best_x)
        final_group.attrs["best_y"] = float(best_y)
        final_group.attrs["stop_reason"] = stop_reason
        final_group.attrs["n_iterations"] = n_iterations

        final_group.create_dataset("train_x", data=final_train_x.numpy())
        final_group.create_dataset("train_y", data=final_train_y.numpy())


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

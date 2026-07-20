"""Optimisation run module for active GPR optimisation.

Orchestrates the active learning loop: fit surrogate, maximise acquisition
function, evaluate objective, repeat until convergence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider

from actgpr import mrr
from actgpr.acquisition import Acquisition
from actgpr.objective_fn import ObjectiveFn
from actgpr.plotting import plot_iteration_snapshot
from actgpr.surrogate import GPyTorchSurrogate


class OptimisationRun:
    """Orchestrates the active GPR optimisation loop.

    Coordinates the ObjectiveFn, Surrogate, and Acquisition components
    to iteratively find the minimum of the ObjectiveFn within the search bounds.

    The loop terminates when either the maximum EI score falls below
    ei_threshold (nothing left to gain) or the number of optimisation
    iterations reaches max_evaluations (budget cap) — whichever fires first.

    Use the classmethods ``with_training`` and ``without_training`` to
    construct an OptimisationRun. The raw ``__init__`` is available for
    advanced use but the classmethods are the preferred entry points.

    Public Methods
    --------------
    with_training()
        Construct an OptimisationRun that optimises GP hyperparameters each iteration.
    without_training()
        Construct an OptimisationRun with fixed GP hyperparameters.
    run()
        Execute the optimisation loop and return the results.
    plot_iterations()
        Open an interactive matplotlib slider to browse GP snapshots per iteration.
    """

    # TODO: add from_config() classmethod to construct from config.json

    def __init__(
        self,
        objective: ObjectiveFn,
        surrogate: GPyTorchSurrogate,
        search_bounds: tuple[float, float],
        initial_train_x: torch.Tensor | list[float],
        max_evaluations: int,
        ei_threshold: float,
        n_candidates: int = 500,
        noise: float = 1e-4,
        store_snapshots: bool = False,
        run_dir: Path | str | None = None,
        *,
        _train_hyperparameters: bool = True,
        _training_iter: int = 50,
        _lengthscale: float = 1.0,
        _outputscale: float = 1.0,
    ) -> None:
        """Initialize the OptimisationRun.

        Prefer using the classmethods ``with_training`` or ``without_training``
        instead of calling ``__init__`` directly.

        Parameters
        ----------
        objective : ObjectiveFn
            The objective function to minimise.
        surrogate : GPyTorchSurrogate
            The GP surrogate model used to approximate the objective.
        search_bounds : tuple[float, float]
            The closed interval (lo, hi) within which input points are considered.
        initial_train_x : torch.Tensor or list[float] of shape (n,)
            The initial input points to seed the optimisation loop. Cast to
            float64 regardless of input dtype, so integer-valued inputs
            (e.g. [1, 2]) don't silently truncate later fractional points
            appended during the optimisation loop.
        max_evaluations : int
            Maximum number of active optimisation iterations — GPR fit
            cycles — to execute (budget cap).
        ei_threshold : float
            The loop stops when the maximum EI score falls below this value.
        n_candidates : int, optional
            Number of candidate points for the acquisition function, by default 500.
        noise : float, optional
            Observation noise variance for the GP likelihood, by default 1e-4.
        store_snapshots : bool, optional
            If True, each iteration stores a snapshot of the GP predictions
            and EI scores for later interactive plotting via plot_iterations(),
            by default False.

        Raises
        ------
        ValueError
            If initial_train_x is empty, max_evaluations is not positive,
            search_bounds is not an increasing interval, or ei_threshold
            is not positive.
        """
        # Cast to float64 regardless of input dtype (list or tensor, int or
        # float) so later torch.cat calls never truncate fractional points.
        self.train_x = torch.as_tensor(initial_train_x, dtype=torch.float64).clone()

        if self.train_x.numel() == 0:
            raise ValueError("initial_train_x must contain at least one point.")
        if max_evaluations <= 0:
            raise ValueError(
                f"max_evaluations ({max_evaluations}) must be a positive integer."
            )
        if not search_bounds[0] < search_bounds[1]:
            raise ValueError(
                f"search_bounds lo ({search_bounds[0]}) must be < "
                f"hi ({search_bounds[1]})"
            )
        if ei_threshold <= 0:
            raise ValueError(f"ei_threshold must be positive, got {ei_threshold}")

        self.objective = objective
        self.surrogate = surrogate
        self.search_bounds = search_bounds
        self.noise = noise
        self.store_snapshots = store_snapshots
        self.max_evaluations = max_evaluations
        self.ei_threshold = ei_threshold
        self._run_dir = Path(run_dir) if run_dir is not None else None

        # Private fit-mode configuration — set by classmethods
        self._train_hyperparameters = _train_hyperparameters
        self._training_iter = _training_iter
        self._lengthscale = _lengthscale
        self._outputscale = _outputscale

        # Evaluate the objective at initial points to get train_y
        self.train_y = torch.tensor(
            self.objective.evaluate(*self.train_x.tolist()), dtype=self.train_x.dtype
        )
        assert self.train_x.numel() == self.train_y.numel(), (
            f"Objective returned {self.train_y.numel()} outputs for "
            f"{self.train_x.numel()} inputs"
        )

        # Create Acquisition once — it holds a reference to the surrogate
        self._acq = Acquisition(surrogate, search_bounds, n_candidates)

        # Deferred-write accumulator for per-iteration data
        self._results: list[dict] = []

    @classmethod
    def with_training(
        cls,
        objective: ObjectiveFn,
        surrogate: GPyTorchSurrogate,
        search_bounds: tuple[float, float],
        initial_train_x: torch.Tensor | list[float],
        max_evaluations: int,
        ei_threshold: float,
        n_candidates: int = 500,
        training_iter: int = 50,
        noise: float = 1e-4,
        store_snapshots: bool = False,
        run_dir: Path | str | None = None,
    ) -> OptimisationRun:
        """Construct an OptimisationRun that optimises GP hyperparameters.

        Each iteration fits the surrogate and optimises the kernel
        lengthscale, outputscale, and noise variance using Adam.

        Parameters
        ----------
        objective : ObjectiveFn
            The objective function to minimise.
        surrogate : GPyTorchSurrogate
            The GP surrogate model used to approximate the objective.
        search_bounds : tuple[float, float]
            The closed interval (lo, hi) within which input points are considered.
        initial_train_x : torch.Tensor or list[float] of shape (n,)
            The initial input points to seed the optimisation loop.
        max_evaluations : int
            Maximum number of active optimisation iterations — GPR fit
            cycles — to execute (budget cap).
        ei_threshold : float
            The loop stops when the maximum EI score falls below this value.
        n_candidates : int, optional
            Number of candidate points for the acquisition function, by default 500.
        training_iter : int, optional
            Number of hyperparameter optimisation iterations per surrogate fit,
            by default 50.
        noise : float, optional
            Initial observation noise variance for the GP likelihood,
            by default 1e-4.
        store_snapshots : bool, optional
            If True, stores GP snapshots for interactive plotting, by default False.

        Returns
        -------
        OptimisationRun
            A configured OptimisationRun that will train hyperparameters.
        """
        return cls(
            objective=objective,
            surrogate=surrogate,
            search_bounds=search_bounds,
            initial_train_x=initial_train_x,
            max_evaluations=max_evaluations,
            ei_threshold=ei_threshold,
            n_candidates=n_candidates,
            noise=noise,
            store_snapshots=store_snapshots,
            run_dir=run_dir,
            _train_hyperparameters=True,
            _training_iter=training_iter,
        )

    @classmethod
    def without_training(
        cls,
        objective: ObjectiveFn,
        surrogate: GPyTorchSurrogate,
        search_bounds: tuple[float, float],
        initial_train_x: torch.Tensor | list[float],
        max_evaluations: int,
        ei_threshold: float,
        n_candidates: int = 500,
        lengthscale: float = 1.0,
        outputscale: float = 1.0,
        noise: float = 1e-4,
        store_snapshots: bool = False,
        run_dir: Path | str | None = None,
    ) -> OptimisationRun:
        """Construct an OptimisationRun with fixed GP hyperparameters.

        Each iteration fits the surrogate with the given lengthscale,
        outputscale, and noise — no hyperparameter optimisation takes place.

        Parameters
        ----------
        objective : ObjectiveFn
            The objective function to minimise.
        surrogate : GPyTorchSurrogate
            The GP surrogate model used to approximate the objective.
        search_bounds : tuple[float, float]
            The closed interval (lo, hi) within which input points are considered.
        initial_train_x : torch.Tensor or list[float] of shape (n,)
            The initial input points to seed the optimisation loop.
        max_evaluations : int
            Maximum number of active optimisation iterations — GPR fit
            cycles — to execute (budget cap).
        ei_threshold : float
            The loop stops when the maximum EI score falls below this value.
        n_candidates : int, optional
            Number of candidate points for the acquisition function, by default 500.
        lengthscale : float, optional
            The RBF kernel lengthscale, by default 1.0.
        outputscale : float, optional
            The kernel outputscale (signal variance), by default 1.0.
        noise : float, optional
            The observation noise variance, by default 1e-4.
        store_snapshots : bool, optional
            If True, stores GP snapshots for interactive plotting, by default False.

        Returns
        -------
        OptimisationRun
            A configured OptimisationRun with fixed hyperparameters.
        """
        return cls(
            objective=objective,
            surrogate=surrogate,
            search_bounds=search_bounds,
            initial_train_x=initial_train_x,
            max_evaluations=max_evaluations,
            ei_threshold=ei_threshold,
            n_candidates=n_candidates,
            noise=noise,
            store_snapshots=store_snapshots,
            run_dir=run_dir,
            _train_hyperparameters=False,
            _lengthscale=lengthscale,
            _outputscale=outputscale,
        )

    def _fit_surrogate(self) -> None:
        """Fit the surrogate using the configured fit mode."""
        if self._train_hyperparameters:
            self.surrogate.fit_and_train(
                self.train_x,
                self.train_y,
                training_iter=self._training_iter,
                noise=self.noise,
            )
        else:
            self.surrogate.fit_no_training(
                self.train_x,
                self.train_y,
                lengthscale=self._lengthscale,
                outputscale=self._outputscale,
                noise=self.noise,
            )

    def _config_dict(self) -> dict[str, object]:
        """Return all configuration parameters for MRR recording."""
        return {
            "fit_mode": "training" if self._train_hyperparameters else "notraining",
            "search_bounds": list(self.search_bounds),
            "initial_train_x": self.train_x.tolist(),
            "max_evaluations": self.max_evaluations,
            "ei_threshold": self.ei_threshold,
            "n_candidates": self._acq.n_candidates,
            "noise": self.noise,
            "training_iter": (
                self._training_iter if self._train_hyperparameters else None
            ),
            "lengthscale": (
                self._lengthscale if not self._train_hyperparameters else None
            ),
            "outputscale": (
                self._outputscale if not self._train_hyperparameters else None
            ),
            "store_snapshots": self.store_snapshots,
        }

    def run(self) -> dict[str, object]:
        """Execute the optimisation loop.

        Iteratively fits the surrogate, finds the next input point via the
        acquisition function, and evaluates the objective until convergence.

        Returns
        -------
        dict
            A dictionary containing the optimisation results:

            - "best_x": float — the input point with the lowest Objective value.
            - "best_y": float — the lowest Objective value found.
            - "train_x": torch.Tensor — all evaluated input points.
            - "train_y": torch.Tensor — all Objective outputs.
            - "n_iterations": int — number of loop iterations executed.
            - "stop_reason": str — "ei_threshold" if EI dropped below
              ei_threshold, "max_evaluations" if budget cap was reached.
        """
        # ── MRR: setup (only if run_dir provided) ──
        logger = logging.getLogger("actgpr")
        file_handler = None
        actual_run_dir = None

        if self._run_dir is not None:
            actual_run_dir = mrr.create_run_dir(
                self._run_dir,
                fit_mode="training" if self._train_hyperparameters else "notraining",
                training_iter=(
                    self._training_iter if self._train_hyperparameters else None
                ),
                ei_threshold=self.ei_threshold,
                max_evaluations=self.max_evaluations,
                noise=self.noise,
                lengthscale=(
                    self._lengthscale if not self._train_hyperparameters else None
                ),
                outputscale=(
                    self._outputscale if not self._train_hyperparameters else None
                ),
            )
            mrr.write_config(actual_run_dir, self._config_dict())
            mrr.write_manifest(actual_run_dir)
            file_handler = mrr.setup_file_logger(actual_run_dir)

        run_start = datetime.now(timezone.utc)

        try:
            fit_mode = "training" if self._train_hyperparameters else "fixed"
            logger.info(
                f"Starting optimisation ({fit_mode}): "
                f"{self.train_x.numel()} initial points, "
                f"max_evaluations={self.max_evaluations}, "
                f"ei_threshold={self.ei_threshold}"
            )

            stop_reason, n_iterations = self._run_loop(logger)

            best_idx = torch.argmin(self.train_y)
            best_x = self.train_x[best_idx].item()
            best_y = self.train_y[best_idx].item()

            run_end = datetime.now(timezone.utc)

            # ── MRR: finalize (only if run_dir provided) ──
            if actual_run_dir is not None:
                mrr.save_hdf5(
                    actual_run_dir,
                    results=self._results,
                    config=self._config_dict(),
                    store_snapshots=self.store_snapshots,
                    final_train_x=self.train_x,
                    final_train_y=self.train_y,
                    best_x=best_x,
                    best_y=best_y,
                    stop_reason=stop_reason,
                    n_iterations=n_iterations,
                )
                mrr.write_meta(
                    actual_run_dir,
                    run_start=run_start,
                    run_end=run_end,
                    best_x=best_x,
                    best_y=best_y,
                    n_iterations=n_iterations,
                    stop_reason=stop_reason,
                )

            return {
                "best_x": best_x,
                "best_y": best_y,
                "train_x": self.train_x,
                "train_y": self.train_y,
                "n_iterations": n_iterations,
                "stop_reason": stop_reason,
            }
        finally:
            # Detach the run.log handler even if the loop raises — a leaked
            # handler would duplicate every log line in a later run() and
            # keep the log file open.
            if file_handler is not None:
                logger.removeHandler(file_handler)
                file_handler.close()

    def _run_loop(self, logger: logging.Logger) -> tuple[str, int]:
        """Execute the optimisation loop and return (stop_reason, n_iterations)."""
        stop_reason = "max_evaluations"
        n_iterations = 0

        while n_iterations < self.max_evaluations:
            n_iterations += 1

            # 1. Fit surrogate to all current training data
            # TODO: consider get_fantasy_model for faster updates
            #       without hyperparameter re-tuning
            self._fit_surrogate()

            # 2. Compute current best and find the next input point
            current_best = self.train_y.min().item()
            next_point = self._acq.find_next_input_point(current_best)
            max_ei = self._acq.ei_scores.max().item()

            # 3. Check EI convergence before evaluating the new point
            if max_ei < self.ei_threshold:
                logger.info(
                    f"Converged after {n_iterations} iterations "
                    f"(max EI {max_ei:.6f} < ei_threshold {self.ei_threshold})"
                )
                stop_reason = "ei_threshold"
                break

            # 4. Evaluate objective at the next point
            new_y = self.objective.evaluate(next_point)[0]

            # 5. Validation metrics
            # improvement Δᵢ = y_best before this iteration − y_best after it;
            # zero when the new point does not improve on current_best.
            best_idx = torch.argmax(self._acq.ei_scores)
            predicted_y = self._acq.f_mean[best_idx].item()
            prediction_error = predicted_y - new_y
            improvement = max(0.0, current_best - new_y)

            logger.info(
                f"Iteration {n_iterations} | "
                f"current_best: {current_best:.4f} | "
                f"next_point: {next_point:.4f} | "
                f"max_ei: {max_ei:.6f} | "
                f"pred_error: {prediction_error:.4f} | "
                f"improvement: {improvement:.4f}"
            )

            # 6. Accumulate per-iteration results
            # Snapshot train_x/train_y BEFORE appending the new point so the
            # next_point marker is not also shown as a training data point.
            iteration_data: dict = {
                "iteration": n_iterations,
                "next_point": next_point,
                "new_y": new_y,
                "current_best": current_best,
                "max_ei": max_ei,
                "prediction_error": prediction_error,
                "improvement": improvement,
            }

            if self.store_snapshots:
                iteration_data.update(
                    {
                        "candidates": self._acq.candidates.clone(),
                        "f_mean": self._acq.f_mean.clone(),
                        "f_var": self._acq.f_var.clone(),
                        "ei_scores": self._acq.ei_scores.clone(),
                        "train_x": self.train_x.clone(),
                        "train_y": self.train_y.clone(),
                    }
                )

            self._results.append(iteration_data)

            # 7. Append to training data (after snapshot)
            self.train_x = torch.cat(
                [self.train_x, torch.tensor([next_point], dtype=self.train_x.dtype)]
            )
            self.train_y = torch.cat(
                [self.train_y, torch.tensor([new_y], dtype=self.train_y.dtype)]
            )

        if stop_reason == "max_evaluations":
            logger.info(
                f"Stopped after {n_iterations} iterations "
                f"(reached max_evaluations={self.max_evaluations})"
            )

        return stop_reason, n_iterations

    def plot_iterations(self) -> None:
        """Open an interactive matplotlib figure to browse iterations.

        Creates a figure with two subplots (GP predictions on top,
        EI landscape on bottom) and a slider to scrub through iterations.

        Raises
        ------
        RuntimeError
            If store_snapshots was False or no snapshots were recorded.
        """
        snapshots = [r for r in self._results if "candidates" in r]
        if not snapshots:
            raise RuntimeError(
                "No snapshots available. Set store_snapshots=True before calling run()."
            )

        fig, (gp_ax, ei_ax) = plt.subplots(2, 1, figsize=(10, 8))
        plt.subplots_adjust(bottom=0.18, hspace=0.35)

        # Draw initial state
        plot_iteration_snapshot(snapshots[0], (gp_ax, ei_ax))

        # Slider axis sits below both subplots
        slider_ax = fig.add_axes([0.15, 0.04, 0.7, 0.04])
        slider = Slider(
            slider_ax,
            "Iteration",
            valmin=1,
            valmax=len(snapshots),
            valinit=1,
            valstep=1,
        )

        def _update(val: float) -> None:
            """Redraw both subplots for the selected iteration."""
            idx = int(val) - 1
            gp_ax.cla()
            ei_ax.cla()
            plot_iteration_snapshot(snapshots[idx], (gp_ax, ei_ax))
            fig.canvas.draw_idle()

        slider.on_changed(_update)

        plt.show()

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the OptimisationRun."""
        fit_mode = "training" if self._train_hyperparameters else "fixed"
        return (
            f"OptimisationRun("
            f"fit={fit_mode}, "
            f"bounds={self.search_bounds}, "
            f"max_eval={self.max_evaluations}, "
            f"ei_thresh={self.ei_threshold}, "
            f"n_points={self.train_x.numel()})"
        )

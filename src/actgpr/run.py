"""Optimisation run module for active GPR optimisation.

Orchestrates the active learning loop: fit surrogate, maximise acquisition
function, evaluate objective, repeat until convergence.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

from actgpr import mrr
from actgpr._points import as_points, format_values, parse_search_bounds
from actgpr.acquisition import Acquisition
from actgpr.objective_fn import Objective
from actgpr.plotting import METRIC_FIELDS, _draw_iteration_slider, _draw_metrics
from actgpr.surrogate import GPyTorchSurrogate


class OptimisationRun:
    """Orchestrates the active GPR optimisation loop.

    Coordinates the ObjectiveFn, Surrogate, and Acquisition components
    to iteratively find the minimum of the ObjectiveFn within the search bounds.

    The number of input dimensions is set by ``search_bounds``: one
    ``(lo, hi)`` pair per input. Every input point, including each entry of
    ``initial_train_x``, then has one coordinate per input, and the
    Objective is called with those coordinates as separate arguments.

    The loop terminates when either the maximum EI score falls below
    ei_threshold (nothing left to gain) or the number of optimisation
    iterations reaches max_iterations (budget cap), whichever fires first.

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
    plot_metrics()
        Plot this run's validation metrics against iteration.
    """

    # TODO: add from_config() classmethod to construct from config.json

    def __init__(
        self,
        objective: Objective,
        surrogate: GPyTorchSurrogate,
        search_bounds: Sequence[float] | Sequence[Sequence[float]],
        initial_train_x: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
        max_iterations: int,
        ei_threshold: float,
        n_candidates: int = 500,
        noise: float = 1e-4,
        store_snapshots: bool = True,
        run_dir: Path | str | None = None,
        *,
        _train_hyperparameters: bool = True,
        _training_iter: int = 50,
        _lengthscale: float | Sequence[float] = 1.0,
        _outputscale: float = 1.0,
    ) -> None:
        """Initialize the OptimisationRun.

        Prefer using the classmethods ``with_training`` or ``without_training``
        instead of calling ``__init__`` directly.

        Parameters
        ----------
        objective : Objective
            The Objective to minimise: any object exposing
            ``evaluate(*x) -> float``, called once per input point with that
            point's coordinates. Its type is never
            checked, only that method is called, so a class of your own
            wrapping a simulation works as well as an ``ObjectiveFn``.
        surrogate : GPyTorchSurrogate
            The GP surrogate model used to approximate the objective.
        search_bounds : sequence of (lo, hi) pairs, or a single (lo, hi) pair
            The closed interval of each input dimension, in order, e.g.
            ``[(-3.0, 3.0), (0.0, 1.0)]`` for two inputs. The number of pairs
            is the number of input dimensions. A single ``(lo, hi)`` pair is
            shorthand for one input dimension.
        initial_train_x : array-like of shape (n, d), or (n,) when d is 1
            The initial input points to seed the optimisation loop, one row
            per point with one coordinate per input dimension, e.g.
            ``[[0.0, 0.5], [1.0, 0.2]]`` for two 2D points. With a single
            input dimension a flat list such as ``[-3.0, 5.0]`` is accepted.
            Cast to float64 regardless of input dtype, so integer-valued
            inputs (e.g. [1, 2]) don't silently truncate later fractional
            points appended during the optimisation loop.
        max_iterations : int
            Maximum number of active optimisation iterations, meaning GPR fit
            cycles, to execute (budget cap).
        ei_threshold : float
            The loop stops when the maximum EI score falls below this value.
        n_candidates : int, optional
            Number of candidate points for the acquisition function, by default 500.
        noise : float, optional
            Observation noise variance for the GP likelihood, by default 1e-4.
        store_snapshots : bool, optional
            If True, each iteration also stores a snapshot of the GP
            predictions and EI scores for later interactive plotting via
            plot_iterations(), by default True. Pass False to skip them (they
            are the bulk of results.h5's size). The prediction_error and
            improvement history used by plotting.load_metrics() is
            recorded either way, regardless of this flag.

        Raises
        ------
        ValueError
            If initial_train_x is empty, its points do not have one
            coordinate per search_bounds pair, or one of them lies outside
            its search bounds; if max_iterations is not positive; if a
            search_bounds pair is not an increasing interval; or if
            ei_threshold is not positive.
        """
        # The number of (lo, hi) pairs is the number of input dimensions; it
        # is the one place the dimension is decided, everything else follows.
        self.search_bounds = parse_search_bounds(search_bounds)
        self.n_dims = len(self.search_bounds)

        # One row per input point, float64 regardless of input dtype (list or
        # tensor, int or float) so later torch.cat calls never truncate
        # fractional points.
        self.train_x = as_points(initial_train_x, n_dims=self.n_dims).clone()

        if self.train_x.shape[0] == 0:
            raise ValueError("initial_train_x must contain at least one point.")

        # The Objective is never evaluated outside the search bounds, so a
        # starting point outside them would be the one exception. Easy to
        # hit with several inputs, whose intervals differ, so check the
        # closed interval of every coordinate before anything runs.
        self._raise_for_points_outside_bounds()
        if max_iterations <= 0:
            raise ValueError(
                f"max_iterations ({max_iterations}) must be a positive integer."
            )
        if ei_threshold <= 0:
            raise ValueError(f"ei_threshold must be positive, got {ei_threshold}")

        self.objective = objective
        self.surrogate = surrogate
        self.noise = noise
        self.store_snapshots = store_snapshots
        self.max_iterations = max_iterations
        self.ei_threshold = ei_threshold
        self._run_dir = Path(run_dir) if run_dir is not None else None

        # Private fit-mode configuration, set by classmethods
        self._train_hyperparameters = _train_hyperparameters
        self._training_iter = _training_iter
        self._lengthscale = _lengthscale
        self._outputscale = _outputscale

        # Evaluate the objective once per initial input point to get train_y
        self.train_y = torch.tensor(
            [self._evaluate(point) for point in self.train_x.tolist()],
            dtype=torch.float64,
        )
        assert self.train_x.shape[0] == self.train_y.shape[0], (
            f"Got {self.train_y.shape[0]} outputs for "
            f"{self.train_x.shape[0]} input points"
        )

        # Create Acquisition once; it holds a reference to the surrogate
        self._acq = Acquisition(surrogate, self.search_bounds, n_candidates)

        # Deferred-write accumulator for per-iteration data
        self._results: list[dict] = []

        # Holds the Slider from plot_iterations() for its lifetime. Matplotlib
        # widgets stop responding if their only reference is garbage
        # collected. See the docstring of plot_iterations() for why this
        # must be an attribute, not a local variable.
        self._active_slider: Slider | None = None

        # GP/EI state of the fit that triggers ei_threshold convergence, if
        # any. That fit's next_point is scored but never evaluated, so it
        # has no place in _results/history, so this is its only record. Set
        # in _run_loop(), only when store_snapshots is True.
        self._convergence_snapshot: dict | None = None

        # The convergence criterion that ended the run, for plot_metrics.
        self._stop_reason: str = "not run"

        # The timestamped directory run() actually wrote to, set once the
        # run starts. run_dir above is only the base path, so without this
        # the caller cannot find its own MRR record afterwards.
        self.run_dir: Path | None = None

    @classmethod
    def with_training(
        cls,
        objective: Objective,
        surrogate: GPyTorchSurrogate,
        search_bounds: Sequence[float] | Sequence[Sequence[float]],
        initial_train_x: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
        max_iterations: int,
        ei_threshold: float,
        n_candidates: int = 500,
        training_iter: int = 50,
        noise: float = 1e-4,
        store_snapshots: bool = True,
        run_dir: Path | str | None = None,
    ) -> OptimisationRun:
        """Construct an OptimisationRun that optimises GP hyperparameters.

        Each iteration fits the surrogate and optimises the kernel
        lengthscale, outputscale, and noise variance using Adam.

        Parameters
        ----------
        objective : Objective
            The Objective to minimise: any object exposing
            ``evaluate(*x) -> float``, called once per input point with that
            point's coordinates. Its type is never
            checked, only that method is called, so a class of your own
            wrapping a simulation works as well as an ``ObjectiveFn``.
        surrogate : GPyTorchSurrogate
            The GP surrogate model used to approximate the objective.
        search_bounds : sequence of (lo, hi) pairs, or a single (lo, hi) pair
            The closed interval of each input dimension, in order, e.g.
            ``[(-3.0, 3.0), (0.0, 1.0)]`` for two inputs. The number of pairs
            is the number of input dimensions. A single ``(lo, hi)`` pair is
            shorthand for one input dimension.
        initial_train_x : array-like of shape (n, d), or (n,) when d is 1
            The initial input points to seed the optimisation loop, one row
            per point with one coordinate per input dimension, e.g.
            ``[[0.0, 0.5], [1.0, 0.2]]`` for two 2D points. With a single
            input dimension a flat list such as ``[-3.0, 5.0]`` is accepted.
        max_iterations : int
            Maximum number of active optimisation iterations, meaning GPR fit
            cycles, to execute (budget cap).
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
            If True, also stores GP snapshots for interactive plotting via
            plot_iterations(), by default True. Pass False to skip them (they
            are the bulk of results.h5's size). The prediction_error and
            improvement history used by plotting.load_metrics() is
            recorded either way, regardless of this flag.

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
            max_iterations=max_iterations,
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
        objective: Objective,
        surrogate: GPyTorchSurrogate,
        search_bounds: Sequence[float] | Sequence[Sequence[float]],
        initial_train_x: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
        max_iterations: int,
        ei_threshold: float,
        n_candidates: int = 500,
        lengthscale: float | Sequence[float] = 1.0,
        outputscale: float = 1.0,
        noise: float = 1e-4,
        store_snapshots: bool = True,
        run_dir: Path | str | None = None,
    ) -> OptimisationRun:
        """Construct an OptimisationRun with fixed GP hyperparameters.

        Each iteration fits the surrogate with the given lengthscale,
        outputscale, and noise, so no hyperparameter optimisation takes place.

        Parameters
        ----------
        objective : Objective
            The Objective to minimise: any object exposing
            ``evaluate(*x) -> float``, called once per input point with that
            point's coordinates. Its type is never
            checked, only that method is called, so a class of your own
            wrapping a simulation works as well as an ``ObjectiveFn``.
        surrogate : GPyTorchSurrogate
            The GP surrogate model used to approximate the objective.
        search_bounds : sequence of (lo, hi) pairs, or a single (lo, hi) pair
            The closed interval of each input dimension, in order, e.g.
            ``[(-3.0, 3.0), (0.0, 1.0)]`` for two inputs. The number of pairs
            is the number of input dimensions. A single ``(lo, hi)`` pair is
            shorthand for one input dimension.
        initial_train_x : array-like of shape (n, d), or (n,) when d is 1
            The initial input points to seed the optimisation loop, one row
            per point with one coordinate per input dimension, e.g.
            ``[[0.0, 0.5], [1.0, 0.2]]`` for two 2D points. With a single
            input dimension a flat list such as ``[-3.0, 5.0]`` is accepted.
        max_iterations : int
            Maximum number of active optimisation iterations, meaning GPR fit
            cycles, to execute (budget cap).
        ei_threshold : float
            The loop stops when the maximum EI score falls below this value.
        n_candidates : int, optional
            Number of candidate points for the acquisition function, by default 500.
        lengthscale : float or sequence of float, optional
            The RBF kernel lengthscale, by default 1.0. A single value is
            used for every input dimension; a sequence gives one value per
            input dimension, in order.
        outputscale : float, optional
            The kernel outputscale (signal variance), by default 1.0.
        noise : float, optional
            The observation noise variance, by default 1e-4.
        store_snapshots : bool, optional
            If True, also stores GP snapshots for interactive plotting via
            plot_iterations(), by default True. Pass False to skip them (they
            are the bulk of results.h5's size). The prediction_error and
            improvement history used by plotting.load_metrics() is
            recorded either way, regardless of this flag.

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
            max_iterations=max_iterations,
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
        """Return all configuration parameters for MRR recording.

        ``"objective"`` is ``repr(self.objective)`` rather than a specific
        field, since the Objective is duck-typed and OptimisationRun has no
        generic way to know what a particular Objective considers worth
        recording (e.g. ObjectiveFn's ``jitter``). repr() is the one thing
        every object provides, and any Objective can put whatever it wants
        in its own __repr__ to show up here.
        """

        return {
            "objective": repr(self.objective),
            "fit_mode": "training" if self._train_hyperparameters else "notraining",
            "n_dims": self.n_dims,
            "search_bounds": [list(pair) for pair in self.search_bounds],
            "initial_train_x": self.train_x.tolist(),
            "max_iterations": self.max_iterations,
            "ei_threshold": self.ei_threshold,
            "n_candidates": self._acq.n_candidates,
            "candidate_seed": self._acq.seed,
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

    def _raise_for_points_outside_bounds(self) -> None:
        """Raise if any initial input point lies outside its search bounds."""
        for row, point in enumerate(self.train_x.tolist()):
            for dim, (value, (lo, hi)) in enumerate(zip(point, self.search_bounds)):
                if not lo <= value <= hi:
                    raise ValueError(
                        f"initial_train_x point {row} has x{dim + 1}={value}, "
                        f"outside its search bounds ({lo}, {hi})."
                    )

    def _evaluate(self, point: list[float]) -> float:
        """Evaluate the Objective at one input point, given its coordinates."""
        output = self.objective.evaluate(*point)
        try:
            return float(output)
        except (TypeError, ValueError) as exc:
            # The most likely cause is an Objective written for actgpr 0.3,
            # whose evaluate() took several points and returned a tuple.
            raise TypeError(
                f"Objective.evaluate() must return one float for one input "
                f"point, got {type(output).__name__} {output!r}. Since actgpr "
                f"0.4, evaluate(*x) receives the coordinates of a single input "
                f"point and returns its single output."
            ) from exc

    def _fitted_hyperparameters(self) -> dict[str, tuple[float, ...]] | None:
        """Return the surrogate's final hyperparameters, or None if unavailable.

        Read through an optional ``hyperparameters()`` method rather than
        reaching into the surrogate's internals, so the backend stays
        swappable: a surrogate that does not offer one simply contributes
        nothing to the record, exactly as before. Also returns None when the
        surrogate was never fitted, which happens if a run crashes before
        completing its first iteration.
        """
        getter = getattr(self.surrogate, "hyperparameters", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except RuntimeError:
            return None

    def _write_mrr_record(
        self,
        actual_run_dir: Path,
        run_start: datetime,
        best_x: tuple[float, ...],
        best_y: float,
        stop_reason: str,
        n_iterations: int,
    ) -> None:
        """Write results.h5 and meta.json for the given outcome.

        Shared by both the normal end-of-run finalization and the
        best-effort checkpoint written on a mid-run crash.
        """
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
            convergence_snapshot=self._convergence_snapshot,
            fitted_hyperparameters=self._fitted_hyperparameters(),
        )
        mrr.write_meta(
            actual_run_dir,
            run_start=run_start,
            run_end=datetime.now(timezone.utc),
            best_x=best_x,
            best_y=best_y,
            n_iterations=n_iterations,
            stop_reason=stop_reason,
        )

    def run(self) -> dict[str, object]:
        """Execute the optimisation loop.

        Iteratively fits the surrogate, finds the next input point via the
        acquisition function, and evaluates the objective until convergence.

        Returns
        -------
        dict
            A dictionary containing the optimisation results:

            - "best_x": tuple of float, the input point with the lowest
              Objective value, one coordinate per input dimension (a
              1-tuple for a single input).
            - "best_y": float, the lowest Objective value found.
            - "train_x": torch.Tensor of shape (n, d), all evaluated input
              points, one row each.
            - "train_y": torch.Tensor, all Objective outputs.
            - "n_iterations": int, number of loop iterations executed.
            - "stop_reason": str, "ei_threshold" if EI dropped below
              ei_threshold, "max_iterations" if budget cap was reached.

        Raises
        ------
        Exception
            Any exception raised while fitting the surrogate or evaluating
            the objective propagates unchanged. If ``run_dir`` was given, a
            best-effort results.h5/meta.json checkpoint covering every
            iteration completed before the failure is written first (with
            ``stop_reason="crashed"``), so a mid-run crash only loses the
            incomplete iteration rather than the whole run.
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
                max_iterations=self.max_iterations,
                noise=self.noise,
                n_dims=self.n_dims,
                lengthscale=(
                    self._lengthscale if not self._train_hyperparameters else None
                ),
                outputscale=(
                    self._outputscale if not self._train_hyperparameters else None
                ),
            )
            self.run_dir = actual_run_dir
            mrr.write_config(actual_run_dir, self._config_dict())
            mrr.write_manifest(actual_run_dir)
            file_handler = mrr.setup_file_logger(actual_run_dir)

        run_start = datetime.now(timezone.utc)

        try:
            fit_mode = "training" if self._train_hyperparameters else "fixed"
            logger.info(
                f"Starting optimisation ({fit_mode}): "
                f"{self.train_x.shape[0]} initial points in {self.n_dims}D, "
                f"max_iterations={self.max_iterations}, "
                f"ei_threshold={self.ei_threshold}"
            )

            stop_reason, n_iterations = self._run_loop(logger)
            self._stop_reason = stop_reason

            best_idx = torch.argmin(self.train_y)
            best_x = tuple(self.train_x[best_idx].tolist())
            best_y = self.train_y[best_idx].item()

            logger.info(
                f"Finished after {n_iterations} iterations ({stop_reason}): "
                f"best_x={format_values(best_x, '.6f')}, best_y={best_y:.6f}"
            )

            fitted = self._fitted_hyperparameters()
            if fitted is not None:
                logger.info(
                    "Final surrogate hyperparameters: "
                    + ", ".join(
                        f"{k}={format_values(v, '.6g')}" for k, v in fitted.items()
                    )
                )

            # ── MRR: finalize (only if run_dir provided) ──
            if actual_run_dir is not None:
                self._write_mrr_record(
                    actual_run_dir, run_start, best_x, best_y, stop_reason, n_iterations
                )

            return {
                "best_x": best_x,
                "best_y": best_y,
                "train_x": self.train_x,
                "train_y": self.train_y,
                "n_iterations": n_iterations,
                "stop_reason": stop_reason,
            }
        except Exception:
            # Best-effort checkpoint: self.train_x/train_y/_results only
            # ever reflect fully-completed iterations (appended after each
            # iteration's data is snapshotted, see _run_loop step 7), so
            # they're safe to persist even mid-crash. Otherwise a failure
            # on iteration 19 of 20 would discard the whole run instead of
            # losing just the incomplete iteration.
            if actual_run_dir is not None:
                crash_best_idx = torch.argmin(self.train_y)
                crash_best_x = tuple(self.train_x[crash_best_idx].tolist())
                crash_best_y = self.train_y[crash_best_idx].item()
                self._write_mrr_record(
                    actual_run_dir,
                    run_start,
                    crash_best_x,
                    crash_best_y,
                    stop_reason="crashed",
                    n_iterations=len(self._results),
                )
                logger.error(
                    f"Run crashed after {len(self._results)} completed "
                    f"iterations, checkpoint written to {actual_run_dir}"
                )
            raise
        finally:
            # Detach the run.log handler even if the loop raises. A leaked
            # handler would duplicate every log line in a later run() and
            # keep the log file open.
            if file_handler is not None:
                logger.removeHandler(file_handler)
                file_handler.close()

    def _run_loop(self, logger: logging.Logger) -> tuple[str, int]:
        """Execute the optimisation loop and return (stop_reason, n_iterations)."""
        stop_reason = "max_iterations"
        n_iterations = 0

        while n_iterations < self.max_iterations:
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
                if self.store_snapshots:
                    # This fit's next_point is never evaluated, so it has
                    # no place among the normal per-iteration snapshots, so
                    # record it separately (see docstring of run.py's
                    # convergence_snapshot in mrr.save_hdf5).
                    self._convergence_snapshot = {
                        "iteration": n_iterations,
                        "next_point": next_point,
                        "current_best": current_best,
                        "max_ei": max_ei,
                        "candidates": self._acq.candidates.clone(),
                        "f_mean": self._acq.f_mean.clone(),
                        "f_var": self._acq.f_var.clone(),
                        "ei_scores": self._acq.ei_scores.clone(),
                        "train_x": self.train_x.clone(),
                        "train_y": self.train_y.clone(),
                    }
                    # Carry the hyperparameters too, so the converged frame
                    # reports the same fields as every other iteration.
                    converged_hyperparameters = self._fitted_hyperparameters()
                    if converged_hyperparameters is not None:
                        self._convergence_snapshot.update(converged_hyperparameters)
                break

            # 4. Evaluate objective at the next point
            new_y = self._evaluate(list(next_point))

            # 5. Validation metrics
            # improvement Δᵢ = y_best before this iteration − y_best after it;
            # zero when the new point does not improve on current_best.
            # next_point_mean is the surrogate's mean at next_point itself
            # (post zoom-refinement), not one of the coarse f_mean entries,
            # which cover the candidate grid rather than next_point exactly.
            predicted_y = self._acq.next_point_mean
            prediction_error = predicted_y - new_y
            improvement = max(0.0, current_best - new_y)

            logger.info(
                f"Iteration {n_iterations} | "
                f"current_best: {current_best:.4f} | "
                f"next_point: {format_values(next_point)} | "
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

            # The hyperparameters of the fit that produced this iteration's
            # EI landscape and next_point. In with_training they are retuned
            # every iteration, so a single final value cannot describe the
            # run; recording them per iteration is the only way to see how
            # the surrogate evolved.
            hyperparameters = self._fitted_hyperparameters()
            if hyperparameters is not None:
                iteration_data.update(hyperparameters)

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

        if stop_reason == "max_iterations":
            logger.info(
                f"Stopped after {n_iterations} iterations "
                f"(reached max_iterations={self.max_iterations})"
            )

        return stop_reason, n_iterations

    def plot_metrics(
        self,
        show: bool = True,
        log_scale: bool = True,
    ) -> tuple[Figure, np.ndarray]:
        """Plot this run's validation metrics against iteration.

        The from-the-run counterpart to ``plotting.load_metrics``, which
        reads the same series back from a saved ``results.h5``. Both draw
        the identical figure, one panel per metric. Use this one while the
        run object is still to hand, including for a run that wrote no MRR
        record and therefore has no directory to read back.

        Parameters
        ----------
        show : bool, optional
            Whether to call plt.show() immediately, by default True.
        log_scale : bool, optional
            If True (the default), the ``max_ei`` panel is log-scaled.

        Returns
        -------
        tuple[Figure, numpy.ndarray]
            The figure and its 2x2 array of axes, one panel per metric.

        Raises
        ------
        RuntimeError
            If the run has not been executed yet, so there is no history.
        """
        if not self._results:
            raise RuntimeError(
                "No history available. Call run() before plot_metrics()."
            )

        best_index = torch.argmin(self.train_y)

        return _draw_metrics(
            iteration=[record["iteration"] for record in self._results],
            series={
                field: [record[field] for record in self._results]
                for field in METRIC_FIELDS
            },
            best_x=tuple(self.train_x[best_index].tolist()),
            best_y=self.train_y[best_index].item(),
            stop_reason=self._stop_reason,
            fitted_hyperparameters=self._fitted_hyperparameters(),
            show=show,
            log_scale=log_scale,
        )

    def plot_iterations(self, show: bool = True, log_scale: bool = True) -> None:
        """Open an interactive matplotlib figure to browse iterations.

        Creates a figure with two subplots (GP predictions on top,
        EI landscape on bottom) and a slider to scrub through iterations.

        The from-the-run counterpart to ``plotting.load_iterations``, which
        opens the identical figure for a run read back from its logs.

        The Slider is kept alive via ``self._active_slider`` for as long as
        the OptimisationRun exists. Matplotlib does not keep its own strong
        reference to a Slider. If the only reference were a local variable
        here, it would be garbage collected as soon as this method returns,
        which happens immediately whenever ``plt.show()`` does not block
        (backend- and environment-dependent). The slider would still be
        drawn, but would silently stop responding to drags.

        Parameters
        ----------
        show : bool, optional
            Whether to call plt.show() immediately, by default True. Pass
            False when opening this alongside another figure: plt.show()
            displays *every* open figure, so calling it once per figure
            re-displays the earlier ones. Build both, then call plt.show()
            once yourself.
        log_scale : bool, optional
            If True, draws the EI subplot's y-axis on a log scale, with the
            ei_threshold convergence criterion marked as a reference line.
            EI often shrinks by orders of magnitude as a run converges,
            which a linear axis compresses into an invisible flat line, so
            log scale keeps that shrinkage visible, so it is the default.
            Pass False for a linear EI axis.

        Notes
        -----
        If the run converged via ei_threshold, the final frame is the fit
        that triggered convergence. Its next_point was scored but never
        evaluated, shown with a title noting "(converged, not evaluated)"
        instead of the usual pred_error/improvement values.

        Raises
        ------
        RuntimeError
            If the run has not been executed yet, or if it was executed
            with store_snapshots=False so no snapshots were recorded.
        ValueError
            If the run has more than one input dimension; its surrogate
            cannot be drawn as a curve. Use ``plot_metrics()`` for such runs.
        """
        # Distinguished from the store_snapshots case below: both leave no
        # snapshots to browse, but telling someone who has not called run()
        # to re-run with store_snapshots=True sends them after the wrong
        # cause. Mirrors plot_metrics.
        if not self._results:
            raise RuntimeError(
                "No iterations available. Call run() before plot_iterations()."
            )

        snapshots = [r for r in self._results if "candidates" in r]
        if self._convergence_snapshot is not None:
            snapshots = snapshots + [self._convergence_snapshot]

        # Always drawn with show=False so the slider is stored before any
        # window opens: with a blocking backend plt.show() does not return
        # until it is closed, and the reference must already be held by then.
        slider = _draw_iteration_slider(
            snapshots, self.ei_threshold, log_scale=log_scale, show=False
        )
        self._active_slider = slider

        if show:
            plt.show()

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the OptimisationRun."""
        fit_mode = "training" if self._train_hyperparameters else "fixed"
        return (
            f"OptimisationRun("
            f"fit={fit_mode}, "
            f"n_dims={self.n_dims}, "
            f"bounds={list(self.search_bounds)}, "
            f"max_iter={self.max_iterations}, "
            f"ei_thresh={self.ei_threshold}, "
            f"n_points={self.train_x.shape[0]})"
        )

"""Optimisation run module for active GPR optimisation.

Orchestrates the active learning loop: fit surrogate, maximise acquisition
function, evaluate objective, repeat until convergence.
"""

import torch

from actgpr.acquisition import Acquisition
from actgpr.objective_fn import ObjectiveFn
from actgpr.surrogate import GPyTorchSurrogate


class OptimisationRun:
    """Orchestrates the active GPR optimisation loop.

    Coordinates the ObjectiveFn, Surrogate, and Acquisition components
    to iteratively find the minimum of the ObjectiveFn within the search bounds.

    The loop terminates when either the maximum EI score falls below
    ei_threshold (nothing left to gain) or the total number of evaluations
    reaches max_evaluations (budget cap) — whichever fires first.

    Public Methods
    --------------
    run()
        Execute the optimisation loop and return the results.
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
        training_iter: int = 50,
        noise: float = 1e-4,
    ) -> None:
        """Initialize the OptimisationRun.

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
            Maximum total number of objective evaluations (budget cap).
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

        Raises
        ------
        ValueError
            If initial_train_x is empty or max_evaluations is less than the
            number of initial points.
        """
        # Cast to float64 regardless of input dtype (list or tensor, int or
        # float) so later torch.cat calls never truncate fractional points.
        self.train_x = torch.as_tensor(initial_train_x, dtype=torch.float64).clone()

        if self.train_x.numel() == 0:
            raise ValueError("initial_train_x must contain at least one point.")
        if max_evaluations <= self.train_x.numel():
            raise ValueError(
                f"max_evaluations ({max_evaluations}) must be greater than the "
                f"number of initial points ({self.train_x.numel()})."
            )

        self.objective = objective
        self.surrogate = surrogate
        self.search_bounds = search_bounds
        self.training_iter = training_iter
        self.noise = noise
        self.max_evaluations = max_evaluations
        self.ei_threshold = ei_threshold

        # Evaluate the objective at initial points to get train_y
        self.train_y = torch.tensor(
            self.objective.evaluate(*self.train_x.tolist()), dtype=self.train_x.dtype
        )

        # Create Acquisition once — it holds a reference to the surrogate
        self._acq = Acquisition(surrogate, search_bounds, n_candidates)

        # Deferred-write accumulator for per-iteration data
        # TODO: add MRR artifact writing (config.json, meta.json, run.log, results.h5)
        self._results: list[dict] = []

    # TODO: max_evaluations validation may need revisiting — should it allow
    #       fewer evaluations than initial points?
    def run(self) -> dict[str, object]:
        """Execute the optimisation loop.

        Iteratively fits the surrogate, finds the next input point via the
        acquisition function, and evaluates the objective until convergence.

        Returns
        -------
        dict
            A dictionary containing the optimisation results:
            - "best_x": float — the input point with the lowest objective value.
            - "best_y": float — the lowest objective value found.
            - "train_x": torch.Tensor — all evaluated input points.
            - "train_y": torch.Tensor — all objective evaluations.
            - "n_iterations": int — number of loop iterations executed.
            - "converged": bool — True if EI dropped below ei_threshold,
              False if max_evaluations was reached.
        """
        converged = False
        n_iterations = 0

        print(
            f"Starting optimisation: {self.train_x.numel()} initial points, "
            f"max_evaluations={self.max_evaluations}, "
            f"ei_threshold={self.ei_threshold}"
        )
        # TODO: replace print with Python logging module

        while self.train_x.numel() < self.max_evaluations:
            n_iterations += 1

            # 1. Fit surrogate to all current training data
            # TODO: consider get_fantasy_model for faster updates
            #       without hyperparameter re-tuning
            self.surrogate.fit_and_train(
                self.train_x,
                self.train_y,
                training_iter=self.training_iter,
                noise=self.noise,
            )

            # 2. Compute current best and find the next input point
            current_best = self.train_y.min().item()
            next_point = self._acq.find_next_input_point(current_best)
            max_ei = self._acq.ei_scores.max().item()

            print(
                f"  Iteration {n_iterations} | "
                f"current_best: {current_best:.4f} | "
                f"next_point: {next_point:.4f} | "
                f"max_ei: {max_ei:.6f}"
            )

            # 3. Check EI convergence before evaluating the new point
            if max_ei < self.ei_threshold:
                print(
                    f"Converged after {n_iterations} iterations "
                    f"(max EI {max_ei:.6f} < ei_threshold {self.ei_threshold})"
                )
                converged = True
                break

            # 4. Evaluate objective at the next point
            new_y = self.objective.evaluate(next_point)[0]

            # 5. Append to training data
            self.train_x = torch.cat(
                [self.train_x, torch.tensor([next_point], dtype=self.train_x.dtype)]
            )
            self.train_y = torch.cat(
                [self.train_y, torch.tensor([new_y], dtype=self.train_y.dtype)]
            )

            # 6. Accumulate per-iteration results
            self._results.append(
                {
                    "iteration": n_iterations,
                    "next_point": next_point,
                    "new_y": new_y,
                    "current_best": current_best,
                    "max_ei": max_ei,
                }
            )

        if not converged:
            print(
                f"Stopped after {n_iterations} iterations "
                f"(reached max_evaluations={self.max_evaluations})"
            )

        best_idx = torch.argmin(self.train_y)
        return {
            "best_x": self.train_x[best_idx].item(),
            "best_y": self.train_y[best_idx].item(),
            "train_x": self.train_x,
            "train_y": self.train_y,
            "n_iterations": n_iterations,
            "converged": converged,
        }

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the OptimisationRun."""
        return (
            f"OptimisationRun("
            f"bounds={self.search_bounds}, "
            f"max_eval={self.max_evaluations}, "
            f"ei_thresh={self.ei_threshold}, "
            f"n_points={self.train_x.numel()})"
        )

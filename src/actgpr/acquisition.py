"""Acquisition function module for active GPR optimisation."""

import torch
from torch.distributions import Normal

from actgpr.surrogate import GPyTorchSurrogate


class Acquisition:
    """Expected Improvement acquisition function for active GPR optimisation.

    Stores a reference to the surrogate, search bounds, and candidate count.
    Scores candidate input points and selects the next input point to evaluate.

    Public Methods
    --------------
    expected_improvement(f_mean, f_var, current_best)
        Compute EI scores for an array of candidate points.
    find_next_input_point(current_best)
        Generate candidates, score them, and return the best input point.
    """

    def __init__(
        self,
        surrogate: GPyTorchSurrogate,
        search_bounds: tuple[float, float],
        n_candidates: int = 1000,
    ) -> None:
        """Initialize the Acquisition function.

        Parameters
        ----------
        surrogate : GPyTorchSurrogate
            The fitted surrogate model used to predict f_mean and f_var.
        search_bounds : tuple[float, float]
            The closed interval (lo, hi) within which candidates are generated.
        n_candidates : int, optional
            Number of evenly spaced candidate points to evaluate, by default 1000.
        """
        self.surrogate = surrogate
        self.search_bounds = search_bounds
        self.n_candidates = n_candidates

    def expected_improvement(
        self,
        f_mean: torch.Tensor,
        f_var: torch.Tensor,
        current_best: float,
    ) -> torch.Tensor:
        """Compute the Expected Improvement at candidate points.

        Parameters
        ----------
        f_mean : torch.Tensor of shape (m,)
            Predicted posterior mean at candidate points.
        f_var : torch.Tensor of shape (m,)
            Predicted posterior variance at candidate points.
        current_best : float
            The smallest objective value observed so far.

        Returns
        -------
        torch.Tensor of shape (m,)
            The EI score for each candidate point.
        """
        f_std = torch.sqrt(f_var)

        # Where std is zero, improvement is zero (no uncertainty)
        ei = torch.zeros_like(f_mean)
        mask = f_std > 0

        improvement = current_best - f_mean[mask]
        z = improvement / f_std[mask]

        normal = Normal(0, 1)
        ei[mask] = improvement * normal.cdf(z) + f_std[mask] * torch.exp(
            normal.log_prob(z)
        )

        # EI must be non-negative
        assert torch.all(ei >= -1e-6), f"EI contains negative values: {ei[ei < -1e-6]}"

        return ei

    def find_next_input_point(self, current_best: float) -> float:
        """Find the next input point to evaluate by maximising Expected Improvement.

        Generates a dense grid of candidate points within the search bounds,
        predicts posterior mean and variance using the surrogate, scores them
        with Expected Improvement, and returns the candidate with the highest score.

        Parameters
        ----------
        current_best : float
            The smallest objective value observed so far.

        Returns
        -------
        float
            The input point with the highest EI score.
        """
        lo, hi = self.search_bounds
        candidates = torch.linspace(lo, hi, self.n_candidates)

        preds = self.surrogate.predict(candidates)
        f_mean = preds["f_mean"]
        f_var = preds["f_var"]

        assert isinstance(f_mean, torch.Tensor)
        assert isinstance(f_var, torch.Tensor)

        ei_scores = self.expected_improvement(f_mean, f_var, current_best)

        best_index = torch.argmax(ei_scores)
        return candidates[best_index].item()

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the Acquisition."""
        lo, hi = self.search_bounds
        return (
            f"Acquisition(method=EI, bounds=({lo}, {hi}), "
            f"n_candidates={self.n_candidates})"
        )

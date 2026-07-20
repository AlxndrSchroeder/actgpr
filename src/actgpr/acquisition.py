"""Acquisition function module for active GPR optimisation.

Implements the Expected Improvement (EI) acquisition function
for selecting the next input point to evaluate.

References
----------
Jones, D. R., Schonlau, M., & Welch, W. J. (1998).
    Efficient Global Optimization of Expensive Black-Box Functions.
    Journal of Global Optimization, 13(4), 455-492.
    https://doi.org/10.1023/A:1008306431147
"""

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
        n_candidates: int = 500,
    ) -> None:
        """Initialize the Acquisition function.

        Parameters
        ----------
        surrogate : GPyTorchSurrogate
            The fitted surrogate model used to predict f_mean and f_var.
        search_bounds : tuple[float, float]
            The closed interval (lo, hi) within which candidates are generated.
        n_candidates : int, optional
            Number of evenly spaced candidate points to evaluate, by default
            500 (matches the OptimisationRun default).
        """
        self.surrogate = surrogate
        self.search_bounds = search_bounds
        self.n_candidates = n_candidates

        # Populated by find_next_input_point for downstream use (e.g. plotting)
        self.candidates: torch.Tensor | None = None
        self.f_mean: torch.Tensor | None = None
        self.f_var: torch.Tensor | None = None
        self.f_covar: torch.Tensor | None = None
        self.ei_scores: torch.Tensor | None = None

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

        Raises
        ------
        ValueError
            If f_mean and f_var have different shapes.
        """
        if f_mean.shape != f_var.shape:
            raise ValueError(
                f"Shape mismatch: f_mean shape {f_mean.shape} must match "
                f"f_var shape {f_var.shape}"
            )

        # Convert variance to standard deviation: σ = √(σ²)
        # The EI formula (Jones et al. 1998) uses σ, not σ²
        f_std = torch.sqrt(f_var)

        # Initialise EI scores to zero for all candidates.
        # Points where f_std == 0 (training points, full certainty) stay at
        # EI = 0 because there is no uncertainty and thus no expected gain.
        ei = torch.zeros_like(f_std)

        # Boolean mask: only compute EI where f_std > 0 to avoid division by zero
        mask = f_std > 0

        # How much better each candidate's predicted mean is compared to current_best.
        # Positive improvement means the model predicts this candidate is better.
        improvement = current_best - f_mean[mask]

        # Standard normal distribution N(0,1) for computing Φ (CDF) and φ (PDF)
        normal = Normal(0, 1)

        # Closed-form EI formula (Jones et al. 1998):
        #   EI(x) = (f_best − μ(x)) · Φ(z) + σ(x) · φ(z)
        #
        # where z = (f_best − μ(x)) / σ(x)  (= improvement / f_std)
        #
        # Term 1: improvement * Φ(z)  →  exploitation
        #   Rewards points where the model confidently predicts a better value.
        #   Φ(z) is the probability that the true value is below current_best.
        #
        # Term 2: σ(x) * φ(z)  →  exploration
        #   Rewards points where the model is uncertain (large f_std).
        #   φ(z) is the bell curve height — computed as exp(log_prob(z)) because
        #   PyTorch's Normal distribution has no .pdf() method, only .log_prob().
        #   exp(log(φ(z))) = φ(z) = (1/√(2π)) · e^(−z²/2)
        z = improvement / f_std[mask]
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

        Raises
        ------
        RuntimeError
            If the surrogate has not been fitted prior to calling.
        """
        lo, hi = self.search_bounds
        self.candidates = torch.linspace(lo, hi, self.n_candidates)

        preds = self.surrogate.predict(self.candidates)
        self.f_mean = preds["f_mean"]
        self.f_var = preds["f_var"]
        self.f_covar = preds["f_covar"]

        assert isinstance(self.f_mean, torch.Tensor)
        assert isinstance(self.f_var, torch.Tensor)

        self.ei_scores = self.expected_improvement(
            self.f_mean, self.f_var, current_best
        )

        best_index = torch.argmax(self.ei_scores)
        return self.candidates[best_index].item()

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the Acquisition."""
        lo, hi = self.search_bounds
        return (
            f"Acquisition(method=EI, bounds=({lo}, {hi}), "
            f"n_candidates={self.n_candidates})"
        )

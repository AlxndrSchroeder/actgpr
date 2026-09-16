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

from collections.abc import Sequence

import torch
from torch.distributions import Normal
from torch.quasirandom import SobolEngine

from actgpr._points import parse_search_bounds
from actgpr.surrogate import GPyTorchSurrogate

# Seed for the candidate sampler used with more than one input dimension.
# Fixed so that a multi-dimensional run is reproducible out of the box, the
# same reasoning as ObjectiveFn's DEFAULT_JITTER_SEED.
DEFAULT_CANDIDATE_SEED = 25


class Acquisition:
    """Expected Improvement acquisition function for active GPR optimisation.

    Stores a reference to the surrogate, search bounds, and candidate count.
    Scores candidate input points and selects the next input point to evaluate.

    Candidates are placed in two stages: a coarse set covering the whole
    search bounds, then a fine set in a small box around the coarse best.
    With one input dimension both sets are evenly spaced grids. With more,
    a grid is unaffordable (500 per axis is 125 million points in 3D), so
    both sets are drawn from a scrambled Sobol sequence, which covers the
    space evenly without growing with the number of dimensions.

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
        search_bounds: Sequence[float] | Sequence[Sequence[float]],
        n_candidates: int = 500,
        seed: int = DEFAULT_CANDIDATE_SEED,
    ) -> None:
        """Initialize the Acquisition function.

        Parameters
        ----------
        surrogate : GPyTorchSurrogate
            The fitted surrogate model used to predict f_mean and f_var.
        search_bounds : sequence of (lo, hi) pairs, or a single (lo, hi) pair
            The closed interval of each input dimension within which
            candidates are generated. A single pair means one dimension.
        n_candidates : int, optional
            Number of candidate points scored in each of the two stages, by
            default 500 (matches the OptimisationRun default). With several
            input dimensions the same count is spread across all of them, so
            raise it for problems with many dimensions.
        seed : int, optional
            Seed for the Sobol candidate sampler used with more than one
            input dimension, by default 25. Unused in one dimension, where
            candidates are an evenly spaced grid.
        """
        self.surrogate = surrogate
        self.search_bounds = parse_search_bounds(search_bounds)
        self.n_dims = len(self.search_bounds)
        self.n_candidates = n_candidates
        self.seed = seed

        self._lower = torch.tensor(
            [lo for lo, _ in self.search_bounds], dtype=torch.float64
        )
        self._upper = torch.tensor(
            [hi for _, hi in self.search_bounds], dtype=torch.float64
        )
        self._sampler = SobolEngine(self.n_dims, scramble=True, seed=seed)

        # Populated by find_next_input_point for downstream use (e.g. plotting)
        self.candidates: torch.Tensor | None = None
        self.f_mean: torch.Tensor | None = None
        self.f_var: torch.Tensor | None = None
        self.ei_scores: torch.Tensor | None = None
        # The surrogate's posterior mean at the returned next_point itself
        # (post zoom-refinement), distinct from f_mean, which covers only
        # the coarse candidates and may not include next_point exactly.
        self.next_point_mean: float | None = None

    def _candidates(self, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """Place n_candidates input points inside the box [lower, upper]."""
        if self.n_dims == 1:
            grid = torch.linspace(lower.item(), upper.item(), self.n_candidates)
            return grid.to(torch.float64).unsqueeze(-1)

        unit = self._sampler.draw(self.n_candidates, dtype=torch.float64)
        return lower + (upper - lower) * unit

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
        #   φ(z) is the bell curve height, computed as exp(log_prob(z)) because
        #   PyTorch's Normal distribution has no .pdf() method, only .log_prob().
        #   exp(log(φ(z))) = φ(z) = (1/√(2π)) · e^(−z²/2)
        z = improvement / f_std[mask]
        ei[mask] = improvement * normal.cdf(z) + f_std[mask] * torch.exp(
            normal.log_prob(z)
        )

        # EI must be non-negative
        assert torch.all(ei >= -1e-6), f"EI contains negative values: {ei[ei < -1e-6]}"

        return ei

    def find_next_input_point(self, current_best: float) -> tuple[float, ...]:
        """Find the next input point to evaluate by maximising Expected Improvement.

        Places candidates across the search bounds, predicts posterior mean
        and variance using the surrogate, scores them with Expected
        Improvement, and locates the candidate with the highest score. The
        coarse candidates' spacing caps how precisely the true EI maximum
        can be located, so a second, much finer set is then scored inside a
        small box around the coarse best point, and the refined maximum is
        returned instead.

        Parameters
        ----------
        current_best : float
            The smallest objective value observed so far.

        Returns
        -------
        tuple of float
            The coordinates of the input point with the highest EI score,
            one per input dimension, refined beyond the coarse candidates'
            resolution.

        Raises
        ------
        RuntimeError
            If the surrogate has not been fitted prior to calling.
        """
        self.candidates = self._candidates(self._lower, self._upper)

        preds = self.surrogate.predict(self.candidates)
        self.f_mean = preds["f_mean"]
        self.f_var = preds["f_var"]

        assert isinstance(self.f_mean, torch.Tensor)
        assert isinstance(self.f_var, torch.Tensor)

        self.ei_scores = self.expected_improvement(
            self.f_mean, self.f_var, current_best
        )
        coarse_best = self.candidates[torch.argmax(self.ei_scores)]

        # Zoom-refine: the coarse candidates only locate the EI maximum to
        # within one spacing. Re-score a much finer set confined to a small
        # box around the coarse best point. This is cheap, since the box is
        # a tiny fraction of the search bounds, and it recovers precision
        # the coarse candidates alone cannot offer. The spacing of n points
        # spread over d dimensions is (hi - lo) / (n^(1/d) - 1), which for a
        # single dimension is exactly the grid step (hi - lo) / (n - 1).
        points_per_axis = self.n_candidates ** (1 / self.n_dims)
        spacing = (self._upper - self._lower) / (points_per_axis - 1)
        window = 2 * spacing
        fine_lower = torch.maximum(self._lower, coarse_best - window)
        fine_upper = torch.minimum(self._upper, coarse_best + window)
        fine_candidates = self._candidates(fine_lower, fine_upper)

        fine_preds = self.surrogate.predict(fine_candidates)
        fine_ei = self.expected_improvement(
            fine_preds["f_mean"], fine_preds["f_var"], current_best
        )

        fine_best_index = torch.argmax(fine_ei)
        next_point = fine_candidates[fine_best_index]

        assert torch.all(next_point >= self._lower) and torch.all(
            next_point <= self._upper
        ), f"next_point {next_point.tolist()} left the search bounds"

        self.next_point_mean = fine_preds["f_mean"][fine_best_index].item()
        return tuple(next_point.tolist())

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the Acquisition."""
        return (
            f"Acquisition(method=EI, bounds={list(self.search_bounds)}, "
            f"n_candidates={self.n_candidates})"
        )

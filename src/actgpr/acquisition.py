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

# Candidates scored per stage. Measurement on a three-input problem over
# twelve seeds showed 4000 gives both a better typical result (median
# best_y 0.0037 against 0.0095 at 500) and a far better worst one (0.024
# against 0.215), for a few seconds per run.
DEFAULT_CANDIDATES = 4000

# How many times the refinement shrinks its box around the best candidate.
# Candidates thin out as dimensions are added, so one stage leaves a box
# covering a quarter of each axis in three dimensions; a second roughly
# halves that again. Three were measured as worse than two, since shrinking
# repeatedly around the first winner stops the search exploring.
REFINEMENT_STAGES = 2


class Acquisition:
    """Expected Improvement acquisition function for active GPR optimisation.

    Stores a reference to the surrogate, search bounds, and candidate count.
    Scores candidate input points and selects the next input point to evaluate.

    Candidates are placed in stages: a coarse set covering the whole search
    bounds, then refinement sets in a shrinking box around the best point
    found so far (see ``refinement_stages``). All of them are drawn from a
    scrambled Sobol sequence, which fills the space evenly for any number
    of input dimensions.
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
        n_candidates: int = DEFAULT_CANDIDATES,
        seed: int = DEFAULT_CANDIDATE_SEED,
        refinement_stages: int = REFINEMENT_STAGES,
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
            Number of candidate points scored in each stage, by default
            4000. Candidates spread over d axes, so the same count thins out
            quickly as inputs are added. Raising it further mainly improves
            the worst case rather than the typical one; prediction cost grows
            faster than linearly with it, so a few thousand is a sensible
            ceiling.
        seed : int, optional
            Seed for the Sobol candidate sampler used with more than one
            input dimension, by default 25. Unused in one dimension, where
            candidates are an evenly spaced grid.
        refinement_stages : int, optional
            How many times the refinement shrinks its box around the best
            candidate, by default 2. One stage leaves a box covering a
            quarter of each axis with three inputs, which barely refines
            anything; three were measured as worse than two.
        """
        self.surrogate = surrogate
        self.search_bounds = parse_search_bounds(search_bounds)
        self.n_dims = len(self.search_bounds)
        self.n_candidates = n_candidates
        self.seed = seed
        self.refinement_stages = refinement_stages

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
        can be located, so a finer set is then scored inside a box around
        the best point, once or twice depending on ``refinement_stages``,
        and the refined maximum is returned instead.

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
        next_point = self.candidates[torch.argmax(self.ei_scores)]

        # Zoom-refine: the coarse candidates only locate the EI maximum to
        # within one spacing. Re-score a finer set confined to a box around
        # the best point so far. This is cheap, since the box is a fraction
        # of the search bounds, and it recovers precision the coarse
        # candidates alone cannot offer. The spacing of n points spread over
        # d dimensions is (hi - lo) / (n^(1/d) - 1), which for a single
        # dimension is exactly the grid step (hi - lo) / (n - 1).
        points_per_axis = self.n_candidates ** (1 / self.n_dims)
        lower, upper = self._lower, self._upper
        for _ in range(self.refinement_stages):
            window = 2 * (upper - lower) / (points_per_axis - 1)
            lower = torch.maximum(self._lower, next_point - window)
            upper = torch.minimum(self._upper, next_point + window)
            fine_candidates = self._candidates(lower, upper)

            fine_preds = self.surrogate.predict(fine_candidates)
            fine_ei = self.expected_improvement(
                fine_preds["f_mean"], fine_preds["f_var"], current_best
            )

            fine_best_index = torch.argmax(fine_ei)
            next_point = fine_candidates[fine_best_index]
            self.next_point_mean = fine_preds["f_mean"][fine_best_index].item()

        assert torch.all(next_point >= self._lower) and torch.all(
            next_point <= self._upper
        ), f"next_point {next_point.tolist()} left the search bounds"

        return tuple(next_point.tolist())

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the Acquisition."""
        return (
            f"Acquisition(method=EI, bounds={list(self.search_bounds)}, "
            f"n_candidates={self.n_candidates})"
        )

"""Unit tests for the Acquisition class."""

import pytest
import torch

from actgpr.acquisition import Acquisition
from actgpr.surrogate import GPyTorchSurrogate

SEED = 25


@pytest.fixture()
def fitted_surrogate() -> GPyTorchSurrogate:
    """Return a GPyTorchSurrogate fitted to a simple quadratic."""
    torch.manual_seed(SEED)
    train_x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    train_y = train_x**2
    surrogate = GPyTorchSurrogate()
    surrogate.fit_and_train(train_x, train_y, training_iter=50)
    return surrogate


@pytest.fixture()
def acquisition(fitted_surrogate: GPyTorchSurrogate) -> Acquisition:
    """Return an Acquisition instance with a fitted surrogate."""
    return Acquisition(
        surrogate=fitted_surrogate,
        search_bounds=(-3.0, 4.0),
        n_candidates=500,
    )


class TestAcquisitionInit:
    """Tests for Acquisition initialisation."""

    def test_stores_surrogate_reference(
        self, acquisition: Acquisition, fitted_surrogate: GPyTorchSurrogate
    ) -> None:
        """Test that the Acquisition stores the surrogate reference."""
        assert acquisition.surrogate is fitted_surrogate

    def test_stores_search_bounds(self, acquisition: Acquisition) -> None:
        """Test that the Acquisition stores the search bounds."""
        assert acquisition.search_bounds == (-3.0, 4.0)

    def test_stores_n_candidates(self, acquisition: Acquisition) -> None:
        """Test that the Acquisition stores the candidate count."""
        assert acquisition.n_candidates == 500

    def test_repr(self, acquisition: Acquisition) -> None:
        """Test the string representation of the Acquisition."""
        assert "EI" in repr(acquisition)
        assert "(-3.0, 4.0)" in repr(acquisition)


class TestExpectedImprovement:
    """Tests for Acquisition.expected_improvement()."""

    def test_ei_output_shape(self, acquisition: Acquisition) -> None:
        """Test that EI returns one score per candidate point."""
        f_mean = torch.tensor([1.0, 2.0, 3.0])
        f_var = torch.tensor([0.5, 0.5, 0.5])
        current_best = 1.5

        ei = acquisition.expected_improvement(f_mean, f_var, current_best)

        assert ei.shape == (3,)

    @pytest.mark.parametrize("current_best", [0.0, 1.0, 5.0, 10.0])
    def test_ei_is_non_negative(
        self, acquisition: Acquisition, current_best: float
    ) -> None:
        """Test scientific invariant: EI scores must be non-negative."""
        f_mean = torch.tensor([0.0, 1.0, 2.0, 5.0])
        f_var = torch.tensor([1.0, 0.5, 2.0, 0.1])

        ei = acquisition.expected_improvement(f_mean, f_var, current_best)

        assert torch.all(ei >= -1e-6)

    def test_ei_higher_where_mean_below_best(self, acquisition: Acquisition) -> None:
        """Test that EI is higher where the predicted mean is below the current best."""
        f_var = torch.tensor([1.0, 1.0])
        current_best = 2.0

        # Point A: mean=1.0 (below best), Point B: mean=3.0 (above best)
        f_mean = torch.tensor([1.0, 3.0])
        ei = acquisition.expected_improvement(f_mean, f_var, current_best)

        assert ei[0] > ei[1]

    def test_ei_higher_where_variance_higher(self, acquisition: Acquisition) -> None:
        """Test that EI is higher where uncertainty is greater (exploration)."""
        f_mean = torch.tensor([2.0, 2.0])
        current_best = 2.0

        # Same mean, but Point A has higher variance
        f_var = torch.tensor([4.0, 0.01])
        ei = acquisition.expected_improvement(f_mean, f_var, current_best)

        assert ei[0] > ei[1]

    def test_ei_zero_when_variance_zero(self, acquisition: Acquisition) -> None:
        """Test that EI is zero where there is no uncertainty."""
        f_mean = torch.tensor([1.0])
        f_var = torch.tensor([0.0])
        current_best = 2.0

        ei = acquisition.expected_improvement(f_mean, f_var, current_best)

        assert ei.item() == 0.0

    def test_ei_raises_on_shape_mismatch(self, acquisition: Acquisition) -> None:
        """Test that EI raises ValueError when f_mean and f_var have different shapes."""
        f_mean = torch.tensor([1.0, 2.0, 3.0])
        f_var = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="Shape mismatch"):
            acquisition.expected_improvement(f_mean, f_var, current_best=1.0)


class TestFindNextInputPoint:
    """Tests for Acquisition.find_next_input_point()."""

    def test_returns_float(self, acquisition: Acquisition) -> None:
        """Test that find_next_input_point returns a scalar float."""
        current_best = 0.0
        result = acquisition.find_next_input_point(current_best)

        assert isinstance(result, float)

    def test_result_within_search_bounds(self, acquisition: Acquisition) -> None:
        """Test that the returned point is within the search bounds."""
        current_best = 0.0
        result = acquisition.find_next_input_point(current_best)

        lo, hi = acquisition.search_bounds
        assert lo <= result <= hi

    def test_deterministic_with_same_seed(
        self, fitted_surrogate: GPyTorchSurrogate
    ) -> None:
        """Test that find_next_input_point is deterministic for same inputs."""
        acq_a = Acquisition(fitted_surrogate, (-3.0, 4.0), n_candidates=500)
        acq_b = Acquisition(fitted_surrogate, (-3.0, 4.0), n_candidates=500)

        result_a = acq_a.find_next_input_point(current_best=0.0)
        result_b = acq_b.find_next_input_point(current_best=0.0)

        assert result_a == result_b


class TestZoomRefinement:
    """Tests for the fine-grid refinement step in find_next_input_point()."""

    def test_stored_candidates_still_span_full_search_bounds(
        self, acquisition: Acquisition
    ) -> None:
        """Test that self.candidates stays the coarse grid used for plotting.

        The fine grid used to refine the returned point is local and
        internal — it must not replace the coarse candidates/f_mean/f_var/
        ei_scores arrays that plot_iteration_snapshot() relies on to draw
        the full EI landscape.
        """
        acquisition.find_next_input_point(current_best=0.0)

        lo, hi = acquisition.search_bounds
        expected_coarse_grid = torch.linspace(lo, hi, acquisition.n_candidates)

        assert acquisition.candidates.shape == (acquisition.n_candidates,)
        assert torch.allclose(acquisition.candidates, expected_coarse_grid)
        assert acquisition.ei_scores.shape == (acquisition.n_candidates,)

    def test_refined_point_can_fall_between_coarse_grid_points(
        self, fitted_surrogate: GPyTorchSurrogate
    ) -> None:
        """Test that refinement is not limited to the coarse grid's spacing.

        With a deliberately coarse grid (few candidates), the true EI
        maximum is very unlikely to sit exactly on a grid point — refinement
        should be able to land strictly between two of them.
        """
        acq = Acquisition(fitted_surrogate, search_bounds=(-3.0, 4.0), n_candidates=6)
        coarse_grid = torch.linspace(-3.0, 4.0, 6)

        result = acq.find_next_input_point(current_best=0.0)

        assert not torch.any(torch.isclose(coarse_grid, torch.tensor(result)))

    def test_refinement_does_not_regress_below_coarse_grid_max(
        self, acquisition: Acquisition
    ) -> None:
        """Test that the refined point's EI is at least as good as the coarse max.

        The fine grid samples densely around the coarse best point, so it
        should always find an EI at least as high as the single coarse-grid
        sample did — refinement should sharpen the estimate, never worsen it.
        """
        current_best = 0.0
        result = acquisition.find_next_input_point(current_best)
        coarse_max_ei = acquisition.ei_scores.max().item()

        preds = acquisition.surrogate.predict(torch.tensor([result]))
        refined_ei = acquisition.expected_improvement(
            preds["f_mean"], preds["f_var"], current_best
        ).item()

        assert refined_ei >= coarse_max_ei - 1e-6

    def test_next_point_mean_matches_prediction_at_next_point(
        self, acquisition: Acquisition
    ) -> None:
        """Test that next_point_mean is the surrogate's mean at next_point itself.

        next_point generally falls strictly between two coarse grid points
        after zoom-refinement, so next_point_mean must not be read off the
        coarse f_mean array (which only covers the coarse grid) — it has to
        come from a prediction at next_point directly.
        """
        current_best = 0.0
        result = acquisition.find_next_input_point(current_best)

        expected_mean = acquisition.surrogate.predict(torch.tensor([result]))[
            "f_mean"
        ].item()

        assert acquisition.next_point_mean == pytest.approx(expected_mean)

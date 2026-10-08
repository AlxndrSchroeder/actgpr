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
def fitted_surrogate_3d() -> GPyTorchSurrogate:
    """Return a GPyTorchSurrogate fitted to a 3D sum of squares."""
    generator = torch.Generator().manual_seed(SEED)
    train_x = torch.rand(10, 3, generator=generator, dtype=torch.float64) * 4 - 2
    train_y = (train_x**2).sum(dim=1)
    surrogate = GPyTorchSurrogate()
    surrogate.fit_no_training(train_x, train_y, lengthscale=1.0)
    return surrogate


BOUNDS_3D = [(-2.0, 2.0), (-1.0, 3.0), (0.0, 0.5)]


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
        assert acquisition.search_bounds == ((-3.0, 4.0),)
        assert acquisition.n_dims == 1

    def test_default_candidate_count_depends_on_the_dimension(
        self,
        fitted_surrogate: GPyTorchSurrogate,
        fitted_surrogate_3d: GPyTorchSurrogate,
    ) -> None:
        """Test that several inputs get more candidates by default.

        The same count spread over d axes thins out quickly: 500 is 500 per
        axis with one input but only 7.9 with three.
        """
        assert Acquisition(fitted_surrogate, (-3.0, 4.0)).n_candidates == 500
        assert Acquisition(fitted_surrogate_3d, BOUNDS_3D).n_candidates == 4000

    def test_explicit_candidate_count_overrides_the_default(
        self, fitted_surrogate_3d: GPyTorchSurrogate
    ) -> None:
        """Test that a caller's own candidate count is kept."""
        assert Acquisition(fitted_surrogate_3d, BOUNDS_3D, 777).n_candidates == 777

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

    def test_returns_one_coordinate_in_one_dimension(
        self, acquisition: Acquisition
    ) -> None:
        """Test that a 1D search returns a point with a single float."""
        result = acquisition.find_next_input_point(current_best=0.0)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_result_within_search_bounds(self, acquisition: Acquisition) -> None:
        """Test that the returned point is within the search bounds."""
        (result,) = acquisition.find_next_input_point(current_best=0.0)

        ((lo, hi),) = acquisition.search_bounds
        assert lo <= result <= hi

    def test_returns_one_coordinate_per_input_dimension(
        self, fitted_surrogate_3d: GPyTorchSurrogate
    ) -> None:
        """Test that a 3D search returns a 3D point inside its bounds."""
        acq = Acquisition(fitted_surrogate_3d, BOUNDS_3D, n_candidates=256)

        result = acq.find_next_input_point(current_best=0.5)

        assert len(result) == 3
        for value, (lo, hi) in zip(result, BOUNDS_3D):
            assert lo <= value <= hi

    def test_candidates_cover_every_dimension_within_bounds(
        self, fitted_surrogate_3d: GPyTorchSurrogate
    ) -> None:
        """Test that 3D candidates are (m, 3) and respect each interval."""
        acq = Acquisition(fitted_surrogate_3d, BOUNDS_3D, n_candidates=256)
        acq.find_next_input_point(current_best=0.5)

        assert acq.candidates.shape == (256, 3)
        for dim, (lo, hi) in enumerate(BOUNDS_3D):
            assert torch.all(acq.candidates[:, dim] >= lo)
            assert torch.all(acq.candidates[:, dim] <= hi)
            # Sobol points fill the interval rather than bunching up.
            assert acq.candidates[:, dim].max() - acq.candidates[:, dim].min() > (
                0.9 * (hi - lo)
            )

    def test_multi_dimensional_search_is_reproducible(
        self, fitted_surrogate_3d: GPyTorchSurrogate
    ) -> None:
        """Test that the seeded Sobol sampler makes 3D runs repeatable.

        The global torch RNG is deliberately disturbed in between, since
        the sampler must not depend on it.
        """
        torch.manual_seed(1)
        result_a = Acquisition(fitted_surrogate_3d, BOUNDS_3D).find_next_input_point(
            0.5
        )
        torch.manual_seed(2)
        result_b = Acquisition(fitted_surrogate_3d, BOUNDS_3D).find_next_input_point(
            0.5
        )

        assert result_a == result_b

    def test_candidate_seed_selects_the_sample(
        self, fitted_surrogate_3d: GPyTorchSurrogate
    ) -> None:
        """Test that a different seed places different 3D candidates."""
        acq_a = Acquisition(fitted_surrogate_3d, BOUNDS_3D, seed=25)
        acq_b = Acquisition(fitted_surrogate_3d, BOUNDS_3D, seed=26)
        acq_a.find_next_input_point(0.5)
        acq_b.find_next_input_point(0.5)

        assert not torch.equal(acq_a.candidates, acq_b.candidates)

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
        ei_scores arrays that _plot_iteration_snapshot() relies on to draw
        the full EI landscape.
        """
        acquisition.find_next_input_point(current_best=0.0)

        ((lo, hi),) = acquisition.search_bounds
        expected_coarse_grid = torch.linspace(
            lo, hi, acquisition.n_candidates, dtype=torch.float64
        )

        assert acquisition.candidates.shape == (acquisition.n_candidates, 1)
        assert torch.equal(acquisition.candidates[:, 0], expected_coarse_grid)

    def test_candidate_grid_has_float64_precision(
        self, acquisition: Acquisition
    ) -> None:
        """Test that the 1D grid is not quietly generated in float32.

        Before 0.4 the grid used torch's float32 default while the rest of
        the package was float64, so candidates carried ~4e-8 rounding error.
        """
        acquisition.find_next_input_point(current_best=0.0)
        ((lo, hi),) = acquisition.search_bounds
        step = (hi - lo) / (acquisition.n_candidates - 1)

        assert acquisition.candidates.dtype == torch.float64
        assert acquisition.candidates[1, 0].item() == pytest.approx(
            lo + step, abs=1e-14
        )
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

    def test_refinement_does_not_regress_in_several_dimensions(
        self, fitted_surrogate_3d: GPyTorchSurrogate
    ) -> None:
        """Test that the refinement box also sharpens a 3D search."""
        acq = Acquisition(fitted_surrogate_3d, BOUNDS_3D, n_candidates=256)
        current_best = 0.5
        result = acq.find_next_input_point(current_best)
        coarse_max_ei = acq.ei_scores.max().item()

        preds = acq.surrogate.predict(torch.tensor([result]))
        refined_ei = acq.expected_improvement(
            preds["f_mean"], preds["f_var"], current_best
        ).item()

        assert refined_ei >= coarse_max_ei - 1e-6

    def test_one_refinement_stage_for_a_single_input(
        self, acquisition: Acquisition
    ) -> None:
        """Test that a 1D search refines once.

        Its box is already 0.8% of the axis; a second stage would shrink it
        to the spacing between the points of a converged run and risk
        evaluating the same point twice.
        """
        assert acquisition.refinement_stages == 1

    def test_two_refinement_stages_for_several_inputs(
        self, fitted_surrogate_3d: GPyTorchSurrogate
    ) -> None:
        """Test that a 3D search refines twice.

        Candidates thin out with dimension: one stage leaves a box covering
        more than half of each axis in 3D, which barely refines anything.
        """
        assert Acquisition(fitted_surrogate_3d, BOUNDS_3D).refinement_stages == 2

    def test_refinement_stages_can_be_set_explicitly(
        self, fitted_surrogate_3d: GPyTorchSurrogate
    ) -> None:
        """Test that the automatic choice can be overridden."""
        acq = Acquisition(fitted_surrogate_3d, BOUNDS_3D, refinement_stages=3)

        assert acq.refinement_stages == 3

    def test_more_stages_narrow_the_search(
        self, fitted_surrogate_3d: GPyTorchSurrogate
    ) -> None:
        """Test that each extra stage lands closer to the EI maximum.

        A later stage searches inside the previous stage's box, so its point
        can only be at least as good, never worse.
        """
        current_best = 0.5
        points, scores = [], []
        for stages in (1, 2):
            acq = Acquisition(
                fitted_surrogate_3d, BOUNDS_3D, 256, refinement_stages=stages
            )
            point = acq.find_next_input_point(current_best)
            preds = fitted_surrogate_3d.predict(torch.tensor([point]))
            scores.append(
                acq.expected_improvement(
                    preds["f_mean"], preds["f_var"], current_best
                ).item()
            )
            points.append(point)

        assert points[0] != points[1]
        assert scores[1] >= scores[0] - 1e-9

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

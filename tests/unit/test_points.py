"""Unit tests for the shared input point and search bounds helpers."""

import pytest
import torch

from actgpr._points import as_points, format_values, parse_search_bounds


class TestAsPoints:
    """Tests for as_points."""

    def test_flat_sequence_is_points_in_one_dimension(self) -> None:
        """Test that a flat list becomes one row per 1D point."""
        points = as_points([-3.0, 5.0])

        assert points.shape == (2, 1)
        assert points.dtype == torch.float64
        assert points[:, 0].tolist() == [-3.0, 5.0]

    def test_rows_are_points_in_several_dimensions(self) -> None:
        """Test that nested rows keep one coordinate per column."""
        points = as_points([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], n_dims=3)

        assert points.shape == (2, 3)
        assert points[1].tolist() == [3.0, 4.0, 5.0]

    def test_dimension_is_taken_from_the_input_when_not_given(self) -> None:
        """Test that n_dims is optional for already two-dimensional input."""
        assert as_points([[0.0, 1.0]]).shape == (1, 2)

    def test_flat_sequence_is_rejected_for_several_dimensions(self) -> None:
        """Test that [x1, x2] is not silently read as two 1D points.

        With two input dimensions a flat pair could mean one 2D point or
        two 1D points, so the caller has to say which by using rows.
        """
        with pytest.raises(ValueError, match="one row per point"):
            as_points([0.0, 1.0], n_dims=2)

    def test_wrong_coordinate_count_is_rejected(self) -> None:
        """Test that a point with the wrong number of coordinates fails."""
        with pytest.raises(ValueError, match="Expected 3 coordinates"):
            as_points([[0.0, 1.0]], n_dims=3)

    def test_three_dimensional_input_is_rejected(self) -> None:
        """Test that a tensor with too many axes fails clearly."""
        with pytest.raises(ValueError, match=r"shape \(n,\) or \(n, d\)"):
            as_points(torch.zeros(2, 2, 2))

    def test_non_numeric_input_is_rejected(self) -> None:
        """Test that non-numeric points raise TypeError."""
        with pytest.raises(TypeError, match="numeric"):
            as_points(["a", "b"])

    def test_integer_input_is_promoted_to_float64(self) -> None:
        """Test that integer input is promoted to float64."""
        assert as_points([1, 2]).dtype == torch.float64


class TestParseSearchBounds:
    """Tests for parse_search_bounds."""

    def test_single_pair_is_one_dimension(self) -> None:
        """Test that (lo, hi) is shorthand for one input dimension."""
        assert parse_search_bounds((-3.0, 5.0)) == ((-3.0, 5.0),)

    def test_list_of_pairs_gives_one_interval_per_dimension(self) -> None:
        """Test that each pair becomes one input dimension, in order."""
        bounds = parse_search_bounds([(-3.0, 5.0), (0, 1), (-1.0, 1.0)])

        assert bounds == ((-3.0, 5.0), (0.0, 1.0), (-1.0, 1.0))
        assert len(bounds) == 3

    def test_single_pair_in_a_list_is_also_one_dimension(self) -> None:
        """Test that [(lo, hi)] and (lo, hi) mean the same thing."""
        assert parse_search_bounds([(-3.0, 5.0)]) == parse_search_bounds((-3.0, 5.0))

    def test_empty_bounds_are_rejected(self) -> None:
        """Test that at least one interval is required."""
        with pytest.raises(ValueError, match="at least one"):
            parse_search_bounds([])

    @pytest.mark.parametrize("bad", [(5.0, -3.0), (1.0, 1.0)])
    def test_non_increasing_interval_is_rejected(self, bad: tuple) -> None:
        """Test that lo must be strictly below hi."""
        with pytest.raises(ValueError, match="must be <"):
            parse_search_bounds(bad)

    def test_error_names_the_offending_dimension(self) -> None:
        """Test that the message says which interval is wrong."""
        with pytest.raises(ValueError, match="dimension 1"):
            parse_search_bounds([(0.0, 1.0), (2.0, 1.0)])

    def test_interval_with_wrong_length_is_rejected(self) -> None:
        """Test that each interval must be exactly a (lo, hi) pair."""
        with pytest.raises(ValueError, match=r"\(lo, hi\) pair"):
            parse_search_bounds([(0.0, 1.0, 2.0)])


class TestFormatValues:
    """Tests for format_values."""

    def test_single_value_is_rendered_bare(self) -> None:
        """Test that one-dimensional output looks as it did before nD."""
        assert format_values([-1.49651]) == "-1.4965"

    def test_several_values_are_rendered_as_a_tuple(self) -> None:
        """Test that a multi-dimensional point is parenthesised."""
        assert format_values([1.0, -2.5]) == "(1.0000, -2.5000)"

    def test_format_spec_is_applied_to_every_value(self) -> None:
        """Test that a custom spec reaches each coordinate."""
        assert format_values([2.0, 0.5], ".4g") == "(2, 0.5)"

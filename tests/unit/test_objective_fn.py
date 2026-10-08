"""Unit tests for the ObjectiveFn class."""

import pytest
import torch

from actgpr.objective_fn import ObjectiveFn

SEED = 25


def test_objective_evaluation(objective: ObjectiveFn) -> None:
    """Test that one call evaluates one 1D input point to one float."""
    assert objective.evaluate(2.0) == 4.0
    assert objective.evaluate(-3.0) == 9.0
    assert objective.evaluate(0.0) == 0.0


def test_several_arguments_are_the_coordinates_of_one_point(
    objective: ObjectiveFn,
) -> None:
    """Test that evaluate(x1, x2, x3) is one 3D point, not three 1D points.

    The default Objective sums the squared coordinates, so a 3D point gives
    a single value rather than one value per argument.
    """
    assert objective.evaluate(2.0, -3.0, 0.0) == 13.0


def test_coordinates_reach_the_function_in_order() -> None:
    """Test that func receives exactly one argument per coordinate."""
    obj = ObjectiveFn(lambda x1, x2, x3: 100 * x1 + 10 * x2 + x3)

    assert obj.evaluate(1.0, 2.0, 3.0) == 123.0


def test_wrong_number_of_coordinates_propagates_the_original_error() -> None:
    """Test that a func expecting 2 inputs fails clearly when given 3."""
    obj = ObjectiveFn(lambda x1, x2: x1 + x2)

    with pytest.raises(TypeError, match="positional argument"):
        obj.evaluate(1.0, 2.0, 3.0)


def test_objective_empty_input(objective: ObjectiveFn) -> None:
    """Test that evaluate() without coordinates raises ValueError."""
    with pytest.raises(ValueError, match="coordinates of one input point"):
        objective.evaluate()


@pytest.mark.parametrize("bad_input", [None, "x", [], {}])
def test_objective_raises_on_wrong_type(
    objective: ObjectiveFn, bad_input: object
) -> None:
    """Test that the objective raises TypeError on non-numeric inputs."""
    with pytest.raises(TypeError, match="Expected float or int"):
        objective.evaluate(bad_input)  # type: ignore[arg-type]


def test_objective_accepts_int_input(objective: ObjectiveFn) -> None:
    """Test that the objective handles int inputs via implicit float conversion."""
    assert objective.evaluate(3) == 9.0


@pytest.mark.parametrize("x", [-5.0, -1.0, 0.0, 1.0, 5.0])
def test_objective_output_is_non_negative(objective: ObjectiveFn, x: float) -> None:
    """Test scientific invariant: x² is always non-negative."""
    assert objective.evaluate(x) >= 0


def test_objective_repr(objective: ObjectiveFn) -> None:
    """Test the string representation of the default ObjectiveFn."""
    assert repr(objective) == "ObjectiveFn(function=sum(x_i^2))"


def test_custom_callable_objective() -> None:
    """Test custom objective initialisation and evaluation."""
    obj = ObjectiveFn(lambda x: (x + 2) ** 2)

    assert obj.evaluate(1.0) == 9.0
    assert obj.evaluate(-2.0) == 0.0
    assert repr(obj) == "ObjectiveFn(function=custom_function)"


def test_custom_named_function_repr() -> None:
    """Test that repr uses function names for normal named functions."""

    def my_cool_function(x: float) -> float:
        return x + 5

    obj = ObjectiveFn(my_cool_function)
    assert repr(obj) == "ObjectiveFn(function=my_cool_function)"


def test_custom_function_error_propagation() -> None:
    """Test that errors inside the Objective propagate with their original type."""

    def failing_func(x: float) -> float:
        raise ValueError("Something went wrong inside the function")

    obj = ObjectiveFn(failing_func)
    with pytest.raises(ValueError, match="Something went wrong"):
        obj.evaluate(1.0)


def test_domain_error_keeps_original_type() -> None:
    """Test that a ZeroDivisionError from the Objective is not relabelled."""
    obj = ObjectiveFn(lambda x: 1.0 / x)
    with pytest.raises(ZeroDivisionError):
        obj.evaluate(0.0)


def test_non_numeric_return_raises_type_error() -> None:
    """Test that an Objective returning a non-numeric value raises TypeError."""
    obj = ObjectiveFn(lambda x: "not a number")  # type: ignore[arg-type,return-value]
    with pytest.raises(TypeError, match="non-numeric value"):
        obj.evaluate(1.0)


class TestJitter:
    """Tests for ObjectiveFn's optional Gaussian jitter."""

    def test_default_jitter_is_zero(self, objective: ObjectiveFn) -> None:
        """Test that jitter defaults to off, matching prior no-jitter behaviour."""
        assert objective.jitter == 0.0

    def test_negative_jitter_raises(self) -> None:
        """Test that a negative jitter is rejected at construction."""
        with pytest.raises(ValueError, match="jitter must be non-negative"):
            ObjectiveFn(jitter=-0.1)

    def test_zero_jitter_is_exact(self) -> None:
        """Test that jitter=0.0 never perturbs the result."""
        torch.manual_seed(SEED)
        obj = ObjectiveFn(lambda x: x**2, jitter=0.0)

        assert obj.evaluate(2.0) == 4.0
        assert obj.evaluate(-3.0) == 9.0

    def test_positive_jitter_perturbs_result(self) -> None:
        """Test that a positive jitter changes the evaluated output."""
        torch.manual_seed(SEED)
        obj = ObjectiveFn(lambda x: x**2, jitter=1.0)

        assert obj.evaluate(2.0) != 4.0

    def test_jitter_reproducible_without_seeding_anything(self) -> None:
        """Test that two ObjectiveFn objects produce the same noise sequence.

        Each owns a generator seeded at construction, so a jittered run
        reproduces without the caller seeding anything. Before this, an
        unseeded jittered run gave a different answer every time and its
        MRR record could not reproduce it.
        """
        obj_a = ObjectiveFn(lambda x: x**2, jitter=0.5)
        obj_b = ObjectiveFn(lambda x: x**2, jitter=0.5)

        assert [obj_a.evaluate(x) for x in (1.0, 2.0, 3.0)] == [
            obj_b.evaluate(x) for x in (1.0, 2.0, 3.0)
        ]

    def test_default_jitter_seed_is_25(self) -> None:
        """Test the documented default seed."""
        assert ObjectiveFn(jitter=0.1).seed == 25

    def test_different_seeds_give_different_noise(self) -> None:
        """Test that the seed actually selects the noise sequence."""
        obj_a = ObjectiveFn(lambda x: x**2, jitter=0.5, seed=25)
        obj_b = ObjectiveFn(lambda x: x**2, jitter=0.5, seed=26)

        assert obj_a.evaluate(1.0) != obj_b.evaluate(1.0)

    def test_jitter_does_not_disturb_global_torch_rng(self) -> None:
        """Test that drawing jitter leaves the global RNG stream untouched.

        Jitter uses its own generator, so adding it to an Objective cannot
        shift any other seeded behaviour in a run.
        """
        torch.manual_seed(SEED)
        expected = torch.randn(1).item()

        torch.manual_seed(SEED)
        noisy = ObjectiveFn(lambda x: x**2, jitter=0.5)
        for x in (1.0, 2.0, 3.0):
            noisy.evaluate(x)

        assert torch.randn(1).item() == expected

    def test_jitter_is_independent_per_call(self) -> None:
        """Test that jitter is drawn afresh for every evaluation.

        The same input point evaluated twice should not get identical noise,
        or jitter would be indistinguishable from a single shared offset
        rather than per-evaluation sensor noise.
        """
        obj = ObjectiveFn(lambda x: x**2, jitter=1.0)

        assert obj.evaluate(2.0) != obj.evaluate(2.0)

    def test_jitter_adds_one_draw_to_a_multi_dimensional_point(self) -> None:
        """Test that a 3D point gets one noise value, not one per coordinate."""
        exact = ObjectiveFn(lambda x1, x2, x3: x1 + x2 + x3)
        noisy = ObjectiveFn(lambda x1, x2, x3: x1 + x2 + x3, jitter=0.5)
        expected_noise = torch.randn(
            1, generator=torch.Generator().manual_seed(25)
        ).item()

        offset = noisy.evaluate(1.0, 2.0, 3.0) - exact.evaluate(1.0, 2.0, 3.0)

        assert offset == pytest.approx(0.5 * expected_noise)

    def test_repr_includes_jitter_when_nonzero(self) -> None:
        """Test that repr surfaces a non-default jitter value."""
        obj = ObjectiveFn(jitter=0.1)
        assert repr(obj) == "ObjectiveFn(function=sum(x_i^2), jitter=0.1, seed=25)"

    def test_repr_omits_jitter_when_zero(self, objective: ObjectiveFn) -> None:
        """Test that repr matches prior output when jitter is off."""
        assert "jitter" not in repr(objective)

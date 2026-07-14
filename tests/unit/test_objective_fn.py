"""Unit tests for the ObjectiveFn class."""

import pytest

from actgpr.objective_fn import ObjectiveFn


def test_objective_evaluation(objective: ObjectiveFn) -> None:
    """Test that the objective evaluates positional inputs correctly."""
    # Single inputs
    assert objective.evaluate(2.0) == (4.0,)
    assert objective.evaluate(-3.0) == (9.0,)
    assert objective.evaluate(0.0) == (0.0,)

    # Multiple inputs
    assert objective.evaluate(2.0, -3.0, 0.0) == (4.0, 9.0, 0.0)


def test_objective_empty_input(objective: ObjectiveFn) -> None:
    """Test that the objective raises ValueError when no inputs are provided."""
    with pytest.raises(ValueError, match="At least one input argument"):
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
    assert objective.evaluate(3) == (9.0,)


@pytest.mark.parametrize("x", [-5.0, -1.0, 0.0, 1.0, 5.0])
def test_objective_output_is_non_negative(objective: ObjectiveFn, x: float) -> None:
    """Test scientific invariant: x² is always non-negative."""
    (result,) = objective.evaluate(x)
    assert result >= 0


def test_objective_repr(objective: ObjectiveFn) -> None:
    """Test the string representation of the default ObjectiveFn."""
    assert repr(objective) == "ObjectiveFn(function=x^2)"


def test_custom_callable_objective() -> None:
    """Test custom objective initialisation and evaluation."""
    custom_func = lambda x: (x + 2) ** 2
    obj = ObjectiveFn(custom_func)

    assert obj.evaluate(1.0) == (9.0,)
    assert obj.evaluate(-2.0) == (0.0,)
    assert repr(obj) == "ObjectiveFn(function=custom_function)"


def test_custom_named_function_repr() -> None:
    """Test that repr uses function names for normal named functions."""
    def my_cool_function(x: float) -> float:
        return x + 5

    obj = ObjectiveFn(my_cool_function)
    assert repr(obj) == "ObjectiveFn(function=my_cool_function)"


def test_custom_function_error_propagation() -> None:
    """Test that errors inside custom functions are caught and raised as TypeError."""
    def failing_func(x: float) -> float:
        raise ValueError("Something went wrong inside the function")

    obj = ObjectiveFn(failing_func)
    with pytest.raises(TypeError, match="Error evaluating objective function"):
        obj.evaluate(1.0)

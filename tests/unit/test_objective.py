"""Unit tests for the Objective class."""

import pytest

from actgpr.objective import Objective


def test_objective_evaluation(objective: Objective) -> None:
    """Test that the objective evaluates positional inputs correctly."""
    # Single inputs
    assert objective.evaluate(2.0) == (4.0,)
    assert objective.evaluate(-3.0) == (9.0,)
    assert objective.evaluate(0.0) == (0.0,)

    # Multiple inputs
    assert objective.evaluate(2.0, -3.0, 0.0) == (4.0, 9.0, 0.0)


def test_objective_empty_input(objective: Objective) -> None:
    """Test that the objective raises ValueError when no inputs are provided."""
    with pytest.raises(ValueError, match="At least one input argument"):
        objective.evaluate()


@pytest.mark.parametrize("bad_input", [None, "x", [], {}])
def test_objective_raises_on_wrong_type(
    objective: Objective, bad_input: object
) -> None:
    """Test that the objective raises TypeError on non-numeric inputs."""
    with pytest.raises(TypeError, match="Expected float or int"):
        objective.evaluate(bad_input)  # type: ignore[arg-type]


def test_objective_accepts_int_input(objective: Objective) -> None:
    """Test that the objective handles int inputs via implicit float conversion."""
    assert objective.evaluate(3) == (9.0,)


@pytest.mark.parametrize("x", [-5.0, -1.0, 0.0, 1.0, 5.0])
def test_objective_output_is_non_negative(objective: Objective, x: float) -> None:
    """Test scientific invariant: x² is always non-negative."""
    (result,) = objective.evaluate(x)
    assert result >= 0


def test_objective_repr(objective: Objective) -> None:
    """Test the string representation of the Objective."""
    assert repr(objective) == "Objective(function=args^2)"

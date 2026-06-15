"""Unit tests for the Objective class."""

import pytest

from actgpr.objective import Objective


def test_objective_evaluation() -> None:
    """Test that the objective evaluates positional inputs correctly."""
    objective = Objective()
    # Single inputs
    assert objective.evaluate(2.0) == (4.0,)
    assert objective.evaluate(-3.0) == (9.0,)
    assert objective.evaluate(0.0) == (0.0,)

    # Multiple inputs
    assert objective.evaluate(2.0, -3.0, 0.0) == (4.0, 9.0, 0.0)


def test_objective_empty_input() -> None:
    """Test that the objective raises ValueError when no inputs are provided."""
    objective = Objective()
    with pytest.raises(ValueError):
        objective.evaluate()


def test_objective_invalid_input_type() -> None:
    """Test that the objective raises TypeError on non-numeric inputs."""
    objective = Objective()
    with pytest.raises(TypeError):
        objective.evaluate("not a number")  # type: ignore[arg-type]


def test_objective_repr() -> None:
    """Test the string representation of the Objective."""
    objective = Objective()
    assert repr(objective) == "Objective(function=args^2)"

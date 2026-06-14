"""Objective function module for active GPR optimisation."""


class Objective:
    """Objective function for active GPR optimisation.

    This class represents the real-valued scalar function being optimised.
    In this implementation, it evaluates the quadratic function:
    f(input_point) = input_point^2.
    """

    def __init__(self) -> None:
        """Initialize the Objective."""
        pass

    def evaluate(self, input_point: float) -> float:
        """Evaluate the objective at a single input point.

        Parameters
        ----------
        input_point : float
            A single value (currently 1D scalar) passed to the Objective.

        Returns
        -------
        float
            The scalar evaluation result of input_point^2.

        Raises
        ------
        TypeError
            If the input_point is not a float or integer.
        """
        try:
            return float(input_point**2)
        except TypeError as exc:
            raise TypeError(
                f"Expected float or int for input_point, got {type(input_point).__name__}"
            ) from exc

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the Objective."""
        return "Objective(function=input_point^2)"

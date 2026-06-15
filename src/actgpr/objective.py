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

    def evaluate(self, *args: float) -> tuple[float, ...]:
        """Evaluate the objective at multiple input points.

        Parameters
        ----------
        *args : float
            Arbitrary positional arguments representing the input values to evaluate.

        Returns
        -------
        tuple of float
            The quadratic evaluation result (value^2) for each input value in the same order.

        Raises
        ------
        ValueError
            If no input arguments are provided.
        TypeError
            If any of the input values cannot be converted to a float.
        """
        if not args:
            raise ValueError("At least one input argument must be provided.")

        results = []
        for i, value in enumerate(args):
            try:
                results.append(float(value**2))
            except TypeError as exc:
                raise TypeError(
                    f"Expected float or int for argument at index {i}, got {type(value).__name__}"
                ) from exc
        return tuple(results)

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the Objective."""
        return "Objective(function=args^2)"

"""Objective function module for active GPR optimisation."""

from typing import Callable


def _default_func(x: float) -> float:
    """Evaluate the default objective function f(x) = x^2."""
    return x**2


DEFAULT_FUNC = _default_func


class ObjectiveFn:
    """Objective function for active GPR optimisation.

    This class represents the real-valued scalar function being optimised.
    It can be configured with an arbitrary single-input function.
    By default, it evaluates the quadratic function: f(x) = x^2.
    """

    def __init__(self, func: Callable[[float], float] | None = None) -> None:
        """Initialize the ObjectiveFn.

        Parameters
        ----------
        func : callable, optional
            A single-input callable that takes a float and returns a float.
            Defaults to lambda x: x**2.
        """
        self.func = func if func is not None else DEFAULT_FUNC

    def evaluate(self, *args: float) -> tuple[float, ...]:
        """Evaluate the objective at multiple input points.

        Parameters
        ----------
        *args : float
            Arbitrary positional arguments representing the input values to evaluate.

        Returns
        -------
        tuple of float
            The function evaluation result for each input value in the same order.

        Raises
        ------
        ValueError
            If no input arguments are provided.
        TypeError
            If any of the input values cannot be converted to a float, or if
            the custom function fails to execute on the input.
        """
        if not args:
            raise ValueError("At least one input argument must be provided.")

        results = []
        for i, value in enumerate(args):
            try:
                float_val = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Expected float or int for argument at index {i}, got {type(value).__name__}"
                ) from exc

            try:
                results.append(float(self.func(float_val)))
            except Exception as exc:
                raise TypeError(
                    f"Error evaluating objective function at index {i} with value {value}: {exc}"
                ) from exc

        return tuple(results)

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the ObjectiveFn."""
        if self.func is DEFAULT_FUNC:
            func_desc = "x^2"
        elif hasattr(self.func, "__name__") and self.func.__name__ != "<lambda>":
            func_desc = self.func.__name__
        else:
            func_desc = "custom_function"

        return f"ObjectiveFn(function={func_desc})"

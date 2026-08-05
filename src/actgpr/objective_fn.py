"""Objective function module for active GPR optimisation."""

from typing import Callable

import torch


def _default_func(x: float) -> float:
    """Evaluate the default Objective: x²."""
    return x**2


DEFAULT_FUNC = _default_func


class ObjectiveFn:
    """Objective function for active GPR optimisation.

    This class represents the real-valued scalar function being optimised.
    It can be configured with an arbitrary single-input function.
    By default, it evaluates the quadratic function: f(x) = x^2.
    Optionally adds Gaussian jitter to each evaluation, to simulate the
    sensor/measurement noise of a real experiment.
    """

    def __init__(
        self,
        func: Callable[[float], float] | None = None,
        jitter: float = 0.0,
    ) -> None:
        """Initialize the ObjectiveFn.

        Parameters
        ----------
        func : callable, optional
            A single-input callable that takes a float and returns a float.
            Defaults to lambda x: x**2.
        jitter : float, optional
            Standard deviation of Gaussian noise added to each evaluation,
            by default 0.0 (no noise). Simulates the sensor/measurement
            noise a real experiment would have; pairs with the surrogate's
            ``noise`` hyperparameter, which models exactly this observation
            noise. Drawn from ``torch``'s RNG, so it is controlled by
            ``torch.manual_seed(...)`` like the rest of the package.

        Raises
        ------
        ValueError
            If jitter is negative.
        """
        if jitter < 0:
            raise ValueError(f"jitter must be non-negative, got {jitter}")

        self.func = func if func is not None else DEFAULT_FUNC
        self.jitter = jitter

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
            If an input value cannot be converted to a float, or if the
            Objective returns a non-numeric value.

        Notes
        -----
        Exceptions raised *inside* the Objective itself (e.g. a ValueError
        from a domain error) propagate unchanged so callers can handle the
        original error type.

        If ``jitter`` is non-zero, independent Gaussian noise with that
        standard deviation is added to each result after ``func`` runs —
        the wrapped function itself always sees the exact, noise-free input.
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

            # Errors raised by the Objective itself propagate unchanged —
            # relabelling them would mask the original error type.
            result = self.func(float_val)

            try:
                results.append(float(result))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Objective returned non-numeric value {result!r} "
                    f"({type(result).__name__}) at index {i}"
                ) from exc

        assert len(results) == len(
            args
        ), f"Expected {len(args)} outputs, got {len(results)}"

        if self.jitter > 0:
            noise = torch.randn(len(results)) * self.jitter
            results = [r + n.item() for r, n in zip(results, noise)]

        return tuple(results)

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the ObjectiveFn."""
        if self.func is DEFAULT_FUNC:
            func_desc = "x^2"
        elif hasattr(self.func, "__name__") and self.func.__name__ != "<lambda>":
            func_desc = self.func.__name__
        else:
            func_desc = "custom_function"

        if self.jitter > 0:
            return f"ObjectiveFn(function={func_desc}, jitter={self.jitter})"
        return f"ObjectiveFn(function={func_desc})"

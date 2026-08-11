"""Objective function module for active GPR optimisation.

Defines the Objective interface actgpr optimises against, and ObjectiveFn,
the convenience wrapper for plain callables.
"""

from typing import Callable, Protocol

import torch


class Objective(Protocol):
    """Anything actgpr can minimise: an object with an ``evaluate`` method.

    This is a :class:`typing.Protocol`, so conformance is *structural*:
    any object providing a matching ``evaluate`` satisfies it, with no base
    class to inherit and no registration step. ``OptimisationRun`` never
    inspects the type of the objective it is given, only calls this method,
    so wrapping a simulation of your own means writing a class with this
    one method on it.

    ``ObjectiveFn`` below satisfies this protocol and is the convenient
    route for a plain function; it is not the only permitted Objective.
    """

    def evaluate(self, *args: float) -> tuple[float, ...]:
        """Evaluate the Objective at one or more input points."""
        ...


def _default_func(x: float) -> float:
    """Evaluate the default Objective: x²."""
    return x**2


DEFAULT_FUNC = _default_func

# Seed for ObjectiveFn's jitter generator. Fixed so that a jittered run is
# reproducible out of the box: without it, the noise would differ between
# runs and the MRR record could not reproduce its own result.
DEFAULT_JITTER_SEED = 25


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
        seed: int = DEFAULT_JITTER_SEED,
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
            noise.
        seed : int, optional
            Seed for the jitter noise, by default 25. The noise is drawn
            from a generator owned by this ObjectiveFn, so a jittered run
            is reproducible without the caller seeding anything, and
            drawing jitter does not disturb the global ``torch`` RNG. Only
            has an effect when ``jitter`` is non-zero.

        Raises
        ------
        ValueError
            If jitter is negative.
        """
        if jitter < 0:
            raise ValueError(f"jitter must be non-negative, got {jitter}")

        self.func = func if func is not None else DEFAULT_FUNC
        self.jitter = jitter
        self.seed = seed
        self._generator = torch.Generator().manual_seed(seed)

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
        standard deviation is added to each result after ``func`` runs. The
        wrapped function itself always sees the exact, noise-free input.
        The noise comes from this ObjectiveFn's own seeded generator, so
        repeated calls advance it (the same input evaluated twice gives
        different noise, as a real sensor would) while two ObjectiveFn
        objects built with the same seed produce the same sequence.
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

            # Errors raised by the Objective itself propagate unchanged;
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
            noise = torch.randn(len(results), generator=self._generator) * self.jitter
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
            # The seed is only reported alongside jitter, because it is what
            # makes a jittered run reproducible and config.json records this
            # string as the run's Objective.
            return (
                f"ObjectiveFn(function={func_desc}, "
                f"jitter={self.jitter}, seed={self.seed})"
            )
        return f"ObjectiveFn(function={func_desc})"

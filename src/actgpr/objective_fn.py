"""Objective function module for active GPR optimisation.

Defines the Objective interface actgpr optimises against, and ObjectiveFn,
the convenience wrapper for plain callables.

An Objective is evaluated at one input point per call. The point's
coordinates are passed as separate arguments, one per input dimension, and
the call returns one output:

>>> ObjectiveFn(lambda x1, x2: x1 + x2).evaluate(1.0, 2.0)
3.0

Classes
-------
Objective
    The interface: any object with ``evaluate(*x) -> float``.
ObjectiveFn
    Wraps a plain function of one or more inputs as an Objective.
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

    ``evaluate`` is called once per input point, with that point's
    coordinates as positional arguments, one per input dimension, and must
    return that point's single output. A three-input simulation therefore
    implements ``def evaluate(self, x1, x2, x3) -> float``.
    """

    def evaluate(self, *x: float) -> float:
        """Evaluate the Objective at one input point and return its output."""
        ...


def _default_func(*x: float) -> float:
    """Evaluate the default Objective: the sum of squared coordinates."""
    return sum(value**2 for value in x)


DEFAULT_FUNC = _default_func

# Seed for ObjectiveFn's jitter generator. Fixed so that a jittered run is
# reproducible out of the box: without it, the noise would differ between
# runs and the MRR record could not reproduce its own result.
DEFAULT_JITTER_SEED = 25


class ObjectiveFn:
    """Objective function for active GPR optimisation.

    This class represents the real-valued scalar function being optimised.
    It wraps a plain function taking one argument per input dimension, e.g.
    ``def f(x1, x2, x3) -> float``. By default it evaluates the sum of
    squared coordinates, which is f(x) = x² with one input dimension.
    Optionally adds Gaussian jitter to each evaluation, to simulate the
    sensor/measurement noise of a real experiment.

    Public Methods
    --------------
    evaluate(x1, ..., xd)
        Evaluate the function at one input point with d coordinates.
    """

    def __init__(
        self,
        func: Callable[..., float] | None = None,
        jitter: float = 0.0,
        seed: int = DEFAULT_JITTER_SEED,
    ) -> None:
        """Initialize the ObjectiveFn.

        Parameters
        ----------
        func : callable, optional
            A function taking one float per input dimension and returning one
            float, e.g. ``lambda x1, x2: x1**2 + x2**2``. Defaults to the sum
            of squared coordinates.
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

    def evaluate(self, *x: float) -> float:
        """Evaluate the Objective at one input point.

        Parameters
        ----------
        *x : float
            The coordinates of the input point, one per input dimension.
            They are passed to ``func`` in the same order, so
            ``evaluate(x1, x2, x3)`` calls ``func(x1, x2, x3)``.

        Returns
        -------
        float
            The Objective output at that input point.

        Raises
        ------
        ValueError
            If no coordinates are provided.
        TypeError
            If a coordinate cannot be converted to a float, or if ``func``
            returns a non-numeric value.

        Notes
        -----
        Exceptions raised *inside* ``func`` (e.g. a ValueError from a domain
        error, or a TypeError because ``func`` expects a different number of
        inputs) propagate unchanged so callers can handle the original
        error type.

        If ``jitter`` is non-zero, Gaussian noise with that standard
        deviation is added to the result after ``func`` runs. The wrapped
        function itself always sees the exact, noise-free input. The noise
        comes from this ObjectiveFn's own seeded generator, so repeated calls
        advance it (the same input evaluated twice gives different noise, as
        a real sensor would) while two ObjectiveFn objects built with the
        same seed produce the same sequence.

        Examples
        --------
        >>> ObjectiveFn().evaluate(3.0)
        9.0
        >>> ObjectiveFn(lambda x1, x2: x1 * x2).evaluate(2.0, 4.0)
        8.0
        """
        if not x:
            raise ValueError(
                "evaluate() needs the coordinates of one input point, got none."
            )

        coordinates = []
        for dim, value in enumerate(x):
            try:
                coordinates.append(float(value))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Expected float or int for coordinate {dim}, "
                    f"got {type(value).__name__}"
                ) from exc

        # Errors raised by the Objective itself propagate unchanged;
        # relabelling them would mask the original error type.
        result = self.func(*coordinates)

        try:
            output = float(result)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Objective returned non-numeric value {result!r} "
                f"({type(result).__name__})"
            ) from exc

        if self.jitter > 0:
            output += torch.randn(1, generator=self._generator).item() * self.jitter

        return output

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the ObjectiveFn."""
        if self.func is DEFAULT_FUNC:
            func_desc = "sum(x_i^2)"
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

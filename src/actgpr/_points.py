"""Input point and search bounds handling shared across actgpr.

An input point has one coordinate per input dimension, so a set of ``n``
input points in ``d`` dimensions is always carried as a tensor of shape
``(n, d)``. This module is the single place that turns what a user passes in
into that shape, so the surrogate, the acquisition function, and the
optimisation run cannot disagree about it.

Functions
---------
as_points
    Return input points as a float64 tensor of shape (n, d).
parse_search_bounds
    Return search bounds as one (lo, hi) pair per input dimension.
format_values
    Render a point or per-dimension values for logs and plot titles.
"""

from collections.abc import Sequence

import torch

Bounds = tuple[tuple[float, float], ...]


def as_points(x: object, n_dims: int | None = None) -> torch.Tensor:
    """Return input points as a float64 tensor of shape (n, d).

    A flat sequence is read as ``n`` points in one dimension. With more than
    one input dimension every point must be given as its own row, since a
    flat sequence would be ambiguous: ``[1.0, 2.0]`` could be two 1D points
    or one 2D point.

    Parameters
    ----------
    x : array-like of shape (n,) or (n, d)
        The input points. A flat sequence is only accepted when ``n_dims``
        is None or 1.
    n_dims : int or None, optional
        The number of input dimensions the points must have. If None, it is
        taken from ``x``.

    Returns
    -------
    torch.Tensor of shape (n, d)
        The input points, one row per point.

    Raises
    ------
    TypeError
        If x cannot be converted to a numeric tensor.
    ValueError
        If x is not one- or two-dimensional, or does not have ``n_dims``
        coordinates per point.

    Examples
    --------
    >>> as_points([-3.0, 5.0]).shape
    torch.Size([2, 1])
    >>> as_points([[0.0, 1.0], [2.0, 3.0]], n_dims=2).shape
    torch.Size([2, 2])
    """
    try:
        points = torch.as_tensor(x, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(
            f"Input points must be numeric, got {type(x).__name__}: {exc}"
        ) from exc

    if points.ndim == 1:
        if n_dims not in (None, 1):
            raise ValueError(
                f"Expected {n_dims} coordinates per input point, so pass one "
                f"row per point, e.g. [[x1, ..., x{n_dims}], ...]; got a flat "
                f"sequence of {points.numel()} values."
            )
        points = points.unsqueeze(-1)
    elif points.ndim != 2:
        raise ValueError(
            f"Input points must have shape (n,) or (n, d), got {tuple(points.shape)}."
        )

    if n_dims is not None and points.shape[1] != n_dims:
        raise ValueError(
            f"Expected {n_dims} coordinates per input point, got {points.shape[1]}."
        )

    return points


def parse_search_bounds(
    search_bounds: Sequence[float] | Sequence[Sequence[float]],
) -> Bounds:
    """Return search bounds as one (lo, hi) pair per input dimension.

    The number of pairs is the number of input dimensions of the run. A
    single ``(lo, hi)`` pair is shorthand for one input dimension.

    Parameters
    ----------
    search_bounds : sequence of (lo, hi) pairs, or a single (lo, hi) pair
        The closed interval of each input dimension, in order.

    Returns
    -------
    tuple of (float, float)
        One ``(lo, hi)`` pair per input dimension.

    Raises
    ------
    ValueError
        If no interval is given, an interval does not have exactly two
        values, or its lower value is not below its upper value.

    Examples
    --------
    >>> parse_search_bounds((-3.0, 5.0))
    ((-3.0, 5.0),)
    >>> parse_search_bounds([(-3.0, 5.0), (0.0, 1.0)])
    ((-3.0, 5.0), (0.0, 1.0))
    """
    if len(search_bounds) == 0:
        raise ValueError("search_bounds must contain at least one (lo, hi) pair.")

    # A single (lo, hi) pair of numbers is shorthand for one dimension; a
    # pair's first element fails float() because it is itself a sequence.
    try:
        float(search_bounds[0])
    except (TypeError, ValueError, RuntimeError):
        intervals = search_bounds
    else:
        intervals = [search_bounds]

    parsed = []
    for dim, interval in enumerate(intervals):
        if len(interval) != 2:
            raise ValueError(
                f"search_bounds for input dimension {dim} must be a (lo, hi) "
                f"pair, got {len(interval)} values."
            )
        lo, hi = float(interval[0]), float(interval[1])
        if not lo < hi:
            raise ValueError(
                f"search_bounds for input dimension {dim}: lo ({lo}) must be "
                f"< hi ({hi})."
            )
        parsed.append((lo, hi))

    return tuple(parsed)


def format_values(values: Sequence[float], spec: str = ".4f") -> str:
    """Render a point or per-dimension values for logs and plot titles.

    A single value is rendered bare, so one-dimensional output reads as it
    always has; several values are rendered as a parenthesised tuple.

    Parameters
    ----------
    values : sequence of float
        The values to render, one per input dimension.
    spec : str, optional
        The format specification applied to each value, by default ".4f".

    Returns
    -------
    str
        The rendered values.

    Examples
    --------
    >>> format_values([1.5])
    '1.5000'
    >>> format_values([1.5, -2.0], ".2f")
    '(1.50, -2.00)'
    """
    rendered = [format(float(value), spec) for value in values]
    if len(rendered) == 1:
        return rendered[0]
    return "(" + ", ".join(rendered) + ")"

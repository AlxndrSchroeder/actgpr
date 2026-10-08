"""Active GPR (Gaussian Process Regression) Optimisation package.

This package finds the minimum of an expensive-to-evaluate objective
function with one or more inputs and a single scalar output, by iteratively
fitting a Gaussian Process surrogate and using active learning. The number of
inputs is set by ``search_bounds``, one ``(lo, hi)`` pair per input.

Exported classes
----------------
OptimisationRun
    Orchestrates the active optimisation loop and MRR artifact writes.
ObjectiveFn
    Wraps a plain function of one or more inputs as the Objective.
GPyTorchSurrogate
    Gaussian Process surrogate backend built on GPyTorch.
Acquisition
    Expected Improvement acquisition function.
"""

from importlib.metadata import PackageNotFoundError, version

from actgpr.acquisition import Acquisition
from actgpr.objective_fn import ObjectiveFn
from actgpr.run import OptimisationRun
from actgpr.surrogate import GPyTorchSurrogate

try:
    # Single source of truth: the version declared in pyproject.toml
    __version__ = version("actgpr")
except PackageNotFoundError:
    # Package is not installed (e.g. source tree without poetry install)
    __version__ = "0.0.0"

__all__ = [
    "Acquisition",
    "ObjectiveFn",
    "OptimisationRun",
    "GPyTorchSurrogate",
    "__version__",
]

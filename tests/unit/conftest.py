"""Shared fixtures for unit tests."""

import math

import matplotlib.pyplot as plt
import pytest
import torch

from actgpr.objective_fn import ObjectiveFn
from actgpr.surrogate import GPyTorchSurrogate

SEED = 42


@pytest.fixture(autouse=True)
def _close_all_figures():
    """Close every matplotlib figure after each test.

    Plotting tests create figures via plt.subplots() without an explicit
    plt.close(); left open, they accumulate across the suite until
    matplotlib's open-figure warning fires — which our strict
    filterwarnings=["error"] policy turns into a failure on whatever test
    happens to tip over the threshold, not the test that actually leaked.
    """
    yield
    plt.close("all")


@pytest.fixture()
def objective() -> ObjectiveFn:
    """Return a fresh ObjectiveFn instance."""
    return ObjectiveFn()


@pytest.fixture()
def training_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a small, seeded (train_x, train_y) pair for testing."""
    torch.manual_seed(SEED)
    train_x = torch.linspace(0, 1, 20)
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(20) * math.sqrt(0.04)
    return train_x, train_y


@pytest.fixture()
def fitted_model(
    training_data: tuple[torch.Tensor, torch.Tensor],
) -> GPyTorchSurrogate:
    """Return a GPyTorchSurrogate that has already been fitted."""
    train_x, train_y = training_data
    model = GPyTorchSurrogate()
    model.fit_and_train(train_x, train_y, training_iter=20)
    return model

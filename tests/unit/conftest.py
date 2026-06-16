"""Shared fixtures for unit tests."""

import math

import gpytorch
import pytest
import torch

from actgpr.objective import Objective
from actgpr.surrogate import ExactGPModel, GPyTorchSurrogate

SEED = 42


@pytest.fixture()
def objective() -> Objective:
    """Return a fresh Objective instance."""
    return Objective()


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
    model.fit(train_x, train_y, training_iter=20)
    return model

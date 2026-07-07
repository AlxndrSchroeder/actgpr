"""Unit tests for the surrogate module (ExactGPModel and GPyTorchSurrogate)."""

import gpytorch
import pytest
import torch

from actgpr.surrogate import ExactGPModel, GPyTorchSurrogate

SEED = 42


# ---------------------------------------------------------------------------
# ExactGPModel
# ---------------------------------------------------------------------------


class TestExactGPModel:
    """Tests for the ExactGPModel prior specification."""

    def test_forward_returns_multivariate_normal(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that forward() produces a MultivariateNormal distribution."""
        train_x, train_y = training_data
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)
        model.train()
        likelihood.train()

        output = model(train_x)

        assert isinstance(output, gpytorch.distributions.MultivariateNormal)

    def test_forward_output_shape(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that forward() mean shape matches the input length."""
        train_x, train_y = training_data
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)
        model.train()
        likelihood.train()

        output = model(train_x)

        assert output.mean.shape == train_x.shape

    def test_has_constant_mean_and_scaled_rbf_kernel(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that the model uses ConstantMean and ScaleKernel(RBFKernel)."""
        train_x, train_y = training_data
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)

        assert isinstance(model.mean_module, gpytorch.means.ConstantMean)
        assert isinstance(model.covar_module, gpytorch.kernels.ScaleKernel)
        assert isinstance(model.covar_module.base_kernel, gpytorch.kernels.RBFKernel)


# ---------------------------------------------------------------------------
# GPyTorchSurrogate — initialisation
# ---------------------------------------------------------------------------


class TestGPyTorchSurrogateInit:
    """Tests for GPyTorchSurrogate initial state."""

    def test_initial_state_is_none(self) -> None:
        """Test that a fresh model has no fitted components."""
        model = GPyTorchSurrogate()

        assert model.model is None
        assert model.likelihood is None
        assert model.train_x is None
        assert model.train_y is None


# ---------------------------------------------------------------------------
# GPyTorchSurrogate.fit
# ---------------------------------------------------------------------------


class TestGPyTorchSurrogateFit:
    """Tests for GPyTorchSurrogate.fit()."""

    def test_fit_populates_model_and_likelihood(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that fit() creates a model and likelihood."""
        train_x, train_y = training_data
        model = GPyTorchSurrogate()
        model.fit(train_x, train_y, training_iter=5)

        assert model.model is not None
        assert model.likelihood is not None
        assert model.train_x is not None
        assert model.train_y is not None

    def test_fit_stores_training_data(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that fit() stores a reference to the original training data."""
        train_x, train_y = training_data
        model = GPyTorchSurrogate()
        model.fit(train_x, train_y, training_iter=5)

        assert torch.equal(model.train_x, train_x)
        assert torch.equal(model.train_y, train_y)

    def test_fit_raises_on_shape_mismatch(self) -> None:
        """Test that fit() raises ValueError when train_x and train_y have different shapes."""
        model = GPyTorchSurrogate()
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 5)

        with pytest.raises(ValueError, match="Shape mismatch"):
            model.fit(train_x, train_y)

    def test_fit_is_deterministic(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that fitting with the same seed produces identical hyperparameters."""
        train_x, train_y = training_data

        torch.manual_seed(SEED)
        model_a = GPyTorchSurrogate()
        model_a.fit(train_x, train_y, training_iter=20)

        torch.manual_seed(SEED)
        model_b = GPyTorchSurrogate()
        model_b.fit(train_x, train_y, training_iter=20)

        assert model_a.model is not None and model_b.model is not None
        ls_a = model_a.model.covar_module.base_kernel.lengthscale
        ls_b = model_b.model.covar_module.base_kernel.lengthscale
        assert torch.allclose(ls_a, ls_b)


# ---------------------------------------------------------------------------
# GPyTorchSurrogate.predict
# ---------------------------------------------------------------------------


class TestGPyTorchSurrogatePredict:
    """Tests for GPyTorchSurrogate.predict()."""

    def test_predict_raises_before_fit(self) -> None:
        """Test that predict() raises RuntimeError on an unfitted model."""
        model = GPyTorchSurrogate()
        test_x = torch.linspace(0, 1, 10)

        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(test_x)

    def test_predict_returns_expected_keys(
        self,
        fitted_model: GPyTorchSurrogate,
    ) -> None:
        """Test that predict() returns all documented dictionary keys."""
        test_x = torch.linspace(0, 1, 10)
        preds = fitted_model.predict(test_x)

        expected_keys = {
            "f_preds",
            "observed_pred",
            "f_mean",
            "f_var",
            "f_covar",
            "f_samples",
        }
        assert set(preds.keys()) == expected_keys

    def test_predict_output_shapes(
        self,
        fitted_model: GPyTorchSurrogate,
    ) -> None:
        """Test that prediction tensor shapes match the number of test points."""
        n_test = 15
        test_x = torch.linspace(0, 1, n_test)
        preds = fitted_model.predict(test_x)

        assert preds["f_mean"].shape == (n_test,)
        assert preds["f_var"].shape == (n_test,)
        assert preds["f_covar"].shape == (n_test, n_test)
        assert preds["f_samples"].shape == (1000, n_test)

    def test_predict_f_mean_is_finite(
        self,
        fitted_model: GPyTorchSurrogate,
    ) -> None:
        """Test scientific invariant: f_mean must contain only finite values."""
        test_x = torch.linspace(0, 1, 10)
        preds = fitted_model.predict(test_x)

        assert torch.all(torch.isfinite(preds["f_mean"]))

    def test_predict_f_var_is_non_negative(
        self,
        fitted_model: GPyTorchSurrogate,
    ) -> None:
        """Test scientific invariant: f_var (variance) must be non-negative."""
        test_x = torch.linspace(0, 1, 10)
        preds = fitted_model.predict(test_x)

        assert torch.all(preds["f_var"] >= 0)

    def test_predict_f_covar_is_symmetric(
        self,
        fitted_model: GPyTorchSurrogate,
    ) -> None:
        """Test scientific invariant: f_covar must be a symmetric matrix."""
        test_x = torch.linspace(0, 1, 10)
        preds = fitted_model.predict(test_x)

        assert torch.allclose(preds["f_covar"], preds["f_covar"].T, atol=1e-5)

    def test_predict_variance_low_at_training_points(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that variance near training points is lower than away from them.

        A fitted GP should be more certain where it has observed data. We compare
        variance at training points against variance at mid-gaps between them.
        """
        train_x, train_y = training_data
        model = GPyTorchSurrogate()
        model.fit(train_x, train_y, training_iter=50)

        preds_at_train = model.predict(train_x)
        var_at_train = preds_at_train["f_var"].mean()

        # Points far outside the training range should have higher variance
        far_points = torch.tensor([5.0, 10.0, -5.0, -10.0])
        preds_far = model.predict(far_points)
        var_far = preds_far["f_var"].mean()

        assert var_at_train < var_far, (
            f"Variance at training points ({var_at_train:.4f}) should be less than "
            f"variance at distant points ({var_far:.4f})"
        )

    def test_predict_distributions_type(
        self,
        fitted_model: GPyTorchSurrogate,
    ) -> None:
        """Test that f_preds and observed_pred are MultivariateNormal distributions."""
        test_x = torch.linspace(0, 1, 10)
        preds = fitted_model.predict(test_x)

        assert isinstance(preds["f_preds"], gpytorch.distributions.MultivariateNormal)
        assert isinstance(
            preds["observed_pred"], gpytorch.distributions.MultivariateNormal
        )

    def test_predict_f_covar_is_positive_semi_definite(
        self,
        fitted_model: GPyTorchSurrogate,
    ) -> None:
        """Test scientific invariant: f_covar eigenvalues must be non-negative."""
        test_x = torch.linspace(0, 1, 10)
        preds = fitted_model.predict(test_x)

        eigenvalues = torch.linalg.eigvalsh(preds["f_covar"])
        assert torch.all(
            eigenvalues >= -1e-5
        ), f"f_covar has negative eigenvalues: {eigenvalues[eigenvalues < -1e-5]}"

    def test_predict_observed_variance_geq_latent_variance(
        self,
        fitted_model: GPyTorchSurrogate,
    ) -> None:
        """Test scientific invariant: observed variance >= latent variance (noise is additive)."""
        test_x = torch.linspace(0, 1, 10)
        preds = fitted_model.predict(test_x)

        observed_var = preds["observed_pred"].variance
        assert torch.all(observed_var >= preds["f_var"] - 1e-5)


# ---------------------------------------------------------------------------
# GPyTorchSurrogate.plot
# ---------------------------------------------------------------------------


class TestGPyTorchSurrogatePlot:
    """Tests for GPyTorchSurrogate.plot()."""

    def test_plot_raises_before_fit(self) -> None:
        """Test that plot() raises RuntimeError on an unfitted model."""
        model = GPyTorchSurrogate()
        test_x = torch.linspace(0, 1, 10)

        with pytest.raises(RuntimeError, match="fitted"):
            model.plot(test_x)

    def test_plot_runs_without_error(
        self,
        fitted_model: GPyTorchSurrogate,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that plot() executes without raising when the model is fitted.

        Uses monkeypatch to suppress plt.show() so no window opens during CI.
        """
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)

        test_x = torch.linspace(0, 1, 10)
        fitted_model.plot(test_x)  # Should not raise

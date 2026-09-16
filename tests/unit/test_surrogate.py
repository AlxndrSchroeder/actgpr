"""Unit tests for the surrogate module (ExactGPModel and GPyTorchSurrogate)."""

import gpytorch
import pytest
import torch

from actgpr.surrogate import ExactGPModel, GPyTorchSurrogate

SEED = 25


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
    """Tests for GPyTorchSurrogate.fit_and_train()."""

    def test_fit_populates_model_and_likelihood(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that fit() creates a model and likelihood."""
        train_x, train_y = training_data
        model = GPyTorchSurrogate()
        model.fit_and_train(train_x, train_y, training_iter=5)

        assert model.model is not None
        assert model.likelihood is not None
        assert model.train_x is not None
        assert model.train_y is not None

    def test_fit_stores_training_data(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that fit() stores the training data, one row per input point."""
        train_x, train_y = training_data
        model = GPyTorchSurrogate()
        model.fit_and_train(train_x, train_y, training_iter=5)

        assert model.train_x.shape == (train_x.shape[0], 1)
        assert torch.equal(model.train_x[:, 0], train_x.double())
        assert torch.equal(model.train_y, train_y.double())

    def test_fit_accepts_several_input_dimensions(self) -> None:
        """Test that input points with three coordinates can be fitted."""
        generator = torch.Generator().manual_seed(SEED)
        train_x = torch.rand(12, 3, generator=generator, dtype=torch.float64)
        train_y = (train_x**2).sum(dim=1)
        model = GPyTorchSurrogate()
        model.fit_no_training(train_x, train_y)

        preds = model.predict(
            torch.rand(7, 3, generator=generator, dtype=torch.float64)
        )

        assert model.train_x.shape == (12, 3)
        assert preds["f_mean"].shape == (7,)
        assert preds["f_var"].shape == (7,)

    def test_predict_rejects_points_with_the_wrong_dimension(self) -> None:
        """Test that predicting in 2D on a 3D fit fails clearly."""
        generator = torch.Generator().manual_seed(SEED)
        model = GPyTorchSurrogate()
        model.fit_no_training(
            torch.rand(5, 3, generator=generator), torch.rand(5, generator=generator)
        )

        with pytest.raises(ValueError, match="Expected 3 coordinates"):
            model.predict(torch.rand(4, 2, generator=generator))

    def test_repr_counts_points_not_coordinates(self) -> None:
        """Test that n_points is the number of rows, not n * d."""
        generator = torch.Generator().manual_seed(SEED)
        model = GPyTorchSurrogate()
        model.fit_no_training(
            torch.rand(5, 3, generator=generator), torch.rand(5, generator=generator)
        )

        assert "n_points=5" in repr(model)

    def test_fit_raises_on_shape_mismatch(self) -> None:
        """Test that fit() raises ValueError when train_x and train_y have different shapes."""
        model = GPyTorchSurrogate()
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 5)

        with pytest.raises(ValueError, match="Shape mismatch"):
            model.fit_and_train(train_x, train_y)

    def test_fit_is_deterministic(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that fitting with the same seed produces identical hyperparameters."""
        train_x, train_y = training_data

        torch.manual_seed(SEED)
        model_a = GPyTorchSurrogate()
        model_a.fit_and_train(train_x, train_y, training_iter=20)

        torch.manual_seed(SEED)
        model_b = GPyTorchSurrogate()
        model_b.fit_and_train(train_x, train_y, training_iter=20)

        assert model_a.model is not None and model_b.model is not None
        ls_a = model_a.model.covar_module.base_kernel.lengthscale
        ls_b = model_b.model.covar_module.base_kernel.lengthscale
        assert torch.allclose(ls_a, ls_b)


# ---------------------------------------------------------------------------
# GPyTorchSurrogate.fit_no_training
# ---------------------------------------------------------------------------


class TestGPyTorchSurrogateFitNoTraining:
    """Tests for GPyTorchSurrogate.fit_no_training() — the without_training() backend."""

    def test_fit_no_training_populates_model_and_likelihood(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that fit_no_training() creates a model and likelihood."""
        train_x, train_y = training_data
        model = GPyTorchSurrogate()
        model.fit_no_training(train_x, train_y)

        assert model.model is not None
        assert model.likelihood is not None
        assert model.train_x is not None
        assert model.train_y is not None

    def test_fit_no_training_sets_given_hyperparameters(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that lengthscale/outputscale/noise are set to the given values, not tuned."""
        train_x, train_y = training_data
        model = GPyTorchSurrogate()
        model.fit_no_training(
            train_x, train_y, lengthscale=2.5, outputscale=3.0, noise=0.02
        )

        assert model.model.covar_module.base_kernel.lengthscale.item() == pytest.approx(
            2.5
        )
        assert model.model.covar_module.outputscale.item() == pytest.approx(3.0)
        assert model.likelihood.noise.item() == pytest.approx(0.02)

    def test_fit_no_training_freezes_all_parameters(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that no parameter is left trainable — without_training() must not tune anything."""
        train_x, train_y = training_data
        model = GPyTorchSurrogate()
        model.fit_no_training(train_x, train_y)

        assert all(not p.requires_grad for p in model.model.parameters())
        assert all(not p.requires_grad for p in model.likelihood.parameters())

    def test_fit_no_training_predictions_reflect_fixed_outputscale(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that the fixed hyperparameters actually drive predictions.

        A larger outputscale should widen posterior variance away from the
        training data relative to a smaller one — confirms fit_no_training's
        values are used at predict time rather than silently overridden by
        some default.
        """
        train_x, train_y = training_data
        small_scale = GPyTorchSurrogate()
        small_scale.fit_no_training(train_x, train_y, outputscale=0.1)
        large_scale = GPyTorchSurrogate()
        large_scale.fit_no_training(train_x, train_y, outputscale=10.0)

        test_x = torch.tensor([5.0])  # away from training data
        var_small = small_scale.predict(test_x)["f_var"].item()
        var_large = large_scale.predict(test_x)["f_var"].item()

        assert var_large > var_small


# ---------------------------------------------------------------------------
# GPyTorchSurrogate.predict
# ---------------------------------------------------------------------------


class TestGPyTorchSurrogateHyperparameters:
    """Tests for GPyTorchSurrogate.hyperparameters()."""

    def test_raises_before_fitting(self) -> None:
        """Test that reading hyperparameters off an unfitted surrogate is an error."""
        model = GPyTorchSurrogate()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.hyperparameters()

    def test_reports_the_values_given_to_fit_no_training(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that fixed hyperparameters are reported back unchanged."""
        train_x, train_y = training_data
        model = GPyTorchSurrogate()
        model.fit_no_training(
            train_x, train_y, lengthscale=2.5, outputscale=3.5, noise=1e-3
        )

        hyperparameters = model.hyperparameters()

        assert hyperparameters["lengthscale"] == pytest.approx((2.5,), rel=1e-4)
        assert hyperparameters["outputscale"] == pytest.approx((3.5,), rel=1e-4)
        assert hyperparameters["noise"] == pytest.approx((1e-3,), rel=1e-4)

    def test_reports_the_values_adam_tuned_to(
        self,
        training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that trained hyperparameters differ from the starting point.

        This is the case the MRR record depends on: config.json is written
        before the loop and can only hold the starting values, so these are
        recoverable nowhere else.
        """
        train_x, train_y = training_data
        torch.manual_seed(SEED)
        model = GPyTorchSurrogate()
        model.fit_and_train(train_x, train_y, training_iter=25, noise=1e-4)

        hyperparameters = model.hyperparameters()

        assert set(hyperparameters) == {"lengthscale", "outputscale", "noise"}
        assert all(isinstance(v, tuple) for v in hyperparameters.values())
        assert all(isinstance(x, float) for v in hyperparameters.values() for x in v)
        # Adam moves the kernel away from GPyTorch's default starting point.
        assert hyperparameters["lengthscale"] != pytest.approx((1.0,), rel=1e-3)

    def test_lengthscale_has_one_value_per_input_dimension(self) -> None:
        """Test that a 3D fit reports three lengthscales, others one value."""
        generator = torch.Generator().manual_seed(SEED)
        model = GPyTorchSurrogate()
        model.fit_no_training(
            torch.rand(6, 3, generator=generator),
            torch.rand(6, generator=generator),
            lengthscale=[0.5, 1.0, 2.0],
        )

        hyperparameters = model.hyperparameters()

        assert hyperparameters["lengthscale"] == pytest.approx((0.5, 1.0, 2.0))
        assert len(hyperparameters["outputscale"]) == 1
        assert len(hyperparameters["noise"]) == 1

    def test_single_lengthscale_is_used_for_every_dimension(self) -> None:
        """Test that a scalar lengthscale is shared across all dimensions."""
        generator = torch.Generator().manual_seed(SEED)
        model = GPyTorchSurrogate()
        model.fit_no_training(
            torch.rand(6, 3, generator=generator),
            torch.rand(6, generator=generator),
            lengthscale=1.5,
        )

        assert model.hyperparameters()["lengthscale"] == pytest.approx((1.5,) * 3)

    def test_lengthscale_count_must_match_the_dimensions(self) -> None:
        """Test that two lengthscales for a 3D fit is rejected."""
        generator = torch.Generator().manual_seed(SEED)
        model = GPyTorchSurrogate()

        with pytest.raises(ValueError, match="one value per input dimension"):
            model.fit_no_training(
                torch.rand(6, 3, generator=generator),
                torch.rand(6, generator=generator),
                lengthscale=[1.0, 2.0],
            )


class TestGPyTorchSurrogatePredict:
    """Tests for GPyTorchSurrogate.predict()."""

    def test_predict_raises_before_fit(self) -> None:
        """Test that predict() raises RuntimeError on an unfitted model."""
        model = GPyTorchSurrogate()
        test_x = torch.linspace(0, 1, 10)

        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(test_x)

    def test_predict_omits_the_dense_covariance(
        self,
        fitted_model: GPyTorchSurrogate,
    ) -> None:
        """Test that predict() does not build the (m, m) covariance eagerly.

        It grows with the square of the candidate count and nothing in the
        optimisation loop reads it; at 32768 candidates it took minutes.
        """
        preds = fitted_model.predict(torch.linspace(0, 1, 10))

        assert "f_covar" not in preds

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
        }
        assert set(preds.keys()) == expected_keys

    def test_predict_skips_sampling_by_default(
        self,
        fitted_model: GPyTorchSurrogate,
    ) -> None:
        """Test that f_samples is absent when n_samples is not requested."""
        test_x = torch.linspace(0, 1, 10)
        preds = fitted_model.predict(test_x)

        assert "f_samples" not in preds

    def test_predict_output_shapes(
        self,
        fitted_model: GPyTorchSurrogate,
    ) -> None:
        """Test that prediction tensor shapes match the number of test points."""
        n_test = 15
        n_samples = 25
        test_x = torch.linspace(0, 1, n_test)
        preds = fitted_model.predict(test_x, n_samples=n_samples)

        assert preds["f_mean"].shape == (n_test,)
        assert preds["f_var"].shape == (n_test,)
        assert preds["f_preds"].covariance_matrix.shape == (n_test, n_test)
        assert preds["f_samples"].shape == (n_samples, n_test)

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
        """Test scientific invariant: the posterior covariance is symmetric."""
        test_x = torch.linspace(0, 1, 10)
        f_covar = fitted_model.predict(test_x)["f_preds"].covariance_matrix

        assert torch.allclose(f_covar, f_covar.T, atol=1e-5)

    # Predicting at exactly the training inputs is the point of this test;
    # gpytorch's "did you forget model.train()?" heuristic is a false positive.
    @pytest.mark.filterwarnings("ignore::gpytorch.utils.warnings.GPInputWarning")
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
        model.fit_and_train(train_x, train_y, training_iter=50)

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
        """Test scientific invariant: posterior covariance eigenvalues are >= 0."""
        test_x = torch.linspace(0, 1, 10)
        f_covar = fitted_model.predict(test_x)["f_preds"].covariance_matrix

        eigenvalues = torch.linalg.eigvalsh(f_covar)
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

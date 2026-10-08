"""Surrogate model module for active GPR optimisation."""

from collections.abc import Sequence

import torch
import gpytorch

from actgpr._points import as_points

# Jitter added to the covariance diagonal during Cholesky decomposition to
# keep the matrix numerically positive definite (float64 throughout).
CHOLESKY_JITTER = 1e-4


class ExactGPModel(gpytorch.models.ExactGP):
    """An exact Gaussian Process model with Constant mean and scaled RBF kernel.

    This class defines the structural prior components (mean and covariance) of
    the GP model. The RBF kernel has one lengthscale per input dimension
    (automatic relevance determination), since inputs of a multi-dimensional
    problem generally vary on different scales.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> None:
        """Initialize the ExactGPModel.

        Parameters
        ----------
        train_x : torch.Tensor of shape (n, d)
            The training input points, one row per point.
        train_y : torch.Tensor of shape (n,)
            The training outputs.
        likelihood : gpytorch.likelihoods.GaussianLikelihood
            The GPyTorch likelihood mapping latent outputs to observed targets.
        """
        super().__init__(train_x, train_y, likelihood)
        n_dims = train_x.shape[-1] if train_x.ndim > 1 else 1
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=n_dims)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Compute the prior distribution at input points x.

        Parameters
        ----------
        x : torch.Tensor of shape (m, d)
            The input points to evaluate the prior mean and covariance at.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            The prior multivariate normal distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the ExactGPModel."""
        n = self.train_inputs[0].shape[0] if self.train_inputs else 0
        return f"ExactGPModel(n_points={n})"


class GPyTorchSurrogate:
    """A Gaussian Process surrogate model backend wrapper using GPyTorch.

    This class packages model definition, parameter optimization, and posterior prediction,
    hiding GPyTorch-specific API details from external callers.
    """

    def __init__(self) -> None:
        """Initialize the GPyTorchSurrogate."""
        self.model: ExactGPModel | None = None
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.train_x: torch.Tensor | None = None
        self.train_y: torch.Tensor | None = None

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the GPyTorchSurrogate."""
        n = self.train_x.shape[0] if self.train_x is not None else 0
        fitted = self.model is not None
        return f"GPyTorchSurrogate(n_points={n}, fitted={fitted})"

    def _setup_model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
    ) -> None:
        """Set up the GP model and likelihood with training data.

        Parameters
        ----------
        train_x : torch.Tensor of shape (n,) or (n, d)
            The input points where the Objective was evaluated, one row per
            point. A flat tensor is read as ``n`` points in one dimension.
        train_y : torch.Tensor of shape (n,)
            The corresponding Objective outputs, one per input point.

        Raises
        ------
        ValueError
            If train_y is not one-dimensional or does not hold one output
            per input point.
        """
        points = as_points(train_x)
        outputs = torch.as_tensor(train_y, dtype=torch.float64)
        if outputs.ndim != 1 or outputs.shape[0] != points.shape[0]:
            raise ValueError(
                f"Shape mismatch: train_x holds {points.shape[0]} input points "
                f"but train_y has shape {tuple(outputs.shape)}; expected "
                f"({points.shape[0]},)."
            )

        self.train_x = points
        self.train_y = outputs

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood).double()

    def fit_and_train(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        training_iter: int = 50,
        lr: float = 0.1,
        noise: float = 1e-4,
    ) -> None:
        """Fit the GP model and optimise hyperparameters automatically.

        Optimises the kernel lengthscale, outputscale, and noise variance
        using PyTorch's Adam optimiser.

        Parameters
        ----------
        train_x : torch.Tensor of shape (n,) or (n, d)
            The input points where the Objective was evaluated, one row per
            point.
        train_y : torch.Tensor of shape (n,)
            The corresponding Objective outputs.
        training_iter : int, optional
            Number of iterations for hyperparameter optimisation, by default 50.
        lr : float, optional
            Learning rate for the optimiser, by default 0.1.
        noise : float, optional
            Initial observation noise variance for the likelihood, by default 1e-4.
        """
        self._setup_model(train_x, train_y)

        self.likelihood.noise = noise
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(training_iter):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            with gpytorch.settings.cholesky_jitter(CHOLESKY_JITTER):
                loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

    def fit_no_training(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        lengthscale: float | Sequence[float] = 1.0,
        outputscale: float = 1.0,
        noise: float = 1e-4,
    ) -> None:
        """Fit the GP model with user-specified hyperparameters (no training).

        Sets the kernel lengthscale, outputscale, and noise to the given values
        and freezes all parameters so no optimisation takes place.

        Parameters
        ----------
        train_x : torch.Tensor of shape (n,) or (n, d)
            The input points where the Objective was evaluated, one row per
            point.
        train_y : torch.Tensor of shape (n,)
            The corresponding Objective outputs.
        lengthscale : float or sequence of float, optional
            The RBF kernel lengthscale, by default 1.0. A single value is used
            for every input dimension; a sequence gives one value per input
            dimension, in order.
        outputscale : float, optional
            The kernel outputscale (signal variance), by default 1.0.
        noise : float, optional
            The observation noise variance, by default 1e-4.

        Raises
        ------
        ValueError
            If lengthscale is a sequence whose length is not the number of
            input dimensions.
        """
        self._setup_model(train_x, train_y)

        n_dims = self.train_x.shape[1]
        lengthscales = torch.as_tensor(lengthscale, dtype=torch.float64)
        if lengthscales.ndim > 0 and lengthscales.numel() != n_dims:
            raise ValueError(
                f"lengthscale needs one value per input dimension ({n_dims}), "
                f"got {lengthscales.numel()}."
            )

        # Set hyperparameters to user-specified values
        self.model.covar_module.base_kernel.lengthscale = lengthscales
        self.model.covar_module.outputscale = outputscale
        self.likelihood.noise = noise

        # Freeze all parameters, no training
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.likelihood.parameters():
            param.requires_grad = False

    def hyperparameters(self) -> dict[str, tuple[float, ...]]:
        """Return the GP's current kernel and likelihood hyperparameters.

        After ``fit_no_training`` these are the values that were passed in.
        After ``fit_and_train`` they are the values Adam arrived at, which
        are otherwise not recoverable: ``config.json`` is written before the
        loop starts and can only record the *starting* point.

        Returns
        -------
        dict[str, tuple of float]
            ``lengthscale`` with one value per input dimension, and
            ``outputscale`` and ``noise`` with one value each. Every entry is
            a tuple so that callers can record and render all three alike.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.model is None or self.likelihood is None:
            raise RuntimeError(
                "The model must be fitted before reading hyperparameters."
            )

        kernel = self.model.covar_module
        return {
            "lengthscale": tuple(kernel.base_kernel.lengthscale.flatten().tolist()),
            "outputscale": (float(kernel.outputscale.item()),),
            "noise": (float(self.likelihood.noise.item()),),
        }

    def predict(
        self,
        test_x: torch.Tensor,
        n_samples: int = 0,
    ) -> dict[str, torch.Tensor | gpytorch.distributions.MultivariateNormal]:
        """Predict the posterior distributions at test points.

        Parameters
        ----------
        test_x : torch.Tensor of shape (m, d), or (m,) when d is 1
            The input points to predict at, with as many coordinates per
            point as the training data.
        n_samples : int, optional
            Number of samples to draw from the latent function's predictive
            posterior, by default 0. Sampling is skipped entirely when 0,
            which keeps predictions cheap inside the optimisation loop.

        Returns
        -------
        dict[str, torch.Tensor | gpytorch.distributions.MultivariateNormal]
            A dictionary containing prediction components:

            - "f_preds": predictive distribution (MultivariateNormal) of the
              latent function f(test_x). Its full covariance matrix is
              available on demand as ``f_preds.covariance_matrix``, shape
              (m, m). It is not computed eagerly, since it grows with m²
              and the optimisation loop needs only the diagonal.
            - "observed_pred": predictive distribution (MultivariateNormal) of
              observed targets y(test_x) = f(test_x) + noise.
            - "f_mean": predicted posterior mean of the latent function,
              torch.Tensor of shape (m,).
            - "f_var": predicted posterior variance of the latent function,
              torch.Tensor of shape (m,).
            - "f_samples": samples drawn from the latent function's predictive
              posterior, torch.Tensor of shape (n_samples, m). Only present
              when n_samples > 0.

        Raises
        ------
        RuntimeError
            If fit_and_train() or fit_no_training() has not been called prior to predicting.
        ValueError
            If test_x does not have one coordinate per input dimension.
        """
        if self.model is None or self.likelihood is None:
            raise RuntimeError("The model must be fitted before predicting.")

        test_x_double = as_points(test_x, n_dims=self.train_x.shape[1])
        self.model.eval()
        self.likelihood.eval()

        with (
            torch.no_grad(),
            gpytorch.settings.fast_pred_var(),
            gpytorch.settings.cholesky_jitter(CHOLESKY_JITTER),
        ):
            f_preds = self.model(test_x_double)
            observed_pred = self.likelihood(f_preds)

            f_mean = f_preds.mean
            f_var = f_preds.variance
            f_samples = (
                f_preds.sample(sample_shape=torch.Size([n_samples]))
                if n_samples > 0
                else None
            )

        # Scientific Invariants / Asserts
        assert torch.all(torch.isfinite(f_mean)), "f_mean contains non-finite values"
        assert torch.all(f_var >= 0), "f_var contains negative variance values"
        assert torch.all(torch.isfinite(f_var)), "f_var contains non-finite values"

        preds: dict[str, torch.Tensor | gpytorch.distributions.MultivariateNormal] = {
            "f_preds": f_preds,
            "observed_pred": observed_pred,
            "f_mean": f_mean,
            "f_var": f_var,
        }

        if f_samples is not None:
            assert torch.all(
                torch.isfinite(f_samples)
            ), "f_samples contains non-finite values"
            preds["f_samples"] = f_samples

        return preds

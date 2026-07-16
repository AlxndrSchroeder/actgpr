"""Surrogate model module for active GPR optimisation."""

import torch
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    """An exact Gaussian Process model with Constant mean and scaled RBF kernel.

    This class defines the structural prior components (mean and covariance) of
    the GP model.
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
        train_x : torch.Tensor of shape (n,)
            The training input points.
        train_y : torch.Tensor of shape (n,)
            The training target evaluations.
        likelihood : gpytorch.likelihoods.GaussianLikelihood
            The GPyTorch likelihood mapping latent outputs to observed targets.
        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Compute the prior distribution at input points x.

        Parameters
        ----------
        x : torch.Tensor of shape (m,)
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
        n = self.train_inputs[0].numel() if self.train_inputs else 0
        return f"ExactGPModel(n_points={n})"


class GPyTorchSurrogate:
    """A Gaussian Process surrogate model backend wrapper using GPyTorch.

    This class packages model definition, parameter optimization, posterior prediction,
    and plotting routines, hiding GPyTorch-specific API details from external callers.
    """

    def __init__(self) -> None:
        """Initialize the GPyTorchSurrogate."""
        self.model: ExactGPModel | None = None
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.train_x: torch.Tensor | None = None
        self.train_y: torch.Tensor | None = None

    def __repr__(self) -> str:
        """Return a concise human-readable summary of the GPyTorchSurrogate."""
        n = self.train_x.numel() if self.train_x is not None else 0
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
        train_x : torch.Tensor of shape (n,)
            The input points where the objective was evaluated.
        train_y : torch.Tensor of shape (n,)
            The corresponding evaluations of the objective.

        Raises
        ------
        ValueError
            If train_x and train_y shapes are not compatible.
        """
        if train_x.shape != train_y.shape:
            raise ValueError(
                f"Shape mismatch: train_x shape {train_x.shape} must match train_y shape {train_y.shape}"
            )

        self.train_x = train_x.double()
        self.train_y = train_y.double()

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
        train_x : torch.Tensor of shape (n,)
            The input points where the objective was evaluated.
        train_y : torch.Tensor of shape (n,)
            The corresponding evaluations of the objective.
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
            with gpytorch.settings.cholesky_jitter(1e-4):
                loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

    def fit_no_training(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        lengthscale: float = 1.0,
        outputscale: float = 1.0,
        noise: float = 1e-4,
    ) -> None:
        """Fit the GP model with user-specified hyperparameters (no training).

        Sets the kernel lengthscale, outputscale, and noise to the given values
        and freezes all parameters so no optimisation takes place.

        Parameters
        ----------
        train_x : torch.Tensor of shape (n,)
            The input points where the objective was evaluated.
        train_y : torch.Tensor of shape (n,)
            The corresponding evaluations of the objective.
        lengthscale : float, optional
            The RBF kernel lengthscale, by default 1.0.
        outputscale : float, optional
            The kernel outputscale (signal variance), by default 1.0.
        noise : float, optional
            The observation noise variance, by default 1e-4.
        """
        self._setup_model(train_x, train_y)

        # Set hyperparameters to user-specified values
        self.model.covar_module.base_kernel.lengthscale = lengthscale
        self.model.covar_module.outputscale = outputscale
        self.likelihood.noise = noise

        # Freeze all parameters — no training
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.likelihood.parameters():
            param.requires_grad = False

    def predict(
        self,
        test_x: torch.Tensor,
    ) -> dict[str, torch.Tensor | gpytorch.distributions.MultivariateNormal]:
        """Predict the posterior distributions at test points.

        Parameters
        ----------
        test_x : torch.Tensor of shape (m,)
            The test input points to predict at.

        Returns
        -------
        dict[str, torch.Tensor | gpytorch.distributions.MultivariateNormal]
            A dictionary containing prediction components:
            - "f_preds": gpytorch.distributions.MultivariateNormal
                Predictive distribution of the latent function f(test_x).
            - "observed_pred": gpytorch.distributions.MultivariateNormal
                Predictive distribution of observed targets y(test_x) = f(test_x) + noise.
            - "f_mean": torch.Tensor of shape (m,)
                Predicted posterior mean of the latent function.
            - "f_var": torch.Tensor of shape (m,)
                Predicted posterior variance of the latent function.
            - "f_covar": torch.Tensor of shape (m, m)
                Predicted posterior covariance matrix.
            - "f_samples": torch.Tensor of shape (1000, m)
                1000 samples drawn from the latent function's predictive posterior.

        Raises
        ------
        RuntimeError
            If fit_and_train() or fit_no_training() has not been called prior to predicting.
        """
        if self.model is None or self.likelihood is None:
            raise RuntimeError("The model must be fitted before predicting.")

        test_x_double = test_x.double()
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
            1e-4
        ):
            f_preds = self.model(test_x_double)
            observed_pred = self.likelihood(f_preds)

            f_mean = f_preds.mean
            f_var = f_preds.variance
            f_covar = f_preds.covariance_matrix
            f_samples = f_preds.sample(sample_shape=torch.Size([1000]))

        # Scientific Invariants / Asserts
        assert torch.all(torch.isfinite(f_mean)), "f_mean contains non-finite values"
        assert torch.all(f_var >= 0), "f_var contains negative variance values"
        assert torch.all(torch.isfinite(f_var)), "f_var contains non-finite values"
        assert torch.all(torch.isfinite(f_covar)), "f_covar contains non-finite values"
        assert torch.all(
            torch.isfinite(f_samples)
        ), "f_samples contains non-finite values"

        return {
            "f_preds": f_preds,
            "observed_pred": observed_pred,
            "f_mean": f_mean,
            "f_var": f_var,
            "f_covar": f_covar,
            "f_samples": f_samples,
        }

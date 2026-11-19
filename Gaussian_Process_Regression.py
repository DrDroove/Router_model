
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    ConstantKernel,
    WhiteKernel
)


class GaussianProcessBoundaryRegression:

    def __init__(
        self,
        kernel="matern",
        length_scale=1.0,
        alpha=1e-6,
        nu=2.5,
        n_restarts_optimizer=10
    ):
        self.kernel_type = kernel
        self.length_scale = length_scale
        self.alpha = alpha
        self.nu = nu
        self.n_restarts_optimizer = n_restarts_optimizer
        self.is_fitted_ = False

    def fit(self, boundary_points, dependent_axis=1):
        points = np.asarray(boundary_points, dtype=float)

        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("Expected an (N, parameters) array.")

        self.n_parameters_ = points.shape[1]
        self.dependent_axis_ = dependent_axis % self.n_parameters_

        self.independent_axes_ = [
            i for i in range(self.n_parameters_)
            if i != self.dependent_axis_
        ]

        X = points[:, self.independent_axes_]
        y = points[:, self.dependent_axis_]

        self.training_X_ = X.copy()
        self.training_y_ = y.copy()

        # Normalize inputs.
        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ == 0] = 1.0

        X_scaled = (X - self.x_mean_) / self.x_scale_

        self.y_mean_ = y.mean()
        self.y_scale_ = y.std()
        if self.y_scale_ == 0:
            self.y_scale_ = 1.0

        y_scaled = (y - self.y_mean_) / self.y_scale_

        if self.kernel_type == "rbf":
            base_kernel = RBF(length_scale=self.length_scale)

        elif self.kernel_type == "matern":
            base_kernel = Matern(
                length_scale=self.length_scale,
                nu=self.nu
            )

        else:
            raise ValueError("kernel must be 'rbf' or 'matern'.")

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * base_kernel
            + WhiteKernel(noise_level=1e-5)
        )

        self.model_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts_optimizer
        )

        self.model_.fit(X_scaled, y_scaled)

        self.n_samples_ = len(y)
        self.is_fitted_ = True

        return self

    def predict(self, independent_points, return_std=False):
        if not self.is_fitted_:
            raise RuntimeError("Fit the model first.")

        X = np.asarray(independent_points, dtype=float)

        if X.ndim == 1:
            if len(self.independent_axes_) == 1:
                X = X.reshape(-1, 1)
            else:
                X = X.reshape(1, -1)

        X = X.reshape(-1, len(self.independent_axes_))

        X_scaled = (X - self.x_mean_) / self.x_scale_

        if return_std:
            mean, std = self.model_.predict(
                X_scaled,
                return_std=True
            )

            return (
                mean * self.y_scale_ + self.y_mean_,
                std * self.y_scale_
            )

        prediction = self.model_.predict(X_scaled)

        return prediction * self.y_scale_ + self.y_mean_

    def get_info(self):
        if not self.is_fitted_:
            raise RuntimeError("Fit the model first.")

        # Predict the training boundary points
        X = self.training_X_
        predictions = self.predict(X)
        errors = predictions - self.training_y_

        return {
            "n_samples": self.n_samples_,
            "n_features": len(self.independent_axes_),
            "dependent_axis": self.dependent_axis_,
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mae": float(np.mean(np.abs(errors))),
            "max_error": float(np.max(np.abs(errors)))
        }
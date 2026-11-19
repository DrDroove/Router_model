
import numpy as np
from scipy.interpolate import RBFInterpolator


class RBFBoundaryRegression:

    def __init__(
        self,
        kernel="thin_plate_spline",
        smoothing=0.0,
        degree=1,
        neighbors=None
    ):
        self.kernel = kernel
        self.smoothing = smoothing
        self.degree = degree
        self.neighbors = neighbors
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

        # Normalize inputs for numerical stability.
        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ == 0] = 1.0

        X_scaled = (X - self.x_mean_) / self.x_scale_

        self.y_mean_ = y.mean()
        self.y_scale_ = y.std()
        if self.y_scale_ == 0:
            self.y_scale_ = 1.0

        y_scaled = (y - self.y_mean_) / self.y_scale_

        self.model_ = RBFInterpolator(
            X_scaled,
            y_scaled,
            kernel=self.kernel,
            smoothing=self.smoothing,
            degree=self.degree,
            neighbors=self.neighbors
        )

        self.n_samples_ = len(y)
        self.is_fitted_ = True

        return self

    def predict(self, independent_points):
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

        y_scaled = self.model_(X_scaled)

        return y_scaled.ravel() * self.y_scale_ + self.y_mean_

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

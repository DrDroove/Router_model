
import numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import lstsq


def _polynomial_exponents(n_features, degree):
    """
    Generate exponent vectors for every monomial of total degree
    from 0 up to `degree`.

    Example: n_features=2, degree=2:
        [0, 0] -> 1
        [1, 0] -> x
        [0, 1] -> y
        [2, 0] -> x^2
        [1, 1] -> xy
        [0, 2] -> y^2
    """
    exponents = []

    for total_degree in range(degree + 1):
        for combination in combinations_with_replacement(
            range(n_features), total_degree
        ):
            powers = [0] * n_features

            for feature_index in combination:
                powers[feature_index] += 1

            exponents.append(powers)

    return np.asarray(exponents, dtype=int)


def _polynomial_features(X, exponents):
    """Evaluate all polynomial monomials for each row of X."""
    return np.prod(
        X[:, None, :] ** exponents[None, :, :],
        axis=2
    )


def _as_2d_array(X, name="X"):
    """Convert input to a finite 2D float array."""
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")

    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return X


class PolynomialBoundaryRegression:
    """
    Fits one coordinate of the boundary as a polynomial function
    of all the other coordinates.

    Example in 2D:
        y_boundary = f(x)

    Example in 3D:
        z_boundary = f(x, y)

    Training data must contain boundary points, not Boolean verdicts.
    """

    def __init__(self, degree=2):
        if degree < 0:
            raise ValueError("degree must be non-negative.")

        self.degree = degree
        self.is_fitted_ = False

    def fit(self, boundary_points, dependent_axis=-1):
        """
        Parameters
        ----------
        boundary_points : array-like, shape (n_samples, n_parameters)
            Coordinates of points on the estimated boundary.

        dependent_axis : int, default=-1
            Which coordinate to predict. All other coordinates are
            used as independent variables.

            For 2D points (x, y), dependent_axis=1 fits y = f(x).
            For 3D points (x, y, z), dependent_axis=2 fits z = f(x, y).
        """
        points = _as_2d_array(boundary_points, "boundary_points")

        if points.shape[0] == 0:
            raise ValueError("At least one boundary point is required.")

        self.n_parameters_ = points.shape[1]
        self.dependent_axis_ = dependent_axis % self.n_parameters_
        self.independent_axes_ = [
            i for i in range(self.n_parameters_)
            if i != self.dependent_axis_
        ]

        X = points[:, self.independent_axes_]
        y = points[:, self.dependent_axis_]

        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ == 0] = 1.0

        self.y_mean_ = y.mean()
        self.y_scale_ = y.std()
        if self.y_scale_ == 0:
            self.y_scale_ = 1.0

        X_scaled = (X - self.x_mean_) / self.x_scale_
        y_scaled = (y - self.y_mean_) / self.y_scale_

        self.exponents_ = _polynomial_exponents(
            X.shape[1], self.degree
        )
        A = _polynomial_features(X_scaled, self.exponents_)

        self.coefficients_, _, self.rank_, self.singular_values_ = (
            lstsq(A, y_scaled)
        )

        fitted_scaled = A @ self.coefficients_
        residuals = y_scaled - fitted_scaled

        self.n_samples_ = len(y)
        self.n_features_ = A.shape[1]

        errors = residuals * self.y_scale_

        self.rmse_ = float(np.sqrt(np.mean(errors ** 2)))
        self.mae_ = float(np.mean(np.abs(errors)))
        self.max_error_ = float(np.max(np.abs(errors)))

        self.is_rank_deficient_ = self.rank_ < self.n_features_
        self.is_fitted_ = True

        return self

    def predict(self, independent_points):
        self._check_fitted()

        X = np.asarray(independent_points, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = (X - self.x_mean_) / self.x_scale_
        A = _polynomial_features(X_scaled, self.exponents_)

        y_scaled = A @ self.coefficients_
        return y_scaled * self.y_scale_ + self.y_mean_

    def predict_boundary(self, independent_points):
        """
        Return complete boundary coordinates, including the
        predicted dependent coordinate.
        """
        predicted = self.predict(independent_points)
        X = _as_2d_array(independent_points, "independent_points")

        result = np.empty((len(X), self.n_parameters_))
        result[:, self.independent_axes_] = X
        result[:, self.dependent_axis_] = predicted

        return result

    def get_info(self):
        self._check_fitted()

        return {
            "degree": self.degree,
            "n_samples": self.n_samples_,
            "n_features": self.n_features_,
            "rank": self.rank_,
            "rank_deficient": self.is_rank_deficient_,
            "rmse": self.rmse_,
            "mae": self.mae_,
            "max_error": self.max_error_,
            "dependent_axis": self.dependent_axis_,
        }

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before using this model.")


class PolynomialBoundaryClassifier:
    """
    Fits a polynomial least-squares classifier to stationary and
    non-stationary samples.

    The polynomial score is positive for stationary predictions
    and negative for non-stationary predictions.
    The estimated boundary is the zero level set of that score.
    """

    def __init__(self, degree=2, class_weight=None):
        if degree < 0:
            raise ValueError("degree must be non-negative.")

        if class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'.")

        self.degree = degree
        self.class_weight = class_weight
        self.is_fitted_ = False

    def fit(self, lambdas, verdicts):
        """
        Parameters
        ----------
        lambdas : array-like, shape (n_samples, n_parameters)
            Parameter coordinates of simulation samples.

        verdicts : array-like, shape (n_samples,)
            Boolean values: True = stationary, False = non-stationary.
        """
        X = _as_2d_array(lambdas, "lambdas")
        y = np.asarray(verdicts, dtype=bool).reshape(-1)

        if len(X) != len(y):
            raise ValueError("lambdas and verdicts must have equal length.")

        if len(y) == 0:
            raise ValueError("At least one sample is required.")

        if len(np.unique(y)) != 2:
            raise ValueError(
                "Both stationary and non-stationary samples are required."
            )

        self.n_parameters_ = X.shape[1]

        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ == 0] = 1.0

        X_scaled = (X - self.x_mean_) / self.x_scale_

        self.exponents_ = _polynomial_exponents(
            X.shape[1], self.degree
        )
        A = _polynomial_features(X_scaled, self.exponents_)

        # Map True/False to +1/-1.
        targets = np.where(y, 1.0, -1.0)

        # Optional class balancing, useful if one class is much
        # more common than the other.
        if self.class_weight == "balanced":
            counts = np.bincount(y.astype(int), minlength=2)
            weights = np.where(
                y,
                len(y) / (2 * counts[1]),
                len(y) / (2 * counts[0])
            )
            sqrt_weights = np.sqrt(weights)
            A_fit = A * sqrt_weights[:, None]
            targets_fit = targets * sqrt_weights
        else:
            A_fit = A
            targets_fit = targets

        self.coefficients_, _, self.rank_, self.singular_values_ = (
            lstsq(A_fit, targets_fit)
        )

        scores = A @ self.coefficients_
        predictions = scores >= 0

        self.n_samples_ = len(y)
        self.n_features_ = A.shape[1]
        self.accuracy_ = float(np.mean(predictions == y))
        self.is_rank_deficient_ = self.rank_ < self.n_features_
        self.is_fitted_ = True

        return self

    def decision_function(self, lambdas):
        """Return the signed polynomial score for each point."""
        self._check_fitted()

        X = _as_2d_array(lambdas, "lambdas")

        if X.shape[1] != self.n_parameters_:
            raise ValueError(
                f"Expected {self.n_parameters_} parameters, "
                f"got {X.shape[1]}."
            )

        X_scaled = (X - self.x_mean_) / self.x_scale_
        A = _polynomial_features(X_scaled, self.exponents_)

        return A @ self.coefficients_

    def predict(self, lambdas):
        """Return True for stationary, False for non-stationary."""
        return self.decision_function(lambdas) >= 0

    def get_info(self):
        self._check_fitted()

        return {
            "degree": self.degree,
            "n_samples": self.n_samples_,
            "n_features": self.n_features_,
            "rank": self.rank_,
            "rank_deficient": self.is_rank_deficient_,
            "training_accuracy": self.accuracy_,
            "class_weight": self.class_weight,
        }

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before using this model.")

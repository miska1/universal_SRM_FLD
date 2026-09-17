from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import LinAlgError, solve


@dataclass
class FisherLearner:
    feature_indices: np.ndarray
    weights: np.ndarray
    intercept: float
    score_scale: float


class FisherLDAEnsemble:
    """Random-subspace ensemble of regularized Fisher linear discriminants.

    The random subspaces avoid estimating a 34,671 x 34,671 covariance matrix.
    Every base learner is a Fisher discriminant; its normalized signed distance
    is averaged to form the final stego score.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        subspace_size: int = 200,
        shrinkage: float = 0.1,
        ridge: float = 1e-6,
        bootstrap: bool = True,
        random_state: int = 12345,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if subspace_size < 1:
            raise ValueError("subspace_size must be positive")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        self.n_estimators = int(n_estimators)
        self.subspace_size = int(subspace_size)
        self.shrinkage = float(shrinkage)
        self.ridge = float(ridge)
        self.bootstrap = bool(bootstrap)
        self.random_state = int(random_state)
        self.threshold_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FisherLDAEnsemble":
        # Keep the full SRM matrix in float32; promote only each small random
        # subspace to float64 while estimating its Fisher direction.
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2-D and y must be a matching 1-D array")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Training data must contain cover=0 and stego=1")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")

        self.n_features_in_ = X.shape[1]
        self.mean_ = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = X.std(axis=0, ddof=1, dtype=np.float64)
        scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
        self.scale_ = scale.astype(np.float32)
        Z = (X - self.mean_) / self.scale_

        class_indices = (np.flatnonzero(y == 0), np.flatnonzero(y == 1))
        subspace_size = min(self.subspace_size, self.n_features_in_)
        rng = np.random.default_rng(self.random_state)
        self.learners_: list[FisherLearner] = []
        for _ in range(self.n_estimators):
            features = np.sort(
                rng.choice(self.n_features_in_, size=subspace_size, replace=False)
            ).astype(np.int32)
            selected: list[np.ndarray] = []
            for indices in class_indices:
                if self.bootstrap:
                    selected.append(rng.choice(indices, size=len(indices), replace=True))
                else:
                    selected.append(indices)
            X0 = Z[selected[0]][:, features].astype(np.float64, copy=False)
            X1 = Z[selected[1]][:, features].astype(np.float64, copy=False)
            mean0 = X0.mean(axis=0)
            mean1 = X1.mean(axis=0)
            centered0 = X0 - mean0
            centered1 = X1 - mean1
            denominator = max(len(X0) + len(X1) - 2, 1)
            covariance = (centered0.T @ centered0 + centered1.T @ centered1) / denominator
            diagonal = np.diag(covariance).copy()
            covariance *= 1.0 - self.shrinkage
            covariance.flat[:: subspace_size + 1] += self.shrinkage * diagonal
            ridge_scale = max(float(diagonal.mean()), 1.0)
            covariance.flat[:: subspace_size + 1] += self.ridge * ridge_scale
            difference = mean1 - mean0
            try:
                weights = solve(covariance, difference, assume_a="pos", check_finite=False)
            except LinAlgError:
                weights = np.linalg.pinv(covariance, hermitian=True) @ difference
            norm = np.linalg.norm(weights)
            if not np.isfinite(norm) or norm < 1e-12:
                continue
            weights /= norm
            intercept = -0.5 * float(weights @ (mean0 + mean1))
            score_variance = float(weights @ covariance @ weights)
            score_scale = max(np.sqrt(max(score_variance, 0.0)), 1e-8)
            self.learners_.append(
                FisherLearner(
                    feature_indices=features,
                    weights=weights.astype(np.float32),
                    intercept=intercept,
                    score_scale=score_scale,
                )
            )
        if not self.learners_:
            raise RuntimeError("All Fisher learners were degenerate")
        return self

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "learners_"):
            raise RuntimeError("Model is not fitted")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected X with {self.n_features_in_} columns")
        return (X - self.mean_) / self.scale_

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Z = self._check_X(X)
        scores = np.zeros(len(Z), dtype=np.float64)
        for learner in self.learners_:
            scores += (
                Z[:, learner.feature_indices] @ learner.weights + learner.intercept
            ) / learner.score_scale
        return scores / len(self.learners_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= self.threshold_).astype(np.int8)

    def set_threshold(self, threshold: float) -> "FisherLDAEnsemble":
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        self.threshold_ = float(threshold)
        return self

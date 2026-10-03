from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Sequence

import numpy as np
from scipy.linalg import LinAlgError, solve

from .logging_utils import LOGGER_NAME, elapsed_text


LOGGER = logging.getLogger(LOGGER_NAME)
AUTOMATIC = "automatic"


@dataclass
class FisherLearner:
    """One MATLAB-style FLD base learner."""

    feature_indices: np.ndarray
    weights: np.ndarray
    bias: float


def _validate_automatic_or_positive(value: int | str, name: str) -> int | str:
    if isinstance(value, str):
        if value.lower() != AUTOMATIC:
            raise ValueError(f"{name} must be a positive integer or 'automatic'")
        return AUTOMATIC
    if int(value) < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _pair_key(file_name: str, expected_class: str) -> str:
    normalized = str(file_name).replace("\\", "/")
    prefix = f"{expected_class}/"
    if not normalized.startswith(prefix) or len(normalized) == len(prefix):
        raise ValueError(
            f"Expected a {expected_class!r} sample path beginning with {prefix!r}, "
            f"got {file_name!r}"
        )
    return normalized[len(prefix) :]


def synchronized_cover_stego(
    X: np.ndarray,
    y: np.ndarray,
    files: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return synchronized cover/stego matrices required by the MATLAB ensemble.

    When file names are available, samples are paired by their path relative to
    the ``cover`` and ``stego`` directories. Without names, the function assumes
    that the rows inside each class are already in corresponding order.
    """

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2-D and y must be a matching 1-D array")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training data must contain cover=0 and stego=1")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values")

    cover_indices = np.flatnonzero(y == 0)
    stego_indices = np.flatnonzero(y == 1)
    if len(cover_indices) != len(stego_indices):
        raise ValueError(
            "The original ensemble requires the same number of synchronized "
            "cover and stego samples"
        )

    if files is None:
        pair_names = tuple(f"pair_{index:08d}" for index in range(len(cover_indices)))
        return X[cover_indices], X[stego_indices], pair_names

    file_array = np.asarray(files).astype(str)
    if file_array.ndim != 1 or len(file_array) != len(X):
        raise ValueError("files must contain one path for every feature row")

    cover_by_name: dict[str, int] = {}
    stego_by_name: dict[str, int] = {}
    for index in cover_indices:
        key = _pair_key(file_array[index], "cover")
        if key in cover_by_name:
            raise ValueError(f"Duplicate cover pair name: {key}")
        cover_by_name[key] = int(index)
    for index in stego_indices:
        key = _pair_key(file_array[index], "stego")
        if key in stego_by_name:
            raise ValueError(f"Duplicate stego pair name: {key}")
        stego_by_name[key] = int(index)

    missing_stego = sorted(set(cover_by_name) - set(stego_by_name))
    missing_cover = sorted(set(stego_by_name) - set(cover_by_name))
    if missing_stego or missing_cover:
        raise ValueError(
            "Cover/stego files are not synchronized; "
            f"missing stego={missing_stego[:3]}, missing cover={missing_cover[:3]}"
        )

    pair_names = tuple(sorted(cover_by_name))
    return (
        X[[cover_by_name[name] for name in pair_names]],
        X[[stego_by_name[name] for name in pair_names]],
        pair_names,
    )


class FisherLDAEnsemble:
    """Python implementation of the DDE Lab MATLAB ensemble classifier 2.0.

    It follows ``ensemble_training.m`` and ``ensemble_testing.m``: synchronized
    cover/stego pairs are bootstrapped together, each FLD uses a random feature
    subspace, its threshold minimizes training false alarms plus missed
    detections, base learners cast binary votes, and out-of-bag (OOB) error can
    automatically select both subspace dimensionality and ensemble length.

    NumPy/SciPy and MATLAB do not generate bit-identical random permutations or
    linear-system solutions, so matching settings need not produce identical
    floating-point models. The training and decision procedure is equivalent.
    """

    def __init__(
        self,
        n_estimators: int | str = AUTOMATIC,
        subspace_size: int | str = AUTOMATIC,
        *,
        seed_subspaces: int = 12345,
        seed_bootstrap: int = 67890,
        tie_break_seed: int = 1,
        d_sub_step: int = 200,
        oob_tolerance: float = 0.02,
        l_kernel: Sequence[float] | None = None,
        l_min_length: int = 25,
        l_memory: int = 50,
        l_epsilon: float = 0.005,
        max_estimators: int = 500,
    ) -> None:
        self.n_estimators = _validate_automatic_or_positive(n_estimators, "n_estimators")
        self.subspace_size = _validate_automatic_or_positive(subspace_size, "subspace_size")
        if d_sub_step < 1:
            raise ValueError("d_sub_step must be positive")
        if oob_tolerance < 0.0:
            raise ValueError("oob_tolerance must be non-negative")
        if l_min_length < 1 or l_memory < 1 or max_estimators < 1:
            raise ValueError("L stopping parameters must be positive")
        if l_epsilon < 0.0:
            raise ValueError("l_epsilon must be non-negative")
        kernel = np.asarray(
            tuple(l_kernel) if l_kernel is not None else np.ones(5) / 5.0,
            dtype=np.float64,
        )
        if kernel.ndim != 1 or len(kernel) < 1 or not np.isfinite(kernel).all():
            raise ValueError("l_kernel must be a finite, non-empty 1-D sequence")
        if float(kernel.sum()) == 0.0:
            raise ValueError("l_kernel must have a non-zero sum")

        self.seed_subspaces = int(seed_subspaces)
        self.seed_bootstrap = int(seed_bootstrap)
        self.tie_break_seed = int(tie_break_seed)
        self.d_sub_step = int(d_sub_step)
        self.oob_tolerance = float(oob_tolerance)
        self.l_kernel = kernel / kernel.sum()
        self.l_min_length = int(l_min_length)
        self.l_memory = int(l_memory)
        self.l_epsilon = float(l_epsilon)
        self.max_estimators = int(max_estimators)
        self.threshold_ = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        files: Sequence[str] | None = None,
    ) -> "FisherLDAEnsemble":
        X_cover, X_stego, pair_names = synchronized_cover_stego(X, y, files)
        self.pair_names_ = pair_names
        return self.fit_paired(X_cover, X_stego)

    def fit_paired(
        self,
        X_cover: np.ndarray,
        X_stego: np.ndarray,
    ) -> "FisherLDAEnsemble":
        fit_started = time.perf_counter()
        X_cover = np.asarray(X_cover, dtype=np.float32)
        X_stego = np.asarray(X_stego, dtype=np.float32)
        if X_cover.ndim != 2 or X_stego.ndim != 2:
            raise ValueError("X_cover and X_stego must be 2-D")
        if X_cover.shape != X_stego.shape:
            raise ValueError(
                "Cover/stego feature matrices must have identical shapes and "
                "contain synchronized pairs"
            )
        if len(X_cover) < 2:
            raise ValueError("At least two synchronized cover/stego pairs are required")
        if X_cover.shape[1] < 1:
            raise ValueError("Feature matrices must contain at least one column")
        if not np.isfinite(X_cover).all() or not np.isfinite(X_stego).all():
            raise ValueError("Training features contain non-finite values")

        self.n_features_in_ = int(X_cover.shape[1])
        self.n_pairs_ = int(len(X_cover))
        if isinstance(self.subspace_size, int) and self.subspace_size > self.n_features_in_:
            raise ValueError("subspace_size cannot exceed the full feature dimension")

        LOGGER.info(
            "Fitting MATLAB-style FLD ensemble: pairs=%d, samples=%d, features=%d",
            self.n_pairs_,
            2 * self.n_pairs_,
            self.n_features_in_,
        )
        LOGGER.info(
            "Training configuration: L=%s, d_sub=%s, seed_subspaces=%d, "
            "seed_bootstrap=%d, paired_bagging=True",
            self.n_estimators,
            self.subspace_size,
            self.seed_subspaces,
            self.seed_bootstrap,
        )

        subspace_rng = np.random.RandomState(self.seed_subspaces)
        bootstrap_rng = np.random.RandomState(self.seed_bootstrap)
        search_points, search_step = self._initialize_subspace_search()
        search_errors: dict[int, float] = {}
        search_results: list[dict[str, float | int]] = []
        best_error = float("inf")
        best_learners: list[FisherLearner] | None = None
        best_progress: list[float] = []

        while True:
            unfinished = [point for point in search_points if point not in search_errors]
            if not unfinished:
                break
            d_sub = unfinished[0]
            candidate_started = time.perf_counter()
            learners, oob_progress = self._train_candidate(
                X_cover,
                X_stego,
                d_sub,
                subspace_rng,
                bootstrap_rng,
            )
            oob_error = float(oob_progress[-1])
            search_errors[d_sub] = oob_error
            elapsed = time.perf_counter() - candidate_started
            search_results.append(
                {
                    "subspace_size": int(d_sub),
                    "oob_error": oob_error,
                    "estimators": len(learners),
                    "elapsed_seconds": elapsed,
                }
            )
            LOGGER.info(
                "Completed d_sub=%d: OOB=%.6f, L=%d, elapsed=%s",
                d_sub,
                oob_error,
                len(learners),
                elapsed_text(elapsed),
            )
            if oob_error < best_error:
                best_error = oob_error
                best_learners = learners
                best_progress = oob_progress

            if self.subspace_size != AUTOMATIC:
                break
            search_points, search_step, finished = self._update_subspace_search(
                search_points,
                search_errors,
                search_step,
            )
            if finished:
                break

        if not best_learners:
            raise RuntimeError("No Fisher base learners were trained")

        self.learners_ = best_learners
        self.optimal_oob_error_ = best_error
        self.optimal_n_estimators_ = len(best_learners)
        best_row = min(search_results, key=lambda row: float(row["oob_error"]))
        self.optimal_subspace_size_ = int(best_row["subspace_size"])
        self.oob_progress_ = np.asarray(best_progress, dtype=np.float64)
        self.search_results_ = search_results
        self.training_time_ = time.perf_counter() - fit_started
        LOGGER.info(
            "MATLAB-style ensemble training finished: optimal d_sub=%d, "
            "OOB=%.6f, L=%d, elapsed=%s",
            self.optimal_subspace_size_,
            self.optimal_oob_error_,
            self.optimal_n_estimators_,
            elapsed_text(self.training_time_),
        )
        return self

    def _initialize_subspace_search(self) -> tuple[list[int], int]:
        if self.subspace_size != AUTOMATIC:
            return [int(self.subspace_size)], self.d_sub_step
        if self.n_features_in_ == 1:
            return [1], 1
        step = self.d_sub_step
        if step >= self.n_features_in_ / 4:
            step = max(1, self.n_features_in_ // 4)
        if self.n_features_in_ < 10:
            step = 1
        if self.n_features_in_ == 2:
            return [1, 2], step
        return [step, 2 * step, 3 * step], step

    @staticmethod
    def _add_gridpoints(points: list[int], additions: Sequence[int], maximum: int) -> list[int]:
        return sorted(set(points).union(int(value) for value in additions if 1 <= value <= maximum))

    def _update_subspace_search(
        self,
        points: list[int],
        errors: dict[int, float],
        step: int,
    ) -> tuple[list[int], int, bool]:
        """Algorithm 2 / ``update_search`` from ensemble_training.m."""

        if any(point not in errors for point in points):
            return points, step, False
        error_values = np.asarray([errors[point] for point in points], dtype=np.float64)
        min_index = int(np.argmin(error_values))
        min_error = float(error_values[min_index])
        if step == 1 or min_error == 0.0:
            return points, step, True

        if min_index == 0:
            step = max(1, step // 2)
            points = self._add_gridpoints(
                points,
                [points[0] - step, points[0] + step],
                self.n_features_in_,
            )
        elif min_index == len(points) - 1:
            next_point = points[-1] + step
            sufficiently_new = min(abs(next_point - point) for point in points) > step / 2
            if next_point <= self.n_features_in_ and sufficiently_new:
                points = self._add_gridpoints(points, [next_point], self.n_features_in_)
            else:
                previous_error = float(error_values[-2])
                stop = (
                    min_error / previous_error >= 1.0 - self.oob_tolerance
                    if previous_error > 0.0
                    else min_error == 0.0
                )
                stop = stop or previous_error - min_error < 5e-3
                stop = stop or step < points[min_index] * 0.05
                if stop:
                    return points, step, True
                step = max(1, step // 2)
                additions = (
                    [points[-1] - step, points[-1] + step]
                    if points[-1] + step <= self.n_features_in_
                    else [points[-1] - step]
                )
                points = self._add_gridpoints(points, additions, self.n_features_in_)
        elif (
            min_index == len(points) - 2
            and points[-1] + step <= self.n_features_in_
            and min(abs(points[-1] + step - point) for point in points) > step / 2
            and not (
                error_values[-1] > error_values[-2]
                and error_values[-1] > error_values[-3]
            )
        ):
            points = self._add_gridpoints(
                points,
                [points[-1] + step],
                self.n_features_in_,
            )
        else:
            around_error = float((error_values[min_index - 1] + error_values[min_index + 1]) / 2)
            stop = (
                min_error / around_error >= 1.0 - self.oob_tolerance
                if around_error > 0.0
                else min_error == 0.0
            )
            stop = stop or around_error - min_error < 5e-3
            stop = stop or step < points[min_index] * 0.05
            if stop:
                return points, step, True
            step = max(1, step // 2)
            points = self._add_gridpoints(
                points,
                [points[min_index] - step, points[min_index] + step],
                self.n_features_in_,
            )
        return points, step, False

    def _train_candidate(
        self,
        X_cover: np.ndarray,
        X_stego: np.ndarray,
        d_sub: int,
        subspace_rng: np.random.RandomState,
        bootstrap_rng: np.random.RandomState,
    ) -> tuple[list[FisherLearner], list[float]]:
        n_pairs = len(X_cover)
        cover_votes = np.zeros(n_pairs, dtype=np.int32)
        stego_votes = np.zeros(n_pairs, dtype=np.int32)
        cover_vote_counts = np.zeros(n_pairs, dtype=np.int32)
        stego_vote_counts = np.zeros(n_pairs, dtype=np.int32)
        tie_rng = np.random.RandomState(self.tie_break_seed)
        learners: list[FisherLearner] = []
        oob_progress: list[float] = []

        fixed_l = int(self.n_estimators) if self.n_estimators != AUTOMATIC else None
        limit = fixed_l if fixed_l is not None else self.max_estimators
        progress_interval = max(1, min(25, limit // 10))
        for learner_index in range(1, limit + 1):
            features = subspace_rng.permutation(self.n_features_in_)[:d_sub].astype(np.int32)
            bootstrap = np.floor(bootstrap_rng.random_sample(n_pairs) * n_pairs).astype(np.int64)
            in_bag = np.zeros(n_pairs, dtype=bool)
            in_bag[bootstrap] = True
            oob_indices = np.flatnonzero(~in_bag)

            learner = self._train_fld(
                X_cover[bootstrap][:, features],
                X_stego[bootstrap][:, features],
                features,
            )
            learners.append(learner)

            if len(oob_indices):
                cover_projection = (
                    X_cover[oob_indices][:, features] @ learner.weights - learner.bias
                )
                stego_projection = (
                    X_stego[oob_indices][:, features] @ learner.weights - learner.bias
                )
                cover_votes[oob_indices] += np.sign(cover_projection).astype(np.int32)
                stego_votes[oob_indices] += np.sign(stego_projection).astype(np.int32)
                cover_vote_counts[oob_indices] += 1
                stego_vote_counts[oob_indices] += 1

            resolved_cover = self._resolve_ties(cover_votes, tie_rng)
            resolved_stego = self._resolve_ties(stego_votes, tie_rng)
            oob_error = float(
                (np.count_nonzero(resolved_cover > 0) + np.count_nonzero(resolved_stego < 0))
                / (2 * n_pairs)
            )
            oob_progress.append(oob_error)

            if learner_index % progress_interval == 0 or learner_index == limit:
                average_oob_votes = float(
                    np.mean(np.concatenate([cover_vote_counts, stego_vote_counts]))
                )
                LOGGER.info(
                    "d_sub=%d, trained L=%d, OOB=%.6f, average OOB votes=%.2f",
                    d_sub,
                    learner_index,
                    oob_error,
                    average_oob_votes,
                )

            if fixed_l is not None:
                if learner_index >= fixed_l:
                    break
            elif self._automatic_l_finished(oob_progress, learner_index):
                break
        return learners, oob_progress

    def _automatic_l_finished(self, progress: Sequence[float], learner_index: int) -> bool:
        if learner_index >= self.max_estimators:
            return True
        if len(progress) < self.l_min_length:
            return False
        recent = np.asarray(progress[-self.l_memory :], dtype=np.float64)
        if len(recent) < len(self.l_kernel):
            return False
        moving_average = np.convolve(recent, self.l_kernel, mode="valid")
        variation = abs(float(moving_average.max() - moving_average.min()))
        return variation < self.l_epsilon

    @staticmethod
    def _train_fld(
        X_cover: np.ndarray,
        X_stego: np.ndarray,
        feature_indices: np.ndarray,
    ) -> FisherLearner:
        X_cover = np.asarray(X_cover, dtype=np.float32)
        X_stego = np.asarray(X_stego, dtype=np.float32)
        width = X_cover.shape[1]

        constant_cover = np.all(X_cover == X_cover[0], axis=0)
        constant_stego = np.all(X_stego == X_stego[0], axis=0)
        remove = constant_cover & constant_stego & (X_cover[0] == X_stego[0])
        active = ~remove
        weights = np.zeros(width, dtype=np.float64)

        if np.any(active):
            cover_active = X_cover[:, active]
            stego_active = X_stego[:, active]
            mean_cover = cover_active.mean(axis=0, dtype=np.float64)
            mean_stego = stego_active.mean(axis=0, dtype=np.float64)
            centered_cover = cover_active.astype(np.float64) - mean_cover
            centered_stego = stego_active.astype(np.float64) - mean_stego
            covariance = (centered_cover.T @ centered_cover) / len(cover_active)
            covariance += (centered_stego.T @ centered_stego) / len(stego_active)
            covariance.flat[:: covariance.shape[0] + 1] += 1e-10
            difference = mean_stego - mean_cover
            try:
                active_weights = solve(
                    covariance,
                    difference,
                    assume_a="sym",
                    check_finite=False,
                )
            except (LinAlgError, ValueError):
                regularization = np.finfo(np.float64).eps
                active_weights = None
                for _ in range(8):
                    try:
                        adjusted = covariance.copy()
                        adjusted.flat[:: adjusted.shape[0] + 1] += regularization
                        active_weights = solve(
                            adjusted,
                            difference,
                            assume_a="sym",
                            check_finite=False,
                        )
                        break
                    except (LinAlgError, ValueError):
                        regularization *= 5.0
                if active_weights is None:
                    active_weights = np.linalg.pinv(covariance, hermitian=True) @ difference
            weights[active] = active_weights

        weights, bias = FisherLDAEnsemble._find_training_threshold(
            X_cover.astype(np.float64),
            X_stego.astype(np.float64),
            weights,
        )
        return FisherLearner(
            feature_indices=feature_indices,
            weights=weights.astype(np.float32),
            bias=float(bias),
        )

    @staticmethod
    def _find_training_threshold(
        X_cover: np.ndarray,
        X_stego: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        projection_cover = X_cover @ weights
        projection_stego = X_stego @ weights
        projections = np.concatenate([projection_cover, projection_stego])
        labels = np.concatenate(
            [-np.ones(len(X_cover), dtype=np.int8), np.ones(len(X_stego), dtype=np.int8)]
        )
        order = np.argsort(projections, kind="stable")
        projections = projections[order]
        labels = labels[order]

        false_alarm = int(np.count_nonzero(labels == -1))
        missed_detection = 0
        missed_detection_reverse = false_alarm
        false_alarm_reverse = missed_detection
        minimum_error = false_alarm + missed_detection
        best_index: int | None = None
        orientation = 1

        for index in range(len(projections) - 1):
            if labels[index] == -1:
                false_alarm -= 1
                missed_detection_reverse += 1
            else:
                false_alarm_reverse -= 1
                missed_detection += 1
            if false_alarm + missed_detection < minimum_error:
                minimum_error = false_alarm + missed_detection
                best_index = index
                orientation = 1
            if false_alarm_reverse + missed_detection_reverse < minimum_error:
                minimum_error = false_alarm_reverse + missed_detection_reverse
                best_index = index
                orientation = -1

        if best_index is None:
            # Degenerate all-tie learner. MATLAB rarely reaches this case;
            # keeping a zero boundary preserves its neutral vote behavior.
            return weights, 0.0
        midpoint = 0.5 * (projections[best_index] + projections[best_index + 1])
        if orientation == -1:
            weights = -weights
        return weights, float(orientation * midpoint)

    @staticmethod
    def _resolve_ties(votes: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        resolved = votes.astype(np.float64, copy=True)
        ties = resolved == 0
        if np.any(ties):
            resolved[ties] = rng.random_sample(int(np.count_nonzero(ties))) - 0.5
        return resolved

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "learners_"):
            raise RuntimeError("Model is not fitted")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected X with {self.n_features_in_} columns")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")
        return X

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the MATLAB-style sum of binary base-learner votes."""

        X = self._check_X(X)
        votes = np.zeros(len(X), dtype=np.int32)
        for learner in self.learners_:
            projection = X[:, learner.feature_indices] @ learner.weights - learner.bias
            votes += np.sign(projection).astype(np.int32)
        # ensemble_testing.m resolves exact majority ties randomly. A fixed
        # MT19937 seed makes saved reports reproducible across repeated calls.
        return self._resolve_ties(votes, np.random.RandomState(self.tie_break_seed))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > 0.0).astype(np.int8)

"""Classical unsupervised baselines for DAS pass-level anomaly detection.

Both baselines are trained on normal passes only.  PCA learns a linear subspace
of the full spatio-temporal pass and flags samples with large projection
residuals.  Isolation Forest operates on compact per-channel summary features,
which also provides a simple channel localisation mechanism.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..dsp.features import extract_feature_matrix
from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


class PCABaseline(BaseModel):
    """PCA reconstruction baseline.

    Each pass is flattened to a ``channels * samples`` vector.  A PCA subspace
    learned from normal pass vectors is used to reconstruct unseen passes; the
    mean squared residual is the anomaly score.

    Parameters
    ----------
    n_components:
        Number of retained principal components.  Automatically clipped to the
        smaller of the feature count and the number of training samples.
    whiten:
        Forwarded to scikit-learn PCA.
    standardize:
        Standardise every flattened feature (recommended when channel gains
        differ).
    """

    name = "pca"

    def __init__(
        self,
        n_components: int = 32,
        whiten: bool = False,
        standardize: bool = True,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(
            n_components=n_components,
            whiten=whiten,
            standardize=standardize,
            random_state=random_state,
        )
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.standardize = bool(standardize)
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.input_shape_: tuple[int, int] | None = None

    def _flatten(self, batch: np.ndarray) -> np.ndarray:
        flat = batch.reshape(batch.shape[0], -1).astype(np.float64)
        if self.standardize:
            if self.scaler_ is None:
                raise RuntimeError("Scaler has not been fitted")
            flat = self.scaler_.transform(flat)
        return flat

    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "PCABaseline":
        train = self.validate_samples(train_data)
        self.input_shape_ = tuple(train.shape[1:])
        flat = train.reshape(train.shape[0], -1).astype(np.float64)

        if self.standardize:
            self.scaler_ = StandardScaler()
            flat = self.scaler_.fit_transform(flat)

        max_components = min(flat.shape[0], flat.shape[1])
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")
        if self.n_components > max_components:
            LOGGER.warning(
                "PCA n_components=%d clipped to %d for %d samples x %d features",
                self.n_components,
                max_components,
                flat.shape[0],
                flat.shape[1],
            )
        components = min(self.n_components, max_components)
        self.pca_ = PCA(
            n_components=components,
            whiten=self.whiten,
            random_state=self.random_state,
            svd_solver="full",
        )
        self.pca_.fit(flat)
        self._is_fitted = True
        self.training_history_ = {
            "explained_variance_ratio": self.pca_.explained_variance_ratio_.tolist()
        }
        LOGGER.info(
            "PCA fitted with %d components (%.2f%% variance explained)",
            components,
            100.0 * float(np.sum(self.pca_.explained_variance_ratio_)),
        )
        return self

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.pca_ is None:
            raise RuntimeError("PCA model is missing")
        batch = self.validate_samples(data)
        if self.input_shape_ is not None and batch.shape[1:] != self.input_shape_:
            raise ValueError(
                f"Expected input shape {self.input_shape_}, received {batch.shape[1:]}"
            )
        flat = self._flatten(batch)
        projected = self.pca_.transform(flat)
        reconstructed = self.pca_.inverse_transform(projected)
        if self.standardize and self.scaler_ is not None:
            reconstructed = self.scaler_.inverse_transform(reconstructed)
        return reconstructed.reshape(batch.shape).astype(np.float32)

    def score(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        batch = self.validate_samples(data)
        residual = np.square(batch - self.reconstruct(batch))
        return residual.mean(axis=(1, 2))


class IsolationForestBaseline(BaseModel):
    """Isolation Forest on per-channel summary features.

    A pass is represented by ``channels`` feature vectors.  The forest is fitted
    on the pooled vectors of all normal training passes.  For inference, each
    channel receives an anomaly score and the pass score is the mean of its
    ``topk_channels`` most anomalous channels.  This makes localisation a
    natural by-product of the baseline.

    Parameters
    ----------
    n_estimators, contamination, random_state:
        Forwarded to ``sklearn.ensemble.IsolationForest``.
    topk_channels:
        Number of channels aggregated into a pass score.
    """

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float | str = "auto",
        random_state: int | None = 42,
        topk_channels: int = 3,
        n_jobs: int = 1,
        wavelet: str = "db4",
        wavelet_level: int = 3,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            topk_channels=topk_channels,
            n_jobs=n_jobs,
            wavelet=wavelet,
            wavelet_level=wavelet_level,
        )
        self.n_estimators = int(n_estimators)
        self.contamination = contamination
        self.random_state = random_state
        self.topk_channels = int(topk_channels)
        self.n_jobs = int(n_jobs)
        self.wavelet = wavelet
        self.wavelet_level = int(wavelet_level)
        self.scaler_: StandardScaler | None = None
        self.forest_: IsolationForest | None = None
        self.input_shape_: tuple[int, int] | None = None

    def _extract_features(self, batch: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
        feature_blocks = [
            extract_feature_matrix(
                sample,
                sampling_rate_hz=sampling_rate_hz,
                wavelet=self.wavelet,
                wavelet_level=self.wavelet_level,
            )
            for sample in batch
        ]
        return np.stack(feature_blocks, axis=0)  # (N, C, F)

    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
        sampling_rate_hz: float = 1000.0,
        **kwargs: Any,
    ) -> "IsolationForestBaseline":
        train = self.validate_samples(train_data)
        self.input_shape_ = tuple(train.shape[1:])
        features = self._extract_features(train, sampling_rate_hz=float(sampling_rate_hz))
        flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
        self.scaler_ = StandardScaler()
        flat = self.scaler_.fit_transform(flat)
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.forest_.fit(flat)
        self._is_fitted = True
        self.metadata_["feature_dim"] = int(features.shape[-1])
        self.training_history_ = {
            "mean_normal_score": float(np.mean(self._raw_scores(self.forest_, flat)))
        }
        LOGGER.info(
            "Isolation Forest fitted on %d channel feature vectors (%d dims)",
            flat.shape[0],
            flat.shape[1],
        )
        return self

    @staticmethod
    def _raw_scores(forest: IsolationForest, normalized: np.ndarray) -> np.ndarray:
        """Convert sklearn decision scores to "larger is more anomalous"."""

        return -forest.decision_function(normalized)

    def channel_scores(
        self,
        data: np.ndarray,
        sampling_rate_hz: float = 1000.0,
    ) -> np.ndarray:
        self._check_fitted()
        if self.forest_ is None or self.scaler_ is None:
            raise RuntimeError("Isolation Forest has not been fitted")
        batch = self.validate_samples(data)
        if self.input_shape_ is not None and batch.shape[1:] != self.input_shape_:
            raise ValueError(
                f"Expected input shape {self.input_shape_}, received {batch.shape[1:]}"
            )
        features = self._extract_features(batch, sampling_rate_hz=float(sampling_rate_hz))
        n_samples, n_channels, feature_dim = features.shape
        normalized = self.scaler_.transform(
            features.reshape(-1, feature_dim).astype(np.float64)
        )
        scores = self._raw_scores(self.forest_, normalized)
        return scores.reshape(n_samples, n_channels).astype(np.float32)

    def score(
        self,
        data: np.ndarray,
        sampling_rate_hz: float = 1000.0,
    ) -> np.ndarray:
        channel_scores = self.channel_scores(
            data, sampling_rate_hz=sampling_rate_hz
        )
        topk = min(max(self.topk_channels, 1), channel_scores.shape[1])
        # Mean of the strongest channels is more stable than a single maximum.
        strongest = np.partition(channel_scores, -topk, axis=1)[:, -topk:]
        return strongest.mean(axis=1)

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Isolation Forest is not a reconstruction model; use channel_scores"
        )

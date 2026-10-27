"""Scoring utilities for reconstruction residuals and repeated train passes.

A scorer is deliberately separate from a model.  This allows the same model to
be calibrated on a clean reference run and then applied to new acquisitions,
while channel-wise normalisation and multi-pass fusion remain configurable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

AggregationMethod = Literal["topk", "max", "mean", "median", "p95"]
FusionMethod = Literal["mean", "median", "max"]


def reconstruction_residual(
    original: np.ndarray,
    reconstructed: np.ndarray,
    reduction: Literal["none", "channel", "sample"] = "none",
) -> np.ndarray:
    """Squared reconstruction error with optional reduction.

    Parameters
    ----------
    original, reconstructed:
        Arrays shaped ``(N, C, T)`` or matching lower-dimensional arrays.
    reduction:
        ``none`` keeps ``(N, C, T)``; ``channel`` averages over time and returns
        ``(N, C)``; ``sample`` averages over channels and time and returns
        ``(N,)``.
    """

    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs reconstruction {reconstructed.shape}"
        )
    residual = np.square(original.astype(np.float64) - reconstructed.astype(np.float64))
    if reduction == "none":
        return residual
    if reduction == "channel":
        if residual.ndim < 2:
            raise ValueError("channel reduction requires at least two dimensions")
        return residual.mean(axis=-1)
    if reduction == "sample":
        return residual.reshape(residual.shape[0], -1).mean(axis=1)
    raise ValueError("reduction must be 'none', 'channel' or 'sample'")


@dataclass
class ScoreResult:
    """Container returned by :class:`AnomalyScorer`."""

    sample_scores: np.ndarray
    channel_scores: np.ndarray
    raw_channel_scores: np.ndarray


class SpatialNormalizer:
    """Per-channel robust standardisation of anomaly scores.

    Different fibre sections have different coupling efficiency and noise
    floors.  Standardising each channel against a clean calibration set makes
    scores comparable across space and prevents high-gain sections from
    dominating pass-level aggregation.

    Parameters
    ----------
    method:
        ``robust`` uses median and scaled MAD; ``zscore`` uses mean and standard
        deviation.
    eps:
        Lower bound for the scale to avoid division by zero on constant channels.
    clip:
        Optional absolute z-score clipping applied in :meth:`transform`.
    """

    def __init__(
        self,
        method: Literal["robust", "zscore"] = "robust",
        eps: float = 1e-6,
        clip: float | None = None,
    ) -> None:
        if method not in {"robust", "zscore"}:
            raise ValueError("method must be 'robust' or 'zscore'")
        self.method = method
        self.eps = float(eps)
        self.clip = clip
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, channel_scores: np.ndarray) -> "SpatialNormalizer":
        """Estimate per-channel statistics from normal data ``(N, C)``."""

        scores = np.asarray(channel_scores, dtype=np.float64)
        if scores.ndim != 2:
            raise ValueError("channel_scores must have shape (samples, channels)")
        if scores.shape[0] == 0:
            raise ValueError("cannot fit a normalizer on zero samples")

        if self.method == "robust":
            center = np.median(scores, axis=0)
            scale = 1.4826 * np.median(np.abs(scores - center), axis=0)
            # A degenerate MAD can occur with too few calibration samples.
            std = scores.std(axis=0)
            scale = np.where(scale > self.eps, scale, std)
        else:
            center = scores.mean(axis=0)
            scale = scores.std(axis=0)
        self.center_ = center
        self.scale_ = np.maximum(scale, self.eps)
        return self

    def transform(self, channel_scores: np.ndarray) -> np.ndarray:
        """Standardise ``(N, C)`` or ``(C,)`` channel scores."""

        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("SpatialNormalizer must be fitted first")
        scores = np.asarray(channel_scores, dtype=np.float64)
        normalized = (scores - self.center_) / self.scale_
        if self.clip is not None:
            normalized = np.clip(normalized, -self.clip, self.clip)
        return normalized.astype(np.float32)

    def inverse_transform(self, normalized) -> np.ndarray:
        """Map standardised scores back to the original score units."""

        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("SpatialNormalizer must be fitted first")
        values = np.asarray(normalized, dtype=np.float64)
        return (values * self.scale_ + self.center_).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "eps": self.eps,
            "clip": self.clip,
            "center": self.center_.tolist() if self.center_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpatialNormalizer":
        """Restore a normalizer saved with :meth:`to_dict`."""

        normalizer = cls(
            method=payload.get("method", "robust"),
            eps=float(payload.get("eps", 1e-6)),
            clip=payload.get("clip", 12.0),
        )
        center = payload.get("center")
        scale = payload.get("scale")
        if center is not None and scale is not None:
            normalizer.center_ = np.asarray(center, dtype=np.float64)
            normalizer.scale_ = np.asarray(scale, dtype=np.float64)
        return normalizer


def aggregate_channel_scores(
    channel_scores: np.ndarray,
    method: AggregationMethod = "topk",
    topk: int = 3,
) -> np.ndarray:
    """Reduce per-channel scores to one score per pass.

    ``topk`` averages the strongest channels, which is more sensitive to a
    localised defect than averaging across the whole fibre while being more
    stable than a single maximum.
    """

    scores = np.asarray(channel_scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[None, :]
    if scores.ndim != 2:
        raise ValueError("channel_scores must have shape (samples, channels)")
    if scores.shape[1] == 0:
        raise ValueError("channel_scores has zero channels")

    if method == "max":
        return scores.max(axis=1).astype(np.float32)
    if method == "mean":
        return scores.mean(axis=1).astype(np.float32)
    if method == "median":
        return np.median(scores, axis=1).astype(np.float32)
    if method == "p95":
        return np.percentile(scores, 95, axis=1).astype(np.float32)
    if method == "topk":
        k = min(max(int(topk), 1), scores.shape[1])
        strongest = np.partition(scores, -k, axis=1)[:, -k:]
        return strongest.mean(axis=1).astype(np.float32)
    raise ValueError("method must be one of topk, max, mean, median, p95")


class AnomalyScorer:
    """Fit normalisation and turn a model into calibrated anomaly scores.

    The model may be a reconstruction model or a baseline exposing
    ``channel_scores``.  In both cases the resulting channel scores are
    normalised using a clean calibration set and aggregated into pass scores.

    Examples
    --------
    >>> scorer = AnomalyScorer(model, topk=3, sampling_rate_hz=1000.0)
    >>> scorer.fit(normal_calibration_passes)
    >>> result = scorer.score(test_passes)
    >>> result.sample_scores.shape
    (n_test_passes,)
    """

    def __init__(
        self,
        model: Any,
        topk_channels: int = 3,
        aggregation: AggregationMethod = "topk",
        normalization: Literal["robust", "zscore", "none"] = "robust",
        clip: float | None = None,
        sampling_rate_hz: float = 1000.0,
    ) -> None:
        self.model = model
        self.topk_channels = int(topk_channels)
        self.aggregation = aggregation
        self.normalization = normalization
        self.clip = clip
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.normalizer_: SpatialNormalizer | None = None
        self._uses_reconstruction: bool | None = None

    # ------------------------------------------------------------------ fitting
    @staticmethod
    def _as_batch(data: np.ndarray) -> np.ndarray:
        """Normalise a single pass or a batch to ``(N, C, T)``."""

        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise ValueError(
                "data must have shape (channels, samples) or (passes, channels, samples)"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("data contains NaN or infinite values")
        return array

    def _raw_channel_scores(self, data: np.ndarray) -> np.ndarray:
        """Get unnormalised per-channel scores from any supported model."""

        if not getattr(self.model, "is_fitted", False):
            raise RuntimeError("The underlying model must be fitted before scoring")
        batch = self._as_batch(data)

        if self._uses_reconstruction is None:
            probe = batch[: min(2, batch.shape[0])]
            try:
                reconstruction = self.model.reconstruct(probe)
                self._uses_reconstruction = reconstruction.shape == probe.shape
            except NotImplementedError:
                self._uses_reconstruction = False

        if self._uses_reconstruction:
            reconstruction = self.model.reconstruct(batch)
            return reconstruction_residual(batch, reconstruction, reduction="channel")

        try:
            scores = self.model.channel_scores(
                batch, sampling_rate_hz=self.sampling_rate_hz
            )
        except TypeError:
            # Some non-reconstruction models expose a score-only API without
            # the optional sampling-rate keyword.
            scores = self.model.channel_scores(batch)
        return np.asarray(scores, dtype=np.float32)

    def fit(self, calibration_data: np.ndarray) -> "AnomalyScorer":
        """Fit the spatial normaliser on clean (normal) passes."""

        raw = self._raw_channel_scores(calibration_data)
        if self.normalization == "none":
            self.normalizer_ = None
        else:
            self.normalizer_ = SpatialNormalizer(
                method=self.normalization,
                clip=self.clip,
            ).fit(raw)
        LOGGER.info(
            "Anomaly scorer calibrated on %d passes / %d channels (%s normalisation)",
            raw.shape[0],
            raw.shape[1],
            self.normalization,
        )
        return self

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        """Serialise calibration state for a training metadata sidecar."""

        return {
            "normalization": self.normalization,
            "topk_channels": self.topk_channels,
            "aggregation": self.aggregation,
            "sampling_rate_hz": self.sampling_rate_hz,
            "normalizer": self.normalizer_.to_dict() if self.normalizer_ else None,
        }

    def load_state(self, state: dict[str, Any]) -> "AnomalyScorer":
        """Restore calibration state produced by :meth:`state_dict`."""

        self.normalization = state.get("normalization", self.normalization)
        self.topk_channels = int(state.get("topk_channels", self.topk_channels))
        self.aggregation = state.get("aggregation", self.aggregation)
        self.sampling_rate_hz = float(
            state.get("sampling_rate_hz", self.sampling_rate_hz)
        )
        normalizer_state = state.get("normalizer")
        if normalizer_state is not None:
            self.normalizer_ = SpatialNormalizer.from_dict(normalizer_state)
        return self

    # ---------------------------------------------------------------- inference
    def raw_channel_scores(self, data: np.ndarray) -> np.ndarray:
        """Expose unnormalised channel scores for diagnostics."""

        return self._raw_channel_scores(data)

    def channel_scores(self, data: np.ndarray) -> np.ndarray:
        """Return normalised channel scores ``(N, C)``."""

        raw = self._raw_channel_scores(data)
        if self.normalizer_ is None:
            return raw.astype(np.float32)
        return self.normalizer_.transform(raw)

    def sample_scores(self, data: np.ndarray) -> np.ndarray:
        """Return aggregated pass-level anomaly scores ``(N,)``."""

        channel_scores = self.channel_scores(data)
        return aggregate_channel_scores(
            channel_scores, method=self.aggregation, topk=self.topk_channels
        )

    def score(self, data: np.ndarray) -> ScoreResult:
        """Return raw and normalised channel scores plus pass scores."""

        raw = self._raw_channel_scores(data)
        normalized = (
            self.normalizer_.transform(raw)
            if self.normalizer_ is not None
            else raw.astype(np.float32)
        )
        samples = aggregate_channel_scores(
            normalized, method=self.aggregation, topk=self.topk_channels
        )
        return ScoreResult(
            sample_scores=samples,
            channel_scores=normalized,
            raw_channel_scores=raw,
        )


class MultiPassAccumulator:
    """Fuse per-channel anomaly profiles from repeated train passes.

    Defects are stationary while wheel/axle excitations vary between passes.
    Aggregating aligned channel profiles (for example by median) improves the
    signal-to-noise ratio and robustness to a single outlier pass.

    Parameters
    ----------
    method:
        Fusion statistic: ``mean``, ``median`` or ``max``.
    """

    def __init__(self, method: FusionMethod = "median") -> None:
        if method not in {"mean", "median", "max"}:
            raise ValueError("method must be mean, median or max")
        self.method = method
        self._profiles: list[np.ndarray] = []
        self._pass_ids: list[Any] = []

    def add(
        self,
        channel_scores: np.ndarray,
        pass_id: Any | None = None,
    ) -> "MultiPassAccumulator":
        """Add one profile ``(C,)`` or a batch of profiles ``(P, C)``."""

        scores = np.asarray(channel_scores, dtype=np.float64)
        if scores.ndim == 1:
            scores = scores[None, :]
        if scores.ndim != 2 or scores.shape[1] == 0:
            raise ValueError("channel_scores must have shape (C,) or (P, C)")
        if self._profiles and scores.shape[1] != self._profiles[0].shape[0]:
            raise ValueError("all profiles must have the same channel count")
        for index, profile in enumerate(scores):
            self._profiles.append(profile.copy())
            self._pass_ids.append(
                pass_id if pass_id is not None else len(self._profiles) - 1
            )
        return self

    @property
    def n_passes(self) -> int:
        return len(self._profiles)

    def fuse(self, method: FusionMethod | None = None) -> dict[str, np.ndarray]:
        """Return fused profile, variability and per-pass count.

        The returned dictionary contains ``profile`` (C,), ``std`` (C,),
        ``n_passes`` as a scalar array and ``pass_ids``.
        """

        if not self._profiles:
            raise RuntimeError("No pass profiles have been added")
        matrix = np.stack(self._profiles, axis=0)
        method = method or self.method
        if method == "mean":
            profile = matrix.mean(axis=0)
        elif method == "median":
            profile = np.median(matrix, axis=0)
        elif method == "max":
            profile = matrix.max(axis=0)
        else:
            raise ValueError("method must be mean, median or max")
        std = matrix.std(axis=0)
        return {
            "profile": profile.astype(np.float32),
            "std": std.astype(np.float32),
            "n_passes": np.asarray([matrix.shape[0]], dtype=np.int32),
            "pass_ids": np.asarray(self._pass_ids, dtype=object),
        }

    def reset(self) -> None:
        self._profiles.clear()
        self._pass_ids.clear()


def accumulate_multi_pass(
    profiles: np.ndarray | Sequence[np.ndarray],
    method: FusionMethod = "median",
) -> dict[str, np.ndarray]:
    """Functional wrapper around :class:`MultiPassAccumulator`."""

    accumulator = MultiPassAccumulator(method=method)
    accumulator.add(np.asarray(profiles))
    return accumulator.fuse()


def select_threshold(
    normal_scores: np.ndarray,
    method: Literal["percentile", "fixed"] = "percentile",
    percentile: float = 99.0,
    fixed_threshold: float | None = None,
) -> float:
    """Choose an anomaly threshold from normal or validation scores.

    Percentile thresholds are set on scores from normal passes; a 99th
    percentile therefore targets roughly 1 % false alarms on the calibration
    set, subject to the sample size.
    """

    if method == "fixed":
        if fixed_threshold is None:
            raise ValueError("fixed_threshold is required when method='fixed'")
        return float(fixed_threshold)
    if method != "percentile":
        raise ValueError("method must be 'percentile' or 'fixed'")
    scores = np.asarray(normal_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("normal_scores must not be empty")
    return float(np.percentile(scores, percentile))

"""Pure energy-threshold baseline.

This is the baseline every learned detector has to beat.  It uses no
reconstruction, no subspace and no temporal model: each channel is summarised
by its band energy over the analysis window, standardised with robust
statistics estimated on normal passes, and the pass score is the mean of the
strongest channels.

If a deep model cannot beat this on a given SNR, the extra capacity is not
buying anything and the benchmark is probably measuring the preprocessing
chain rather than the model.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


class EnergyThresholdBaseline(BaseModel):
    """Per-channel energy z-score with a top-k pass aggregation.

    Parameters
    ----------
    topk_channels:
        Number of strongest channels averaged into the pass score.
    frame_length:
        Samples per RMS frame.  Channel energy is the mean power over frames
        (``statistic="mean"``) or the strongest frame (``statistic="max"``).
    statistic:
        ``mean`` is the classical energy detector; ``max`` is a transient-aware
        variant that reacts to short impacts.
    eps:
        Floor added before the logarithm.
    """

    name = "energy"

    def __init__(
        self,
        topk_channels: int = 3,
        frame_length: int = 128,
        statistic: Literal["mean", "max"] = "mean",
        eps: float = 1e-10,
    ) -> None:
        super().__init__(
            topk_channels=topk_channels,
            frame_length=frame_length,
            statistic=statistic,
            eps=eps,
        )
        if statistic not in {"mean", "max"}:
            raise ValueError("statistic must be 'mean' or 'max'")
        self.topk_channels = int(topk_channels)
        self.frame_length = int(frame_length)
        self.statistic = statistic
        self.eps = float(eps)
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    # ------------------------------------------------------------------ energy
    def _channel_energy(self, batch: np.ndarray) -> np.ndarray:
        """Log energy per channel, shape ``(N, C)``."""

        n_samples, n_channels, time_samples = batch.shape
        frame = max(1, min(self.frame_length, time_samples))
        usable = (time_samples // frame) * frame
        trimmed = batch[:, :, :usable].reshape(
            n_samples, n_channels, time_samples // frame, frame
        )
        power = np.mean(np.square(trimmed.astype(np.float64)), axis=-1)
        if self.statistic == "max":
            energy = power.max(axis=-1)
        else:
            energy = power.mean(axis=-1)
        return 10.0 * np.log10(np.maximum(energy, self.eps))

    # -------------------------------------------------------------------- API
    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "EnergyThresholdBaseline":
        train = self.validate_samples(train_data)
        energy = self._channel_energy(train)
        self.center_ = np.median(energy, axis=0)
        scale = 1.4826 * np.median(np.abs(energy - self.center_), axis=0)
        fallback = energy.std(axis=0)
        self.scale_ = np.maximum(np.where(scale > 1e-8, scale, fallback), 1e-8)
        self._is_fitted = True
        self.metadata_["n_channels"] = int(train.shape[1])
        self.training_history_ = {
            "mean_energy_db": [float(np.mean(self.center_))]
        }
        LOGGER.info(
            "Energy baseline calibrated on %d windows / %d channels",
            train.shape[0],
            train.shape[1],
        )
        return self

    def channel_scores(
        self,
        data: np.ndarray,
        sampling_rate_hz: float = 1000.0,
    ) -> np.ndarray:
        self._check_fitted()
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Energy baseline has not been calibrated")
        batch = self.validate_samples(data)
        energy = self._channel_energy(batch)
        return ((energy - self.center_) / self.scale_).astype(np.float32)

    def score(
        self,
        data: np.ndarray,
        sampling_rate_hz: float = 1000.0,
    ) -> np.ndarray:
        scores = self.channel_scores(data, sampling_rate_hz=sampling_rate_hz)
        topk = min(max(self.topk_channels, 1), scores.shape[1])
        strongest = np.partition(scores, -topk, axis=1)[:, -topk:]
        return strongest.mean(axis=1)

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "The energy baseline is not a reconstruction model; use channel_scores"
        )

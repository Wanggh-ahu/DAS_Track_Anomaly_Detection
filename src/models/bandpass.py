"""Band-limited transient detector — the physically matched baseline.

The injected defects are narrow-band: 70–165 Hz impact carriers and a 120 Hz
resonance with its 1.45× harmonic.  The moving-load background, by contrast, is
concentrated near 65 Hz and is broadband elsewhere.  A detector that simply
integrates total energy therefore compares the defect against a much larger
background; one that first isolates the defect band does not.

This baseline applies a Butterworth band-pass, converts to a short-frame
envelope and keeps the strongest frame per channel.  It is the honest classical
competitor for `ae_1d`: if a learned model cannot beat it, the extra capacity is
not buying anything a two-line filter would not.

It is deliberately *not* tuned per SNR — the band is fixed from the defect
physics, so the comparison stays fair.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


class BandEnergyBaseline(BaseModel):
    """Band-pass envelope z-score with a top-k pass aggregation.

    Parameters
    ----------
    band_hz:
        Pass band.  The default covers the impact carriers and the resonance.
    frame_length:
        Samples per envelope frame.  ``max`` over frames reacts to short
        transients, ``mean`` behaves like a classical band-energy detector.
    statistic:
        ``max`` or ``mean`` over frames.
    order:
        Butterworth order of the band-pass.
    """

    name = "bandpass"

    def __init__(
        self,
        band_hz: tuple[float, float] | list[float] = (80.0, 160.0),
        frame_length: int = 32,
        statistic: Literal["mean", "max"] = "max",
        order: int = 4,
        topk_channels: int = 3,
        sampling_rate_hz: float = 1000.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            band_hz=list(band_hz),
            frame_length=frame_length,
            statistic=statistic,
            order=order,
            topk_channels=topk_channels,
            sampling_rate_hz=sampling_rate_hz,
            eps=eps,
        )
        if statistic not in {"mean", "max"}:
            raise ValueError("statistic must be 'mean' or 'max'")
        if len(band_hz) != 2 or float(band_hz[0]) >= float(band_hz[1]):
            raise ValueError("band_hz must be an increasing pair")
        self.band_hz = (float(band_hz[0]), float(band_hz[1]))
        self.frame_length = int(frame_length)
        self.statistic = statistic
        self.order = int(order)
        self.topk_channels = int(topk_channels)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.eps = float(eps)
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self._sos: np.ndarray | None = None
        self._sos_rate: float | None = None

    # ------------------------------------------------------------------ filter
    def _filter(self, sampling_rate_hz: float) -> np.ndarray:
        """Build (and cache) the Butterworth sections for a sampling rate."""

        from scipy.signal import butter

        if self._sos is not None and self._sos_rate == float(sampling_rate_hz):
            return self._sos
        nyquist = 0.5 * float(sampling_rate_hz)
        low = max(self.band_hz[0], 1e-3)
        high = min(self.band_hz[1], 0.98 * nyquist)
        if low >= high:
            raise ValueError(
                f"band {self.band_hz} does not fit a {sampling_rate_hz:.0f} Hz sampling rate"
            )
        self._sos = butter(
            self.order, [low, high], btype="bandpass", fs=float(sampling_rate_hz), output="sos"
        )
        self._sos_rate = float(sampling_rate_hz)
        return self._sos

    def _channel_response(
        self, batch: np.ndarray, sampling_rate_hz: float
    ) -> np.ndarray:
        """Log envelope per channel, shape ``(N, C)``."""

        from scipy.signal import sosfiltfilt

        n_samples, n_channels, time_samples = batch.shape
        sos = self._filter(sampling_rate_hz)
        filtered = sosfiltfilt(sos, batch.astype(np.float64), axis=-1)
        frame = max(1, min(self.frame_length, time_samples))
        usable = (time_samples // frame) * frame
        trimmed = filtered[:, :, :usable].reshape(
            n_samples, n_channels, usable // frame, frame
        )
        power = np.mean(np.square(trimmed), axis=-1)
        energy = power.max(axis=-1) if self.statistic == "max" else power.mean(axis=-1)
        return 10.0 * np.log10(np.maximum(energy, self.eps))

    # -------------------------------------------------------------------- API
    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "BandEnergyBaseline":
        sampling_rate_hz = float(kwargs.get("sampling_rate_hz", self.sampling_rate_hz))
        self.sampling_rate_hz = sampling_rate_hz
        train = self.validate_samples(train_data)
        response = self._channel_response(train, sampling_rate_hz)
        self.center_ = np.median(response, axis=0)
        scale = 1.4826 * np.median(np.abs(response - self.center_), axis=0)
        fallback = response.std(axis=0)
        self.scale_ = np.maximum(np.where(scale > 1e-8, scale, fallback), 1e-8)
        self._is_fitted = True
        self.metadata_["n_channels"] = int(train.shape[1])
        self.training_history_ = {"mean_band_db": [float(np.mean(self.center_))]}
        LOGGER.info(
            "Band energy baseline %s calibrated on %d windows / %d channels",
            self.band_hz,
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
            raise RuntimeError("Band energy baseline has not been calibrated")
        batch = self.validate_samples(data)
        response = self._channel_response(batch, float(sampling_rate_hz))
        return ((response - self.center_) / self.scale_).astype(np.float32)

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
            "The band-energy baseline is not a reconstruction model; "
            "use channel_scores"
        )

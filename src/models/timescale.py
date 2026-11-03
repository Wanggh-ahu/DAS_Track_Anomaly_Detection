"""Detectors that look at how the defect moves in time, not at its static spectrum.

Section 7 of ``reports/snr_benchmark.md`` showed that the STFT autoencoder loses to
a fixed band-pass envelope as soon as the defect stops being a *stationary tone*:

* ``impulse_train`` -- impacts repeat at the wheel-rotation rate, so the envelope
  carries a *line spectrum*; averaging the spectrogram over the pass erases it.
* ``chirp`` -- the tone sweeps through the band during one response; averaging
  over frames smears it across bins.

Both failures have a textbook remedy that needs no learning, and that is the
point: if these two detectors recover the lost margin, then the organising
variable is *the time scale the detector integrates over*, not whether the
representation was learned.  If they do not, the "time scale" story is empty and
the project should stop building representations.

``envspec`` -- envelope-spectrum detector
    Band-pass, take the Hilbert envelope, then look at the *spectrum of the
    envelope* in the modulation band (3-45 Hz).  Repetition at 9 Hz puts a line
    there; a single non-repeating transient does not.  This is the standard
    demodulation / envelope-analysis tool for periodic faults, and it is exactly
    the observable the STAE family cannot form.

``tftraj`` -- time-frequency trajectory detector
    Track the dominant frequency of each window frame by frame and measure how
    *coherently* it moves: a straight sweep gives a large, well-fitted slope,
    whereas noise gives a slope with no explanatory power.  The score is
    ``|slope| * R^2``, so it responds to motion that is real rather than to
    jitter.

Both are calibrated the same way as every other baseline in this project: per
(channel, band) median/MAD over normal windows only, so "unusual" is always a
question about a channel against itself.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

import numpy as np

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)

DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (5.0, 45.0),
    (45.0, 90.0),
    (90.0, 140.0),
    (140.0, 200.0),
)


def _butter_band(
    low: float, high: float, sampling_rate_hz: float, order: int
) -> np.ndarray:
    from scipy.signal import butter

    nyquist = 0.5 * float(sampling_rate_hz)
    low = max(float(low), 1e-3)
    high = min(float(high), 0.98 * nyquist)
    if low >= high:
        raise ValueError(f"empty band ({low}, {high}) at {sampling_rate_hz} Hz")
    return butter(order, [low, high], btype="bandpass", fs=float(sampling_rate_hz), output="sos")


class _CalibratedStatistic(BaseModel):
    """Shared plumbing: compute a statistic per (window, channel), calibrate it."""

    def __init__(self, topk_channels: int = 3, sampling_rate_hz: float = 1000.0) -> None:
        super().__init__(topk_channels=topk_channels, sampling_rate_hz=sampling_rate_hz)
        self.topk_channels = int(topk_channels)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _raw_statistic(self, batch: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
        raise NotImplementedError

    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "_CalibratedStatistic":
        sampling_rate_hz = float(kwargs.get("sampling_rate_hz", self.sampling_rate_hz))
        self.sampling_rate_hz = sampling_rate_hz
        train = self.validate_samples(train_data)
        raw = self._raw_statistic(train, sampling_rate_hz)
        self.center_ = np.median(raw, axis=0)
        scale = 1.4826 * np.median(np.abs(raw - self.center_), axis=0)
        fallback = raw.std(axis=0)
        self.scale_ = np.maximum(np.where(scale > 1e-8, scale, fallback), 1e-8)
        self._is_fitted = True
        self.metadata_["n_channels"] = int(train.shape[1])
        LOGGER.info("%s calibrated on %d windows", self.name, train.shape[0])
        return self

    def channel_scores(
        self, data: np.ndarray, sampling_rate_hz: float = 1000.0
    ) -> np.ndarray:
        self._check_fitted()
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError(f"{self.name} has not been calibrated")
        batch = self.validate_samples(data)
        raw = self._raw_statistic(batch, float(sampling_rate_hz))
        return ((raw - self.center_) / self.scale_).astype(np.float32)

    def score(self, data: np.ndarray, sampling_rate_hz: float = 1000.0) -> np.ndarray:
        scores = self.channel_scores(data, sampling_rate_hz=sampling_rate_hz)
        topk = min(max(self.topk_channels, 1), scores.shape[1])
        return np.partition(scores, -topk, axis=1)[:, -topk:].mean(axis=1)

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{self.name} is not a reconstruction model; use channel_scores"
        )


class EnvelopeSpectrumBaseline(_CalibratedStatistic):
    """Modulation energy in the envelope spectrum.

    For each candidate band the Hilbert envelope is formed, its spectrum is
    taken in the modulation band, and the energy there is the statistic.  A
    periodic impact train produces a strong 9 Hz line plus harmonics; a single
    ring or a tone produces a broadband envelope, so the statistic stays low.
    """

    name = "envspec"

    def __init__(
        self,
        bands: Sequence[Sequence[float]] | None = None,
        modulation_hz: tuple[float, float] = (3.0, 45.0),
        order: int = 3,
        band_aggregation: Literal["max", "mean"] = "max",
        topk_channels: int = 3,
        sampling_rate_hz: float = 1000.0,
    ) -> None:
        super().__init__(topk_channels=topk_channels, sampling_rate_hz=sampling_rate_hz)
        self.bands = [tuple(float(v) for v in b) for b in (bands or DEFAULT_BANDS)]
        self.modulation_hz = (float(modulation_hz[0]), float(modulation_hz[1]))
        self.order = int(order)
        self.band_aggregation = band_aggregation

    def _raw_statistic(self, batch: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
        from scipy.signal import hilbert, sosfiltfilt

        n_samples, n_channels, time_samples = batch.shape
        data = batch.astype(np.float64)
        # A Hann taper before the envelope FFT keeps the finite window from
        # leaking a 1/f skirt across the modulation band.
        taper = np.hanning(time_samples)
        out = np.empty((n_samples, n_channels, len(self.bands)))
        for index, (low, high) in enumerate(self.bands):
            sos = _butter_band(low, high, sampling_rate_hz, self.order)
            filtered = sosfiltfilt(sos, data, axis=-1)
            envelope = np.abs(hilbert(filtered, axis=-1))
            envelope = envelope - envelope.mean(axis=-1, keepdims=True)
            spectrum = np.abs(np.fft.rfft(envelope * taper, axis=-1)) ** 2
            freqs = np.fft.rfftfreq(time_samples, d=1.0 / float(sampling_rate_hz))
            mask = (freqs >= self.modulation_hz[0]) & (freqs <= self.modulation_hz[1])
            if not mask.any():
                mask = np.ones_like(freqs, dtype=bool)
            out[..., index] = np.log10(np.maximum(spectrum[..., mask].mean(axis=-1), 1e-12))
        if self.band_aggregation == "max":
            return out.max(axis=-1)
        return out.mean(axis=-1)


class TrajectoryBaseline(_CalibratedStatistic):
    """How coherently the dominant frequency moves within a window.

    Score is ``|slope| * R^2`` of a straight-line fit to the per-frame dominant
    frequency, in Hz per window.  A sweep is coherent (large slope, R^2 near 1);
    a stationary tone or noise is not (either slope ~ 0 or no explanatory
    power), so the product separates the two without a threshold on either.
    """

    name = "tftraj"

    def __init__(
        self,
        band_hz: tuple[float, float] = (20.0, 200.0),
        frame_length: int = 32,
        order: int = 3,
        topk_channels: int = 3,
        sampling_rate_hz: float = 1000.0,
    ) -> None:
        super().__init__(topk_channels=topk_channels, sampling_rate_hz=sampling_rate_hz)
        self.band_hz = (float(band_hz[0]), float(band_hz[1]))
        self.frame_length = int(frame_length)
        self.order = int(order)

    def _raw_statistic(self, batch: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
        from scipy.signal import hilbert, sosfiltfilt

        n_samples, n_channels, time_samples = batch.shape
        frame = max(8, min(self.frame_length, time_samples))
        usable = (time_samples // frame) * frame
        sos = _butter_band(self.band_hz[0], self.band_hz[1], sampling_rate_hz, self.order)
        data = batch.astype(np.float64)
        filtered = sosfiltfilt(sos, data, axis=-1)
        envelope = np.abs(hilbert(filtered, axis=-1))
        frames = envelope[:, :, :usable].reshape(
            n_samples, n_channels, usable // frame, frame
        )
        # Instantaneous frequency from the phase increment of the dominant
        # component is noisy; the envelope-weighted spectral centroid per frame
        # is far more stable at this signal length.
        spectrum = np.abs(np.fft.rfft(frames * np.hanning(frame), axis=-1)) ** 2
        freqs = np.fft.rfftfreq(frame, d=1.0 / float(sampling_rate_hz))
        band = (freqs >= self.band_hz[0]) & (freqs <= self.band_hz[1])
        if not band.any():
            band = np.ones_like(freqs, dtype=bool)
        power = spectrum[..., band]
        weights = power.sum(axis=-1, keepdims=True)
        centroids = (power * freqs[band]).sum(axis=-1, keepdims=True) / np.maximum(
            weights, 1e-12
        )
        centroids = centroids[..., 0]  # (N, C, n_frames)
        n_frames = centroids.shape[-1]
        if n_frames < 3:
            return np.zeros((n_samples, n_channels))
        t = np.arange(n_frames, dtype=np.float64)
        t -= t.mean()
        denom = float(np.sum(t * t))
        centred = centroids - centroids.mean(axis=-1, keepdims=True)
        slope = (centred * t).sum(axis=-1) / max(denom, 1e-12)
        fitted = slope[..., None] * t
        ss_res = np.sum(np.square(centred - fitted), axis=-1)
        ss_tot = np.sum(np.square(centred), axis=-1)
        r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
        return np.abs(slope) * np.clip(r2, 0.0, 1.0)

"""Spectral-kurtosis baseline: an adaptive band selector with no learning at all.

`bandpass` hard-codes 80-160 Hz, so losing to it at fixed morphology and beating
it under frequency jitter is *guaranteed by construction*.  That makes the
frequency-jitter result circular unless a classical method that also adapts its
band is beaten too.  Antoni's spectral kurtosis -- the standard tool for bearing
and gear diagnostics -- is exactly that method: it looks for the band where the
signal is most impulsive and analyses the envelope there.

The point of this baseline is to decide between two readings of the jitter
result:

* "learning a frequency-domain representation buys robustness" -- then this
  baseline should lose, because it adapts without learning;
* "adapting the band buys robustness" -- then this baseline should match the
  autoencoder, and the credit belongs to band selection, not to the network.

Implementation choices, stated because they are choices:

* Bands are two overlapping dyadic families (31.25 Hz and 62.5 Hz wide, half-width
  hop) covering 30-200 Hz.  A defect tone that straddles a boundary is still
  covered by the neighbouring band.
* The selector is the *excess kurtosis of the band envelope*, the practical
  variant of spectral kurtosis.  It is computed per window and per channel, so
  the selected band may differ from pass to pass.
* The detection statistic in the selected band is the log envelope energy.
  Kurtosis says *where*, energy says *how loud* -- using kurtosis itself as the
  score would mostly measure how few samples the band has.
* Both statistics are standardised per (channel, band) against normal windows
  only, the same calibration the filterbank gets.  Without it, argmax over raw
  kurtosis would just pick the band with the fewest effective samples.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

import numpy as np

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)

#: Band widths in Hz.  Two overlapping dyadic families; 31.25 Hz is 1/32 of the
#: 1000 Hz sampling rate, so the set lines up with the STFT resolution used by
#: the autoencoder and the comparison stays about *selection*, not resolution.
DEFAULT_WIDTHS: tuple[float, ...] = (31.25, 62.5)


def _merge_bands(
    widths: Sequence[float], low_hz: float, high_hz: float
) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    for width in widths:
        width = float(width)
        hop = width / 2.0
        start = low_hz
        while start + width <= high_hz + 1e-9:
            bands.append((start, start + width))
            start += hop
    # De-duplicate while keeping order deterministic.
    seen: set[tuple[float, float]] = set()
    unique: list[tuple[float, float]] = []
    for band in bands:
        key = (round(band[0], 3), round(band[1], 3))
        if key not in seen:
            seen.add(key)
            unique.append(band)
    return unique


class KurtogramBaseline(BaseModel):
    """Adaptive band selection by envelope kurtosis, then envelope energy."""

    name = "kurtogram"

    def __init__(
        self,
        band_widths: Sequence[float] | None = None,
        low_hz: float = 30.0,
        high_hz: float = 200.0,
        frame_length: int = 32,
        order: int = 3,
        selection: Literal["kurtosis", "energy"] = "kurtosis",
        statistic: Literal["max", "mean"] = "max",
        topk_channels: int = 3,
        sampling_rate_hz: float = 1000.0,
        eps: float = 1e-12,
        n_channels: int | None = None,
    ) -> None:
        widths = list(band_widths) if band_widths else list(DEFAULT_WIDTHS)
        bands = _merge_bands(widths, float(low_hz), float(high_hz))
        super().__init__(
            band_widths=widths,
            low_hz=float(low_hz),
            high_hz=float(high_hz),
            frame_length=frame_length,
            order=order,
            selection=selection,
            statistic=statistic,
            topk_channels=topk_channels,
            sampling_rate_hz=sampling_rate_hz,
            eps=eps,
            n_channels=n_channels,
        )
        if selection not in {"kurtosis", "energy"}:
            raise ValueError("selection must be 'kurtosis' or 'energy'")
        if statistic not in {"max", "mean"}:
            raise ValueError("statistic must be 'max' or 'mean'")
        self.band_widths = [float(w) for w in widths]
        self.bands = bands
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.frame_length = int(frame_length)
        self.order = int(order)
        self.selection = selection
        self.statistic = statistic
        self.topk_channels = int(topk_channels)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.eps = float(eps)
        self.energy_center_: np.ndarray | None = None
        self.energy_scale_: np.ndarray | None = None
        self.kurt_center_: np.ndarray | None = None
        self.kurt_scale_: np.ndarray | None = None
        self._sos: list[np.ndarray] | None = None
        self._sos_rate: float | None = None

    # ------------------------------------------------------------------ filter
    def _filters(self, sampling_rate_hz: float) -> list[np.ndarray]:
        from scipy.signal import butter

        if self._sos is not None and self._sos_rate == float(sampling_rate_hz):
            return self._sos
        nyquist = 0.5 * float(sampling_rate_hz)
        sections: list[np.ndarray] = []
        for low, high in self.bands:
            low = max(low, 1e-3)
            high = min(high, 0.98 * nyquist)
            if low >= high:
                continue
            sections.append(
                butter(
                    self.order,
                    [low, high],
                    btype="bandpass",
                    fs=float(sampling_rate_hz),
                    output="sos",
                )
            )
        if not sections:
            raise ValueError(
                f"no band fits a {sampling_rate_hz:.0f} Hz sampling rate: {self.bands}"
            )
        self._sos = sections
        self._sos_rate = float(sampling_rate_hz)
        return sections

    def _band_statistics(
        self, batch: np.ndarray, sampling_rate_hz: float, chunk: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(energy, kurtosis)`` with shape ``(N, C, B)``.

        Windows are processed in chunks.  A full training split is ~2600 windows
        by 96 channels by 128 samples, which is 256 MB per band in float64; the
        band-pass and Hilbert transform are happy to allocate several copies of
        that, and the resulting page churn made this baseline ~15x slower than
        the same arithmetic done in chunks.
        """

        from scipy.signal import hilbert, sosfiltfilt
        from scipy.stats import kurtosis

        n_samples, n_channels, time_samples = batch.shape
        frame = max(1, min(self.frame_length, time_samples))
        usable = (time_samples // frame) * frame
        filters = self._filters(sampling_rate_hz)
        energy = np.empty((n_samples, n_channels, len(filters)))
        kurt = np.empty((n_samples, n_channels, len(filters)))
        for start in range(0, n_samples, max(1, chunk)):
            stop = min(start + max(1, chunk), n_samples)
            data = batch[start:stop].astype(np.float64)
            for index, sos in enumerate(filters):
                filtered = sosfiltfilt(sos, data, axis=-1)
                envelope = np.abs(hilbert(filtered, axis=-1))
                trimmed = envelope[:, :, :usable].reshape(
                    data.shape[0], n_channels, usable // frame, frame
                )
                power = np.mean(np.square(trimmed), axis=-1)
                energy[start:stop, :, index] = (
                    power.max(axis=-1)
                    if self.statistic == "max"
                    else power.mean(axis=-1)
                )
                # Excess kurtosis of the envelope: high when the band is quiet
                # most of the time and loud briefly, which is what a transient
                # defect is.
                kurt[start:stop, :, index] = kurtosis(
                    envelope, axis=-1, fisher=True, bias=False, nan_policy="omit"
                )
        energy = 10.0 * np.log10(np.maximum(energy, self.eps))
        return energy, np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)

    # -------------------------------------------------------------------- API
    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "KurtogramBaseline":
        sampling_rate_hz = float(kwargs.get("sampling_rate_hz", self.sampling_rate_hz))
        self.sampling_rate_hz = sampling_rate_hz
        train = self.validate_samples(train_data)
        energy, kurt = self._band_statistics(train, sampling_rate_hz)

        self.energy_center_ = np.median(energy, axis=0)
        energy_scale = 1.4826 * np.median(np.abs(energy - self.energy_center_), axis=0)
        self.energy_scale_ = np.maximum(
            np.where(energy_scale > 1e-8, energy_scale, energy.std(axis=0)), 1e-8
        )
        self.kurt_center_ = np.median(kurt, axis=0)
        kurt_scale = 1.4826 * np.median(np.abs(kurt - self.kurt_center_), axis=0)
        self.kurt_scale_ = np.maximum(
            np.where(kurt_scale > 1e-8, kurt_scale, kurt.std(axis=0)), 1e-8
        )

        self._is_fitted = True
        self.metadata_["n_channels"] = int(train.shape[1])
        self.metadata_["n_bands"] = int(len(self.bands))
        self.metadata_["selection"] = self.selection
        LOGGER.info(
            "Kurtogram baseline %d bands (%.1f-%.1f Hz, widths %s), selection=%s",
            len(self.bands),
            self.low_hz,
            self.high_hz,
            self.band_widths,
            self.selection,
        )
        return self

    def band_zscores(
        self, data: np.ndarray, sampling_rate_hz: float = 1000.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standardised ``(energy_z, kurt_z)`` per (window, channel, band)."""

        self._check_fitted()
        batch = self.validate_samples(data)
        energy, kurt = self._band_statistics(batch, float(sampling_rate_hz))
        energy_z = (energy - self.energy_center_) / self.energy_scale_
        kurt_z = (kurt - self.kurt_center_) / self.kurt_scale_
        return energy_z.astype(np.float32), kurt_z.astype(np.float32)

    def channel_scores(
        self, data: np.ndarray, sampling_rate_hz: float = 1000.0
    ) -> np.ndarray:
        energy_z, kurt_z = self.band_zscores(data, sampling_rate_hz)
        if self.selection == "energy":
            # Ablation: no selection at all, just the loudest band.
            return energy_z.max(axis=-1).astype(np.float32)
        # Pick the most impulsive band per (window, channel) -- the kurtogram
        # decision -- then score how loud that band's envelope is.
        pick = np.argmax(kurt_z, axis=-1)[..., None]
        selected = np.take_along_axis(energy_z, pick, axis=-1)[..., 0]
        return selected.astype(np.float32)

    def score(
        self, data: np.ndarray, sampling_rate_hz: float = 1000.0
    ) -> np.ndarray:
        scores = self.channel_scores(data, sampling_rate_hz=sampling_rate_hz)
        topk = min(max(self.topk_channels, 1), scores.shape[1])
        strongest = np.partition(scores, -topk, axis=1)[:, -topk:]
        return strongest.mean(axis=1)

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "The kurtogram baseline is not a reconstruction model; use channel_scores"
        )


class LoudestBandBaseline(KurtogramBaseline):
    """Adaptive selection with a criterion matched to a tonal defect.

    The kurtogram selects the band that is most *impulsive*, which is the right
    question for a bearing spall and the wrong one for a narrow-band ring: the
    ring is not impulsive, so kurtosis picks whichever background band happens
    to look spiky in that window.  This variant keeps the same adaptive
    machinery but selects the band whose calibrated envelope energy is highest,
    which is the natural classical answer for a tone.

    It exists to separate two explanations that the kurtogram alone cannot:
    "adapting the band is useless" versus "adapting the band with the wrong
    statistic is useless".
    """

    name = "bandselect"

    def __init__(
        self,
        band_widths: Sequence[float] | None = None,
        low_hz: float = 30.0,
        high_hz: float = 200.0,
        frame_length: int = 32,
        order: int = 3,
        statistic: Literal["max", "mean"] = "max",
        topk_channels: int = 3,
        sampling_rate_hz: float = 1000.0,
        eps: float = 1e-12,
        n_channels: int | None = None,
    ) -> None:
        super().__init__(
            band_widths=band_widths,
            low_hz=low_hz,
            high_hz=high_hz,
            frame_length=frame_length,
            order=order,
            selection="energy",
            statistic=statistic,
            topk_channels=topk_channels,
            sampling_rate_hz=sampling_rate_hz,
            eps=eps,
            n_channels=n_channels,
        )

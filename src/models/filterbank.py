"""Multi-band envelope detector — the adaptive classical baseline.

`bandpass` is matched to *one* assumed spectrum.  If the defect band moves from
pass to pass, a fixed filter is no longer matched and its advantage may be an
artefact of the simulator rather than a property of DAS.  This baseline answers
that objection without any learning: run a small filterbank, score every band,
and keep the strongest one per channel.

If the filterbank recovers the accuracy the fixed filter loses, the lesson is
"adapt to the band", not "use a network".  If it does not, the deformation is
genuinely beyond a fixed time-frequency template.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

import numpy as np

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)

#: Contiguous 30 Hz bands from 60 to 240 Hz.  Chosen to tile the whole band the
#: sensor and the acquisition filter leave usable, not to fit the nominal defect.
DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (60.0, 90.0),
    (90.0, 120.0),
    (120.0, 150.0),
    (150.0, 180.0),
    (180.0, 240.0),
)


class FilterbankBaseline(BaseModel):
    """Mean of the per-band envelope z-scores, aggregated with a top-k mean.

    The default is ``mean`` rather than ``max``: taking the maximum over five
    noisy bands picks whichever band happened to fluctuate most, which inflates
    the score of every *healthy* channel and destroys the contrast with the
    defect channel (measured on the control data: profile AUC 0.52 for ``max``
    vs 0.66 for ``mean``, against 0.88 for a single well-chosen band).  This
    loss is intrinsic to combining bands blindly and is the point of keeping the
    filterbank in the benchmark.
    """

    name = "filterbank"

    def __init__(
        self,
        bands: Sequence[Sequence[float]] | None = None,
        frame_length: int = 32,
        statistic: Literal["max", "mean"] = "max",
        band_aggregation: Literal["max", "mean"] = "mean",
        order: int = 4,
        topk_channels: int = 3,
        sampling_rate_hz: float = 1000.0,
        eps: float = 1e-12,
    ) -> None:
        resolved = (
            [list(band) for band in bands] if bands else [list(b) for b in DEFAULT_BANDS]
        )
        super().__init__(
            bands=resolved,
            frame_length=frame_length,
            statistic=statistic,
            band_aggregation=band_aggregation,
            order=order,
            topk_channels=topk_channels,
            sampling_rate_hz=sampling_rate_hz,
            eps=eps,
        )
        if statistic not in {"mean", "max"}:
            raise ValueError("statistic must be 'mean' or 'max'")
        if band_aggregation not in {"max", "mean"}:
            raise ValueError("band_aggregation must be 'max' or 'mean'")
        for band in resolved:
            if len(band) != 2 or float(band[0]) >= float(band[1]):
                raise ValueError(f"invalid band {band!r}")
        self.bands = [(float(lo), float(hi)) for lo, hi in resolved]
        self.frame_length = int(frame_length)
        self.statistic = statistic
        self.band_aggregation = band_aggregation
        self.order = int(order)
        self.topk_channels = int(topk_channels)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.eps = float(eps)
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
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

    def _band_response(
        self, batch: np.ndarray, sampling_rate_hz: float
    ) -> np.ndarray:
        """Log envelope per channel and band, shape ``(N, C, B)``."""

        from scipy.signal import sosfiltfilt

        n_samples, n_channels, time_samples = batch.shape
        frame = max(1, min(self.frame_length, time_samples))
        usable = (time_samples // frame) * frame
        filters = self._filters(sampling_rate_hz)
        energies = np.empty((n_samples, n_channels, len(filters)))
        for index, sos in enumerate(filters):
            filtered = sosfiltfilt(sos, batch.astype(np.float64), axis=-1)
            trimmed = filtered[:, :, :usable].reshape(
                n_samples, n_channels, usable // frame, frame
            )
            power = np.mean(np.square(trimmed), axis=-1)
            energies[..., index] = (
                power.max(axis=-1) if self.statistic == "max" else power.mean(axis=-1)
            )
        return 10.0 * np.log10(np.maximum(energies, self.eps))

    # -------------------------------------------------------------------- API
    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "FilterbankBaseline":
        sampling_rate_hz = float(kwargs.get("sampling_rate_hz", self.sampling_rate_hz))
        self.sampling_rate_hz = sampling_rate_hz
        train = self.validate_samples(train_data)
        response = self._band_response(train, sampling_rate_hz)
        # Normalise every (channel, band) pair against its own normal-pass
        # distribution, so "which band looks unusual" is a per-channel question
        # and never depends on the batch being scored.
        self.center_ = np.median(response, axis=0)
        scale = 1.4826 * np.median(np.abs(response - self.center_), axis=0)
        fallback = response.std(axis=0)
        self.scale_ = np.maximum(np.where(scale > 1e-8, scale, fallback), 1e-8)
        self._is_fitted = True
        self.metadata_["n_channels"] = int(train.shape[1])
        self.training_history_ = {"n_bands": [float(len(self.bands))]}
        LOGGER.info(
            "Filterbank baseline %d bands (%s) calibrated on %d windows",
            len(self.bands),
            self.bands[0],
            train.shape[0],
        )
        return self

    def band_zscores(
        self,
        data: np.ndarray,
        sampling_rate_hz: float = 1000.0,
    ) -> np.ndarray:
        """Per-(channel, band) z-scores, shape ``(N, C, B)``.

        Exposed so the frequency-attribution benchmark can ask *which band* a
        window looks unusual in; the filterbank is the honest classical rival
        there because it does carry a coarse spectral estimate.
        """

        self._check_fitted()
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Filterbank baseline has not been calibrated")
        batch = self.validate_samples(data)
        response = self._band_response(batch, float(sampling_rate_hz))
        return ((response - self.center_) / self.scale_).astype(np.float32)

    def channel_scores(
        self,
        data: np.ndarray,
        sampling_rate_hz: float = 1000.0,
    ) -> np.ndarray:
        self._check_fitted()
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Filterbank baseline has not been calibrated")
        batch = self.validate_samples(data)
        response = self._band_response(batch, float(sampling_rate_hz))
        z = (response - self.center_) / self.scale_
        if self.band_aggregation == "max":
            return z.max(axis=-1).astype(np.float32)
        return z.mean(axis=-1).astype(np.float32)

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
            "The filterbank baseline is not a reconstruction model; "
            "use channel_scores"
        )

"""Spatial-shape matched filter: does requiring a *contiguous* defect profile help?

Every detector in this project reduces 96 channels to one number by looking at
magnitudes only -- the filterbank takes the mean of the ``k`` loudest channels,
the band-pass envelope does the same.  That discards the one piece of spatial
structure a track defect has: it is a *localised source*, so its residual rises
and falls smoothly over a few adjacent channels.

A background window can easily have three equally loud channels; it is much
rarer for those three to be adjacent and to fall off on both sides.  This model
tests whether that shape prior is worth anything, holding everything else fixed:

* per-channel calibration is reused verbatim from ``FilterbankBaseline``
  (per (channel, band) median/MAD, then mean over bands), so the amplitude
  statistic is bit-identical to the baseline;
* the only change is the spatial reduction -- a raised-cosine profile of width
  ``w`` is slid along the array and the best weighted mean is taken instead of
  the top-3 mean.

The maximum over 96 positions and several widths is itself a selection, and
section 7.7 showed that selections must be calibrated or they inflate the null.
So the same max is computed on the training passes and the score is standardised
against that null before being compared across windows.

Prediction, if the shape prior is real: fewer false alarms on normal windows
because scattered loud channels no longer score, so the pass-level AUC should
rise even though the per-channel evidence is unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from .base_model import BaseModel
from .filterbank import FilterbankBaseline

LOGGER = logging.getLogger(__name__)

DEFAULT_WIDTHS: tuple[int, ...] = (2, 3, 4, 5)


def _raised_cosine(width: int) -> np.ndarray:
    """Unit-sum profile of the given width, peaked in the middle."""

    if width <= 1:
        return np.ones(1, dtype=np.float64)
    x = np.linspace(-np.pi / 2.0, np.pi / 2.0, width)
    profile = np.cos(x) ** 2
    return profile / profile.sum()


class SpatialProfileBaseline(BaseModel):
    """Profile-matched spatial reduction on top of the calibrated filterbank."""

    name = "spatial"

    def __init__(
        self,
        widths: Sequence[int] | None = None,
        frame_length: int = 32,
        topk_channels: int = 3,
        sampling_rate_hz: float = 1000.0,
        n_channels: int | None = None,
    ) -> None:
        super().__init__(topk_channels=topk_channels, sampling_rate_hz=sampling_rate_hz)
        self.widths = tuple(int(w) for w in (widths or DEFAULT_WIDTHS))
        self.frame_length = int(frame_length)
        self.topk_channels = int(topk_channels)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.n_channels = n_channels
        self.inner_: FilterbankBaseline | None = None
        self.center_: float | None = None
        self.scale_: float | None = None

    # ------------------------------------------------------------------ inner
    def _build_inner(self) -> FilterbankBaseline:
        inner = FilterbankBaseline(
            frame_length=self.frame_length,
            topk_channels=self.topk_channels,
            band_aggregation="mean",
        )
        return inner

    def _best_profile(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Locate the best-matching contiguous window for every sample.

        Returns ``(best_width, best_centre, best_projection)``.  ``convolve1d``
        is a true convolution, but every profile here is symmetric, so output
        ``i`` is the weighted mean of channels ``i - w//2 .. i + w//2``.
        """

        from scipy.ndimage import convolve1d

        n_samples = z.shape[0]
        best_width = np.zeros(n_samples, dtype=int)
        best_centre = np.zeros(n_samples, dtype=int)
        best_projection = np.full(n_samples, -np.inf, dtype=np.float64)
        for width in self.widths:
            if width > z.shape[1]:
                continue
            kernel = _raised_cosine(width)
            smoothed = convolve1d(z, weights=kernel, axis=1, mode="constant", cval=0.0)
            centre = smoothed.argmax(axis=1)
            value = smoothed[np.arange(n_samples), centre]
            better = value > best_projection
            best_projection = np.where(better, value, best_projection)
            best_width = np.where(better, width, best_width)
            best_centre = np.where(better, centre, best_centre)
        return best_width, best_centre, best_projection

    def _profile_max(self, channel_z: np.ndarray) -> np.ndarray:
        """Best weighted mean over contiguous channel windows, per window."""

        return self._best_profile(channel_z)[2]

    def _contributions(self, channel_z: np.ndarray) -> np.ndarray:
        """Per-channel score = weighted contribution inside the best window.

        The evaluation harness builds window scores by aggregating
        ``channel_scores`` and never calls :meth:`score`, so the spatial
        statistic has to live in the channel vector to be measured at all.
        Channels outside the winning window get zero, which also localises the
        detection to the channels that actually matched the profile.
        """

        widths, centres, _ = self._best_profile(channel_z)
        out = np.zeros_like(channel_z, dtype=np.float64)
        for width in np.unique(widths):
            kernel = _raised_cosine(int(width))
            half = int(width) // 2
            rows = np.flatnonzero(widths == width)
            for offset in range(-half, half + 1):
                column = centres[rows] + offset
                valid = (column >= 0) & (column < channel_z.shape[1])
                selected_rows = rows[valid]
                selected_cols = column[valid]
                weight = kernel[offset + half]
                out[selected_rows, selected_cols] = (
                    weight * channel_z[selected_rows, selected_cols]
                )
        return out

    # -------------------------------------------------------------------- API
    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "SpatialProfileBaseline":
        inner = self._build_inner()
        inner.fit(train_data, val_data, **kwargs)
        self.inner_ = inner
        z = inner.band_zscores(train_data, sampling_rate_hz=self.sampling_rate_hz)
        channel_z = z.mean(axis=2)
        raw = self._profile_max(channel_z)
        self.center_ = float(np.mean(raw))
        self.scale_ = float(max(np.std(raw), 1e-8))
        self._is_fitted = True
        self.metadata_["n_channels"] = int(train_data.shape[1])
        LOGGER.info(
            "%s calibrated on %d windows (profile null %.4f +- %.4f)",
            self.name,
            train_data.shape[0],
            self.center_,
            self.scale_,
        )
        return self

    def _channel_z(self, data: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
        self._check_fitted()
        if self.inner_ is None:
            raise RuntimeError("SpatialProfileBaseline has not been calibrated")
        z = self.inner_.band_zscores(data, sampling_rate_hz=sampling_rate_hz)
        return z.mean(axis=2)

    def channel_scores(
        self, data: np.ndarray, sampling_rate_hz: float = 1000.0
    ) -> np.ndarray:
        """Per-channel evidence: weighted contribution inside the best profile.

        This is what the evaluation harness aggregates, so the spatial prior is
        expressed here rather than only in :meth:`score`.
        """

        channel_z = self._channel_z(data, float(sampling_rate_hz))
        return self._contributions(channel_z).astype(np.float32)

    def score(self, data: np.ndarray, sampling_rate_hz: float = 1000.0) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("SpatialProfileBaseline has not been calibrated")
        channel_z = self._channel_z(data, float(sampling_rate_hz))
        raw = self._profile_max(channel_z)
        return ((raw - self.center_) / self.scale_).astype(np.float32)

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "SpatialProfileBaseline is not a reconstruction model; use channel_scores"
        )

"""Plotting helpers for DAS waterfalls, spectra and anomaly profiles.

All functions return Matplotlib ``(figure, axes)`` objects and never call
``show()``.  This makes them usable both in Jupyter notebooks (where the last
figure is displayed automatically) and in headless batch scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _new_axes(
    ax: Axes | None,
    figsize: tuple[float, float] = (10.0, 5.0),
) -> tuple[Figure, Axes]:
    if ax is None:
        figure, axes = plt.subplots(figsize=figsize)
    else:
        axes = ax
        figure = axes.figure
    return figure, axes


def _robust_limits(
    data: np.ndarray,
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[float, float]:
    finite = np.asarray(data, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = np.percentile(finite, percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        centre = float(np.median(finite))
        return centre - 1.0, centre + 1.0
    return float(low), float(high)


def plot_waterfall(
    recording: np.ndarray,
    sampling_rate_hz: float,
    channel_spacing_m: float = 1.0,
    ax: Axes | None = None,
    to_db: bool = True,
    cmap: str = "seismic",
    robust_percentiles: tuple[float, float] = (1.0, 99.0),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a ``(channels, samples)`` DAS recording as a waterfall image.

    Time is on the horizontal axis and distance along the fibre on the vertical
    axis.  Colours are clipped using robust percentiles to prevent a few loud
    samples from flattening the display.
    """

    data = np.asarray(recording, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("recording must have shape (channels, samples)")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")

    if to_db:
        data = 20.0 * np.log10(np.maximum(np.abs(data), 1e-12))
        color_label = "Amplitude (dB)"
    else:
        color_label = "Amplitude"

    low, high = _robust_limits(data, robust_percentiles)
    duration = data.shape[1] / float(sampling_rate_hz)
    length_m = max(data.shape[0] - 1, 1) * float(channel_spacing_m)

    figure, axes = _new_axes(ax)
    image = axes.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=(0.0, duration, 0.0, length_m),
        cmap=cmap,
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Distance along fibre (m)")
    axes.set_title(title or "DAS waterfall")
    figure.colorbar(image, ax=axes, label=color_label)
    figure.tight_layout()
    return figure, axes


def plot_channel_waveform(
    recording: np.ndarray,
    channel: int,
    sampling_rate_hz: float,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot one channel's raw waveform against time."""

    data = np.asarray(recording)
    if data.ndim != 2 or not 0 <= channel < data.shape[0]:
        raise ValueError("channel is out of range for a (channels, samples) array")
    time = np.arange(data.shape[1], dtype=np.float64) / float(sampling_rate_hz)
    figure, axes = _new_axes(ax, figsize=(10.0, 3.5))
    axes.plot(time, data[channel], linewidth=0.8, color="#1f77b4")
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Amplitude")
    axes.set_title(title or f"DAS channel {channel}")
    axes.grid(True, alpha=0.25)
    figure.tight_layout()
    return figure, axes


def plot_heatmap(
    matrix: np.ndarray,
    ax: Axes | None = None,
    x_values: Sequence[float] | None = None,
    y_values: Sequence[float] | None = None,
    cmap: str = "viridis",
    xlabel: str = "",
    ylabel: str = "",
    title: str | None = None,
    colorbar_label: str = "",
    robust_percentiles: tuple[float, float] | None = (1.0, 99.0),
) -> tuple[Figure, Axes]:
    """Generic heatmap for STFTs, residual maps or feature matrices."""

    data = np.asarray(matrix, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    figure, axes = _new_axes(ax)
    kwargs: dict[str, Any] = {"aspect": "auto", "origin": "lower", "cmap": cmap}
    if robust_percentiles is not None:
        kwargs["vmin"], kwargs["vmax"] = _robust_limits(data, robust_percentiles)
    if x_values is not None or y_values is not None:
        x_extent = (
            (float(x_values[0]), float(x_values[-1]))
            if x_values is not None
            else (0.0, float(data.shape[1] - 1))
        )
        y_extent = (
            (float(y_values[0]), float(y_values[-1]))
            if y_values is not None
            else (0.0, float(data.shape[0] - 1))
        )
        kwargs["extent"] = (*x_extent, *y_extent)
    image = axes.imshow(data, **kwargs)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title or "Heatmap")
    figure.colorbar(image, ax=axes, label=colorbar_label)
    figure.tight_layout()
    return figure, axes


def plot_spectrogram(
    frequencies: np.ndarray,
    times: np.ndarray,
    magnitude: np.ndarray,
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = "magma",
    colorbar_label: str = "Magnitude (dB)",
) -> tuple[Figure, Axes]:
    """Plot a single-channel STFT magnitude with physical axes."""

    figure, axes = _new_axes(ax)
    image = axes.pcolormesh(
        np.asarray(times),
        np.asarray(frequencies),
        np.asarray(magnitude),
        shading="auto",
        cmap=cmap,
    )
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Frequency (Hz)")
    axes.set_title(title or "STFT spectrogram")
    figure.colorbar(image, ax=axes, label=colorbar_label)
    figure.tight_layout()
    return figure, axes


def plot_score_profile(
    scores: np.ndarray,
    positions_m: Sequence[float] | np.ndarray | None = None,
    threshold: float | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    label: str = "Anomaly score",
) -> tuple[Figure, Axes]:
    """Plot a channel-wise score profile and optional detection threshold."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if positions_m is None:
        positions = np.arange(values.size, dtype=np.float64)
        xlabel = "Channel"
    else:
        positions = np.asarray(positions_m, dtype=np.float64).reshape(-1)
        if positions.size != values.size:
            raise ValueError("positions_m and scores must have the same length")
        xlabel = "Distance along fibre (m)"

    figure, axes = _new_axes(ax, figsize=(10.0, 4.0))
    axes.plot(positions, values, linewidth=1.1, color="#d62728", label=label)
    if threshold is not None:
        axes.axhline(
            float(threshold), color="black", linestyle="--", linewidth=1.0, label="Threshold"
        )
    axes.set_xlabel(xlabel)
    axes.set_ylabel("Score")
    axes.set_title(title or "Spatial anomaly profile")
    axes.grid(True, alpha=0.25)
    axes.legend(loc="best")
    figure.tight_layout()
    return figure, axes


def plot_sample_scores(
    scores: np.ndarray,
    labels: Sequence[int] | np.ndarray | None = None,
    threshold: float | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot pass-level scores, optionally coloured by ground-truth labels."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    indices = np.arange(values.size)
    figure, axes = _new_axes(ax, figsize=(10.0, 4.0))
    if labels is None:
        axes.plot(indices, values, marker="o", markersize=3, linewidth=1.0)
    else:
        truth = np.asarray(labels).reshape(-1).astype(int)
        if truth.size != values.size:
            raise ValueError("labels and scores must have the same length")
        for value, marker in ((0, "o"), (1, "x")):
            mask = truth == value
            axes.scatter(
                indices[mask],
                values[mask],
                marker=marker,
                s=24,
                label="Anomalous" if value else "Normal",
            )
        axes.legend(loc="best")
    if threshold is not None:
        axes.axhline(float(threshold), color="black", linestyle="--", linewidth=1.0)
    axes.set_xlabel("Pass index")
    axes.set_ylabel("Anomaly score")
    axes.set_title(title or "Pass-level anomaly scores")
    axes.grid(True, alpha=0.25)
    figure.tight_layout()
    return figure, axes


def plot_training_curves(
    history: Mapping[str, Sequence[float]] | Any,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot training and validation losses from a model history object."""

    if isinstance(history, Mapping):
        values = history
    else:
        values = getattr(history, "training_history_", history)
    figure, axes = _new_axes(ax, figsize=(8.0, 4.0))
    for key in ("train_loss", "val_loss"):
        curve = values.get(key) if isinstance(values, Mapping) else None
        if curve:
            axes.plot(np.arange(1, len(curve) + 1), curve, label=key)
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.set_title(title or "Training history")
    axes.grid(True, alpha=0.25)
    axes.legend(loc="best")
    figure.tight_layout()
    return figure, axes


def save_figure(
    figure: Figure,
    path: str | Path,
    dpi: int = 150,
    close: bool = True,
) -> Path:
    """Save a figure, creating parent directories when needed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(figure)
    return destination

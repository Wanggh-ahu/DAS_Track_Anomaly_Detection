"""Temporal and spatial filters for DAS matrices."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, sosfilt, sosfiltfilt

LOGGER = logging.getLogger(__name__)


def _as_float_array(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    if array.size == 0:
        raise ValueError("data must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("data contains NaN or infinite values")
    return array


def bandpass_filter(
    data: np.ndarray,
    sampling_rate_hz: float,
    low_hz: float | None = 5.0,
    high_hz: float | None = 200.0,
    order: int = 4,
    axis: int = -1,
    zero_phase: bool = True,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter along one axis.

    Parameters
    ----------
    data:
        Input array, typically ``(channels, samples)``.
    sampling_rate_hz:
        Sampling frequency along ``axis``.
    low_hz, high_hz:
        Pass-band edges.  ``low_hz=None``/``0`` selects a low-pass filter;
        ``high_hz=None`` selects a high-pass filter.
    order:
        Butterworth order.
    axis:
        Axis containing time samples.
    zero_phase:
        Use forward-backward filtering (``sosfiltfilt``) when enough samples are
        available.  Falls back to a causal filter for very short segments.

    Notes
    -----
    The filter is computed in second-order sections for numerical stability.
    The output dtype follows the input for floating point inputs.
    """

    array = _as_float_array(data)
    fs = float(sampling_rate_hz)
    if fs <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if order < 1:
        raise ValueError("order must be >= 1")

    nyquist = 0.5 * fs
    high = None if high_hz is None else float(high_hz)
    low = None if low_hz is None or float(low_hz) <= 0 else float(low_hz)
    if high is not None and high >= nyquist:
        high = None
    if high is not None and low is not None and low >= high:
        raise ValueError(f"low_hz ({low}) must be smaller than high_hz ({high})")
    if low is None and high is None:
        return array.copy()

    if low is None:
        sos = butter(order, high / nyquist, btype="lowpass", output="sos")
    elif high is None:
        sos = butter(order, low / nyquist, btype="highpass", output="sos")
    else:
        sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")

    # sosfiltfilt requires the input to be longer than its internal padding.
    padlen = 3 * (2 * len(sos) + 1)
    n_samples = array.shape[axis]
    if zero_phase and n_samples > padlen:
        filtered = sosfiltfilt(sos, array, axis=axis)
    else:
        if zero_phase:
            LOGGER.warning(
                "Segment along axis %d has %d samples (padlen=%d); using causal filtering",
                axis,
                n_samples,
                padlen,
            )
        filtered = sosfilt(sos, array, axis=axis)
    if np.issubdtype(np.asarray(data).dtype, np.floating):
        filtered = filtered.astype(np.asarray(data).dtype, copy=False)
    return np.ascontiguousarray(filtered)


def spatial_common_mode_removal(
    data: np.ndarray,
    window: int = 5,
    axis: int = 0,
    method: Literal["median", "mean"] = "median",
    mode: Literal["nearest", "reflect", "constant"] = "nearest",
) -> np.ndarray:
    """Remove spatially coherent noise shared by neighbouring DAS channels.

    DAS interrogators often imprint the same laser/clock noise on many adjacent
    channels.  A running spatial median is a robust estimate of that common
    component; subtracting it preserves local transient events better than a
    global mean.  ``window=1`` disables the operation.

    Parameters
    ----------
    data:
        Array whose ``axis`` enumerates channels (default axis 0).
    window:
        Number of adjacent channels in the running statistic.  The project
        configuration calls this ``Cw``.
    method:
        ``median`` (recommended) or ``mean``.
    mode:
        Boundary handling forwarded to SciPy.
    """

    array = _as_float_array(data)
    if window is None or int(window) <= 1:
        return array.copy()
    window = int(window)
    if not -array.ndim <= axis < array.ndim:
        raise ValueError(f"axis {axis} is out of bounds for an array with {array.ndim} dims")
    axis = axis % array.ndim
    if window > array.shape[axis] + 2:
        LOGGER.warning(
            "Spatial window %d is larger than channel count %d; using all channels",
            window,
            array.shape[axis],
        )

    if method == "median":
        size = [1] * array.ndim
        size[axis] = window
        common = median_filter(array, size=size, mode=mode)
    elif method == "mean":
        common = uniform_filter1d(
            array, size=window, axis=axis, mode=mode
        )
    else:
        raise ValueError("method must be 'median' or 'mean'")

    filtered = array - common
    if np.issubdtype(np.asarray(data).dtype, np.floating):
        filtered = filtered.astype(np.asarray(data).dtype, copy=False)
    return np.ascontiguousarray(filtered)


def preprocess_recording(
    recording: np.ndarray,
    sampling_rate_hz: float,
    bandpass_hz: tuple[float, float] | list[float] | None = (5.0, 200.0),
    order: int = 4,
    common_mode_window: int = 5,
    spatial_axis: int = 0,
) -> np.ndarray:
    """Convenience pipeline: temporal band-pass followed by spatial median.

    Input and output have shape ``(channels, samples)``.  Keeping the order
    explicit matters: temporal filtering first removes low-frequency drift,
    after which the spatial common-mode estimate is not dominated by slow
    quasi-static loading.
    """

    array = _as_float_array(recording)
    if array.ndim != 2:
        raise ValueError("recording must have shape (channels, samples)")
    if bandpass_hz is not None:
        low, high = (list(bandpass_hz) + [None, None])[:2]
        array = bandpass_filter(
            array,
            sampling_rate_hz=sampling_rate_hz,
            low_hz=low,
            high_hz=high,
            order=order,
            axis=-1,
        )
    return spatial_common_mode_removal(
        array, window=common_mode_window, axis=spatial_axis
    )

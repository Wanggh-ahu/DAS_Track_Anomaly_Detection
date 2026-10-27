"""Feature extractors for DAS waveforms and time-frequency representations."""

from __future__ import annotations

import logging

import numpy as np
import pywt
from scipy.signal import stft

LOGGER = logging.getLogger(__name__)


def _as_recording(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"expected (channels, samples), received shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("recording must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("recording contains NaN or infinite values")
    return array.astype(np.float64, copy=False)


def rms_feature(
    recording: np.ndarray,
    frame_length: int = 128,
    hop_length: int | None = None,
) -> np.ndarray:
    """Compute a frame-wise root-mean-square envelope per channel.

    Returns
    -------
    numpy.ndarray
        Shape ``(channels, n_frames)``.  Frames are fully contained in the
        recording; short recordings are edge-padded to one frame.
    """

    array = _as_recording(recording)
    frame_length = int(frame_length)
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    hop_length = int(hop_length or max(frame_length // 2, 1))
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")

    n_samples = array.shape[1]
    if n_samples < frame_length:
        pad = frame_length - n_samples
        array = np.pad(array, ((0, 0), (0, pad)), mode="reflect")
        n_samples = frame_length

    starts = np.arange(0, n_samples - frame_length + 1, hop_length, dtype=int)
    cumulative = np.concatenate(
        [np.zeros((array.shape[0], 1)), np.cumsum(np.square(array), axis=1)], axis=1
    )
    energy = (
        cumulative[:, starts + frame_length] - cumulative[:, starts]
    ) / float(frame_length)
    return np.sqrt(np.maximum(energy, 0.0)).astype(np.float32)


def stft_feature(
    recording: np.ndarray,
    sampling_rate_hz: float,
    nperseg: int = 128,
    noverlap: int | None = None,
    window: str = "hann",
    to_db: bool = True,
    channel: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a short-time Fourier transform for every DAS channel.

    Parameters
    ----------
    recording:
        ``(channels, samples)`` time series.
    sampling_rate_hz:
        Sampling frequency.
    nperseg:
        STFT window length in samples.  When longer than the recording it is
        clipped and a zero-padded transform is used.
    noverlap:
        Overlap in samples; defaults to ``nperseg // 2``.
    to_db:
        Convert magnitude to ``20 * log10(magnitude + eps)``.
    channel:
        Optional single channel index.  When supplied the returned magnitude has
        shape ``(frequencies, frames)`` instead of ``(channels, frequencies,
        frames)``.

    Returns
    -------
    frequencies, times, magnitude:
        Frequency axis in hertz, frame times in seconds, and the (optionally
        decibel) STFT magnitude.
    """

    array = _as_recording(recording)
    if channel is not None:
        if not 0 <= channel < array.shape[0]:
            raise IndexError(f"channel {channel} out of range")
        array = array[channel : channel + 1]
    n_samples = array.shape[1]
    nperseg = max(8, min(int(nperseg), n_samples))
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")

    frequencies, times, spectrum = stft(
        array,
        fs=float(sampling_rate_hz),
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
        axis=-1,
    )
    magnitude = np.abs(spectrum).astype(np.float32)
    if to_db:
        magnitude = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    if channel is not None:
        magnitude = magnitude[0]
    return frequencies.astype(np.float32), times.astype(np.float32), magnitude


def wavelet_packet_features(
    recording: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    normalize: bool = False,
) -> np.ndarray:
    """Extract wavelet-packet band energies for each channel.

    A full binary wavelet-packet decomposition is used.  Every leaf at ``level``
    becomes one feature containing the sum of squared coefficients in that
    frequency band.

    Returns
    -------
    numpy.ndarray
        Shape ``(channels, 2 ** level_used)``.  The level is automatically
        reduced for short signals to satisfy the wavelet boundary condition.
    """

    array = _as_recording(recording) if np.asarray(recording).ndim == 2 else np.asarray(recording)[None, :]
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    if level < 0:
        raise ValueError("level must be non-negative")

    try:
        wavelet_obj = pywt.Wavelet(wavelet)
    except ValueError as exc:  # pragma: no cover - depends on pywt version
        raise ValueError(f"Unknown wavelet {wavelet!r}") from exc

    max_level = pywt.dwt_max_level(array.shape[1], wavelet_obj.dec_len)
    used_level = int(min(level, max_level))
    if used_level < level:
        LOGGER.warning(
            "Wavelet level reduced from %d to %d for %d-sample segments",
            level,
            used_level,
            array.shape[1],
        )
    # Very short segments need a minimum length for the packet transform.
    minimum_length = max(wavelet_obj.dec_len, 2 ** used_level)
    if array.shape[1] < minimum_length:
        array = np.pad(
            array, ((0, 0), (0, minimum_length - array.shape[1])), mode="reflect"
        )

    if used_level == 0:
        return np.sum(np.square(array), axis=1, keepdims=True).astype(np.float32)

    features = np.empty((array.shape[0], 2**used_level), dtype=np.float64)
    for channel in range(array.shape[0]):
        packet = pywt.WaveletPacket(
            data=array[channel], wavelet=wavelet_obj, mode="symmetric", maxlevel=used_level
        )
        nodes = packet.get_level(used_level, order="freq")
        energies = np.asarray(
            [np.sum(np.square(node.data), dtype=np.float64) for node in nodes]
        )
        if normalize:
            total = energies.sum()
            energies = energies / total if total > 0 else energies
        features[channel] = energies
    return features.astype(np.float32)


def extract_feature_matrix(
    recording: np.ndarray,
    sampling_rate_hz: float,
    frame_length: int = 128,
    hop_length: int | None = None,
    wavelet: str = "db4",
    wavelet_level: int = 3,
    n_spectral_bins: int = 16,
    return_names: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """Summarise each channel with a compact hand-crafted feature vector.

    The matrix combines RMS statistics, spectral band energies, spectral
    centroid/bandwidth, wavelet-packet energies, kurtosis and zero-crossing
    rate.  It is primarily intended for the statistical baselines and for
    quick notebook inspection.

    Returns
    -------
    features:
        Shape ``(channels, n_features)``.
    names:
        Optional feature names when ``return_names=True``.
    """

    array = _as_recording(recording)
    hop_length = int(hop_length or max(frame_length // 2, 1))
    rms = rms_feature(array, frame_length=frame_length, hop_length=hop_length)
    names: list[str] = [
        "rms_mean",
        "rms_std",
        "rms_max",
        "rms_p90",
        "kurtosis",
        "zero_crossing_rate",
    ]
    blocks: list[np.ndarray] = [
        rms.mean(axis=1),
        rms.std(axis=1),
        rms.max(axis=1),
        np.percentile(rms, 90, axis=1),
    ]

    # Distribution shape of the raw waveform.
    centered = array - array.mean(axis=1, keepdims=True)
    scale = np.maximum(centered.std(axis=1), 1e-12)
    kurtosis = np.mean(np.power(centered / scale[:, None], 4), axis=1) - 3.0
    zero_crossing = np.mean(
        np.abs(np.diff(np.signbit(centered).astype(np.int8), axis=1)), axis=1
    )
    blocks.extend([kurtosis, zero_crossing])

    # Spectral descriptors from the time-averaged STFT power spectrum.
    frequencies, _, magnitude = stft_feature(
        array,
        sampling_rate_hz=sampling_rate_hz,
        nperseg=min(128, array.shape[1]),
        noverlap=None,
        to_db=False,
    )
    power = np.square(magnitude).mean(axis=-1)  # (channels, frequencies)
    power_sum = np.maximum(power.sum(axis=1), 1e-12)
    centroid = (power * frequencies[None, :]).sum(axis=1) / power_sum
    bandwidth = np.sqrt(
        (power * np.square(frequencies[None, :] - centroid[:, None])).sum(axis=1)
        / power_sum
    )
    edges = np.linspace(0, power.shape[1], int(n_spectral_bins) + 1, dtype=int)
    band_energy = np.stack(
        [
            power[:, edges[i] : max(edges[i] + 1, edges[i + 1])].mean(axis=1)
            for i in range(int(n_spectral_bins))
        ],
        axis=1,
    )
    band_energy = np.log10(np.maximum(band_energy, 1e-12))
    names.extend([f"band_energy_{i:02d}" for i in range(int(n_spectral_bins))])
    names.extend(["spectral_centroid_hz", "spectral_bandwidth_hz"])
    blocks.extend([band_energy, centroid[:, None], bandwidth[:, None]])

    # Wavelet packet energies, normalised per channel so that loud passes do
    # not dominate the shape descriptor.
    wavelet_energy = wavelet_packet_features(
        array, wavelet=wavelet, level=wavelet_level, normalize=True
    )
    names.extend([f"wavelet_packet_{i:02d}" for i in range(wavelet_energy.shape[1])])
    blocks.append(wavelet_energy)

    features = np.concatenate(
        [
            np.asarray(block, dtype=np.float32).reshape(array.shape[0], -1)
            for block in blocks
        ],
        axis=1,
    )
    if return_names:
        return features, names
    return features

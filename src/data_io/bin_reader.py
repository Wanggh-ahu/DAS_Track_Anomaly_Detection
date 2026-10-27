"""Read DAS recordings stored in HDF5 containers.

Raw DAS files from different interrogators use inconsistent layouts.  This
module keeps the rest of the project independent of those details by exposing
one canonical representation:

``recording.shape == (n_channels, n_samples)`` and ``recording.dtype`` is a
floating point type.  The reader supports both channel-major and time-major
arrays, auto-detects the largest two-dimensional dataset, and can stream a
recording in bounded chunks for files that do not fit into RAM.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import h5py
import numpy as np

LOGGER = logging.getLogger(__name__)

Orientation = Literal["auto", "channels_time", "time_channels"]

# Common attribute names used by DAS vendors.  Matching is case-insensitive.
_SAMPLING_KEYS = (
    "sampling_rate",
    "sampling_rate_hz",
    "sample_rate",
    "sample_rate_hz",
    "fs",
    "f_s",
    "frequency",
)
_DT_KEYS = ("dt", "sample_interval", "sampling_interval", "time_step")
_SPACING_KEYS = (
    "channel_spacing",
    "channel_spacing_m",
    "spatial_spacing",
    "spatial_sampling",
    "dx",
    "gauge_length",
)
_START_KEYS = ("start_time", "starttime", "timestamp", "acquisition_start", "time_origin")
_CHANNEL_COUNT_KEYS = ("n_channels", "num_channels", "channel_count", "channels")
_SAMPLE_COUNT_KEYS = ("n_samples", "num_samples", "sample_count", "samples", "nt")


@dataclass
class DASMetadata:
    """Metadata required to interpret a DAS matrix.

    Attributes
    ----------
    source_path:
        Absolute path of the HDF5 file.
    dataset_key:
        HDF5 path to the two-dimensional DAS dataset.
    n_channels, n_samples:
        Dimensions after orientation correction.
    sampling_rate_hz:
        Temporal sampling frequency in hertz.
    channel_spacing_m:
        Distance between adjacent fibre channels in metres.
    start_time:
        Optional acquisition start timestamp (kept as a string).
    dtype:
        Storage dtype of the source dataset.
    attrs:
        Flattened, JSON-friendly HDF5 attributes from the file and dataset.
    """

    source_path: str
    dataset_key: str
    n_channels: int
    n_samples: int
    sampling_rate_hz: float = 1000.0
    channel_spacing_m: float = 1.0
    start_time: str | None = None
    dtype: str = "float32"
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Recording duration in seconds."""

        return self.n_samples / max(float(self.sampling_rate_hz), 1e-12)

    @property
    def fibre_length_m(self) -> float:
        """Approximate instrumented length in metres."""

        return max(self.n_channels - 1, 0) * float(self.channel_spacing_m)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation."""

        return asdict(self)


def _attr_value(value: Any) -> Any:
    """Convert HDF5 attribute values into plain Python/JSON-friendly values."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collect_attrs(*objects: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    """Merge attributes from several HDF5 objects, later objects taking priority."""

    attrs: dict[str, Any] = {}
    for obj in objects:
        for key, value in obj.attrs.items():
            attrs[str(key)] = _attr_value(value)
    return attrs


def _find_attr(attrs: dict[str, Any], candidates: tuple[str, ...]) -> Any | None:
    """Case-insensitive lookup of the first matching attribute."""

    lower_map = {str(key).lower(): value for key, value in attrs.items()}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    return None


def _as_float(value: Any) -> float | None:
    """Best-effort scalar conversion used for metadata attributes."""

    try:
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size == 1 and np.isfinite(array[0]):
            return float(array[0])
    except (TypeError, ValueError):
        return None
    return None


def _choose_dataset(
    h5_file: h5py.File, dataset_key: str | None = None
) -> tuple[str, h5py.Dataset]:
    """Locate the DAS matrix inside an HDF5 file.

    If ``dataset_key`` is supplied it is used directly.  Otherwise all numeric
    two-dimensional datasets are scanned and the one with the largest number of
    elements is selected.  This is a pragmatic convention that works for the
    usual ``/data``, ``/raw/data`` and vendor-specific layouts.
    """

    if dataset_key:
        key = dataset_key if dataset_key.startswith("/") else f"/{dataset_key}"
        if key not in h5_file:
            raise KeyError(f"Dataset {key!r} does not exist in {h5_file.filename!r}")
        dataset = h5_file[key]
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 2:
            raise ValueError(f"Dataset {key!r} must be a 2-D HDF5 dataset")
        return key, dataset

    candidates: list[tuple[str, h5py.Dataset, int]] = []

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if not isinstance(obj, h5py.Dataset) or obj.ndim != 2:
            return
        if obj.size == 0 or obj.dtype.kind not in "fiu":
            return
        # Prefer descriptive dataset names when sizes are close, but always let
        # the element count dominate the decision.
        score = int(obj.size)
        lower_name = name.lower()
        if any(token in lower_name for token in ("das", "data", "raw", "signal")):
            score = int(score * 1.01)
        candidates.append((f"/{name.lstrip('/')}", obj, score))

    h5_file.visititems(visitor)
    if not candidates:
        raise ValueError(
            f"No numeric 2-D dataset found in {h5_file.filename!r}; "
            "please provide an explicit dataset_key."
        )
    key, dataset, _ = max(candidates, key=lambda item: item[2])
    return key, dataset


def _infer_orientation(
    shape: tuple[int, int],
    attrs: dict[str, Any],
    orientation: Orientation = "auto",
) -> Literal["channels_time", "time_channels"]:
    """Infer whether the array axes are ``(channels, time)`` or ``(time, channels)``."""

    if orientation == "channels_time":
        return orientation
    if orientation == "time_channels":
        return orientation
    if orientation != "auto":
        raise ValueError(
            "orientation must be one of 'auto', 'channels_time', 'time_channels'"
        )

    n_channels = _as_float(_find_attr(attrs, _CHANNEL_COUNT_KEYS))
    n_samples = _as_float(_find_attr(attrs, _SAMPLE_COUNT_KEYS))
    first, second = shape

    if n_channels is not None:
        if int(n_channels) == first:
            return "channels_time"
        if int(n_channels) == second:
            return "time_channels"
    if n_samples is not None:
        if int(n_samples) == second:
            return "channels_time"
        if int(n_samples) == first:
            return "time_channels"

    # DAS acquisitions normally contain far more time samples than channels.
    if first < second:
        return "channels_time"
    if second < first:
        return "time_channels"
    LOGGER.warning(
        "HDF5 dataset is square (%d x %d); assuming channel-major layout", first, second
    )
    return "channels_time"


class HDF5DASReader:
    """Random-access reader for a single HDF5 DAS recording.

    Parameters
    ----------
    path:
        HDF5 file path.
    dataset_key:
        Optional explicit path to the 2-D dataset.
    orientation:
        ``channels_time``, ``time_channels`` or ``auto``.
    sampling_rate_hz, channel_spacing_m:
        Fallbacks used when the file does not carry acquisition metadata.
    """

    def __init__(
        self,
        path: str | Path,
        dataset_key: str | None = None,
        orientation: Orientation = "auto",
        sampling_rate_hz: float = 1000.0,
        channel_spacing_m: float = 1.0,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"DAS file not found: {self.path}")

        self.dataset_key = dataset_key
        self.orientation = orientation
        self._fallback_sampling_rate = float(sampling_rate_hz)
        self._fallback_spacing = float(channel_spacing_m)
        self._layout: Literal["channels_time", "time_channels"] = "channels_time"
        self.metadata = self._read_metadata()

    # ------------------------------------------------------------------ metadata
    def _read_metadata(self) -> DASMetadata:
        with h5py.File(self.path, "r") as h5_file:
            key, dataset = _choose_dataset(h5_file, self.dataset_key)
            attrs = _collect_attrs(h5_file, dataset)
            layout = _infer_orientation(dataset.shape, attrs, self.orientation)
            self._layout = layout

            if layout == "channels_time":
                n_channels, n_samples = dataset.shape
            else:
                n_samples, n_channels = dataset.shape

            sampling_rate = _as_float(_find_attr(attrs, _SAMPLING_KEYS))
            if sampling_rate is None:
                dt = _as_float(_find_attr(attrs, _DT_KEYS))
                sampling_rate = 1.0 / dt if dt and dt > 0 else self._fallback_sampling_rate
            spacing = _as_float(_find_attr(attrs, _SPACING_KEYS))
            if spacing is None or spacing <= 0:
                spacing = self._fallback_spacing
            start_time = _find_attr(attrs, _START_KEYS)
            start_time = str(start_time) if start_time is not None else None

            return DASMetadata(
                source_path=str(self.path),
                dataset_key=key,
                n_channels=int(n_channels),
                n_samples=int(n_samples),
                sampling_rate_hz=float(sampling_rate),
                channel_spacing_m=float(spacing),
                start_time=start_time,
                dtype=str(dataset.dtype),
                attrs=attrs,
            )

    # -------------------------------------------------------------------- access
    def _dataset_slice(
        self,
        dataset: h5py.Dataset,
        channel_slice: slice,
        time_slice: slice,
    ) -> np.ndarray:
        """Read a canonical ``(channels, samples)`` block from an HDF5 dataset."""

        if self._layout == "channels_time":
            block = dataset[channel_slice, time_slice]
        else:
            block = dataset[time_slice, channel_slice].T
        return np.asarray(block)

    def read(
        self,
        channel_slice: slice | None = None,
        time_slice: slice | None = None,
        dtype: str | np.dtype = "float32",
    ) -> np.ndarray:
        """Read a full or partial recording as ``(channels, samples)``."""

        channel_slice = channel_slice or slice(0, self.metadata.n_channels)
        time_slice = time_slice or slice(0, self.metadata.n_samples)
        with h5py.File(self.path, "r") as h5_file:
            dataset = h5_file[self.metadata.dataset_key]
            block = self._dataset_slice(dataset, channel_slice, time_slice)
        return np.asarray(block, dtype=dtype)

    def read_channels(self, start: int, stop: int, dtype: str = "float32") -> np.ndarray:
        """Read a channel interval; useful for spatial windows."""

        return self.read(channel_slice=slice(start, stop), dtype=dtype)

    def read_time_window(
        self, start_sample: int, end_sample: int, dtype: str = "float32"
    ) -> np.ndarray:
        """Read a sample interval across all channels."""

        return self.read(time_slice=slice(start_sample, end_sample), dtype=dtype)

    def iter_chunks(
        self,
        chunk_channels: int = 256,
        chunk_samples: int = 4096,
        dtype: str = "float32",
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield ``(channel_start, channel_stop, block)`` chunks.

        Channels are never split across chunks; time is traversed first so that
        the common case of a full fibre in one channel chunk remains cache
        friendly.
        """

        if chunk_channels <= 0 or chunk_samples <= 0:
            raise ValueError("chunk_channels and chunk_samples must be positive")

        for c0 in range(0, self.metadata.n_channels, chunk_channels):
            c1 = min(c0 + chunk_channels, self.metadata.n_channels)
            for t0 in range(0, self.metadata.n_samples, chunk_samples):
                t1 = min(t0 + chunk_samples, self.metadata.n_samples)
                yield c0, c1, self.read(slice(c0, c1), slice(t0, t1), dtype=dtype)

    def close(self) -> None:  # pragma: no cover - API compatibility
        """No persistent handle is held, so this is a no-op."""

    def __enter__(self) -> "HDF5DASReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def load_das_recording(
    path: str | Path,
    dataset_key: str | None = None,
    orientation: Orientation = "auto",
    sampling_rate_hz: float | None = None,
    channel_spacing_m: float | None = None,
    dtype: str = "float32",
) -> tuple[np.ndarray, DASMetadata]:
    """Convenience wrapper returning ``(recording, metadata)``.

    Metadata in the file takes precedence; explicit arguments are only fallbacks.
    """

    reader = HDF5DASReader(
        path=path,
        dataset_key=dataset_key,
        orientation=orientation,
        sampling_rate_hz=sampling_rate_hz or 1000.0,
        channel_spacing_m=channel_spacing_m or 1.0,
    )
    return reader.read(dtype=dtype), reader.metadata

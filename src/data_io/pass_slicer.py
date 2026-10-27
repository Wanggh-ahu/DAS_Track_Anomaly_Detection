"""Detect train-pass events and cut them into fixed-size model samples.

The detector intentionally works on a very simple robust short-time energy
envelope.  It is not tied to a particular axle signature, which makes it useful
as a first-stage gate for many DAS installations.  More advanced detectors can
be substituted without changing the downstream ``(channels, samples)`` data
contract.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.ndimage import median_filter

LOGGER = logging.getLogger(__name__)


@dataclass
class PassEvent:
    """A contiguous high-energy interval associated with one train pass."""

    start_sample: int
    end_sample: int
    peak_sample: int
    score: float
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    window_start_sample: int | None = None
    window_end_sample: int | None = None
    window_samples: int | None = None
    anchor: str | None = None

    @property
    def n_samples(self) -> int:
        return max(self.end_sample - self.start_sample, 0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PassEvent":
        return cls(**values)


class PassSlicer:
    """Short-time-energy pass detector and slicer.

    Parameters
    ----------
    sampling_rate_hz:
        Temporal sampling rate of the recording.
    frame_seconds:
        Length of the energy analysis frame.
    energy_percentile:
        Percentile of the energy envelope used as an additional threshold
        floor.  A value of 90 means that the quietest 90 % of frames can never
        trigger an event, which suppresses isolated noisy frames.
    threshold_sigma:
        Multiplier applied to a robust (median absolute deviation) estimate of
        the background envelope.
    min_duration_s, merge_gap_s, pad_s:
        Post-processing constraints applied to detected intervals.
    """

    def __init__(
        self,
        sampling_rate_hz: float,
        frame_seconds: float = 0.05,
        energy_percentile: float = 90.0,
        threshold_sigma: float = 2.5,
        min_duration_s: float = 0.3,
        merge_gap_s: float = 0.5,
        pad_s: float = 0.2,
    ) -> None:
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.frame_length = max(int(round(frame_seconds * self.sampling_rate_hz)), 8)
        self.hop_length = max(self.frame_length // 2, 1)
        self.energy_percentile = float(np.clip(energy_percentile, 0.0, 100.0))
        self.threshold_sigma = float(threshold_sigma)
        self.min_samples = max(int(round(min_duration_s * self.sampling_rate_hz)), 1)
        self.merge_gap = max(int(round(merge_gap_s * self.sampling_rate_hz)), 0)
        self.pad = max(int(round(pad_s * self.sampling_rate_hz)), 0)

    # ------------------------------------------------------------------ detection
    def energy_envelope(self, recording: np.ndarray) -> np.ndarray:
        """Compute a robust across-channel RMS envelope.

        ``recording`` must be ``(channels, samples)``.  The per-frame RMS is
        averaged across channels, then lightly median-filtered to reject single
        frame spikes.  The returned envelope is aligned with frame starts.
        """

        self._validate_recording(recording)
        n_samples = recording.shape[1]
        if n_samples < self.frame_length:
            return np.asarray(
                [float(np.sqrt(np.mean(np.square(recording, dtype=np.float64))))]
            )

        n_frames = 1 + (n_samples - self.frame_length) // self.hop_length
        energies = np.empty(n_frames, dtype=np.float64)
        for index in range(n_frames):
            start = index * self.hop_length
            frame = recording[:, start : start + self.frame_length]
            energies[index] = np.sqrt(
                np.mean(np.square(frame, dtype=np.float64), dtype=np.float64)
            )
        # A width of three removes isolated spikes while retaining onsets.
        return median_filter(energies, size=3, mode="nearest")

    def detect(self, recording: np.ndarray) -> list[PassEvent]:
        """Return non-overlapping pass events sorted by start sample."""

        self._validate_recording(recording)
        n_samples = recording.shape[1]
        if n_samples == 0:
            return []

        envelope = self.energy_envelope(recording)
        if envelope.size == 0 or not np.any(np.isfinite(envelope)):
            return []

        median = float(np.median(envelope))
        mad = float(np.median(np.abs(envelope - median)))
        robust_sigma = max(1.4826 * mad, 1e-12)
        percentile_floor = float(np.percentile(envelope, self.energy_percentile))
        threshold = max(
            median + self.threshold_sigma * robust_sigma,
            percentile_floor,
            np.finfo(np.float32).eps,
        )
        active = envelope > threshold
        if not np.any(active):
            return []

        # Convert contiguous active frames into sample intervals.
        intervals: list[list[int]] = []
        index = 0
        while index < active.size:
            if not active[index]:
                index += 1
                continue
            first = index
            while index + 1 < active.size and active[index + 1]:
                index += 1
            last = index
            start = first * self.hop_length
            end = min(last * self.hop_length + self.frame_length, n_samples)
            intervals.append([start, end])
            index += 1

        intervals = self._merge_intervals(intervals)
        events: list[PassEvent] = []
        for start, end in intervals:
            start = max(0, start - self.pad)
            end = min(n_samples, end + self.pad)
            if end - start < self.min_samples:
                continue
            frame_start = start // self.hop_length
            frame_end = min(
                envelope.size, max(frame_start + 1, end // self.hop_length + 1)
            )
            local = envelope[frame_start:frame_end]
            peak_offset = int(np.argmax(local)) * self.hop_length
            peak_sample = min(start + peak_offset, end - 1)
            score = float(local.max() / max(median, 1e-12))
            events.append(
                PassEvent(
                    start_sample=int(start),
                    end_sample=int(end),
                    peak_sample=int(peak_sample),
                    score=score,
                    duration_s=float((end - start) / self.sampling_rate_hz),
                    metadata={
                        "detector": "short_time_energy",
                        "threshold": threshold,
                        "frame_length": self.frame_length,
                        "hop_length": self.hop_length,
                    },
                )
            )
        return events

    def _merge_intervals(self, intervals: Sequence[Sequence[int]]) -> list[list[int]]:
        """Merge intervals whose quiet gap is shorter than ``merge_gap_s``."""

        if not intervals:
            return []
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start - merged[-1][1] <= self.merge_gap:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([int(start), int(end)])
        return merged

    # --------------------------------------------------------------------- slicing
    def slice_events(
        self,
        recording: np.ndarray,
        events: Sequence[PassEvent] | None = None,
        window_samples: int | None = None,
        anchor: str = "peak",
        dtype: str | np.dtype = "float32",
    ) -> tuple[np.ndarray, list[PassEvent]]:
        """Cut one recording into equally shaped pass samples.

        Parameters
        ----------
        recording:
            Array with shape ``(channels, samples)``.
        events:
            Events from :meth:`detect`; detected automatically when omitted.
        window_samples:
            Fixed output length.  When omitted the longest event length is used.
            All returned windows always share the same length, which is required
            for batched model training.
        anchor:
            ``peak`` centres the window on the event peak, ``start`` keeps the
            detected onset at the beginning of the window.
        """

        self._validate_recording(recording)
        events = list(events) if events is not None else self.detect(recording)
        if not events:
            return np.empty((0, recording.shape[0], 0), dtype=dtype), []

        if window_samples is None:
            window_samples = max(event.n_samples for event in events)
        window_samples = int(window_samples)
        if window_samples <= 0:
            raise ValueError("window_samples must be positive")
        if anchor not in {"peak", "start", "center"}:
            raise ValueError("anchor must be 'peak', 'start' or 'center'")

        windows = np.zeros(
            (len(events), recording.shape[0], window_samples), dtype=dtype
        )
        adjusted_events: list[PassEvent] = []
        for index, event in enumerate(events):
            if anchor == "start":
                start = event.start_sample
            elif anchor == "peak":
                start = event.peak_sample - window_samples // 2
            else:
                start = (event.start_sample + event.end_sample) // 2 - window_samples // 2
            windows[index] = self._extract_window(
                recording, start, window_samples, dtype=dtype
            )
            adjusted_events.append(
                self._window_event(event, start, window_samples, anchor)
            )
        return windows, adjusted_events

    def tile_events(
        self,
        recording: np.ndarray,
        events: Sequence[PassEvent] | None = None,
        window_samples: int = 128,
        hop_samples: int | None = None,
        dtype: str | np.dtype = "float32",
    ) -> tuple[np.ndarray, list[PassEvent]]:
        """Cut long pass events into overlapping fixed-length windows.

        One train pass can last several seconds, while a model may consume only a
        short time window.  Tiling produces multiple training samples per pass
        without changing the event labels.  Short events yield one window.

        Parameters
        ----------
        window_samples:
            Output length (``Tw`` in the project configuration).
        hop_samples:
            Window start increment; defaults to half the window length.
        """

        self._validate_recording(recording)
        events = list(events) if events is not None else self.detect(recording)
        window_samples = int(window_samples)
        if window_samples <= 0:
            raise ValueError("window_samples must be positive")
        hop_samples = int(hop_samples or max(window_samples // 2, 1))
        if hop_samples <= 0:
            raise ValueError("hop_samples must be positive")
        if not events:
            return np.empty((0, recording.shape[0], window_samples), dtype=dtype), []

        windows: list[np.ndarray] = []
        adjusted_events: list[PassEvent] = []
        n_samples = recording.shape[1]
        for event in events:
            if event.n_samples <= window_samples:
                start = max(
                    0,
                    min(
                        event.peak_sample - window_samples // 2,
                        max(n_samples - window_samples, 0),
                    ),
                )
                starts = [start]
            else:
                last_start = max(event.start_sample, event.end_sample - window_samples)
                starts = list(
                    range(event.start_sample, last_start + 1, hop_samples)
                )
                if starts[-1] != last_start:
                    starts.append(last_start)
            for start in starts:
                windows.append(
                    self._extract_window(recording, start, window_samples, dtype=dtype)
                )
                adjusted_events.append(
                    self._window_event(event, start, window_samples, "tile")
                )
        return np.stack(windows, axis=0), adjusted_events

    @staticmethod
    def _extract_window(
        recording: np.ndarray,
        start: int,
        window_samples: int,
        dtype: str | np.dtype = "float32",
    ) -> np.ndarray:
        """Zero-padded extraction of one fixed window."""

        output = np.zeros((recording.shape[0], window_samples), dtype=dtype)
        stop = start + window_samples
        src_start = max(start, 0)
        src_stop = min(stop, recording.shape[1])
        if src_stop > src_start:
            dst_start = src_start - start
            output[:, dst_start : dst_start + (src_stop - src_start)] = recording[
                :, src_start:src_stop
            ]
        return output

    @staticmethod
    def _window_event(
        event: PassEvent,
        start: int,
        window_samples: int,
        anchor: str,
    ) -> PassEvent:
        """Attach window coordinates to an existing event description."""

        payload = event.to_dict()
        payload.update(
            {
                "window_start_sample": int(start),
                "window_end_sample": int(start + window_samples),
                "window_samples": int(window_samples),
                "anchor": anchor,
            }
        )
        return PassEvent.from_dict(payload)

    def save_slices(
        self,
        output_dir: str | Path,
        windows: np.ndarray,
        events: Sequence[PassEvent],
        metadata: dict[str, Any] | None = None,
        file_prefix: str = "pass",
    ) -> list[Path]:
        """Write one ``.npz`` file per pass sample plus an event manifest.

        Each saved file contains ``data`` of shape ``(channels, samples)`` and a
        JSON string in ``event``.  Metadata is stored under ``metadata``.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(windows) != len(events):
            raise ValueError("windows and events must have the same length")

        saved: list[Path] = []
        manifest: list[dict[str, Any]] = []
        for index, (window, event) in enumerate(zip(windows, events)):
            path = output_dir / f"{file_prefix}_{index:05d}.npz"
            np.savez_compressed(
                path,
                data=np.asarray(window, dtype=np.float32),
                event=np.asarray(json.dumps(event.to_dict(), ensure_ascii=False)),
                metadata=np.asarray(json.dumps(metadata or {}, ensure_ascii=False)),
            )
            saved.append(path)
            manifest.append({"file": path.name, **event.to_dict()})
        (output_dir / f"{file_prefix}_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOGGER.info("Saved %d pass samples to %s", len(saved), output_dir)
        return saved

    @staticmethod
    def load_slice(path: str | Path) -> tuple[np.ndarray, PassEvent, dict[str, Any]]:
        """Load a file written by :meth:`save_slices`."""

        with np.load(Path(path), allow_pickle=False) as payload:
            data = np.asarray(payload["data"], dtype=np.float32)
            event = PassEvent.from_dict(json.loads(str(payload["event"])))
            metadata = json.loads(str(payload["metadata"]))
        return data, event, metadata

    @staticmethod
    def _validate_recording(recording: np.ndarray) -> None:
        array = np.asarray(recording)
        if array.ndim != 2:
            raise ValueError(
                "recording must have shape (channels, samples); "
                f"received {array.shape}"
            )
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("recording must not be empty")
        if not np.issubdtype(array.dtype, np.floating):
            # The caller may pass integer ADC counts; computations below cast
            # explicitly but a warning helps catch accidental misuse.
            LOGGER.debug("Recording dtype %s is not floating point", array.dtype)


def slice_dataset(
    recordings: Iterable[np.ndarray],
    slicer: PassSlicer,
    window_samples: int | None = None,
) -> tuple[np.ndarray, list[PassEvent]]:
    """Detect and slice several recordings, returning concatenated samples."""

    all_windows: list[np.ndarray] = []
    all_events: list[PassEvent] = []
    for recording in recordings:
        windows, events = slicer.slice_events(
            recording, window_samples=window_samples
        )
        if windows.size:
            all_windows.append(windows)
            all_events.extend(events)
    if not all_windows:
        return np.empty((0, 0, 0), dtype=np.float32), all_events
    # Fixed window length is guaranteed by slice_events, but channel counts may
    # differ between recordings; only concatenate compatible arrays.
    n_channels = all_windows[0].shape[1]
    compatible = [w for w in all_windows if w.shape[1] == n_channels]
    if len(compatible) != len(all_windows):
        LOGGER.warning("Dropping recordings with incompatible channel counts")
    return np.concatenate(compatible, axis=0), all_events

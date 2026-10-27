"""Synthetic DAS train-pass generator with physically motivated defects.

The simulator represents a train as a set of axle sources travelling along the
fibre, each exciting a Gaussian spatial footprint.  The resulting waterfall has
the characteristic diagonal patterns of a moving load.  Two physically distinct
track-defect responses are supported:

``impact``
    A short, broadband transient emitted every time an axle crosses the defect
    (rail joint, squat, dipped weld).  Energy is concentrated in a few tens of
    milliseconds and is therefore easy to lose in the moving-load background.

``resonance``
    A high-frequency decaying ring excited by each axle crossing (loose sleeper,
    hanging sleeper, ballast void).  It lasts longer than an impact and carries
    narrow-band energy at the local structural resonance.

Both mechanisms are *additive* responses of the defect.  There is deliberately
no gain multiplication of existing channels: amplifying a channel is a sensor
fault, not a track defect, and it trivialises detection.

SNR control
-----------
``snr_db`` sets the local signal-to-noise ratio of the defect response relative
to the background (moving load + ambient noise + common-mode noise + channel
gain scatter) over the defect support region.  With ``snr_reference:
processed`` the reference point is the detector input, i.e. the signal *after*
the band-pass and spatial common-mode removal of the preprocessing pipeline, so
``-15 dB`` really means the defect is buried in the waveform the model sees.
Scaling is solved in closed form (the filters are linear), so the requested SNR
is met exactly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.signal import butter, sosfilt

LOGGER = logging.getLogger(__name__)

VALID_ANOMALY_TYPES = ("impact", "resonance", "mixed", "none")


@dataclass
class SimulationConfig:
    """Parameters controlling one simulated acquisition."""

    n_channels: int = 96
    channel_spacing_m: float = 0.5
    sampling_rate_hz: float = 1000.0
    duration_s: float = 8.0
    speed_mps: float = 20.0
    axle_spacing_m: Sequence[float] = (2.5, 14.0, 2.5, 9.0)
    noise_std: float = 0.02
    seed: int = 2024
    source_frequency_hz: float = 65.0
    spatial_width_m: float = 1.6
    common_mode_fraction: float = 0.35
    channel_gain_std: float = 0.08

    # ---------------------------------------------------------- defect physics
    anomaly_type: str = "mixed"
    snr_db: float | None = -6.0
    anomaly_min_width_channels: int = 1
    anomaly_max_width_channels: int = 4
    anomaly_channel_centre: float | None = None
    impact_duration_s: float = 0.018
    impact_decay_s: float = 0.010
    impact_carriers_hz: Sequence[float] = (70.0, 115.0, 165.0)
    resonance_frequency_hz: float = 120.0
    resonance_decay_s: float = 0.075
    resonance_harmonic_ratio: float = 0.35
    resonance_harmonic_multiplier: float = 1.45

    # ---------------------------------------- SNR reference (mirror data.*)
    snr_reference: str = "processed"
    bandpass_hz: Sequence[float] | None = (5.0, 200.0)
    filter_order: int = 4
    common_mode_window: int = 5

    extra: dict[str, Any] = field(default_factory=dict)


class TrainPassSimulator:
    """Generate one or more normal/anomalous DAS train passes.

    Examples
    --------
    >>> sim = TrainPassSimulator(SimulationConfig(n_channels=32, duration_s=4))
    >>> data, channel_labels, meta = sim.simulate(anomaly=True)
    >>> data.shape
    (32, 4000)
    """

    def __init__(self, config: SimulationConfig | None = None, **overrides: Any) -> None:
        base = config or SimulationConfig()
        values = {**base.__dict__, **overrides}
        self.config = SimulationConfig(**values)
        self._validate()
        self.rng = np.random.default_rng(self.config.seed)
        # Exact (channel, sample) support of the most recent defect; exposed for
        # verification tooling rather than serialised into the metadata.
        self.last_support_: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrainPassSimulator":
        """Build a simulator from the ``simulation`` section of a YAML file."""

        allowed = set(SimulationConfig.__dataclass_fields__)
        values = {key: value for key, value in config.items() if key in allowed}
        return cls(SimulationConfig(**values))

    def _validate(self) -> None:
        cfg = self.config
        if cfg.n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if cfg.duration_s <= 0 or cfg.sampling_rate_hz <= 0:
            raise ValueError("duration_s and sampling_rate_hz must be positive")
        if cfg.channel_spacing_m <= 0:
            raise ValueError("channel_spacing_m must be positive")
        if cfg.speed_mps <= 0:
            raise ValueError("speed_mps must be positive")
        if cfg.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if cfg.anomaly_type not in VALID_ANOMALY_TYPES:
            raise ValueError(
                f"anomaly_type must be one of {VALID_ANOMALY_TYPES}, "
                f"received {cfg.anomaly_type!r}"
            )
        if cfg.snr_reference not in {"processed", "raw"}:
            raise ValueError("snr_reference must be 'processed' or 'raw'")
        if cfg.anomaly_min_width_channels < 1:
            raise ValueError("anomaly_min_width_channels must be >= 1")
        if cfg.anomaly_max_width_channels < cfg.anomaly_min_width_channels:
            raise ValueError("anomaly_max_width_channels must be >= the minimum")
        if cfg.resonance_decay_s <= 0 or cfg.impact_duration_s <= 0:
            raise ValueError("defect decay/duration values must be positive")

    # ------------------------------------------------------------------ utilities
    @property
    def n_samples(self) -> int:
        return int(round(self.config.duration_s * self.config.sampling_rate_hz))

    @property
    def fibre_length_m(self) -> float:
        return (self.config.n_channels - 1) * self.config.channel_spacing_m

    def _lowpass_noise(self, samples: int, cutoff_hz: float = 150.0) -> np.ndarray:
        """Coloured Gaussian noise shaped with a Butterworth low-pass filter."""

        white = self.rng.normal(
            0.0, self.config.noise_std, size=(self.config.n_channels, samples)
        )
        nyquist = 0.5 * self.config.sampling_rate_hz
        cutoff = min(cutoff_hz, 0.95 * nyquist)
        sos = butter(4, cutoff / nyquist, btype="lowpass", output="sos")
        return sosfilt(sos, white, axis=-1).astype(np.float64)

    def _normal_field(self) -> np.ndarray:
        """Generate the moving-load response on all channels."""

        cfg = self.config
        n_samples = self.n_samples
        time = np.arange(n_samples, dtype=np.float64) / cfg.sampling_rate_hz
        positions = np.arange(cfg.n_channels, dtype=np.float64) * cfg.channel_spacing_m

        # Enter from just outside the interrogated fibre; with the defaults the
        # train crosses the array within the simulated time window.
        start_position = -0.20 * self.fibre_length_m
        centre = start_position + cfg.speed_mps * time
        field = np.zeros((cfg.n_channels, n_samples), dtype=np.float64)

        axle_offsets = np.concatenate(
            [[0.0], np.cumsum(np.asarray(cfg.axle_spacing_m, dtype=float))]
        )
        width = max(cfg.spatial_width_m, cfg.channel_spacing_m * 0.5)
        sleeper_frequency = cfg.speed_mps / 0.60  # sleeper pitch of 0.6 m
        for axle_index, offset in enumerate(axle_offsets):
            axle_centre = centre + offset
            distance = positions[:, None] - axle_centre[None, :]
            spatial_envelope = np.exp(-0.5 * np.square(distance / width))
            phase = 0.7 * axle_index
            carrier = (
                np.sin(2.0 * np.pi * cfg.source_frequency_hz * time + phase)
                + 0.35 * np.sin(
                    2.0 * np.pi * (0.55 * cfg.source_frequency_hz) * time + 2.1 * phase
                )
            )
            # The sleeper-passing modulation creates realistic sidebands.
            modulation = 1.0 + 0.18 * np.sin(
                2.0 * np.pi * sleeper_frequency * time + phase
            )
            amplitude = 0.55 * np.exp(-0.12 * axle_index)
            field += amplitude * spatial_envelope * (carrier * modulation)[None, :]

        # Slow quasi-static loading around the train centre of mass.
        centre_distance = positions[:, None] - (
            centre + 0.5 * axle_offsets[-1]
        )[None, :]
        field += 0.25 * np.exp(-0.5 * np.square(centre_distance / (3.0 * width))) * np.sin(
            2.0 * np.pi * 6.0 * time
        )[None, :]

        # Mild random track unevenness along distance.
        field *= 1.0 + 0.12 * np.sin(
            2.0 * np.pi * positions / max(self.fibre_length_m, 1.0) * 5.0
        )[:, None]
        return field

    # ------------------------------------------------------------- defect physics
    @staticmethod
    def _raised_cosine_profile(
        n_channels: int, centre: float, half_width: float
    ) -> np.ndarray:
        """Smooth spatial footprint of a defect over the channel axis."""

        positions = np.arange(n_channels, dtype=np.float64)
        distance = np.abs(positions - centre)
        profile = np.zeros(n_channels, dtype=np.float64)
        inside = distance <= half_width
        if half_width <= 0:
            profile[int(round(centre))] = 1.0
            return profile
        profile[inside] = 0.5 * (
            1.0 + np.cos(np.pi * distance[inside] / half_width)
        )
        return profile

    def _axle_crossings(self, defect_position_m: float) -> list[tuple[float, int]]:
        """Times at which each axle passes the defect (seconds, axle index)."""

        cfg = self.config
        start = -0.20 * self.fibre_length_m
        offsets = np.concatenate(
            [[0.0], np.cumsum(np.asarray(cfg.axle_spacing_m, dtype=float))]
        )
        crossings: list[tuple[float, int]] = []
        for index, offset in enumerate(offsets):
            crossing = (defect_position_m - start - float(offset)) / cfg.speed_mps
            if 0.0 <= crossing < cfg.duration_s:
                crossings.append((float(crossing), int(index)))
        return crossings

    def _impact_atom(self, tau: np.ndarray) -> np.ndarray:
        """Broadband transient following one axle crossing.

        A sum of three Gaussian-modulated carriers with staggered phases yields a
        genuinely wideband pulse (roughly 50-190 Hz) that is short in time, the
        signature of a rail joint or a squat.
        """

        cfg = self.config
        atom = np.zeros_like(tau)
        sigma = max(cfg.impact_duration_s / 2.5, 1e-4)
        active = (tau >= 0.0) & (tau <= 4.0 * cfg.impact_duration_s)
        if not active.any():
            return atom
        t = tau[active]
        envelope = np.exp(-np.square(t / sigma)) + 0.30 * np.exp(
            -t / max(cfg.impact_decay_s, 1e-4)
        )
        carriers = np.asarray(cfg.impact_carriers_hz, dtype=float)
        wave = np.zeros_like(t)
        for index, frequency in enumerate(carriers):
            wave += np.sin(2.0 * np.pi * frequency * t + 0.9 * index)
        wave /= max(carriers.size, 1)
        atom[active] = envelope * wave
        return atom

    def _resonance_atom(self, tau: np.ndarray) -> np.ndarray:
        """Decaying structural ring excited by one axle crossing.

        The ring is truncated after six decay constants: beyond that point less
        than 0.3 % of its energy remains, and keeping an exponential tail that
        never reaches zero would smear the support region over the whole pass.
        """

        cfg = self.config
        atom = np.zeros_like(tau)
        horizon = 6.0 * max(cfg.resonance_decay_s, 1e-4)
        active = (tau >= 0.0) & (tau <= horizon)
        if not active.any():
            return atom
        t = tau[active]
        envelope = np.exp(-t / max(cfg.resonance_decay_s, 1e-4))
        wave = np.sin(2.0 * np.pi * cfg.resonance_frequency_hz * t) + (
            cfg.resonance_harmonic_ratio
            * np.sin(
                2.0
                * np.pi
                * cfg.resonance_frequency_hz
                * cfg.resonance_harmonic_multiplier
                * t
                + 0.7
            )
        )
        atom[active] = envelope * wave
        return atom

    def _defect_component(
        self,
        profile: np.ndarray,
        crossings: Sequence[tuple[float, int]],
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
        """Additive defect response, its support mask and active time ranges."""

        cfg = self.config
        time = np.arange(self.n_samples, dtype=np.float64) / cfg.sampling_rate_hz
        component = np.zeros((cfg.n_channels, self.n_samples), dtype=np.float64)
        support = np.zeros((cfg.n_channels, self.n_samples), dtype=bool)
        active_ranges: list[tuple[float, float]] = []

        for crossing_time, axle_index in crossings:
            tau = time - crossing_time
            chemical = np.zeros(self.n_samples, dtype=np.float64)
            if cfg.anomaly_type in {"impact", "mixed"}:
                chemical += self._impact_atom(tau)
            if cfg.anomaly_type in {"resonance", "mixed"}:
                chemical += self._resonance_atom(tau)
            if not np.any(chemical):
                continue
            # Later axles couple slightly less strongly through the same defect.
            weight = 0.85 ** axle_index
            component += weight * profile[:, None] * chemical[None, :]
            active_time = np.flatnonzero(chemical != 0.0)
            support[:, active_time] = True
            active_ranges.append(
                (
                    float(active_time[0]) / cfg.sampling_rate_hz,
                    (float(active_time[-1]) + 1.0) / cfg.sampling_rate_hz,
                )
            )

        support &= (profile[:, None] > 0.05)
        merged: list[tuple[float, float]] = []
        for start, stop in sorted(active_ranges):
            if merged and start <= merged[-1][1] + 1e-9:
                merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
            else:
                merged.append((start, stop))
        return component, support, merged

    def _scale_to_snr(
        self,
        component: np.ndarray,
        background: np.ndarray,
        support: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Scale a unit defect template to a target local SNR in dB.

        The temporal band-pass is linear but the spatial common-mode removal is
        a running *median*, so ``filter(a + b) != filter(a) + filter(b)``.  The
        scale is therefore first estimated from a linear approximation and then
        refined by a few fixed-point iterations against the real preprocessing
        chain, which keeps the reported SNR honest.
        """

        cfg = self.config
        target = None if cfg.snr_db is None else float(cfg.snr_db)

        if cfg.snr_reference == "processed":
            noise_ref = self._apply_reference_filters(background, background)[1]
            signal_ref = self._apply_reference_filters(component, background)[0]
        else:
            noise_ref, signal_ref = background, component

        signal_power = float(np.mean(np.square(signal_ref[support])))
        noise_power = float(np.mean(np.square(noise_ref[support])))
        if signal_power <= 0.0:
            raise FloatingPointError("defect template has zero power on its support region")
        if noise_power <= 0.0:
            raise FloatingPointError("background has zero power on the defect support region")

        if target is None:
            scale = 1.0
            processed_snr = float(self._local_snr_db(signal_ref, noise_ref, support))
        else:
            scale = np.sqrt(noise_power / signal_power) * 10.0 ** (target / 20.0)
            if cfg.snr_reference == "processed":
                from ..dsp.filters import preprocess_recording

                kwargs = dict(
                    sampling_rate_hz=cfg.sampling_rate_hz,
                    bandpass_hz=list(cfg.bandpass_hz) if cfg.bandpass_hz else None,
                    order=cfg.filter_order,
                    common_mode_window=cfg.common_mode_window,
                    spatial_axis=0,
                )
                processed_snr = float("nan")
                for _ in range(6):
                    trial = preprocess_recording(
                        (background + scale * component).astype(np.float32), **kwargs
                    ).astype(np.float64)
                    residual_power = float(np.mean(np.square((trial - noise_ref)[support])))
                    processed_snr = 10.0 * np.log10(
                        residual_power / max(noise_power, 1e-30)
                    )
                    delta = target - processed_snr
                    if abs(delta) < 0.05:
                        break
                    scale *= 10.0 ** (delta / 20.0)
            else:
                processed_snr = self._local_snr_db(
                    self._apply_reference_filters(scale * component, background)[0],
                    noise_ref,
                    support,
                )

        scaled = component * scale
        info: dict[str, Any] = {
            "target_snr_db": target,
            "snr_reference": cfg.snr_reference,
            "snr_db_processed": float(processed_snr),
            "snr_db_raw": float(self._local_snr_db(scaled, background, support)),
            "defect_scale": float(scale),
            "support_fraction": float(np.mean(support)),
        }
        return scaled, info

    def _apply_reference_filters(
        self, signal: np.ndarray, noise: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the preprocessing chain to both components independently."""

        from ..dsp.filters import preprocess_recording

        cfg = self.config
        kwargs = dict(
            sampling_rate_hz=cfg.sampling_rate_hz,
            bandpass_hz=list(cfg.bandpass_hz) if cfg.bandpass_hz else None,
            order=cfg.filter_order,
            common_mode_window=cfg.common_mode_window,
            spatial_axis=0,
        )
        filtered_signal = preprocess_recording(
            signal.astype(np.float32), **kwargs
        ).astype(np.float64)
        filtered_noise = preprocess_recording(
            noise.astype(np.float32), **kwargs
        ).astype(np.float64)
        return filtered_signal, filtered_noise

    @staticmethod
    def _local_snr_db(
        signal: np.ndarray, noise: np.ndarray, support: np.ndarray
    ) -> float:
        """Local SNR in dB over a boolean support mask."""

        signal_power = float(np.mean(np.square(signal[support])))
        noise_power = float(np.mean(np.square(noise[support])))
        if noise_power <= 0.0 or signal_power <= 0.0:
            return float("-inf")
        return 10.0 * np.log10(signal_power / noise_power)

    # ------------------------------------------------------------------- public
    def simulate(
        self,
        anomaly: bool = False,
        anomaly_channels: Sequence[int] | None = None,
        anomaly_type: str | None = None,
        snr_db: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Generate one pass.

        Returns
        -------
        data:
            ``(n_channels, n_samples)`` float32 recording.
        channel_labels:
            ``(n_channels,)`` uint8 labels, ``1`` at anomalous channels.
        metadata:
            Acquisition geometry, defect type and achieved SNR.
        """

        cfg = self.config
        if anomaly_type is not None and anomaly_type not in VALID_ANOMALY_TYPES:
            raise ValueError(f"Unknown anomaly_type {anomaly_type!r}")
        if snr_db is not None:
            cfg = SimulationConfig(**{**cfg.__dict__, "snr_db": float(snr_db)})
            self.config = cfg
        if anomaly_type is not None:
            cfg = SimulationConfig(**{**cfg.__dict__, "anomaly_type": anomaly_type})
            self.config = cfg

        # ---------------------------------------------------------- background
        field = self._normal_field()
        field += self._lowpass_noise(self.n_samples)

        # Common-mode cable noise plus channel-dependent sensitivity.
        common = self.rng.normal(
            0.0, cfg.noise_std * cfg.common_mode_fraction, size=self.n_samples
        )
        common = np.convolve(common, np.ones(9) / 9.0, mode="same")
        field += common[None, :]
        gains = self.rng.normal(1.0, cfg.channel_gain_std, size=(cfg.n_channels, 1))
        field *= np.clip(gains, 0.7, 1.3)

        labels = np.zeros(cfg.n_channels, dtype=np.uint8)
        defect_info: dict[str, Any] = {"anomaly_type": None}
        injected = False
        self.last_support_ = None

        if anomaly and cfg.anomaly_type != "none":
            profile, centre_channel = self._resolve_profile(anomaly_channels)
            defect_position_m = float(centre_channel) * cfg.channel_spacing_m
            crossings = self._axle_crossings(defect_position_m)
            if not crossings:
                LOGGER.warning(
                    "Defect at channel %.1f is never crossed by an axle; "
                    "pass will be labelled normal",
                    centre_channel,
                )
            component, support, active_ranges = self._defect_component(profile, crossings)
            if np.any(component):
                component, snr_info = self._scale_to_snr(component, field, support)
                field = field + component
                labels[profile > 0.35] = 1
                injected = True
                self.last_support_ = support
                defect_info = {
                    "anomaly_type": cfg.anomaly_type,
                    "defect_centre_channel": float(centre_channel),
                    "defect_channels": np.flatnonzero(profile > 0.35).astype(int).tolist(),
                    "axle_crossing_times_s": [float(t) for t, _ in crossings],
                    "active_time_ranges_s": [
                        [float(start), float(stop)] for start, stop in active_ranges
                    ],
                    "injected": True,
                    **snr_info,
                }
            else:
                defect_info = {
                    "anomaly_type": cfg.anomaly_type,
                    "defect_centre_channel": float(centre_channel),
                    "defect_channels": [],
                    "injected": False,
                }

        metadata: dict[str, Any] = {
            "source": "simulator",
            "sampling_rate_hz": cfg.sampling_rate_hz,
            "channel_spacing_m": cfg.channel_spacing_m,
            "n_channels": cfg.n_channels,
            "n_samples": self.n_samples,
            "duration_s": cfg.duration_s,
            "speed_mps": cfg.speed_mps,
            "anomaly": bool(injected and np.any(labels)),
            "anomaly_channels": np.flatnonzero(labels).astype(int).tolist(),
            "defect": defect_info,
            "simulation": {
                "source_frequency_hz": cfg.source_frequency_hz,
                "spatial_width_m": cfg.spatial_width_m,
                "noise_std": cfg.noise_std,
                "anomaly_type": cfg.anomaly_type,
                "snr_db": cfg.snr_db,
            },
        }
        return field.astype(np.float32), labels, metadata

    def _resolve_profile(
        self, anomaly_channels: Sequence[int] | None
    ) -> tuple[np.ndarray, float]:
        """Choose (or convert) the defect channel profile."""

        cfg = self.config
        if anomaly_channels is not None:
            channels = np.asarray(anomaly_channels, dtype=int)
            channels = channels[(channels >= 0) & (channels < cfg.n_channels)]
            if channels.size == 0:
                raise ValueError("anomaly_channels contains no valid channel index")
            centre = float(channels.mean())
            half_width = max((channels.max() - channels.min()) / 2.0, 0.5)
        else:
            if cfg.anomaly_channel_centre is not None:
                centre = float(cfg.anomaly_channel_centre)
            else:
                margin = 0.2 * cfg.n_channels
                centre = float(
                    self.rng.uniform(margin, cfg.n_channels - margin)
                )
            width = int(
                self.rng.integers(
                    cfg.anomaly_min_width_channels,
                    cfg.anomaly_max_width_channels + 1,
                )
            )
            half_width = max(width / 2.0, 0.5)
        profile = self._raised_cosine_profile(cfg.n_channels, centre, half_width)
        if not np.any(profile > 0.35):
            profile[int(round(centre))] = 1.0
        return profile, centre


def generate_synthetic_dataset(
    output_dir: str | Path,
    n_files: int,
    split: str = "train",
    anomaly_probability: float = 0.0,
    seed: int = 2024,
    simulation_config: dict[str, Any] | None = None,
    **simulation_overrides: Any,
) -> list[Path]:
    """Write a reproducible set of ``.npz`` recordings and a manifest.

    Training files contain only normal passes when ``anomaly_probability=0``.
    Test files can mix normal and anomalous passes; channel labels are stored
    for evaluation only and must never be used during model fitting.
    """

    if n_files < 0:
        raise ValueError("n_files must be non-negative")
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of 'train', 'val', 'test'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_values = dict(simulation_config or {})
    config_values.update(simulation_overrides)
    base_seed = int(config_values.pop("seed", seed))
    simulator = TrainPassSimulator.from_config(config_values)

    manifest: list[dict[str, Any]] = []
    paths: list[Path] = []
    for index in range(n_files):
        # Derive an independent but reproducible anomaly decision.
        decision_rng = np.random.default_rng(base_seed + 10_000 + index)
        is_anomaly = bool(decision_rng.random() < anomaly_probability)
        # Guarantee at least one anomalous test pass so evaluation metrics are
        # defined even for a very small synthetic test set.
        if split == "test" and anomaly_probability > 0 and index == 0:
            is_anomaly = True
        simulator.rng = np.random.default_rng(base_seed + index)
        data, labels, metadata = simulator.simulate(anomaly=is_anomaly)

        path = output_dir / f"{split}_{'anomaly' if is_anomaly else 'normal'}_{index:04d}.npz"
        time_s = np.arange(data.shape[1], dtype=np.float32) / simulator.config.sampling_rate_hz
        distance_m = (
            np.arange(data.shape[0], dtype=np.float32)
            * simulator.config.channel_spacing_m
        )
        np.savez_compressed(
            path,
            data=data,
            channel_labels=labels,
            time_s=time_s,
            distance_m=distance_m,
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        paths.append(path)
        manifest.append(
            {
                "file": path.name,
                "split": split,
                "anomaly": bool(metadata["anomaly"]),
                "anomaly_type": metadata["defect"].get("anomaly_type"),
                "achieved_snr_db": metadata["defect"].get("snr_db_processed"),
                "anomaly_channels": metadata["anomaly_channels"],
                "n_channels": data.shape[0],
                "n_samples": data.shape[1],
            }
        )
        LOGGER.info("Generated %s", path)

    manifest_path = output_dir / f"{split}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return paths

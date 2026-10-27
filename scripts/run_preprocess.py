"""Batch preprocessing for raw HDF5 DAS files or synthetic recordings.

Pipeline
--------
1. Read an HDF5 recording (or a synthetic ``.npz`` file).
2. Band-pass filter in time and remove spatially common noise.
3. Detect train-pass events with a robust short-time energy envelope.
4. Tile each event into fixed ``time_window_tw`` samples.
5. Save one model-ready ``.npz`` per window and a JSON manifest.

Examples
--------
Process all HDF5 files in ``data/raw`` into ``data/processed/train``::

    python scripts/run_preprocess.py --split train

Generate and process a small synthetic dataset (train and test)::

    python scripts/run_preprocess.py --synthetic --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import HDF5DASReader, PassSlicer, generate_synthetic_dataset  # noqa: E402
from src.dsp.filters import preprocess_recording  # noqa: E402
from src.utils import get_logger  # noqa: E402

LOGGER = logging.getLogger("das.preprocess")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must be a mapping")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_recording(
    recording: np.ndarray,
    method: str = "robust",
    clip_sigma: float | None = 6.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalise a whole recording while preserving relative anomaly amplitude."""

    data = np.asarray(recording, dtype=np.float32)
    method = (method or "none").lower()
    if method == "none":
        return data, {"method": "none"}
    if method not in {"robust", "zscore"}:
        raise ValueError("normalization must be 'robust', 'zscore' or 'none'")

    if method == "robust":
        centre = float(np.median(data))
        scale = float(1.4826 * np.median(np.abs(data - centre)))
        if scale <= 1e-8:
            scale = float(data.std())
    else:
        centre = float(data.mean())
        scale = float(data.std())
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    normalized = (data - centre) / scale
    if clip_sigma is not None and clip_sigma > 0:
        normalized = np.clip(normalized, -float(clip_sigma), float(clip_sigma))
    return normalized.astype(np.float32), {
        "method": method,
        "centre": centre,
        "scale": scale,
        "clip_sigma": clip_sigma,
    }


class Preprocessor:
    """Apply the configured filtering, detection, tiling and save pipeline."""

    def __init__(
        self,
        config: dict[str, Any],
        output_root: Path,
        overwrite: bool = False,
        save_interim: bool = False,
    ) -> None:
        self.config = config
        data_cfg = config.get("data", {})
        prep_cfg = data_cfg.get("preprocessing", {})
        detection_cfg = data_cfg.get("pass_detection", {})
        self.sampling_rate_hz = float(data_cfg.get("sampling_rate_hz", 1000.0))
        self.channel_spacing_m = float(data_cfg.get("spatial_spacing_m", 1.0))
        self.time_window_tw = int(data_cfg.get("time_window_tw", 128))
        self.hop_samples = max(self.time_window_tw // 2, 1)
        self.bandpass_hz = data_cfg.get("bandpass_hz", [5.0, 200.0])
        self.filter_order = int(data_cfg.get("filter_order", 4))
        self.common_mode_window = int(
            data_cfg.get(
                "common_mode_window", data_cfg.get("spatial_window_cw", 5)
            )
        )
        self.normalization = str(prep_cfg.get("normalization", "robust"))
        self.clip_sigma = prep_cfg.get("clip_sigma", 6.0)
        self.save_dtype = str(prep_cfg.get("save_dtype", "float32"))
        self.overwrite = overwrite
        self.save_interim = save_interim
        self.output_root = output_root
        self.slicer = PassSlicer(
            sampling_rate_hz=self.sampling_rate_hz,
            frame_seconds=float(detection_cfg.get("frame_seconds", 0.05)),
            energy_percentile=float(detection_cfg.get("energy_percentile", 90.0)),
            threshold_sigma=float(detection_cfg.get("threshold_sigma", 2.5)),
            min_duration_s=float(detection_cfg.get("min_duration_s", 0.3)),
            merge_gap_s=float(detection_cfg.get("merge_gap_s", 0.5)),
            pad_s=float(detection_cfg.get("pad_s", 0.2)),
        )
        self.interim_dir = resolve_path(config.get("paths", {}).get("interim_dir", "data/interim"))

    def process_array(
        self,
        recording: np.ndarray,
        source_stem: str,
        split: str,
        sampling_rate_hz: float | None = None,
        channel_spacing_m: float | None = None,
        channel_labels: np.ndarray | None = None,
        source_metadata: dict[str, Any] | None = None,
    ) -> int:
        """Process one recording and return the number of saved windows."""

        data = np.asarray(recording, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"{source_stem}: expected (channels, samples), got {data.shape}")
        fs = float(sampling_rate_hz or self.sampling_rate_hz)
        spacing = float(channel_spacing_m or self.channel_spacing_m)
        n_channels = data.shape[0]
        has_labels = channel_labels is not None
        if channel_labels is None:
            labels = np.zeros(n_channels, dtype=np.uint8)
        else:
            labels = np.asarray(channel_labels, dtype=np.uint8).reshape(-1)
            if labels.size != n_channels:
                raise ValueError(
                    f"{source_stem}: channel_labels has {labels.size} entries "
                    f"for {n_channels} channels"
                )
        is_anomaly = int(labels.any())

        filtered = preprocess_recording(
            data,
            sampling_rate_hz=fs,
            bandpass_hz=self.bandpass_hz,
            order=self.filter_order,
            common_mode_window=self.common_mode_window,
            spatial_axis=0,
        )
        normalized, normalization_stats = normalize_recording(
            filtered, method=self.normalization, clip_sigma=self.clip_sigma
        )

        if self.save_interim:
            self.interim_dir.mkdir(parents=True, exist_ok=True)
            interim_path = self.interim_dir / f"{source_stem}.npz"
            np.savez_compressed(
                interim_path,
                data=normalized.astype(self.save_dtype),
                channel_labels=labels,
                metadata=np.asarray(
                    json.dumps(
                        {
                            "source": source_stem,
                            "sampling_rate_hz": fs,
                            "channel_spacing_m": spacing,
                            **(source_metadata or {}),
                        },
                        ensure_ascii=False,
                        default=_json_default,
                    )
                ),
            )

        events = self.slicer.detect(normalized)
        if not events:
            LOGGER.warning("%s: no train pass detected, skipping", source_stem)
            return 0
        windows, window_events = self.slicer.tile_events(
            normalized,
            events=events,
            window_samples=self.time_window_tw,
            hop_samples=self.hop_samples,
            dtype=self.save_dtype,
        )

        output_dir = self.output_root / split / source_stem
        if output_dir.exists():
            existing = list(output_dir.glob("*.npz"))
            if existing and not self.overwrite:
                LOGGER.info(
                    "%s: %d processed window(s) already exist, skipping (use --overwrite)",
                    source_stem,
                    len(existing),
                )
                return 0
            for old_file in existing:
                old_file.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest: list[dict[str, Any]] = []
        for index, (window, event) in enumerate(zip(windows, window_events)):
            path = output_dir / f"window_{index:05d}.npz"
            metadata = {
                "source": source_stem,
                "split": split,
                "sampling_rate_hz": fs,
                "channel_spacing_m": spacing,
                "normalization": normalization_stats,
                "event": event.to_dict(),
                "source_metadata": source_metadata or {},
            }
            np.savez_compressed(
                path,
                data=np.asarray(window, dtype=self.save_dtype),
                channel_labels=labels,
                is_anomaly=np.asarray([is_anomaly], dtype=np.uint8),
                has_labels=np.asarray([int(has_labels)], dtype=np.uint8),
                sampling_rate_hz=np.asarray([fs], dtype=np.float32),
                channel_spacing_m=np.asarray([spacing], dtype=np.float32),
                source=np.asarray([source_stem]),
                event=np.asarray(json.dumps(event.to_dict(), ensure_ascii=False)),
                metadata=np.asarray(
                    json.dumps(metadata, ensure_ascii=False, default=_json_default)
                ),
            )
            manifest.append(
                {
                    "file": path.name,
                    "is_anomaly": is_anomaly,
                    "anomaly_channels": np.flatnonzero(labels).astype(int).tolist(),
                    "window_start_sample": event.window_start_sample,
                    "window_samples": self.time_window_tw,
                }
            )

        if manifest:
            (output_dir / "windows_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        LOGGER.info(
            "%s: %d event(s), %d window(s), anomaly=%s",
            source_stem,
            len(events),
            len(windows),
            bool(is_anomaly),
        )
        return len(windows)

    def process_hdf5(self, path: Path, split: str) -> int:
        data_cfg = self.config.get("data", {})
        hdf5_cfg = data_cfg.get("hdf5", {})
        reader = HDF5DASReader(
            path,
            dataset_key=hdf5_cfg.get("dataset_key"),
            orientation=hdf5_cfg.get("orientation", "auto"),
            sampling_rate_hz=self.sampling_rate_hz,
            channel_spacing_m=self.channel_spacing_m,
        )
        recording = reader.read(dtype="float32")
        meta = reader.metadata
        LOGGER.info(
            "%s: dataset=%s shape=%s fs=%.2f Hz spacing=%.3f m",
            path.name,
            meta.dataset_key,
            recording.shape,
            meta.sampling_rate_hz,
            meta.channel_spacing_m,
        )
        return self.process_array(
            recording,
            source_stem=path.stem,
            split=split,
            sampling_rate_hz=meta.sampling_rate_hz,
            channel_spacing_m=meta.channel_spacing_m,
            source_metadata={"format": "hdf5", "dataset_key": meta.dataset_key},
        )

    def process_synthetic_file(self, path: Path, split: str) -> int:
        with np.load(path, allow_pickle=False) as payload:
            data = np.asarray(payload["data"], dtype=np.float32)
            labels = (
                np.asarray(payload["channel_labels"])
                if "channel_labels" in payload
                else None
            )
            metadata_payload: dict[str, Any] = {}
            if "metadata" in payload:
                try:
                    metadata_payload = json.loads(str(payload["metadata"]))
                except json.JSONDecodeError:
                    metadata_payload = {}
        fs = float(metadata_payload.get("sampling_rate_hz", self.sampling_rate_hz))
        spacing = float(metadata_payload.get("channel_spacing_m", self.channel_spacing_m))
        metadata_payload["synthetic_file"] = path.name
        return self.process_array(
            data,
            source_stem=path.stem,
            split=split,
            sampling_rate_hz=fs,
            channel_spacing_m=spacing,
            channel_labels=labels,
            source_metadata=metadata_payload,
        )


def _find_hdf5_files(directory: Path, limit: int | None = None) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.h5", "*.hdf5", "*.H5", "*.HDF5"):
        files.extend(directory.rglob(pattern))
    unique = sorted({path.resolve() for path in files})
    if limit is not None:
        unique = unique[: max(limit, 0)]
    return unique


def _find_npz_files(directory: Path, limit: int | None = None) -> list[Path]:
    files = sorted(directory.glob("*.npz"))
    if limit is not None:
        files = files[: max(limit, 0)]
    return files


def _create_synthetic_data(
    config: dict[str, Any],
    overwrite: bool,
    limit: int | None,
) -> tuple[list[Path], list[Path]]:
    sim_cfg = dict(config.get("simulation", {}))
    base_seed = int(sim_cfg.get("seed", 2024))
    synthetic_root = resolve_path(
        config.get("paths", {}).get("synthetic_dir", "data/synthetic")
    )
    train_dir = synthetic_root / "train"
    test_dir = synthetic_root / "test"
    train_files = _find_npz_files(train_dir, limit)
    test_files = _find_npz_files(test_dir, limit)

    if overwrite or not train_files:
        train_cfg = {**sim_cfg, "seed": base_seed}
        train_files = generate_synthetic_dataset(
            train_dir,
            n_files=int(train_cfg.get("train_files", 4)),
            split="train",
            anomaly_probability=0.0,
            simulation_config=train_cfg,
        )
    if overwrite or not test_files:
        test_cfg = {**sim_cfg, "seed": base_seed + 1}
        test_files = generate_synthetic_dataset(
            test_dir,
            n_files=int(test_cfg.get("test_files", 2)),
            split="test",
            anomaly_probability=float(test_cfg.get("anomaly_probability", 0.5)),
            simulation_config=test_cfg,
        )
    return train_files, test_files


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess DAS recordings into fixed pass windows."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "exp_default.yaml"),
        help="YAML experiment configuration.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate/process the synthetic train and test datasets.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Override the raw HDF5 input directory (ignored with --synthetic).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the processed root directory (a split subfolder is added).",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Split subfolder for raw HDF5 processing.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    parser.add_argument("--save-interim", action="store_true", help="Also save filtered recordings.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of source files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = load_config(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = resolve_path(config.get("paths", {}).get("log_dir", "outputs/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"preprocess_{timestamp}.log"
    global LOGGER
    LOGGER = get_logger("das.preprocess", log_file=log_file, level=config.get("logging", {}).get("level", "INFO"))

    processed_root = (
        resolve_path(args.output_dir)
        if args.output_dir
        else resolve_path(config.get("paths", {}).get("processed_dir", "data/processed"))
    )
    preprocessor = Preprocessor(
        config,
        output_root=processed_root,
        overwrite=args.overwrite,
        save_interim=args.save_interim,
    )
    LOGGER.info("Project root: %s", PROJECT_ROOT)
    LOGGER.info("Processed output root: %s", processed_root)

    total = 0
    if args.synthetic:
        train_files, test_files = _create_synthetic_data(
            config, overwrite=args.overwrite, limit=args.limit
        )
        for split, files in (("train", train_files), ("test", test_files)):
            if not files:
                LOGGER.warning("No synthetic files for split %s", split)
                continue
            for path in tqdm(files, desc=f"preprocess {split}", unit="file"):
                total += preprocessor.process_synthetic_file(path, split=split)
    else:
        raw_dir = (
            resolve_path(args.input_dir)
            if args.input_dir
            else resolve_path(config.get("paths", {}).get("raw_dir", "data/raw"))
        )
        hdf5_files = _find_hdf5_files(raw_dir, args.limit)
        if not hdf5_files:
            LOGGER.error(
                "No HDF5 files found under %s. Add raw data or rerun with --synthetic.",
                raw_dir,
            )
            return 1
        for path in tqdm(hdf5_files, desc=f"preprocess {args.split}", unit="file"):
            try:
                total += preprocessor.process_hdf5(path, split=args.split)
            except Exception:
                LOGGER.exception("Failed to process %s", path)

    LOGGER.info("Preprocessing finished: %d window(s) written", total)
    print(f"Preprocessing finished: {total} window(s) written to {processed_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

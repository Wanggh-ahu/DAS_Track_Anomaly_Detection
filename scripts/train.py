"""Train an unsupervised DAS anomaly detector from a YAML configuration.

The script loads processed fixed-length pass windows, splits normal data into
training/validation partitions, builds the selected model, calibrates spatial
score normalisation on the training set, and saves a checkpoint plus a JSON
metadata sidecar for :mod:`scripts.evaluate`.  Besides the window threshold it
also calibrates a pass-level threshold from the normal source recordings, so a
whole train pass can be accepted or rejected without relying on one 128-sample
window.

Examples
--------
    python scripts/train.py
    python scripts/train.py --model pca
    python scripts/train.py --model ae_1d --epochs 20 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (  # noqa: E402
    AnomalyScorer,
    conformal_threshold_report,
    select_threshold,
    split_conformal_quantile,
)
from src.models import BaseTorchModel, available_models, build_model  # noqa: E402
from src.utils import ExperimentLogger, plot_training_curves, save_figure  # noqa: E402

LOGGER = logging.getLogger("das.train")


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


def _extract_scalar(payload: Any, default: Any = None) -> Any:
    array = np.asarray(payload).reshape(-1)
    return array[0] if array.size else default


@dataclass
class ProcessedData:
    """In-memory collection of processed pass windows."""

    data: np.ndarray
    channel_labels: np.ndarray
    sample_labels: np.ndarray
    sources: np.ndarray
    sampling_rate_hz: float
    channel_spacing_m: float
    files: list[Path]

    def __len__(self) -> int:
        return int(self.data.shape[0])


def load_processed_split(
    root: str | Path,
    limit: int | None = None,
    only_normal: bool = False,
) -> ProcessedData:
    """Load all ``.npz`` windows under a processed split directory.

    Files are expected to contain ``data`` with shape ``(channels, samples)``.
    Optional fields ``channel_labels``, ``is_anomaly``, ``source``,
    ``sampling_rate_hz`` and ``channel_spacing_m`` are used when present.
    """

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Processed data directory not found: {root}")
    files = sorted(root.rglob("*.npz"))
    if limit is not None:
        files = files[: max(0, int(limit))]
    if not files:
        raise FileNotFoundError(
            f"No processed .npz windows under {root}. "
            "Run scripts/run_preprocess.py first."
        )

    arrays: list[np.ndarray] = []
    channel_labels: list[np.ndarray] = []
    sample_labels: list[int] = []
    sources: list[str] = []
    expected_shape: tuple[int, int] | None = None
    sampling_rate = 1000.0
    spacing = 0.5
    metadata_initialised = False
    skipped = 0

    for path in files:
        try:
            with np.load(path, allow_pickle=False) as payload:
                if "data" not in payload:
                    LOGGER.warning("%s has no 'data' field; skipping", path)
                    skipped += 1
                    continue
                window = np.asarray(payload["data"], dtype=np.float32)
                if window.ndim != 2:
                    LOGGER.warning("%s has non-2-D data %s; skipping", path, window.shape)
                    skipped += 1
                    continue
                if expected_shape is None:
                    expected_shape = window.shape
                elif window.shape != expected_shape:
                    LOGGER.warning(
                        "%s shape %s differs from %s; skipping",
                        path,
                        window.shape,
                        expected_shape,
                    )
                    skipped += 1
                    continue

                n_channels = window.shape[0]
                labels = (
                    np.asarray(payload["channel_labels"], dtype=np.uint8).reshape(-1)
                    if "channel_labels" in payload
                    else np.zeros(n_channels, dtype=np.uint8)
                )
                if labels.size != n_channels:
                    LOGGER.warning("%s has invalid channel labels; skipping", path)
                    skipped += 1
                    continue
                if "is_anomaly" in payload:
                    flag = int(_extract_scalar(payload["is_anomaly"], int(labels.any())))
                else:
                    flag = int(labels.any())
                if only_normal and flag:
                    continue
                fs = float(_extract_scalar(payload["sampling_rate_hz"], sampling_rate)) if "sampling_rate_hz" in payload else sampling_rate
                spacing_value = (
                    float(_extract_scalar(payload["channel_spacing_m"], spacing))
                    if "channel_spacing_m" in payload
                    else spacing
                )
                source = (
                    str(_extract_scalar(payload["source"], path.parent.name))
                    if "source" in payload
                    else path.parent.name
                )
                if not metadata_initialised:
                    sampling_rate = fs
                    spacing = spacing_value
                    metadata_initialised = True
                elif fs != sampling_rate or spacing_value != spacing:
                    LOGGER.warning(
                        "Sampling metadata differs in %s; using first values", path
                    )
                arrays.append(window)
                channel_labels.append(labels)
                sample_labels.append(flag)
                sources.append(source)
        except (OSError, ValueError, KeyError) as exc:
            LOGGER.warning("Could not read %s: %s", path, exc)
            skipped += 1

    if not arrays:
        raise RuntimeError(f"No valid processed windows found under {root}")
    if skipped:
        LOGGER.warning("Skipped %d unreadable or incompatible file(s)", skipped)
    return ProcessedData(
        data=np.stack(arrays, axis=0).astype(np.float32),
        channel_labels=np.stack(channel_labels, axis=0).astype(np.uint8),
        sample_labels=np.asarray(sample_labels, dtype=np.uint8),
        sources=np.asarray(sources),
        sampling_rate_hz=sampling_rate,
        channel_spacing_m=spacing,
        files=files,
    )


def split_train_validation(
    dataset: ProcessedData,
    validation_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Split by source recording when possible to reduce window leakage."""

    n_samples = len(dataset)
    if n_samples < 2 or validation_split <= 0:
        return np.arange(n_samples), None

    rng = np.random.default_rng(seed)
    unique_sources = np.unique(dataset.sources)
    if unique_sources.size >= 2:
        shuffled = unique_sources.copy()
        rng.shuffle(shuffled)
        n_val_groups = int(round(validation_split * len(shuffled)))
        n_val_groups = min(max(n_val_groups, 1), len(shuffled) - 1)
        val_sources = set(shuffled[-n_val_groups:].tolist())
        val_mask = np.asarray([source in val_sources for source in dataset.sources])
        train_indices = np.flatnonzero(~val_mask)
        val_indices = np.flatnonzero(val_mask)
    else:
        permutation = rng.permutation(n_samples)
        n_val = min(max(int(round(validation_split * n_samples)), 1), n_samples - 1)
        val_indices = permutation[:n_val]
        train_indices = permutation[n_val:]

    if train_indices.size == 0 or val_indices.size == 0:
        return np.arange(n_samples), None
    return np.sort(train_indices), np.sort(val_indices)


def split_train_calibration_validation(
    dataset: ProcessedData,
    validation_split: float,
    calibration_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Three-way, source-level split.

    ``train`` fits the model, ``calibration`` fits the score normaliser and the
    conformal threshold, and ``validation`` is only used for early stopping.
    Keeping these disjoint matters: a threshold computed on the residuals the
    model was fitted on is optimistic by construction.
    """

    n_samples = len(dataset)
    if n_samples < 3:
        return np.arange(n_samples), None, None

    rng = np.random.default_rng(seed)
    unique_sources = np.unique(dataset.sources)
    calibration_split = float(np.clip(calibration_split, 0.0, 0.9))
    validation_split = float(np.clip(validation_split, 0.0, 0.9))

    if unique_sources.size >= 3:
        shuffled = unique_sources.copy()
        rng.shuffle(shuffled)
        n_val = int(round(validation_split * shuffled.size))
        n_cal = int(round(calibration_split * shuffled.size))
        n_val = min(max(n_val, 0), shuffled.size - 2)
        n_cal = min(max(n_cal, 0), shuffled.size - n_val - 1)
        val_sources = set(shuffled[:n_val].tolist())
        cal_sources = set(shuffled[n_val : n_val + n_cal].tolist())
        source_array = np.asarray(dataset.sources)
        val_mask = np.isin(source_array, list(val_sources))
        cal_mask = np.isin(source_array, list(cal_sources))
        train_indices = np.flatnonzero(~val_mask & ~cal_mask)
        calibration_indices = np.flatnonzero(cal_mask)
        validation_indices = np.flatnonzero(val_mask)
    else:
        permutation = rng.permutation(n_samples)
        n_val = int(round(validation_split * n_samples))
        n_cal = int(round(calibration_split * n_samples))
        n_val = min(max(n_val, 0), n_samples - 2)
        n_cal = min(max(n_cal, 1), n_samples - n_val - 1)
        validation_indices = np.sort(permutation[:n_val])
        calibration_indices = np.sort(permutation[n_val : n_val + n_cal])
        train_indices = np.sort(permutation[n_val + n_cal :])

    if train_indices.size == 0 or calibration_indices.size == 0:
        LOGGER.warning(
            "Not enough data for a disjoint calibration split; "
            "falling back to a two-way split (threshold guarantee unavailable)"
        )
        train_idx, val_idx = split_train_validation(dataset, validation_split, seed)
        return train_idx, None, val_idx

    return (
        np.sort(train_indices),
        np.sort(calibration_indices),
        np.sort(validation_indices) if validation_indices.size else None,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an unsupervised DAS anomaly model.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "exp_default.yaml"),
        help="YAML experiment configuration.",
    )
    parser.add_argument(
        "--model",
        choices=available_models(),
        default=None,
        help="Model name; defaults to model.name in the configuration.",
    )
    parser.add_argument("--train-dir", default=None, help="Override processed/train.")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", default=None, help="auto, cpu or cuda.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--val-split", type=float, default=None, help="Validation fraction.")
    parser.add_argument("--limit", type=int, default=None, help="Limit input .npz files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = load_config(args.config)
    model_name = args.model or str(config.get("model", {}).get("name", "stae_convlstm"))
    if model_name not in available_models():
        raise ValueError(f"Unsupported model {model_name!r}")

    training_cfg = dict(config.get("training", {}))
    if args.epochs is not None:
        training_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        training_cfg["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        training_cfg["lr"] = float(args.lr)
    if args.device is not None:
        training_cfg["device"] = args.device
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    training_cfg["seed"] = seed
    validation_split = (
        float(args.val_split)
        if args.val_split is not None
        else float(training_cfg.get("validation_split", 0.15))
    )

    log_dir = resolve_path(config.get("paths", {}).get("log_dir", "outputs/logs"))
    experiment = ExperimentLogger(
        "train", log_dir, config=config, level=str(config.get("logging", {}).get("level", "INFO"))
    )
    global LOGGER
    LOGGER = experiment.logger
    # Let module-level loggers from src.* write to the same experiment file.
    root_logger = logging.getLogger()
    root_logger.setLevel(experiment.logger.level)
    for handler in experiment.logger.handlers:
        if handler not in root_logger.handlers:
            root_logger.addHandler(handler)

    try:
        processed_root = resolve_path(
            config.get("paths", {}).get("processed_dir", "data/processed")
        )
        train_dir = (
            resolve_path(args.train_dir)
            if args.train_dir
            else processed_root / "train"
        )
        LOGGER.info("Loading processed training windows from %s", train_dir)
        dataset = load_processed_split(train_dir, limit=args.limit, only_normal=True)
        if len(dataset) == 0:
            raise RuntimeError("No normal training windows available after filtering")
        LOGGER.info(
            "Loaded %d normal windows from %d source file(s); shape=%s fs=%.2f Hz",
            len(dataset),
            len(np.unique(dataset.sources)),
            dataset.data.shape[1:],
            dataset.sampling_rate_hz,
        )
        train_indices, calibration_indices, val_indices = (
            split_train_calibration_validation(
                dataset,
                validation_split=validation_split,
                calibration_split=float(
                    config.get("evaluation", {}).get("calibration_split", 0.2)
                ),
                seed=seed,
            )
        )
        train_data = dataset.data[train_indices]
        calibration_data = (
            dataset.data[calibration_indices]
            if calibration_indices is not None
            else None
        )
        val_data = dataset.data[val_indices] if val_indices is not None else None
        LOGGER.info(
            "Train windows: %d | calibration windows: %d | validation windows: %d",
            train_data.shape[0],
            0 if calibration_data is None else calibration_data.shape[0],
            0 if val_data is None else val_data.shape[0],
        )

        model_params = dict(
            config.get("model", {}).get("params", {}).get(model_name, {})
        )
        evaluation_cfg = config.get("evaluation", {})
        if model_name == "isolation_forest" and "topk_channels" not in model_params:
            model_params["topk_channels"] = int(evaluation_cfg.get("topk_channels", 3))
        LOGGER.info("Building model %s with %s", model_name, model_params)
        model = build_model(
            model_name,
            model_params=model_params,
            training_config=training_cfg,
        )
        LOGGER.info("Model: %r", model)
        model.fit(
            train_data,
            val_data=val_data,
            sampling_rate_hz=dataset.sampling_rate_hz,
        )

        normalization = str(
            config.get("data", {})
            .get("preprocessing", {})
            .get("normalization", "robust")
        ).lower()
        if normalization not in {"robust", "zscore", "none"}:
            LOGGER.warning(
                "Unknown preprocessing normalization %r; using robust", normalization
            )
            normalization = "robust"
        scorer = AnomalyScorer(
            model,
            topk_channels=int(evaluation_cfg.get("topk_channels", 3)),
            aggregation="topk",
            normalization=normalization,
            sampling_rate_hz=dataset.sampling_rate_hz,
        )
        calibration_source = (
            calibration_data if calibration_data is not None else train_data
        )
        if calibration_data is None:
            LOGGER.warning(
                "No disjoint calibration split available; calibrating on training "
                "windows (the false-alarm guarantee does not hold)"
            )
        scorer.fit(calibration_source)
        normal_scores = scorer.sample_scores(calibration_source)

        threshold_method = str(evaluation_cfg.get("threshold_method", "conformal"))
        conformal_alpha = float(evaluation_cfg.get("conformal_alpha", 0.05))
        fixed_threshold = evaluation_cfg.get("fixed_threshold")
        if threshold_method == "fixed" and fixed_threshold is None:
            LOGGER.warning("fixed_threshold is null; falling back to percentile threshold")
            threshold_method = "percentile"

        window_conformal = None
        if threshold_method == "conformal":
            window_conformal = split_conformal_quantile(
                normal_scores, alpha=conformal_alpha
            )
            threshold = window_conformal.threshold
            LOGGER.info(
                "Window threshold (conformal alpha=%.3f): %.6f | %s",
                conformal_alpha,
                threshold,
                window_conformal.reason,
            )
        else:
            threshold = select_threshold(
                normal_scores,
                method=threshold_method,
                percentile=float(evaluation_cfg.get("threshold_percentile", 99.0)),
                fixed_threshold=fixed_threshold,
            )

        calibration_false_alarms = float(np.mean(normal_scores >= threshold))
        LOGGER.info(
            "Calibration threshold=%.6f (empirical false-alarm fraction %.4f over %d windows)",
            threshold,
            calibration_false_alarms,
            normal_scores.size,
        )
        if val_data is not None:
            val_scores = scorer.sample_scores(val_data)
            LOGGER.info(
                "Validation score mean=%.6f std=%.6f | above-threshold fraction=%.4f",
                float(val_scores.mean()),
                float(val_scores.std()),
                float(np.mean(val_scores >= threshold)),
            )

        # Calibrate a separate, pass-level threshold.  A maintenance alert is
        # raised for a whole train pass, not for an isolated 128-sample window.
        # Per-pass averaging removes most short background bursts, but its
        # score scale is different from the window-level threshold.
        pass_score_aggregation = str(
            evaluation_cfg.get("source_score_aggregation", "mean")
        ).lower()
        pass_reducer = {
            "mean": np.mean,
            "median": np.median,
            "max": np.max,
        }.get(pass_score_aggregation)
        pass_threshold = float(threshold)
        pass_conformal = None
        normal_pass_scores: list[float] = []
        if pass_reducer is not None:
            calibration_scores = scorer.sample_scores(calibration_source)
            calibration_sources = (
                dataset.sources[calibration_indices]
                if calibration_indices is not None
                else dataset.sources[train_indices]
            )
            normal_pass_scores = [
                float(pass_reducer(calibration_scores[calibration_sources == source]))
                for source in np.unique(calibration_sources)
            ]
            if normal_pass_scores:
                if threshold_method == "conformal":
                    pass_conformal = split_conformal_quantile(
                        normal_pass_scores, alpha=conformal_alpha
                    )
                    pass_threshold = pass_conformal.threshold
                    LOGGER.info(
                        "Pass-level threshold (conformal): %.6f over %d passes | %s",
                        pass_threshold,
                        len(normal_pass_scores),
                        pass_conformal.reason,
                    )
                else:
                    margin = max(
                        float(evaluation_cfg.get("pass_threshold_margin", 0.01)), 0.0
                    )
                    pass_threshold = float(max(normal_pass_scores) * (1.0 + margin))
                    LOGGER.info(
                        "Pass-level threshold=%.6f (max normal pass=%.6f, margin=%.3f)",
                        pass_threshold,
                        max(normal_pass_scores),
                        margin,
                    )

        checkpoint_dir = (
            resolve_path(args.checkpoint_dir)
            if args.checkpoint_dir
            else resolve_path(
                config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints")
            )
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".pt" if isinstance(model, BaseTorchModel) else ".pkl"
        checkpoint_path = checkpoint_dir / f"{model_name}{suffix}"
        metadata_path = checkpoint_dir / f"{model_name}_metadata.json"
        metadata: dict[str, Any] = {
            "model_name": model_name,
            "model_class": type(model).__name__,
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT))
            if checkpoint_path.is_relative_to(PROJECT_ROOT)
            else str(checkpoint_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sampling_rate_hz": dataset.sampling_rate_hz,
            "channel_spacing_m": dataset.channel_spacing_m,
            "input_shape": list(train_data.shape[1:]),
            "threshold": threshold,
            "pass_threshold": pass_threshold,
            "threshold_method": threshold_method,
            "pass_score_aggregation": pass_score_aggregation,
            "normal_pass_scores": normal_pass_scores,
            "calibration_false_alarm_fraction": calibration_false_alarms,
            "calibration": {
                "source": "disjoint_split" if calibration_data is not None else "train",
                "windows": int(calibration_source.shape[0]),
                "calibration_sources": int(
                    len(
                        np.unique(
                            dataset.sources[calibration_indices]
                            if calibration_indices is not None
                            else dataset.sources[train_indices]
                        )
                    )
                ),
            },
            "conformal": conformal_threshold_report(
                window_conformal
                or split_conformal_quantile(normal_scores, alpha=conformal_alpha),
                pass_conformal
                or split_conformal_quantile(
                    normal_pass_scores or normal_scores, alpha=conformal_alpha
                ),
            ),
            "scorer": scorer.state_dict(),
            "training_config": training_cfg,
            "model_params": model_params,
            "training_history": model.training_history_,
            "train_windows": int(train_data.shape[0]),
            "validation_windows": 0 if val_data is None else int(val_data.shape[0]),
            "seed": seed,
        }
        checkpoint = model.save(checkpoint_path, metadata=metadata)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        experiment.log_metrics(
            {
                "threshold": threshold,
                "normal_score_mean": float(normal_scores.mean()),
                "normal_score_std": float(normal_scores.std()),
                "train_windows": int(train_data.shape[0]),
                "validation_windows": 0 if val_data is None else int(val_data.shape[0]),
            },
            step=1,
        )
        experiment.info("Checkpoint saved to %s", checkpoint)
        experiment.info("Metadata saved to %s", metadata_path)

        history = getattr(model, "training_history_", {})
        if isinstance(history, dict) and history.get("train_loss"):
            curve_path = log_dir / f"training_curves_{model_name}.png"
            figure, _ = plot_training_curves(history)
            save_figure(figure, curve_path)
            experiment.info("Training curves saved to %s", curve_path)

        print(f"Trained {model_name}: {checkpoint}")
        return 0
    finally:
        experiment.close()


if __name__ == "__main__":
    raise SystemExit(main())

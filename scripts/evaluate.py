"""Evaluate a trained DAS anomaly detector on a processed test split.

The script restores the checkpoint and the spatial calibration state saved by
``scripts/train.py``.  If ground-truth channel labels are present it reports:

* window-level detection;
* pass-level detection after fusing all windows of one source recording;
* point localisation using the per-window top-1 channel vote;
* interval localisation (IoU, coverage and boundary errors).

Otherwise only scores and prediction arrays are produced.

Examples
--------
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint outputs/checkpoints/ae_1d.pt
    python scripts/evaluate.py --threshold 2.5
    python scripts/evaluate.py --no-plots
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (  # noqa: E402
    AnomalyScorer,
    aggregate_channel_profile,
    detection_metrics_by_source,
    evaluate_all,
    interval_metrics_by_source,
    localization_metrics_by_source,
    pass_profile_scores,
    select_threshold,
)
from src.models import available_models, build_model  # noqa: E402
from src.utils import (  # noqa: E402
    ExperimentLogger,
    plot_sample_scores,
    plot_score_profile,
    save_figure,
)

LOGGER = logging.getLogger("das.evaluate")


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
class EvaluationData:
    """Processed test windows and optional ground truth."""

    data: np.ndarray
    channel_labels: np.ndarray
    sample_labels: np.ndarray
    has_labels: bool
    sources: np.ndarray
    sampling_rate_hz: float
    channel_spacing_m: float
    files: list[Path]

    def __len__(self) -> int:
        return int(self.data.shape[0])


def load_evaluation_data(root: str | Path, limit: int | None = None) -> EvaluationData:
    """Load processed windows, preserving labels when they are available."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Processed data directory not found: {root}")
    files = sorted(root.rglob("*.npz"))
    if limit is not None:
        files = files[: max(int(limit), 0)]
    if not files:
        raise FileNotFoundError(
            f"No processed windows under {root}. Run scripts/run_preprocess.py first."
        )

    arrays: list[np.ndarray] = []
    channel_labels: list[np.ndarray] = []
    sample_labels: list[int] = []
    sources: list[str] = []
    expected_shape: tuple[int, int] | None = None
    sampling_rate = 1000.0
    spacing = 0.5
    metadata_initialised = False
    any_labels = False
    skipped = 0

    for path in files:
        try:
            with np.load(path, allow_pickle=False) as payload:
                if "data" not in payload:
                    skipped += 1
                    continue
                window = np.asarray(payload["data"], dtype=np.float32)
                if window.ndim != 2:
                    skipped += 1
                    continue
                if expected_shape is None:
                    expected_shape = window.shape
                elif window.shape != expected_shape:
                    LOGGER.warning("Skipping %s with shape %s", path, window.shape)
                    skipped += 1
                    continue

                n_channels = window.shape[0]
                labels = (
                    np.asarray(payload["channel_labels"], dtype=np.uint8).reshape(-1)
                    if "channel_labels" in payload
                    else np.zeros(n_channels, dtype=np.uint8)
                )
                if labels.size != n_channels:
                    LOGGER.warning("Skipping %s with invalid labels", path)
                    skipped += 1
                    continue
                has_labels = (
                    bool(int(_extract_scalar(payload["has_labels"], 0)))
                    if "has_labels" in payload
                    else bool(labels.any())
                )
                if labels.any():
                    has_labels = True
                if has_labels:
                    any_labels = True
                flag = (
                    int(_extract_scalar(payload["is_anomaly"], int(labels.any())))
                    if "is_anomaly" in payload
                    else int(labels.any())
                )
                fs = (
                    float(_extract_scalar(payload["sampling_rate_hz"], sampling_rate))
                    if "sampling_rate_hz" in payload
                    else sampling_rate
                )
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
                    LOGGER.warning("Sampling metadata differs in %s", path)

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
    return EvaluationData(
        data=np.stack(arrays, axis=0).astype(np.float32),
        channel_labels=np.stack(channel_labels, axis=0).astype(np.uint8),
        sample_labels=np.asarray(sample_labels, dtype=np.uint8),
        has_labels=any_labels,
        sources=np.asarray(sources),
        sampling_rate_hz=sampling_rate,
        channel_spacing_m=spacing,
        files=files,
    )


def _find_checkpoint(checkpoint_dir: Path, model_name: str) -> Path:
    for suffix in (".pt", ".pth", ".pkl"):
        candidate = checkpoint_dir / f"{model_name}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint for model {model_name!r} found in {checkpoint_dir}; "
        "run scripts/train.py first or pass --checkpoint."
    )


def _load_metadata(checkpoint_path: Path) -> dict[str, Any]:
    candidates = [
        checkpoint_path.with_name(f"{checkpoint_path.stem}_metadata.json"),
        checkpoint_path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                LOGGER.warning("Could not parse metadata file %s", candidate)
    return {}


def _restore_model(checkpoint_path: Path, metadata: dict[str, Any], config: dict[str, Any]):
    checkpoint_stem = checkpoint_path.stem
    model_name = str(
        metadata.get("model_name")
        or (checkpoint_stem if checkpoint_stem in available_models() else None)
        or config.get("model", {}).get("name")
        or checkpoint_stem
    )
    if model_name not in available_models():
        raise ValueError(f"Unsupported model name {model_name!r} in metadata")
    model_params = dict(
        metadata.get("model_params")
        or config.get("model", {}).get("params", {}).get(model_name, {})
    )
    prototype = build_model(
        model_name,
        model_params=model_params,
        training_config=config.get("training", {}),
    )
    model = type(prototype).load(checkpoint_path)
    LOGGER.info("Loaded %s from %s", model_name, checkpoint_path)
    return model_name, model


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained DAS anomaly model.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "exp_default.yaml"),
        help="YAML experiment configuration.",
    )
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path.")
    parser.add_argument(
        "--model",
        choices=available_models(),
        default=None,
        help="Model name used to locate the default checkpoint.",
    )
    parser.add_argument("--test-dir", default=None, help="Override processed/test.")
    parser.add_argument("--train-dir", default=None, help="Processed/train for calibration fallback.")
    parser.add_argument("--output-dir", default=None, help="Prediction output directory.")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold.")
    parser.add_argument("--limit", type=int, default=None, help="Limit test .npz files.")
    parser.add_argument(
        "--save-plots",
        dest="save_plots",
        action="store_true",
        default=True,
        help="Save score and channel-profile plots (enabled by default).",
    )
    parser.add_argument(
        "--no-plots",
        dest="save_plots",
        action="store_false",
        help="Skip plot generation and only write arrays/metrics.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = load_config(args.config)
    log_dir = resolve_path(config.get("paths", {}).get("log_dir", "outputs/logs"))
    experiment = ExperimentLogger(
        "evaluate",
        log_dir,
        config=config,
        level=str(config.get("logging", {}).get("level", "INFO")),
    )
    global LOGGER
    LOGGER = experiment.logger
    root_logger = logging.getLogger()
    root_logger.setLevel(experiment.logger.level)
    for handler in experiment.logger.handlers:
        if handler not in root_logger.handlers:
            root_logger.addHandler(handler)

    try:
        checkpoint_path = (
            resolve_path(args.checkpoint)
            if args.checkpoint
            else _find_checkpoint(
                resolve_path(
                    config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints")
                ),
                str(
                    args.model
                    or config.get("model", {}).get("name", "stae_convlstm")
                ),
            )
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        metadata = _load_metadata(checkpoint_path)
        model_name, model = _restore_model(checkpoint_path, metadata, config)

        processed_root = resolve_path(
            config.get("paths", {}).get("processed_dir", "data/processed")
        )
        test_dir = (
            resolve_path(args.test_dir) if args.test_dir else processed_root / "test"
        )
        dataset = load_evaluation_data(test_dir, limit=args.limit)
        LOGGER.info(
            "Loaded %d test windows from %d source(s); shape=%s fs=%.2f Hz",
            len(dataset),
            len(np.unique(dataset.sources)),
            dataset.data.shape[1:],
            dataset.sampling_rate_hz,
        )

        evaluation_cfg = config.get("evaluation", {})
        scorer_state = metadata.get("scorer") or {}
        scorer = AnomalyScorer(
            model,
            topk_channels=int(
                scorer_state.get(
                    "topk_channels", evaluation_cfg.get("topk_channels", 3)
                )
            ),
            aggregation=str(scorer_state.get("aggregation", "topk")),
            normalization=str(scorer_state.get("normalization", "robust")),
            sampling_rate_hz=float(
                scorer_state.get("sampling_rate_hz", dataset.sampling_rate_hz)
            ),
        )
        calibration_scores: np.ndarray | None = None
        if scorer_state:
            scorer.load_state(scorer_state)
            LOGGER.info("Restored spatial calibration from training metadata")
        else:
            train_dir = (
                resolve_path(args.train_dir)
                if args.train_dir
                else processed_root / "train"
            )
            if train_dir.is_dir():
                LOGGER.warning(
                    "Checkpoint has no calibration metadata; calibrating on %s",
                    train_dir,
                )
                calibration_data = load_evaluation_data(train_dir, limit=None)
                normal = calibration_data.sample_labels == 0
                if normal.any():
                    scorer.fit(calibration_data.data[normal])
                    calibration_scores = scorer.sample_scores(calibration_data.data[normal])
                else:
                    LOGGER.warning("No normal windows available for calibration")
            else:
                LOGGER.warning("No calibration data available; using unnormalised scores")

        scores = scorer.score(dataset.data)
        sample_scores = scores.sample_scores.astype(np.float32)
        channel_scores = scores.channel_scores.astype(np.float32)

        threshold: float | None = None
        if args.threshold is not None:
            threshold = float(args.threshold)
        elif metadata.get("threshold") is not None:
            threshold = float(metadata["threshold"])
        elif (
            str(evaluation_cfg.get("threshold_method", "percentile")) == "fixed"
            and evaluation_cfg.get("fixed_threshold") is not None
        ):
            threshold = float(evaluation_cfg["fixed_threshold"])
        elif calibration_scores is not None:
            threshold = select_threshold(
                calibration_scores,
                method="percentile",
                percentile=float(evaluation_cfg.get("threshold_percentile", 99.0)),
            )
        else:
            threshold = float(
                np.percentile(
                    sample_scores,
                    float(evaluation_cfg.get("threshold_percentile", 99.0)),
                )
            )
            LOGGER.warning(
                "No threshold in checkpoint or CLI; using test-set percentile %.1f",
                float(evaluation_cfg.get("threshold_percentile", 99.0)),
            )

        metrics: dict[str, Any] = {}
        if dataset.has_labels:
            tolerance_channels = int(
                evaluation_cfg.get("localization_tolerance_channels", 2)
            )
            metrics = evaluate_all(
                dataset.sample_labels,
                sample_scores,
                threshold=threshold,
                channel_labels=dataset.channel_labels,
                channel_scores=channel_scores,
                channel_spacing_m=dataset.channel_spacing_m,
                tolerance_channels=tolerance_channels,
            )
            localization_aggregation = str(
                evaluation_cfg.get("localization_aggregation", "vote")
            )
            # ``vote`` needs the unnormalised residual so that the per-window
            # winner reflects absolute energy.  Averaging profiles instead uses
            # the robustly normalised channel scores, which removes the large
            # per-channel scale differences of the moving-load background and is
            # what makes weak, transient defects accumulate over a pass.
            localization_scores = np.asarray(
                scores.raw_channel_scores
                if localization_aggregation == "vote"
                else scores.channel_scores,
                dtype=np.float64,
            )
            source_score_aggregation = str(
                metadata.get(
                    "pass_score_aggregation",
                    evaluation_cfg.get("source_score_aggregation", "mean"),
                )
            )
            pass_threshold = metadata.get("pass_threshold")
            metrics["detection_by_source"] = detection_metrics_by_source(
                dataset.sample_labels,
                sample_scores,
                dataset.sources,
                score_aggregation=source_score_aggregation,
                threshold=float(pass_threshold) if pass_threshold is not None else None,
            )
            metrics["localization_by_source"] = localization_metrics_by_source(
                dataset.channel_labels,
                localization_scores,
                dataset.sources,
                channel_spacing_m=dataset.channel_spacing_m,
                tolerance_channels=tolerance_channels,
                aggregation=localization_aggregation,
                sample_scores=sample_scores,
                threshold=threshold,
                detected_only=bool(
                    evaluation_cfg.get("localization_detected_only", False)
                ),
            )
            interval_cfg = evaluation_cfg.get("localization_interval", {}) or {}
            metrics["localization_intervals"] = interval_metrics_by_source(
                dataset.channel_labels,
                localization_scores,
                dataset.sources,
                channel_spacing_m=dataset.channel_spacing_m,
                tolerance_channels=tolerance_channels,
                aggregation=localization_aggregation,
                threshold_mad=float(interval_cfg.get("threshold_mad", 1.0)),
                merge_gap=int(interval_cfg.get("merge_gap", 2)),
                min_width=int(interval_cfg.get("min_width", 1)),
                smoothing_sigma=float(interval_cfg.get("smoothing_sigma", 1.0)),
                max_intervals=int(interval_cfg.get("max_intervals", 3)),
            )
            experiment.info(
                "Detection: ROC AUC=%.4f F1=%.4f FAR=%.4f | localisation mean error=%.2f ch",
                metrics.get("roc_auc", float("nan")),
                metrics.get("f1", float("nan")),
                metrics.get("false_alarm_rate", float("nan")),
                metrics.get("localization", {}).get(
                    "mean_error_channels", float("nan")
                ),
            )
            pass_detection = metrics["detection_by_source"]
            experiment.info(
                "Pass-level detection: ROC AUC=%.4f F1=%.4f "
                "threshold=%.3f over %d source(s)",
                pass_detection.get("roc_auc", float("nan")),
                pass_detection.get("f1", float("nan")),
                pass_detection.get("threshold", float("nan")),
                pass_detection.get("n_sources", 0),
            )
            pass_localization = metrics["localization_by_source"]
            # Time-integrated pass detector: aggregate the channel residual over
            # every window of a pass, then keep the strongest channels.  This is
            # robust to the transient nature of impact/resonance defects, which
            # occupy only part of a pass and are averaged away by window-score
            # fusion.
            profile_scores = pass_profile_scores(
                localization_scores,
                dataset.sources,
                aggregation=localization_aggregation,
                statistic="topk_mean",
                topk=3,
            )
            metrics["detection_by_source_profile"] = detection_metrics_by_source(
                dataset.sample_labels,
                [profile_scores[str(source)] for source in dataset.sources],
                dataset.sources,
                score_aggregation="mean",
            )
            profile_detection = metrics["detection_by_source_profile"]
            experiment.info(
                "Pass-level profile detection: ROC AUC=%.4f over %d source(s)",
                profile_detection.get("roc_auc", float("nan")),
                profile_detection.get("n_sources", 0),
            )
            experiment.info(
                "Pass-level localisation: mean error=%.2f ch over %d source(s)",
                pass_localization.get("mean_error_channels", float("nan")),
                pass_localization.get("n_sources", 0),
            )
            interval_localization = metrics["localization_intervals"]
            experiment.info(
                "Interval localisation: hit=%.2f IoU=%.2f coverage=%.2f "
                "start/end error=%.2f/%.2f ch",
                interval_localization.get("hit_rate", float("nan")),
                interval_localization.get("mean_iou", float("nan")),
                interval_localization.get("mean_coverage", float("nan")),
                interval_localization.get("mean_start_error_channels", float("nan")),
                interval_localization.get("mean_end_error_channels", float("nan")),
            )
        else:
            LOGGER.warning(
                "Test data has no ground-truth labels; writing scores without metrics"
            )
            metrics = {
                "n_samples": int(len(dataset)),
                "threshold": float(threshold),
                "score_mean": float(sample_scores.mean()),
                "score_std": float(sample_scores.std()),
                "labels_available": False,
            }

        prediction_dir = (
            resolve_path(args.output_dir)
            if args.output_dir
            else resolve_path(
                config.get("paths", {}).get("prediction_dir", "outputs/predictions")
            )
        )
        prediction_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = prediction_dir / f"predictions_{model_name}.npz"
        metrics_path = prediction_dir / f"metrics_{model_name}.json"
        np.savez_compressed(
            predictions_path,
            sample_scores=sample_scores,
            channel_scores=channel_scores,
            raw_channel_scores=scores.raw_channel_scores.astype(np.float32),
            sample_labels=dataset.sample_labels,
            channel_labels=dataset.channel_labels,
            has_labels=np.asarray([int(dataset.has_labels)], dtype=np.uint8),
            sources=dataset.sources,
            threshold=np.asarray([threshold], dtype=np.float32),
            sampling_rate_hz=np.asarray([dataset.sampling_rate_hz], dtype=np.float32),
            channel_spacing_m=np.asarray([dataset.channel_spacing_m], dtype=np.float32),
        )
        metrics_payload = {
            "model_name": model_name,
            "checkpoint": str(checkpoint_path),
            "threshold": float(threshold),
            "pass_threshold": metadata.get("pass_threshold"),
            "metrics": metrics,
        }
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        interval_rows = metrics.get("localization_intervals", {}).get("sources", [])
        if interval_rows:
            interval_path = prediction_dir / f"intervals_{model_name}.csv"
            with interval_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "source",
                        "predicted_start_channel",
                        "predicted_end_channel",
                        "true_start_channel",
                        "true_end_channel",
                        "iou",
                        "coverage",
                        "channel_precision",
                        "hit",
                        "distance_channels",
                        "n_predicted_intervals",
                        "n_true_intervals",
                        "predicted_intervals",
                        "true_intervals",
                    ],
                )
                writer.writeheader()
                for row in interval_rows:
                    writer.writerow(
                        {
                            "source": row["source"],
                            "predicted_start_channel": row["predicted_interval"][0],
                            "predicted_end_channel": row["predicted_interval"][1],
                            "true_start_channel": row["true_interval"][0],
                            "true_end_channel": row["true_interval"][1],
                            "iou": f"{row['iou']:.4f}",
                            "coverage": f"{row['coverage']:.4f}",
                            "channel_precision": f"{row.get('channel_precision', 0.0):.4f}",
                            "hit": int(bool(row["hit"])),
                            "distance_channels": row["distance_channels"],
                            "n_predicted_intervals": row.get("n_intervals", 1),
                            "n_true_intervals": row.get("n_true_intervals", 1),
                            "predicted_intervals": ";".join(
                                f"{a}-{b}" for a, b in row.get("predicted_intervals", [])
                            ),
                            "true_intervals": ";".join(
                                f"{a}-{b}" for a, b in row.get("true_intervals", [])
                            ),
                        }
                    )
            experiment.info("Anomaly intervals saved to %s", interval_path)
        source_rows = metrics.get("detection_by_source", {}).get("sources", [])
        if source_rows:
            source_path = prediction_dir / f"source_scores_{model_name}.csv"
            with source_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["source", "label", "score", "n_windows"],
                )
                writer.writeheader()
                for row in source_rows:
                    writer.writerow(
                        {
                            "source": row["source"],
                            "label": row["label"],
                            "score": f"{row['score']:.6f}",
                            "n_windows": row["n_windows"],
                        }
                    )
            experiment.info("Pass-level scores saved to %s", source_path)
        experiment.log_metrics(
            {
                "threshold": float(threshold),
                "n_samples": int(len(dataset)),
                **{
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                },
            },
            step=1,
        )
        experiment.info("Predictions saved to %s", predictions_path)
        experiment.info("Metrics saved to %s", metrics_path)

        if args.save_plots:
            labels = dataset.sample_labels if dataset.has_labels else None
            figure, _ = plot_sample_scores(
                sample_scores, labels=labels, threshold=threshold
            )
            save_figure(figure, prediction_dir / f"sample_scores_{model_name}.png")
            if dataset.has_labels and np.any(dataset.sample_labels):
                profile_index = int(np.flatnonzero(dataset.sample_labels == 1)[0])
            else:
                profile_index = int(np.argmax(sample_scores))
            positions = (
                np.arange(dataset.data.shape[1], dtype=np.float64)
                * dataset.channel_spacing_m
            )
            figure, _ = plot_score_profile(
                channel_scores[profile_index],
                positions_m=positions,
                title=f"Channel anomaly profile (window {profile_index})",
            )
            save_figure(figure, prediction_dir / f"profile_{model_name}.png")
            source_details = metrics.get("localization_by_source", {}).get(
                "sources", []
            )
            interval_sources = metrics.get("localization_intervals", {}).get(
                "sources", []
            )
            if source_details:
                source = source_details[0]["source"]
                source_mask = dataset.sources.astype(str) == str(source)
                if source_mask.any():
                    if localization_aggregation == "vote":
                        winners = np.argmax(
                            localization_scores[source_mask], axis=1
                        )
                        profile = np.bincount(
                            winners, minlength=localization_scores.shape[1]
                        ).astype(np.float64)
                        profile /= max(int(source_mask.sum()), 1)
                        profile_label = "Top-1 channel vote fraction"
                    else:
                        profile = aggregate_channel_profile(
                            localization_scores[source_mask],
                            localization_aggregation,
                        )
                        profile_label = (
                            f"{localization_aggregation.title()} anomaly score"
                        )
                    figure, axes = plot_score_profile(
                        profile,
                        positions_m=positions,
                        title=f"Pass-level anomaly profile - {source}",
                        label=profile_label,
                    )
                    if localization_aggregation == "vote":
                        axes.set_ylabel("Vote fraction")
                    true_channels = np.asarray(
                        source_details[0]["true_channels"], dtype=np.float64
                    )
                    if true_channels.size:
                        axes.axvspan(
                            (true_channels.min() - 0.5) * dataset.channel_spacing_m,
                            (true_channels.max() + 0.5) * dataset.channel_spacing_m,
                            color="#2ca02c",
                            alpha=0.18,
                            label="True region",
                        )
                    if interval_sources:
                        predicted = interval_sources[0]["predicted_interval"]
                        axes.axvspan(
                            (predicted[0] - 0.5) * dataset.channel_spacing_m,
                            (predicted[1] + 0.5) * dataset.channel_spacing_m,
                            color="#ff7f0e",
                            alpha=0.18,
                            label="Predicted interval",
                        )
                    axes.legend(loc="best")
                    save_figure(
                        figure, prediction_dir / f"profile_pass_{model_name}.png"
                    )

        print(json.dumps(metrics_payload, indent=2, ensure_ascii=False, default=_json_default))
        return 0
    finally:
        experiment.close()


if __name__ == "__main__":
    raise SystemExit(main())

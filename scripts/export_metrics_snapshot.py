"""Collect every evaluated point into one flat table.

The per-arm ``metrics_*.json`` files are the source of truth for the report, but
they live under ``outputs/`` which is not tracked.  This script folds them into a
single small CSV so the numbers quoted in ``reports/snr_benchmark.md`` can be
version-controlled and diffed without shipping 9 GB of artifacts.

Usage::

    python scripts/export_metrics_snapshot.py
    python scripts/export_metrics_snapshot.py --arms morph_j00_s5 oof_impulse_train
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import sys
from typing import Any, Iterable

LOGGER = logging.getLogger("das.export")

FIELDS = (
    "arm",
    "point",
    "seed",
    "model",
    "channel_auc",
    "channel_auc_active_windows",
    "profile_auc",
    "pass_auc",
    "pass_recall",
    "pass_false_alarm_rate",
    "anomaly_type",
    "defect_jitter",
)


def _seed_of(path: pathlib.Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("seed_"):
            return part
    return "seed_0"


def _point_of(path: pathlib.Path, arm_dir: pathlib.Path) -> str:
    rel = path.relative_to(arm_dir)
    # outputs/<arm>/<point>/seed_x/predictions/metrics_y.json
    return rel.parts[0] if len(rel.parts) > 1 else "root"


def _config_for(metrics_path: pathlib.Path) -> dict[str, Any]:
    for parent in metrics_path.parents:
        candidate = parent / "config.yaml"
        if candidate.exists():
            try:
                import yaml

                return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            except Exception:  # noqa: BLE001 - a missing config must not stop export
                return {}
    return {}


def collect(arms: Iterable[pathlib.Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm_dir in arms:
        for metrics_path in sorted(arm_dir.rglob("metrics_*.json")):
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics") or {}
            config = _config_for(metrics_path)
            simulation = config.get("simulation") or {}
            source = metrics.get("detection_by_source") or {}
            profile = metrics.get("detection_by_source_profile") or {}
            rows.append(
                {
                    "arm": arm_dir.name,
                    "point": _point_of(metrics_path, arm_dir),
                    "seed": _seed_of(metrics_path),
                    "model": payload.get("model_name") or metrics_path.stem.split("_", 1)[-1],
                    "channel_auc": metrics.get("channel_auc"),
                    "channel_auc_active_windows": metrics.get("channel_auc_active_windows"),
                    "profile_auc": profile.get("roc_auc"),
                    "pass_auc": source.get("roc_auc"),
                    "pass_recall": source.get("recall"),
                    "pass_false_alarm_rate": source.get("false_alarm_rate"),
                    "anomaly_type": simulation.get("anomaly_type"),
                    "defect_jitter": simulation.get("defect_frequency_jitter"),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=pathlib.Path, default=root / "outputs")
    parser.add_argument(
        "--dest", type=pathlib.Path, default=root / "results" / "metrics_snapshot.csv"
    )
    parser.add_argument("--arms", nargs="*", default=None, help="restrict to these arms")
    args = parser.parse_args(argv)

    if not args.outputs.is_dir():
        LOGGER.error("outputs directory not found: %s", args.outputs)
        return 1
    if args.arms:
        arms = [args.outputs / name for name in args.arms]
        missing = [str(p) for p in arms if not p.is_dir()]
        if missing:
            LOGGER.error("unknown arms: %s", ", ".join(missing))
            return 1
    else:
        arms = sorted(p for p in args.outputs.iterdir() if p.is_dir())

    rows = collect(arms)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    with args.dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("wrote %d rows from %d arms to %s", len(rows), len(arms), args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

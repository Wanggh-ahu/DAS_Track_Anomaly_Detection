"""Diagnostics for a processed DAS split.

Answers the question that decides whether an SNR benchmark is meaningful at
all: *is the labelled defect actually inside the retained windows?*  If a pass
is labelled anomalous but its defect falls outside the tiled time range, the
pass contributes a positive label with no signal and every model's AUC is
deflated by an amount nobody can see.

Two independent checks are reported per pass:

``cover``
    Fraction of the defect's active time (taken from the simulator metadata)
    that overlaps the union of retained windows.  This is exact.

``ratio``
    Peak per-window ratio between the RMS of the labelled channels and the
    median RMS of healthy channels.  A value near 1.0 means the defect is
    invisible in the processed data even when it is covered.

Usage::

    python scripts/diagnose_processed.py data/processed/test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Length of the intersection of two intervals."""

    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _load_window(path: Path) -> tuple[np.ndarray, dict[str, Any], int]:
    with np.load(path, allow_pickle=False) as payload:
        data = np.asarray(payload["data"], dtype=np.float32)
        labels = np.asarray(payload["channel_labels"], dtype=np.uint8)
        event = json.loads(str(payload["event"])) if "event" in payload else {}
        metadata = json.loads(str(payload["metadata"])) if "metadata" in payload else {}
    metadata["channel_labels"] = labels
    start = int(event.get("window_start_sample", 0))
    return data, metadata, start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose a processed split.")
    parser.add_argument("split_dir", nargs="?", default="data/processed/test")
    parser.add_argument("--min-coverage", type=float, default=0.75)
    args = parser.parse_args(argv)

    root = Path(args.split_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    source_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not source_dirs:
        raise SystemExit(f"no source directories under {root}")

    print(
        f"{'source':>34} {'win':>4} {'anom':>5} {'cover':>6} {'ratio':>7} {'peak_z':>7}"
    )
    coverages: list[float] = []
    ratios: list[float] = []

    for source_dir in source_dirs:
        files = sorted(source_dir.glob("window_*.npz"))
        if not files:
            continue
        windows: list[np.ndarray] = []
        labels = None
        metadata: dict[str, Any] = {}
        starts: list[int] = []
        for path in files:
            data, meta, start = _load_window(path)
            windows.append(data)
            labels = meta["channel_labels"]
            metadata = meta
            starts.append(start)

        source_meta = metadata.get("source_metadata", {}) or {}
        defect = source_meta.get("defect", {}) or {}
        fs = float(source_meta.get("sampling_rate_hz", 1000.0))
        active = defect.get("active_time_ranges_s") or []
        flagged = bool(labels is not None and labels.any())

        window_len = int(windows[0].shape[1])
        coverage = float("nan")
        if flagged and active:
            window_intervals = [
                (start / fs, (start + window_len) / fs) for start in starts
            ]
            active_total = sum(stop - start for start, stop in active)
            covered = 0.0
            for start, stop in active:
                pieces = [
                    _intervals_overlap((start, stop), window) for window in window_intervals
                ]
                covered += min(sum(pieces), stop - start)
            coverage = covered / max(active_total, 1e-9)

        ratio = float("nan")
        peak_z = float("nan")
        if flagged:
            best = 0.0
            best_z = -np.inf
            for window in windows:
                rms = np.sqrt(np.mean(np.square(window), axis=1))
                defect_rms = float(rms[labels.astype(bool)].mean())
                healthy_rms = float(np.median(rms[~labels.astype(bool)]))
                best = max(best, defect_rms / max(healthy_rms, 1e-12))
                spread = float(np.median(np.abs(rms - np.median(rms))) * 1.4826)
                best_z = max(
                    best_z, (defect_rms - float(np.median(rms))) / max(spread, 1e-12)
                )
            ratio, peak_z = best, best_z
            coverages.append(coverage)
            ratios.append(ratio)

        print(
            f"{source_dir.name:>34} {len(windows):>4} {int(flagged):>5} "
            f"{coverage:>6.2f} {ratio:>7.3f} {peak_z:>7.2f}"
        )

    if coverages:
        cover = np.asarray(coverages, dtype=float)
        ratio = np.asarray(ratios, dtype=float)
        print(
            f"\nanomalous passes: n={cover.size} "
            f"median coverage={np.nanmedian(cover):.2f} "
            f"min={np.nanmin(cover):.2f}"
        )
        print(f"passes with coverage >= {args.min_coverage:.2f}: {np.mean(cover >= args.min_coverage):.0%}")
        print(f"median peak channel ratio: {np.nanmedian(ratio):.3f}")
        if np.mean(cover >= args.min_coverage) < 0.9:
            print(
                "WARNING: the defect often falls outside the retained windows. "
                "Widen pass_detection (pad_s / merge_gap_s, lower threshold_sigma) "
                "before trusting any AUC number.",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

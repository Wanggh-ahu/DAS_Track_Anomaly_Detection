"""Multi-seed analysis of the morphology sweep, with the statistics stated honestly.

Every claim in the benchmark about "model A beats model B" has so far rested on a
single run, and the retraining studies showed AUC moves by up to 0.04 between
runs of the *same* configuration.  A difference smaller than that is not a
finding; it is noise with a story attached.

This script reads the per-seed metric files of several sweep directories and
reports, for each comparison:

* mean +- sd over seeds (what the seed spread actually is),
* the paired per-seed difference (same seed, same data for both models, so the
  comparison is paired and far less noisy than the marginal spread suggests),
* how many seeds agree in sign.

With five seeds a sign test bottoms out at p = 2 * 2**-5 = 0.0625 (one-sided
0.031); with three seeds it cannot go below 0.25.  That floor is printed so the
reader can see when a comparison is simply underpowered rather than negative.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _collect(directory: Path) -> dict[str, dict[str, dict[str, float]]]:
    """``directory`` -> ``model`` -> ``seed`` -> metrics."""

    table: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for path in sorted(directory.rglob("predictions/metrics_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
        except (OSError, KeyError, ValueError):
            continue
        model = str(payload.get("model_name") or path.stem.replace("metrics_", ""))
        # Seed directories are named seed_0, seed_1, ...; a single-seed sweep
        # has no seed directory at all.
        seed = next(
            (part.name for part in path.parents if part.name.startswith("seed_")),
            "seed_0",
        )
        profile = metrics.get("detection_by_source_profile", {}) or {}
        table[model][seed] = {
            "channel_auc": float(metrics.get("channel_auc", float("nan"))),
            "profile_auc": float(profile.get("roc_auc", float("nan"))),
            "profile_recall": float(profile.get("recall", float("nan"))),
            "profile_far": float(profile.get("false_alarm_rate", float("nan"))),
        }
    return table


def _sign_test(differences: np.ndarray) -> float:
    """Exact two-sided sign-test p-value (no ties dropped)."""

    nonzero = differences[differences != 0]
    n = int(nonzero.size)
    if n == 0:
        return 1.0
    positive = int(np.sum(nonzero > 0))
    from math import comb

    tail = sum(comb(n, k) for k in range(0, min(positive, n - positive) + 1))
    return float(min(1.0, 2.0 * tail / (2**n)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="+",
        help="Entries of the form <label>=<sweep dir>, e.g. j00=outputs/morph_j00_s5",
    )
    parser.add_argument(
        "--metric",
        default="channel_auc",
        choices=["channel_auc", "profile_auc", "profile_recall", "profile_far"],
    )
    parser.add_argument(
        "--compare",
        nargs="*",
        default=["stft_ae:bandpass", "stft_ae:ae_1d", "ae_1d:bandpass"],
        help="Pairs 'A:B' meaning A minus B.",
    )
    args = parser.parse_args(argv)

    arms: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for entry in args.directories:
        if "=" not in entry:
            raise SystemExit(f"expected <label>=<dir>, got {entry!r}")
        label, directory = entry.split("=", 1)
        arms[label] = _collect(Path(directory))

    models = sorted({m for arm in arms.values() for m in arm})
    print(f"metric: {args.metric}\n")
    header = f"{'arm':>6} | " + " | ".join(f"{m:>22}" for m in models)
    print(header)
    print("-" * len(header))
    for label, arm in arms.items():
        cells = []
        for model in models:
            values = [
                seed[args.metric]
                for seed in arm.get(model, {}).values()
                if np.isfinite(seed[args.metric])
            ]
            if not values:
                cells.append(f"{'-':>22}")
            else:
                cells.append(f"{np.mean(values):8.4f} +- {np.std(values, ddof=1):6.4f} (n={len(values)})")
        print(f"{label:>6} | " + " | ".join(cells))

    for label, arm in arms.items():
        print(f"\n=== {label}: paired comparisons ({args.metric}) ===")
        for pair in args.compare:
            first, second = (part.strip() for part in pair.split(":"))
            common = sorted(set(arm.get(first, {})) & set(arm.get(second, {})))
            if len(common) < 2:
                print(f"  {first} - {second}: not enough common seeds")
                continue
            differences = np.asarray(
                [
                    arm[first][seed][args.metric] - arm[second][seed][args.metric]
                    for seed in common
                ]
            )
            p_value = _sign_test(differences)
            print(
                f"  {first:>10} - {second:<10}: "
                f"{differences.mean():+.4f} +- {differences.std(ddof=1):.4f} "
                f"(n={len(common)}), per-seed "
                + ", ".join(f"{value:+.3f}" for value in differences)
                + f" | sign-test p={p_value:.3f}"
            )

    labels = list(arms)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            left, right = labels[i], labels[j]
            print(f"\n=== across arms: {left} vs {right} ({args.metric}) ===")
            for model in models:
                common = sorted(set(arms[left].get(model, {})) & set(arms[right].get(model, {})))
                if len(common) < 2:
                    continue
                differences = np.asarray(
                    [
                        arms[left][model][seed][args.metric]
                        - arms[right][model][seed][args.metric]
                        for seed in common
                    ]
                )
                print(
                    f"  {model:>10}: {differences.mean():+.4f} "
                    f"+- {differences.std(ddof=1):.4f} (n={len(common)}), "
                    f"sign-test p={_sign_test(differences):.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

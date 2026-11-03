"""How many normal passes does a pass-level threshold actually need?

Section 3 of the report showed the current estimator fails to fire in roughly a
third of calibration draws, because at n = 21 and alpha = 0.05 the conformal
quantile is the sample maximum.  The fix proposed there was "collect more normal
passes".  This script measures how many, instead of guessing.

It takes a single dataset whose normal pool is deliberately large (400 passes,
so 140 reach calibration at the 0.35 split), then asks the same question at
increasing calibration sizes:

* how far does the realised false-alarm rate wander across calibration draws,
* how often does the detector end up never firing,
* and what does the resulting operating point look like on a disjoint test set.

Everything is measured against normal passes only, so no anomaly label is used
to choose the answer.
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

from scripts.train import load_config, resolve_path  # noqa: E402
from scripts.validate_pass_threshold import _model, _profile_scores, _load_windows  # noqa: E402
from src.evaluation.conformal import split_conformal_quantile  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="*", default=["bandpass", "stft_ae"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--sizes", nargs="*", type=int, default=[21, 40, 70, 100, 140, 200, 300]
    )
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)

    config = load_config(resolve_path(args.config))
    train_set = _load_windows(config, "train")
    test_set = _load_windows(config, "test")
    rng = np.random.default_rng(args.seed)
    report: dict[str, Any] = {}

    for model_name in args.models:
        model = _model(config, model_name)
        normal_scores, _ = _profile_scores(model, train_set)
        test_scores, test_sources = _profile_scores(model, test_set)
        labels = np.asarray(
            [
                1 if any(tok in str(s) for tok in ("anomaly", "anomal")) else 0
                for s in test_sources
            ]
        )
        pool = int(normal_scores.size)
        print(
            f"\n{model_name}: {pool} normal passes, "
            f"{int(labels.sum())} anomalous / {int((labels == 0).sum())} normal test passes, "
            f"alpha={args.alpha}"
        )
        print(
            f"{'n_cal':>6} | {'mean FAR':>9} | {'sd FAR':>7} | {'P(FAR=0)':>8} | "
            f"{'P(FAR>2a)':>9} | {'test FAR':>9} | {'test recall':>11}"
        )
        rows = []
        for size in args.sizes:
            if size >= pool:
                print(f"{size:>6} | skipped (pool is {pool})")
                continue
            fars, test_fars, test_recalls = [], [], []
            for _ in range(args.draws):
                perm = rng.permutation(pool)
                calibration = normal_scores[perm[:size]]
                held_out = normal_scores[perm[size:]]
                threshold = split_conformal_quantile(
                    calibration, alpha=args.alpha
                ).threshold
                fars.append(float(np.mean(held_out > threshold)))
                if labels.sum() and (labels == 0).sum():
                    test_fars.append(float(np.mean(test_scores[labels == 0] > threshold)))
                    test_recalls.append(
                        float(np.mean(test_scores[labels == 1] > threshold))
                    )
            fars = np.asarray(fars)
            row = {
                "n_cal": size,
                "mean_far": float(fars.mean()),
                "sd_far": float(fars.std()),
                "p_zero": float(np.mean(fars == 0.0)),
                "p_above_2alpha": float(np.mean(fars > 2 * args.alpha)),
                "test_far": float(np.mean(test_fars)) if test_fars else float("nan"),
                "test_far_sd": float(np.std(test_fars)) if test_fars else float("nan"),
                "test_recall": float(np.mean(test_recalls)) if test_recalls else float("nan"),
                "test_recall_sd": float(np.std(test_recalls)) if test_recalls else float("nan"),
            }
            rows.append(row)
            print(
                f"{size:>6} | {row['mean_far']:9.4f} | {row['sd_far']:7.4f} | "
                f"{row['p_zero']:8.2f} | {row['p_above_2alpha']:9.2f} | "
                f"{row['test_far']:9.4f} | {row['test_recall']:11.3f} "
                f"+- {row['test_recall_sd']:5.3f}"
            )
        report[model_name] = {"pool": pool, "rows": rows}

    out = PROJECT_ROOT / "outputs" / "calibration_size_curve.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

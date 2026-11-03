"""Does the pass-level threshold estimator actually hold its false-alarm rate?

The complaint that started this was concrete: the same model, retrained on the
same data, produced pass-level false-alarm rates of 10.5 % and 5.3 %, and after
a residual calibration it produced 0 % with recall 0.  Every one of those numbers
came from a conformal threshold that, at n = 21 calibration passes and
alpha = 0.05, is simply the largest calibration score.

This script measures that directly, without any test-set labels.  It takes the
real per-pass scores of the *normal* passes, splits them into 21 calibration and
the rest as pseudo-test, and asks each estimator for the false-alarm rate it
would have produced.  Repeat over many random splits and you get the stability
curve that the training log only shows one sample of.

Selection is made on this curve alone -- normal passes only, no anomaly labels --
so it is not tuned against the test set.
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
from src.evaluation.conformal import split_conformal_quantile  # noqa: E402
from src.evaluation.tail_threshold import gpd_threshold  # noqa: E402


def _load_windows(config: dict[str, Any], split: str):
    from scripts.evaluate import load_evaluation_data

    root = resolve_path(config["paths"]["processed_dir"]) / split
    return load_evaluation_data(root, window_label_mode="pass")


def _profile_scores(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    from src.evaluation.metrics import pass_profile_scores

    try:
        scores = model.channel_scores(
            dataset.data, sampling_rate_hz=dataset.sampling_rate_hz
        )
    except TypeError:
        # Waveform-space models do not need the sampling rate.
        scores = model.channel_scores(dataset.data)
    per_pass = pass_profile_scores(
        scores, dataset.sources, aggregation="mean", statistic="topk_mean", topk=3
    )
    sources = np.asarray(sorted(per_pass))
    return np.asarray([per_pass[s] for s in sources], dtype=float), sources


def _model(config: dict[str, Any], name: str):
    from src.models import build_model

    checkpoint_dir = resolve_path(config["paths"]["checkpoint_dir"])
    for suffix in (".pt", ".pth", ".pkl"):
        path = checkpoint_dir / f"{name}{suffix}"
        if path.is_file():
            break
    else:
        raise FileNotFoundError(f"no checkpoint for {name} in {checkpoint_dir}")
    params = dict(config.get("model", {}).get("params", {}).get(name, {}) or {})
    return type(build_model(name, params)).load(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="*", required=True, help="Point configs to pool.")
    parser.add_argument("--models", nargs="*", default=["stft_ae", "bandpass", "ae_1d"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-calibration", type=int, default=21)
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    estimators = {
        "conformal (current)": lambda s: split_conformal_quantile(
            s, alpha=args.alpha
        ).threshold,
        "GPD tail": lambda s: gpd_threshold(s, alpha=args.alpha).threshold,
    }
    summary: dict[str, Any] = {}

    for config_path in args.config:
        config = load_config(resolve_path(config_path))
        train_set = _load_windows(config, "train")
        test_set = _load_windows(config, "test")
        for model_name in args.models:
            model = _model(config, model_name)
            normal_scores, _ = _profile_scores(model, train_set)
            test_scores, test_sources = _profile_scores(model, test_set)
            labels = np.asarray(
                [
                    1
                    if any(tok in str(s) for tok in ("anomaly", "anomal"))
                    else 0
                    for s in test_sources
                ]
            )
            n_total = normal_scores.size
            if n_total <= args.n_calibration:
                print(f"{model_name}: only {n_total} normal passes, skipped")
                continue

            print(
                f"\n{config_path} :: {model_name}  "
                f"({n_total} normal passes, {labels.sum()} anomalous / "
                f"{(labels == 0).sum()} normal test passes)"
            )
            print(
                f"{'estimator':>20} | {'mean FAR':>9} | {'sd FAR':>7} | "
                f"{'P(FAR=0)':>8} | {'P(FAR>2a)':>9} | test recall @ full fit"
            )
            for label, estimator in estimators.items():
                fars = []
                for _ in range(args.splits):
                    perm = rng.permutation(n_total)
                    cal = normal_scores[perm[: args.n_calibration]]
                    rest = normal_scores[perm[args.n_calibration :]]
                    threshold = estimator(cal)
                    fars.append(float(np.mean(rest > threshold)))
                fars = np.asarray(fars)
                full_threshold = estimator(normal_scores)
                if labels.sum() > 0:
                    recall = float(np.mean(test_scores[labels == 1] > full_threshold))
                    far_test = float(np.mean(test_scores[labels == 0] > full_threshold))
                else:
                    recall, far_test = float("nan"), float("nan")
                print(
                    f"{label:>20} | {fars.mean():9.4f} | {fars.std():7.4f} | "
                    f"{np.mean(fars == 0.0):8.2f} | "
                    f"{np.mean(fars > 2 * args.alpha):9.2f} | "
                    f"{recall:.3f} (FAR {far_test:.3f})"
                )
                summary[f"{config_path}|{model_name}|{label}"] = {
                    "mean_far": float(fars.mean()),
                    "sd_far": float(fars.std()),
                    "p_zero": float(np.mean(fars == 0.0)),
                    "p_above_2alpha": float(np.mean(fars > 2 * args.alpha)),
                    "test_recall_full_fit": recall,
                    "test_far_full_fit": far_test,
                    "full_threshold": float(full_threshold),
                }

    out = PROJECT_ROOT / "outputs" / "threshold_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Automated SNR sensitivity benchmark for DAS anomaly detectors.

For every requested SNR the script

1. rewrites a copy of the benchmark configuration (data paths and
   ``simulation.snr_db``),
2. regenerates and preprocesses the synthetic dataset,
3. trains and evaluates every requested model,
4. parses the per-model ``metrics_<model>.json`` and extracts the window-level
   and pass-level ROC AUC,
5. finally plots AUC against SNR and writes a CSV/JSON summary.

The synthetic seed is held fixed across SNR, so every point of the curve is
evaluated on the *same* passes with only the defect amplitude changed.  That
makes the comparison paired and removes dataset variance from the curve.

Examples
--------
Full sweep with the default models::

    python scripts/run_snr_sweep.py

Quick check on two SNR points::

    python scripts/run_snr_sweep.py --snr -15 0 --models pca energy

Regenerate the figure from an existing summary::

    python scripts/run_snr_sweep.py --analyze-only
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODELS = ("pca", "energy", "ae_1d", "spatial_masked_ae")
MODEL_LABELS = {
    "pca": "PCA reconstruction",
    "energy": "Energy threshold",
    "isolation_forest": "Isolation Forest",
    "ae_1d": "1-D autoencoder",
    "stae_convlstm": "STAE-ConvLSTM",
    "spatial_masked_ae": "Masked AE + memory",
}
MODEL_STYLES = {
    "pca": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "energy": {"color": "#7f7f7f", "marker": "s", "linestyle": "--"},
    "isolation_forest": {"color": "#bcbd22", "marker": "^", "linestyle": "-."},
    "ae_1d": {"color": "#2ca02c", "marker": "D", "linestyle": "-"},
    "stae_convlstm": {"color": "#9467bd", "marker": "v", "linestyle": "-"},
    "spatial_masked_ae": {"color": "#d62728", "marker": "*", "linestyle": "-"},
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def snr_tag(snr: float) -> str:
    sign = "p" if snr >= 0 else "m"
    return f"snr_{sign}{abs(snr):.0f}"


def run_command(command: Sequence[str], log_path: Path) -> tuple[int, str]:
    """Run a subprocess with its output redirected to a log file."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        process = subprocess.run(
            list(command),
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - started
    tail = ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        tail = "\n".join(text[-25:])
    except OSError:
        pass
    print(f"    exit={process.returncode} in {elapsed:.1f}s -> {log_path.name}", flush=True)
    return process.returncode, tail


def _portable(path: Path) -> str:
    """Prefer a project-relative path so the config stays reproducible."""

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def patch_config(
    base: dict[str, Any],
    workdir: Path,
    snr: float,
    seed: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.setdefault("paths", {})
    config["paths"].update(
        {
            "synthetic_dir": _portable(workdir / "synthetic"),
            "processed_dir": _portable(workdir / "processed"),
            "checkpoint_dir": _portable(workdir / "checkpoints"),
            "prediction_dir": _portable(workdir / "predictions"),
            "log_dir": _portable(workdir / "logs"),
        }
    )
    config.setdefault("simulation", {})["snr_db"] = float(snr)
    config["simulation"]["seed"] = int(seed)
    return config


def parse_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("metrics", {}) or {}
    pass_detection = metrics.get("detection_by_source", {}) or {}
    profile_detection = metrics.get("detection_by_source_profile", {}) or {}
    intervals = metrics.get("localization_intervals", {}) or {}
    localization = metrics.get("localization", {}) or {}
    return {
        "window_auc": _as_float(metrics.get("roc_auc")),
        "window_f1": _as_float(metrics.get("f1")),
        "false_alarm_rate": _as_float(metrics.get("false_alarm_rate")),
        "pass_auc": _as_float(pass_detection.get("roc_auc")),
        "pass_f1": _as_float(pass_detection.get("f1")),
        "profile_auc": _as_float(profile_detection.get("roc_auc")),
        "profile_f1": _as_float(profile_detection.get("f1")),
        "profile_far": _as_float(profile_detection.get("false_alarm_rate")),
        "n_test_sources": int(pass_detection.get("n_sources", 0) or 0),
        "interval_f1": _as_float(intervals.get("interval_f1")),
        "interval_iou": _as_float(intervals.get("mean_iou")),
        "interval_coverage": _as_float(intervals.get("mean_coverage")),
        "localization_error_channels": _as_float(
            localization.get("mean_error_channels")
        ),
        "threshold": _as_float(payload.get("threshold")),
    }


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def evaluate_point(
    config_path: Path,
    workdir: Path,
    models: Sequence[str],
    epochs: int | None,
    skip_existing: bool,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    python = sys.executable

    preprocess_log = workdir / "logs" / "preprocess.txt"
    code, tail = run_command(
        [
            python,
            str(PROJECT_ROOT / "scripts" / "run_preprocess.py"),
            "--config",
            str(config_path),
            "--synthetic",
            "--overwrite",
        ],
        preprocess_log,
    )
    if code != 0:
        for model in models:
            results[model] = {"error": f"preprocess failed: {tail[-400:]}"}
        return results

    for model in models:
        metrics_path = workdir / "predictions" / f"metrics_{model}.json"
        if skip_existing and metrics_path.is_file():
            try:
                results[model] = parse_metrics(metrics_path)
                results[model]["cached"] = True
                print(f"    {model}: cached", flush=True)
                continue
            except (OSError, json.JSONDecodeError):
                pass

        train_cmd = [
            python,
            str(PROJECT_ROOT / "scripts" / "train.py"),
            "--config",
            str(config_path),
            "--model",
            model,
        ]
        if epochs is not None:
            train_cmd += ["--epochs", str(epochs)]
        code, tail = run_command(
            train_cmd, workdir / "logs" / f"train_{model}.txt"
        )
        if code != 0:
            results[model] = {"error": f"train failed: {tail[-400:]}"}
            continue

        eval_cmd = [
            python,
            str(PROJECT_ROOT / "scripts" / "evaluate.py"),
            "--config",
            str(config_path),
            "--model",
            model,
            "--no-plots",
        ]
        code, tail = run_command(
            eval_cmd, workdir / "logs" / f"evaluate_{model}.txt"
        )
        if code != 0 or not metrics_path.is_file():
            results[model] = {"error": f"evaluate failed: {tail[-400:]}"}
            continue

        results[model] = parse_metrics(metrics_path)
    return results


def aggregate_repeats(
    per_repeat: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    models: set[str] = set()
    for run in per_repeat:
        models.update(run.keys())
    aggregated: dict[str, dict[str, Any]] = {}
    for model in sorted(models):
        entries = [run.get(model, {}) for run in per_repeat]
        errors = [entry.get("error") for entry in entries if entry.get("error")]
        numeric_keys = [
            "window_auc",
            "pass_auc",
            "profile_auc",
            "window_f1",
            "pass_f1",
            "profile_f1",
            "false_alarm_rate",
            "interval_f1",
            "interval_iou",
            "interval_coverage",
        ]
        summary: dict[str, Any] = {"n_repeats": len(entries)}
        if errors:
            summary["error"] = errors[0]
        for key in numeric_keys:
            values = [
                float(entry[key])
                for entry in entries
                if key in entry and np.isfinite(float(entry[key]))
            ]
            summary[key] = float(np.mean(values)) if values else float("nan")
            summary[f"{key}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
        summary["n_test_sources"] = int(
            np.mean([entry.get("n_test_sources", 0) or 0 for entry in entries])
        )
        aggregated[model] = summary
    return aggregated


def plot_sweep(
    results: dict[float, dict[str, dict[str, Any]]],
    models: Sequence[str],
    output_path: Path,
    alpha: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_values = sorted(results.keys())
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    panels = [
        ("window_auc", "Window-level ROC AUC"),
        ("profile_auc", "Pass-level ROC AUC (time-integrated)"),
        ("interval_f1", "Interval localisation F1"),
    ]

    for axis, (key, title) in zip(axes, panels):
        for model in models:
            xs, ys, errs = [], [], []
            for snr in snr_values:
                entry = results[snr].get(model, {})
                value = entry.get(key, float("nan"))
                if not np.isfinite(value):
                    continue
                xs.append(snr)
                ys.append(value)
                errs.append(entry.get(f"{key}_std", 0.0) or 0.0)
            if not xs:
                continue
            style = MODEL_STYLES.get(model, {})
            axis.errorbar(
                xs,
                ys,
                yerr=errs if any(errs) else None,
                label=MODEL_LABELS.get(model, model),
                capsize=3,
                linewidth=1.8,
                markersize=7,
                **style,
            )
        axis.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
        axis.set_title(title)
        axis.set_xlabel("Defect SNR at detector input (dB)")
        axis.set_ylabel("score")
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.25)
        axis.set_xticks(snr_values)

    axes[0].text(
        0.02,
        0.04,
        f"chance = 0.50 | conformal $\\alpha$ = {alpha:.2f}",
        transform=axes[0].transAxes,
        fontsize=8,
        alpha=0.7,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    figure.suptitle("DAS track-defect detection: sensitivity to defect SNR", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"figure written to {output_path}", flush=True)


def write_summary(
    results: dict[float, dict[str, dict[str, Any]]],
    models: Sequence[str],
    output_dir: Path,
) -> None:
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "snr_db": [float(snr) for snr in sorted(results)],
        "models": list(models),
        "results": {
            f"{snr:+.0f}": {model: entry for model, entry in results[snr].items()}
            for snr in sorted(results)
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "snr_db",
                "model",
                "window_auc",
                "window_auc_std",
                "pass_auc",
                "pass_auc_std",
                "profile_auc",
                "profile_auc_std",
                "window_f1",
                "false_alarm_rate",
                "interval_f1",
                "interval_iou",
                "interval_coverage",
                "n_test_sources",
                "error",
            ]
        )
        for snr in sorted(results):
            for model in models:
                entry = results[snr].get(model, {})
                writer.writerow(
                    [
                        f"{snr:+.0f}",
                        model,
                        _fmt(entry.get("window_auc")),
                        _fmt(entry.get("window_auc_std")),
                        _fmt(entry.get("pass_auc")),
                        _fmt(entry.get("pass_auc_std")),
                        _fmt(entry.get("profile_auc")),
                        _fmt(entry.get("profile_auc_std")),
                        _fmt(entry.get("window_f1")),
                        _fmt(entry.get("false_alarm_rate")),
                        _fmt(entry.get("interval_f1")),
                        _fmt(entry.get("interval_iou")),
                        _fmt(entry.get("interval_coverage")),
                        entry.get("n_test_sources", ""),
                        entry.get("error", ""),
                    ]
                )


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(number) else f"{number:.4f}"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SNR sensitivity benchmark.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "exp_snr_sweep.yaml"),
        help="Benchmark configuration (paths and SNR are rewritten).",
    )
    parser.add_argument(
        "--snr", nargs="*", type=float, default=[-15.0, -10.0, -5.0, 0.0, 5.0]
    )
    parser.add_argument(
        "--models", nargs="*", default=list(DEFAULT_MODELS), help="Model names to compare."
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "snr_sweep"),
    )
    parser.add_argument("--repeats", type=int, default=1, help="Dataset seeds per SNR point.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--seed", type=int, default=2024, help="Base simulation seed.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse finished metrics.")
    parser.add_argument("--analyze-only", action="store_true", help="Only replot summary.json.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    base = load_config(config_path)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    models = list(dict.fromkeys(args.models))
    alpha = float(base.get("evaluation", {}).get("conformal_alpha", 0.05))

    if not args.analyze_only:
        results: dict[float, dict[str, dict[str, Any]]] = {}
        started = time.time()
        for snr in args.snr:
            print(f"\n=== SNR {snr:+.0f} dB ===", flush=True)
            per_repeat: list[dict[str, dict[str, Any]]] = []
            for repeat in range(max(int(args.repeats), 1)):
                workdir = output_dir / snr_tag(snr)
                if args.repeats > 1:
                    workdir = workdir / f"seed_{repeat}"
                workdir.mkdir(parents=True, exist_ok=True)
                config = patch_config(
                    base, workdir, snr=snr, seed=int(args.seed) + repeat
                )
                point_config = workdir / "config.yaml"
                point_config.write_text(
                    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                per_repeat.append(
                    evaluate_point(
                        point_config,
                        workdir,
                        models=models,
                        epochs=args.epochs,
                        skip_existing=args.skip_existing,
                    )
                )
            results[snr] = aggregate_repeats(per_repeat)
            for model in models:
                entry = results[snr][model]
                print(
                    f"  {model:<20} window_auc={_fmt(entry.get('window_auc')):>7} "
                    f"profile_auc={_fmt(entry.get('profile_auc')):>7} "
                    f"interval_f1={_fmt(entry.get('interval_f1')):>7}"
                    + (f"  ERROR: {entry['error'][:120]}" if entry.get("error") else ""),
                    flush=True,
                )
        print(f"\ntotal wall time: {(time.time() - started) / 60:.1f} min", flush=True)
    else:
        summary_path = output_dir / "summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"no summary to analyse: {summary_path}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        results = {
            float(key): value for key, value in payload["results"].items()
        }
        models = list(payload.get("models", models))

    write_summary(results, models, output_dir)
    plot_sweep(results, models, output_dir / "auc_vs_snr.png", alpha=alpha)

    # Console pivot table: rows SNR, columns model.
    header = "SNR(dB) | " + " | ".join(f"{model:>18}" for model in models)
    print("\nWindow-level ROC AUC")
    print(header)
    print("-" * len(header))
    for snr in sorted(results):
        row = " | ".join(
            f"{_fmt(results[snr].get(model, {}).get('window_auc')):>18}"
            for model in models
        )
        print(f"{snr:>7.0f} | {row}")
    print("\nPass-level ROC AUC (time-integrated channel profile)")
    print(header)
    print("-" * len(header))
    for snr in sorted(results):
        row = " | ".join(
            f"{_fmt(results[snr].get(model, {}).get('profile_auc')):>18}"
            for model in models
        )
        print(f"{snr:>7.0f} | {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

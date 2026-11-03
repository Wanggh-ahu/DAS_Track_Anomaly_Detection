"""Plot detection quality against defect-morphology heterogeneity.

Each input directory is one condition produced by

    python scripts/run_snr_sweep.py --snr 5 --defect-jitter <J> --output-dir <DIR>

The figure answers a single question: does a fixed band-pass keep its advantage
when the defect spectrum is no longer the same in every pass?
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    "pca": "PCA",
    "bandpass": "Band-pass 80-160 Hz (fixed)",
    "filterbank": "Filterbank (5 bands)",
    "ae_1d": "1-D AE",
    "spatial_masked_ae": "Masked AE",
    "energy": "Energy threshold",
}
COLORS = {
    "pca": "#1f77b4",
    "bandpass": "#ff7f0e",
    "filterbank": "#8c564b",
    "ae_1d": "#2ca02c",
    "spatial_masked_ae": "#d62728",
    "energy": "#7f7f7f",
}


def read_summary(directory: Path) -> dict[str, dict[str, float]]:
    path = directory / "summary.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        model = row["model"]
        table[model] = {
            "channel_auc": _float(row.get("channel_auc")),
            "channel_auc_std": _float(row.get("channel_auc_std")),
            "profile_auc": _float(row.get("profile_auc")),
            "profile_auc_std": _float(row.get("profile_auc_std")),
            "profile_recall": _float(row.get("profile_recall")),
        }
    return table


def _float(value: str | None) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "conditions",
        nargs="+",
        help="Pairs of <jitter>=<directory>, e.g. 0=outputs/morph_j00",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "morphology_vs_auc.png"),
    )
    args = parser.parse_args(argv)

    jitters: list[float] = []
    tables: list[dict[str, dict[str, float]]] = []
    for spec in args.conditions:
        jitter_text, directory_text = spec.split("=", 1)
        jitters.append(float(jitter_text))
        tables.append(read_summary(PROJECT_ROOT / directory_text))
    order = sorted(range(len(jitters)), key=lambda index: jitters[index])
    jitters = [jitters[index] for index in order]
    tables = [tables[index] for index in order]

    models = [m for m in LABELS if any(m in table for table in tables)]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    for model in models:
        color = COLORS.get(model, "#333333")
        label = LABELS.get(model, model)
        for axis, key, std_key in (
            (axes[0], "channel_auc", "channel_auc_std"),
            (axes[1], "profile_auc", "profile_auc_std"),
            (axes[2], "profile_recall", None),
        ):
            xs, ys, errs = [], [], []
            for jitter, table in zip(jitters, tables):
                if model not in table:
                    continue
                value = table[model].get(key, float("nan"))
                if value != value:  # NaN
                    continue
                xs.append(jitter)
                ys.append(value)
                if std_key:
                    errs.append(table[model].get(std_key, 0.0) or 0.0)
            if not xs:
                continue
            axis.errorbar(
                xs,
                ys,
                yerr=errs if errs and any(errs) else None,
                marker="o",
                color=color,
                label=label,
                capsize=3,
            )

    axes[0].set_title("Channel-level AUC (active windows)")
    axes[1].set_title("Pass-level profile AUC")
    axes[2].set_title("Recall at conformal 5% false-alarm rate")
    for axis in axes:
        axis.set_xlabel("defect frequency jitter (fraction of nominal)")
        axis.grid(alpha=0.3)
        axis.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("AUC")
    axes[1].axhline(0.5, color="k", lw=0.8, ls=":")
    axes[2].axhline(0.5, color="k", lw=0.8, ls=":")
    axes[1].legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    print(f"figure written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

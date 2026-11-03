"""Collapse and selectivity diagnostics for the spectrogram autoencoder.

The masked-AE episode showed that a reconstruction model can train to a
plausible-looking loss while learning nothing useful.  Before any benchmark run
the spectrogram model must therefore answer three questions:

1. **Is it learning at all?**  Reconstruction correlation and the reconstruction
   to input standard-deviation ratio expose a model that outputs a constant or
   a blurred copy.
2. **Is it collapsed in frequency?**  A dead frequency bin would make the
   residual blind exactly where the defect lives; per-bin residual spread on
   normal data shows this directly.
3. **Is the residual selective?**  For an anomalous window the residual peak
   should sit at the defect's own frequency, not smeared over the band.  This is
   the property a waveform bottleneck cannot have, so it is the whole point of
   the model.

Usage::

    python scripts/diagnose_stft_ae.py --config <point config> --test-dir <processed/test>
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

from scripts.evaluate import EvaluationData, load_evaluation_data  # noqa: E402
from scripts.train import load_config, resolve_path  # noqa: E402
from src.evaluation import (  # noqa: E402
    compute_roc_auc,
    detection_metrics_by_source,
    pass_profile_scores,
)
from src.models import build_model  # noqa: E402


def _defect_frequencies(root: Path) -> dict[str, float]:
    """Map ``source`` -> realised resonance frequency (Hz), when recorded."""

    table: dict[str, float] = {}
    for path in sorted(root.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata"]))
                source = str(np.asarray(payload["source"]).ravel()[0])
        except (OSError, ValueError, KeyError, IndexError):
            continue
        defect = (metadata.get("source_metadata") or {}).get("defect") or {}
        frequency = defect.get("resonance_frequency_hz")
        if frequency is not None:
            table[source] = float(frequency)
    return table


def _load_model(config: dict[str, Any], model_name: str, checkpoint_dir: Path):
    for suffix in (".pt", ".pth", ".pkl"):
        path = checkpoint_dir / f"{model_name}{suffix}"
        if path.is_file():
            break
    else:
        raise FileNotFoundError(f"no checkpoint for {model_name} in {checkpoint_dir}")
    metadata_path = checkpoint_dir / f"{model_name}_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    params = dict(config.get("model", {}).get("params", {}).get(model_name, {}) or {})
    prototype = build_model(model_name, params)
    # ``load`` is a classmethod returning a new instance; discarding it would
    # leave an unfitted object behind.
    model = type(prototype).load(path)
    return model, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="stft_ae")
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(resolve_path(args.config))
    checkpoint_dir = resolve_path(
        config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints")
    )
    model, metadata = _load_model(config, args.model, checkpoint_dir)
    test_dir = (
        resolve_path(args.test_dir)
        if args.test_dir
        else resolve_path(config.get("paths", {}).get("processed_dir", "data/processed"))
        / "test"
    )
    evaluation_cfg = config.get("evaluation", {})
    dataset = load_evaluation_data(
        test_dir,
        window_label_mode=str(evaluation_cfg.get("window_labels", "activity")),
        activity_overlap=float(evaluation_cfg.get("activity_overlap", 0.3)),
    )
    if args.max_windows:
        keep = slice(0, int(args.max_windows))
        dataset = EvaluationData(
            data=dataset.data[keep],
            channel_labels=dataset.channel_labels[keep],
            sample_labels=dataset.sample_labels[keep],
            has_labels=dataset.has_labels,
            sources=dataset.sources[keep],
            sampling_rate_hz=dataset.sampling_rate_hz,
            channel_spacing_m=dataset.channel_spacing_m,
            files=dataset.files[: int(args.max_windows)],
            pass_labels=(
                dataset.pass_labels[keep] if dataset.pass_labels is not None else None
            ),
        )

    target, error = model.spectrogram_error(dataset.data)
    correlation, ratio = model.reconstruction_quality(dataset.data)
    print(f"windows={len(dataset)} shape={dataset.data.shape[1:]}")
    print(f"reconstruction: corr={correlation:.3f}  std_ratio={ratio:.3f}")

    # ---------------------------------------------------- frequency collapse
    active = dataset.sample_labels.astype(bool)
    channel_residual = error.mean(axis=-1)  # (N, C, F) mean over frames
    frequencies = model.frequency_axis(dataset.sampling_rate_hz)
    band = (frequencies >= 5.0) & (frequencies <= 200.0)
    normal_profile = channel_residual[~active].mean(axis=(0, 1))
    active_profile = channel_residual[active].mean(axis=(0, 1))
    # Averaging over all 96 channels buries the 1-4 defective ones, so also
    # report the profile restricted to the channels that actually carry a defect.
    defect_channel_profile = np.zeros(len(frequencies))
    defect_windows = 0
    for index in range(len(dataset)):
        if not active[index]:
            continue
        channels = np.flatnonzero(dataset.channel_labels[index])
        if channels.size == 0:
            continue
        defect_channel_profile += channel_residual[index][channels].mean(axis=0)
        defect_windows += 1
    if defect_windows:
        defect_channel_profile /= defect_windows
    order = np.argsort(np.where(band, defect_channel_profile, -np.inf))[::-1][:5]
    print("\nstrongest residual on defect channels (top 5 bins within 5-200 Hz):")
    for index in order:
        print(
            f"  {frequencies[index]:6.1f} Hz  defect-channel={defect_channel_profile[index]:.4f} "
            f"all-channel(anomalous)={active_profile[index]:.4f} "
            f"all-channel(normal)={normal_profile[index]:.4f}"
        )
    dead = int((normal_profile[band] < 0.25 * np.median(normal_profile[band])).sum())
    print(f"dead bins in band (<25 % of median response): {dead}/{int(band.sum())}")

    # ------------------------------------------------ frequency selectivity
    defect_frequency = _defect_frequencies(test_dir)
    hits = 0
    total = 0
    errors_bins: list[float] = []
    for index in range(len(dataset)):
        if not active[index]:
            continue
        source = str(dataset.sources[index])
        frequency = defect_frequency.get(source)
        if frequency is None:
            continue
        channels = np.flatnonzero(dataset.channel_labels[index])
        if channels.size == 0:
            continue
        residual = channel_residual[index][channels].max(axis=0)
        residual = np.where(band, residual, -np.inf)
        peak = int(np.argmax(residual))
        truth = int(np.argmin(np.abs(frequencies - frequency)))
        errors_bins.append(abs(peak - truth))
        hits += int(abs(peak - truth) <= 1)
        total += 1
    if total:
        print(
            f"\ndefect-frequency localisation: {hits}/{total} within +-1 bin "
            f"({100.0 * hits / total:.1f} %), median |error|={np.median(errors_bins):.1f} bins"
        )
    else:
        print("\ndefect-frequency localisation: no defect metadata found")

    # ------------------------------------------------------------- detection
    pass_labels = dataset.pass_labels if dataset.pass_labels is not None else dataset.sample_labels
    # Two ways of collapsing the frequency axis: averaging (what a waveform
    # bottleneck implicitly does) versus keeping the strongest bin (what the
    # explicit frequency axis is supposed to buy).
    collapses = {
        "frequency-mean": channel_residual.mean(axis=-1),
        "frequency-max": channel_residual.max(axis=-1),
    }
    channel_auc = float("nan")
    for label, scores in collapses.items():
        profile = pass_profile_scores(scores, dataset.sources, aggregation="mean")
        detection = detection_metrics_by_source(
            pass_labels, [profile[str(source)] for source in dataset.sources], dataset.sources
        )
        if label == "frequency-mean" and active.any():
            channel_auc = float(
                compute_roc_auc(
                    dataset.channel_labels[active].ravel(), scores[active].ravel()
                )
            )
        print(
            f"detection[{label:>14}]: channel_auc="
            f"{(float(compute_roc_auc(dataset.channel_labels[active].ravel(), scores[active].ravel())) if active.any() else float('nan')):.4f} "
            f"profile_auc={detection['roc_auc']:.4f} "
            f"profile_f1={detection.get('f1', float('nan')):.4f}"
        )
    print(f"\nreference channel_auc(frequency-mean)={channel_auc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Frequency attribution: can a model say *which* defect it is looking at?

Every benchmark so far asks "is something wrong here".  That question can be
answered without any spectral estimate — a broadband energy detector answers it.
It therefore cannot distinguish a model that has learned *where the defect lives
in frequency* from one that merely happens to cover the right band.

This script builds the distinguishing question.  Each anomalous pass carries two
defect sites at disjoint channel ranges and different frequencies; both are
scaled to the same local SNR so neither can be found by being louder.  Detection
is then trivial and the only way to score is to attach the correct frequency to
the correct channel:

    for every (window, defect channel):
        truth = the frequency of the site that owns that channel
        prediction = the frequency axis position with the largest residual

A fixed band-pass has no prediction to make (reported as N/A).  The filterbank
*does* — it carries one z-score per band — so it is the honest classical rival,
not a strawman.

Chance level for a two-site design is 50 %.
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

from scripts.evaluate import load_evaluation_data  # noqa: E402
from scripts.train import load_config, resolve_path  # noqa: E402
from src.models import build_model  # noqa: E402


def _load_region_map(root: Path) -> dict[str, list[dict[str, Any]]]:
    """``source`` -> list of ``{"channels": set[int], "frequency_hz": float}``."""

    table: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata"]))
                source = str(np.asarray(payload["source"]).ravel()[0])
        except (OSError, ValueError, KeyError, IndexError):
            continue
        defect = (metadata.get("source_metadata") or {}).get("defect") or {}
        sites = defect.get("regions") or []
        if not sites:
            continue
        table[source] = [
            {
                "channels": set(int(c) for c in site.get("defect_channels", [])),
                "frequency_hz": float(site.get("resonance_frequency_hz", float("nan"))),
            }
            for site in sites
        ]
    return table


def _load_model(config: dict[str, Any], model_name: str):
    checkpoint_dir = resolve_path(
        config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints")
    )
    for suffix in (".pt", ".pth", ".pkl"):
        path = checkpoint_dir / f"{model_name}{suffix}"
        if path.is_file():
            break
    else:
        raise FileNotFoundError(f"no checkpoint for {model_name} in {checkpoint_dir}")
    params = dict(config.get("model", {}).get("params", {}).get(model_name, {}) or {})
    model = type(build_model(model_name, params)).load(path)
    return model


def _spectral_scores(
    model, data: np.ndarray, sampling_rate_hz: float
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]] | None]:
    """Return ``(scores (N, C, K), centres (K,), band_edges or None)``."""

    name = getattr(model, "name", "")
    if name == "stft_ae":
        # Exactly the feature detection uses, with its frequency axis intact.
        scores = model.frequency_profile(data, sampling_rate_hz=sampling_rate_hz)
        centres = model.frequency_axis(sampling_rate_hz)
        return scores, centres, None
    if name == "filterbank":
        scores = model.band_zscores(data, sampling_rate_hz=sampling_rate_hz)
        edges = [(lo, hi) for lo, hi in model.bands]
        centres = np.asarray([0.5 * (lo + hi) for lo, hi in edges], dtype=float)
        return scores, centres, edges
    raise ValueError(f"{name} exposes no frequency axis")


def _expected_index(
    centres: np.ndarray, edges: list[tuple[float, float]] | None, frequency: float
) -> int | None:
    """Which axis position *should* be selected for ``frequency``."""

    if edges is not None:
        for index, (low, high) in enumerate(edges):
            if low <= frequency < high:
                return index
        return None
    return int(np.argmin(np.abs(centres - frequency)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="*", default=["stft_ae", "filterbank"])
    parser.add_argument("--test-dir", default=None)
    args = parser.parse_args(argv)

    config = load_config(resolve_path(args.config))
    test_dir = (
        resolve_path(args.test_dir)
        if args.test_dir
        else resolve_path(config.get("paths", {}).get("processed_dir")) / "test"
    )
    evaluation_cfg = config.get("evaluation", {})
    dataset = load_evaluation_data(
        test_dir,
        window_label_mode=str(evaluation_cfg.get("window_labels", "activity")),
        activity_overlap=float(evaluation_cfg.get("activity_overlap", 0.3)),
    )
    region_map = _load_region_map(test_dir)
    multi_site = sum(1 for sites in region_map.values() if len(sites) > 1)
    print(
        f"windows={len(dataset)} passes_with_metadata={len(region_map)} "
        f"two_site_passes={multi_site}"
    )
    if not multi_site:
        print("no pass carries two sites; run with a dual-defect configuration")
        return 1

    results: dict[str, dict[str, float]] = {}
    per_pass: dict[str, dict[str, tuple[float, int]]] = {}
    for model_name in args.models:
        model = _load_model(config, model_name)
        scores, centres, edges = _spectral_scores(
            model, dataset.data, dataset.sampling_rate_hz
        )
        # Two-alternative forced choice: a channel that sits on one site must
        # respond more strongly at *its own* site's frequency than at the other
        # site's.  Comparing against the exact expected bin would mostly measure
        # how precisely the peak localises (33 bins, ~10 % hit rate) rather than
        # whether the two sites are told apart, which is the question here.
        correct = 0
        total = 0
        margins: list[float] = []
        exact_hits = 0
        confusion: dict[tuple[int, int], int] = {}
        per_source: dict[str, list[float]] = {}
        for index in range(len(dataset)):
            sites = region_map.get(str(dataset.sources[index]))
            if not sites or len(sites) < 2:
                continue
            labelled = np.flatnonzero(dataset.channel_labels[index])
            if labelled.size == 0:
                continue
            for channel in labelled:
                owner = None
                for site_index, site in enumerate(sites):
                    if int(channel) in site["channels"]:
                        owner = site_index
                        break
                if owner is None:
                    continue
                other = 1 - owner if owner < 2 else 0
                i_true = _expected_index(centres, edges, sites[owner]["frequency_hz"])
                i_other = _expected_index(centres, edges, sites[other]["frequency_hz"])
                if i_true is None or i_other is None or i_true == i_other:
                    continue
                row = scores[index, int(channel)]
                margin = float(row[i_true] - row[i_other])
                margins.append(margin)
                correct += int(margin > 0.0) + 0.5 * int(margin == 0.0)
                total += 1
                peak = int(np.argmax(row))
                confusion[(owner, peak)] = confusion.get((owner, peak), 0) + 1
                exact_hits += int(peak == i_true)
                bucket = per_source.setdefault(str(dataset.sources[index]), [0.0, 0.0])
                bucket[0] += int(margin > 0.0) + 0.5 * int(margin == 0.0)
                bucket[1] += 1
        if not total:
            print(f"{model_name:>12}: no attributable channel found")
            continue
        accuracy = correct / total
        results[model_name] = {"accuracy": accuracy, "n": float(total)}
        per_pass[model_name] = {
            source: (values[0] / values[1], int(values[1]))
            for source, values in per_source.items()
            if values[1] > 0
        }
        print(
            f"{model_name:>12}: 2-alternative attribution accuracy {accuracy:.3f} "
            f"over {total} channel-windows  (chance 0.500), "
            f"median margin {np.median(margins):+.3f}, "
            f"peak-bin exact hit {exact_hits / total:.3f}"
        )

    # Channel-windows from the same pass are strongly correlated, so the pass is
    # the unit of analysis: compare per-pass accuracies, not pooled windows.
    if len(per_pass) >= 2:
        names = list(per_pass)
        for name in names:
            values = np.asarray([v[0] for v in per_pass[name].values()])
            print(
                f"    per-pass {name:>12}: mean {values.mean():.3f} "
                f"+- {values.std(ddof=1):.3f} over {values.size} passes"
            )
        base = names[0]
        for name in names[1:]:
            shared = sorted(set(per_pass[base]) & set(per_pass[name]))
            if len(shared) < 5:
                continue
            first = np.asarray([per_pass[base][s][0] for s in shared])
            second = np.asarray([per_pass[name][s][0] for s in shared])
            difference = first - second
            try:
                from scipy.stats import wilcoxon

                p_value = float(wilcoxon(difference).pvalue)
            except (ImportError, ValueError):
                p_value = float("nan")
            print(
                f"    paired {base} - {name}: {difference.mean():+.3f} "
                f"+- {difference.std(ddof=1) / np.sqrt(difference.size):.3f} "
                f"over {len(shared)} passes, Wilcoxon p={p_value:.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

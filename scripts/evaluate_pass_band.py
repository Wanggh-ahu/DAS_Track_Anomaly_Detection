"""Select the analysis band per pass instead of per window.

Section 7.3 of ``reports/snr_benchmark.md`` found that every per-window band
selector (spectral kurtosis, loudest band) loses badly to a *fixed* band, and
explained it by the time scale: the defect band is constant within one train
pass and drifts between passes, so selecting inside a 128 ms window destroys the
one piece of structure the pass-level score needs.

That explanation makes a sharp prediction which had never been tested: picking
the band **once per pass** should recover the loss.  Testing it honestly needs
three controls, because "pick the largest of 13 bands x 96 channels" is biased
upward on *every* pass, defective or not:

``hedge``
    Take the maximum over bands per channel -- i.e. exactly what the filterbank
    baseline already does.  This reproduces the published number and validates
    the harness; it is the reference the selectors must beat.

``per_pass_raw``
    One band per pass, chosen by the largest calibrated z.  No correction for
    the selection bias, so this is expected to look bad -- it quantifies the bias.

``per_pass_cal``
    One band per pass, where the selection statistic is standardised against its
    own null distribution measured on **training** passes (all normal).  This is
    the fair version of the hypothesis: a band only wins if it stands out more
    than the best-of-13 noise band usually does.

``per_pass_channel_cal``
    Same idea per (pass, channel), with per-channel nulls.

``per_pass_oracle``
    Upper bound: the band is chosen **using the labels**.  If even this cannot
    beat ``hedge``, then no label-free single-band selector can, and the whole
    "adapt the band" family is closed.

All variants reuse an already-trained ``filterbank`` checkpoint's calibration,
so nothing is retrained.

Usage::

    python scripts/evaluate_pass_band.py --arms morph_j00_s5 morph_j70_s5
    python scripts/evaluate_pass_band.py --arms oof_impulse_train --dest out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import Any, Iterable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import evaluate as evalmod  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

LOGGER = logging.getLogger("das.passband")


def _topk_mean(values: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, values.shape[-1]))
    return np.partition(values, -k, axis=-1)[..., -k:].mean(axis=-1)


def _load(seed_dir: pathlib.Path, split: str, limit: int | None) -> tuple[np.ndarray, np.ndarray, Any]:
    config_path = seed_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config: {config_path}")
    config = evalmod.load_config(config_path)
    evaluation_cfg = config.get("evaluation", {})
    checkpoint_dir = evalmod.resolve_path(
        config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints")
    )
    checkpoint_path = evalmod._find_checkpoint(checkpoint_dir, "filterbank")
    metadata = evalmod._load_metadata(checkpoint_path)
    _, model = evalmod._restore_model(checkpoint_path, metadata, config)
    processed_root = evalmod.resolve_path(
        config.get("paths", {}).get("processed_dir", "data/processed")
    )
    dataset = evalmod.load_evaluation_data(
        processed_root / split,
        limit=limit,
        window_label_mode=str(evaluation_cfg.get("window_labels", "pass")),
        activity_overlap=float(evaluation_cfg.get("activity_overlap", 0.3)),
    )
    z = model.band_zscores(dataset.data, sampling_rate_hz=float(dataset.sampling_rate_hz))
    return z.astype(np.float64), np.asarray(dataset.sources), dataset


def _group_mean(z: np.ndarray, sources: np.ndarray, names: list[str]) -> np.ndarray:
    return np.stack([z[sources == name].mean(axis=0) for name in names])


def _null_from_train(
    mean_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-band and per-(channel, band) null of the pass-level statistic.

    Training passes are all normal, so any spread here is pure noise, and the
    standardisation below asks "is this band more extreme than the best-of-k
    noise band usually is?"
    """

    band_strength = mean_z.max(axis=1)  # (n_train_passes, n_bands)
    band_mu = band_strength.mean(axis=0)
    band_sd = np.maximum(band_strength.std(axis=0), 1e-6)
    chan_mu = mean_z.mean(axis=0)
    chan_sd = np.maximum(mean_z.std(axis=0), 1e-6)
    return band_mu, band_sd, chan_mu, chan_sd


def _score(
    z: np.ndarray,
    sources: np.ndarray,
    mean_z: np.ndarray,
    names: list[str],
    bands: np.ndarray,
    topk_channels: int,
    reduce: str | None = None,
) -> np.ndarray:
    n_channels = mean_z.shape[1]
    out = np.empty(len(names))
    for index, name in enumerate(names):
        rows = sources == name
        window_z = z[rows]
        if reduce is not None:
            selected = getattr(window_z, reduce)(axis=2)  # (n_windows, n_channels)
        else:
            band_row = np.asarray(bands)[index]
            index_array = np.broadcast_to(
                np.asarray(band_row)[None, :], (window_z.shape[0], window_z.shape[1])
            )[..., None]
            selected = np.take_along_axis(window_z, index_array, axis=2)[..., 0]
        out[index] = _topk_mean(selected, topk_channels).mean()
    return out


def _channel_map(mean_z: np.ndarray, bands: np.ndarray, reduce: str | None = None) -> np.ndarray:
    if reduce is not None:
        return getattr(mean_z, reduce)(axis=2)
    index = np.broadcast_to(np.asarray(bands)[..., None], mean_z.shape[:2] + (1,))
    return np.take_along_axis(mean_z, index, axis=2)[..., 0]


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores))


def run_seed(seed_dir: pathlib.Path, topk_channels: int = 3, train_limit: int = 4000) -> dict[str, Any]:
    z_test, src_test, test = _load(seed_dir, "test", None)
    z_train, src_train, train = _load(seed_dir, "train", train_limit)

    names_test = sorted(set(src_test.tolist()))
    names_train = sorted(set(src_train.tolist()))
    mean_test = _group_mean(z_test, src_test, names_test)
    mean_train = _group_mean(z_train, src_train, names_train)
    band_mu, band_sd, chan_mu, chan_sd = _null_from_train(mean_train)
    if not train.has_labels:
        train_label = np.zeros(len(names_train), dtype=int)
    else:
        train_label = np.array([int(train.channel_labels[src_train == n].max()) for n in names_train])
    if train_label.any():
        LOGGER.warning("%d training passes carry defect labels; null may leak", int(train_label.sum()))

    pass_label = np.array([int(test.channel_labels[src_test == n].max()) for n in names_test])
    n_channels = mean_test.shape[1]
    strength = mean_test.max(axis=1)  # (n_passes, n_bands)
    result: dict[str, Any] = {
        "n_passes": len(names_test),
        "n_pos_passes": int(pass_label.sum()),
        "n_bands": int(strength.shape[1]),
        "train_passes": len(names_train),
    }
    if pass_label.min() == pass_label.max():
        result.update({"skipped": "test set has one class"})
        return result

    # reference: hedge over bands (what filterbank already does)
    # reference: reduce across bands instead of choosing one (what filterbank does)
    result["bands_mean"] = _auc(
        _score(z_test, src_test, mean_test, names_test, None, topk_channels, reduce="mean"),
        pass_label,
    )
    result["bands_max"] = _auc(
        _score(z_test, src_test, mean_test, names_test, None, topk_channels, reduce="max"),
        pass_label,
    )

    # raw per-pass selection: no bias correction
    raw_bands = np.argmax(strength, axis=1)
    result["per_pass_raw"] = _auc(
        _score(z_test, src_test, mean_test, names_test, raw_bands[:, None].repeat(n_channels, 1), topk_channels),
        pass_label,
    )

    # calibrated per-pass selection
    cal_strength = (strength - band_mu) / band_sd
    cal_bands = np.argmax(cal_strength, axis=1)
    cal_scores = _score(
        z_test, src_test, mean_test, names_test, cal_bands[:, None].repeat(n_channels, 1), topk_channels
    )
    result["per_pass_cal"] = _auc(cal_scores, pass_label)
    result["per_pass_cal_band_hist"] = {
        int(b): int((cal_bands == b).sum()) for b in np.unique(cal_bands)
    }
    result["per_pass_cal_channel_auc"] = float(
        roc_auc_score(
            np.array([test.channel_labels[src_test == n].max(axis=0) for n in names_test]).ravel(),
            _channel_map(mean_test, cal_bands[:, None].repeat(n_channels, 1)).ravel(),
        )
    )

    # per (pass, channel) calibrated
    cal_std = (mean_test - chan_mu) / chan_sd
    pc_bands = np.argmax(cal_std, axis=2)
    result["per_pass_channel_cal"] = _auc(
        _score(z_test, src_test, mean_test, names_test, pc_bands, topk_channels), pass_label
    )

    # oracle: band chosen with labels
    gap = strength - band_mu
    is_defect = pass_label == 1
    oracle_bands = np.where(is_defect, np.argmax(gap, axis=1), np.argmin(gap, axis=1)).reshape(-1)
    result["per_pass_oracle"] = _auc(
        _score(z_test, src_test, mean_test, names_test, oracle_bands[:, None].repeat(n_channels, 1), topk_channels),
        pass_label,
    )
    result["oracle_channel_auc"] = float(
        roc_auc_score(
            np.array([test.channel_labels[src_test == n].max(axis=0) for n in names_test]).ravel(),
            _channel_map(mean_test, oracle_bands[:, None].repeat(n_channels, 1)).ravel(),
        )
    )

    # best single band pooled over the test set (label-free, used only as a
    # "how good is one fixed band at most" reference)
    pooled = strength.mean(axis=0)
    best_band = int(np.argmax(pooled))
    result["best_fixed_band"] = best_band
    result["fixed_best_pooled"] = _auc(
        _score(z_test, src_test, mean_test, names_test, np.full((len(names_test), n_channels), best_band), topk_channels),
        pass_label,
    )
    return result


def _seed_dirs(arm_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    return sorted(p.parent for p in arm_dir.glob("*/seed_*/config.yaml"))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=pathlib.Path, default=ROOT / "outputs")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--topk-channels", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=4000)
    parser.add_argument("--dest", type=pathlib.Path, default=ROOT / "outputs" / "pass_band.json")
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {}
    keys = ("bands_mean", "bands_max", "fixed_best_pooled", "per_pass_raw", "per_pass_cal", "per_pass_channel_cal", "per_pass_oracle")
    for arm in args.arms:
        arm_dir = args.outputs / arm
        if not arm_dir.is_dir():
            LOGGER.error("missing arm: %s", arm_dir)
            return 1
        per_seed = {}
        for seed_dir in _seed_dirs(arm_dir):
            LOGGER.info("evaluating %s/%s", arm, seed_dir.name)
            per_seed[seed_dir.name] = run_seed(seed_dir, args.topk_channels, args.train_limit)
        payload[arm] = per_seed
        summary = {}
        for key in keys:
            values = [s[key] for s in per_seed.values() if s.get(key) is not None]
            if values:
                summary[key] = f"{np.mean(values):.3f}+-{np.std(values):.3f}"
        LOGGER.info("%s: %s", arm, summary)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    args.dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

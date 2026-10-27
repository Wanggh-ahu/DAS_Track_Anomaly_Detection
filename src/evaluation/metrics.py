"""Evaluation metrics for pass-level detection and channel-level localisation."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

LOGGER = logging.getLogger(__name__)


def _as_binary_vector(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if not np.all(np.isin(array, [0, 1, False, True])):
        raise ValueError(f"{name} must contain binary values (0/1)")
    return array.astype(np.int8)


def aggregate_channel_profile(
    window_scores: np.ndarray,
    aggregation: str,
) -> np.ndarray:
    """Fuse per-window channel scores into one profile for a pass.

    ``vote`` counts how often each channel is the strongest channel of a
    window.  It is invariant to per-window amplitude changes and is therefore
    more stable than averaging normalised residuals when a pass contains a
    changing moving-load background.
    """

    aggregation = str(aggregation).lower()
    scores = np.asarray(window_scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("window_scores must have shape (n_windows, n_channels)")
    if aggregation == "vote":
        winners = np.argmax(scores, axis=1)
        counts = np.bincount(winners, minlength=scores.shape[1]).astype(np.float64)
        return counts / max(scores.shape[0], 1)
    if aggregation == "mean":
        return scores.mean(axis=0)
    if aggregation == "median":
        return np.median(scores, axis=0)
    if aggregation == "max":
        return scores.max(axis=0)
    raise ValueError("aggregation must be 'mean', 'median', 'max' or 'vote'")


def compute_roc_auc(y_true: Sequence[int] | np.ndarray, scores: Sequence[float]) -> float:
    """Area under the ROC curve; returns ``nan`` for a single-class input."""

    labels = _as_binary_vector(y_true, "y_true")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape[0] != values.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    if np.unique(labels).size < 2:
        LOGGER.warning("ROC AUC is undefined for a single-class evaluation set")
        return float("nan")
    return float(roc_auc_score(labels, values))


def compute_roc_curve(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float],
) -> dict[str, np.ndarray | float]:
    """Return ``fpr``, ``tpr``, ``thresholds`` and AUC for plotting."""

    labels = _as_binary_vector(y_true, "y_true")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if np.unique(labels).size < 2:
        return {
            "fpr": np.asarray([]),
            "tpr": np.asarray([]),
            "thresholds": np.asarray([]),
            "auc": float("nan"),
        }
    fpr, tpr, thresholds = roc_curve(labels, values)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": float(roc_auc_score(labels, values)),
    }


def compute_f1(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    average: str = "binary",
) -> float:
    """F1 score with a safe zero-division policy."""

    labels = _as_binary_vector(y_true, "y_true")
    predictions = _as_binary_vector(y_pred, "y_pred")
    if labels.shape[0] != predictions.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    return float(f1_score(labels, predictions, average=average, zero_division=0))


def compute_false_alarm_rate(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> float:
    """Fraction of normal samples incorrectly flagged as anomalous.

    This is the sample-level false-positive rate ``FP / (FP + TN)``.  It is
    often called the false alarm rate in condition-monitoring literature.
    """

    labels = _as_binary_vector(y_true, "y_true")
    predictions = _as_binary_vector(y_pred, "y_pred")
    if labels.shape[0] != predictions.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    negatives = labels == 0
    n_negatives = int(negatives.sum())
    if n_negatives == 0:
        LOGGER.warning("False alarm rate is undefined without normal samples")
        return float("nan")
    false_positives = int(np.sum((predictions == 1) & negatives))
    return float(false_positives / n_negatives)


def compute_localization_error(
    true_channels: Sequence[Sequence[int] | int | np.ndarray],
    predicted_channels: Sequence[int] | np.ndarray,
    channel_spacing_m: float = 0.5,
    tolerance_channels: int = 2,
) -> dict[str, Any]:
    """Compute distance from a predicted anomalous channel to the true region.

    Each sample may contain one or several true anomalous channels (for example
    a short defective sleeper region).  The error is the distance from the
    predicted channel to the nearest true channel.

    Returns
    -------
    dict
        Mean/median/RMSE error in channels and metres, the hit rate within
        ``tolerance_channels``, and the per-sample distances.
    """

    predicted = np.asarray(predicted_channels, dtype=int).reshape(-1)
    if len(true_channels) != predicted.shape[0]:
        raise ValueError("true_channels and predicted_channels must have equal length")
    distances: list[int] = []
    for truth, prediction in zip(true_channels, predicted):
        channels = np.atleast_1d(np.asarray(truth, dtype=int)).reshape(-1)
        channels = channels[np.isfinite(channels)]
        if channels.size == 0 or not np.isfinite(prediction):
            continue
        distances.append(int(np.min(np.abs(channels - int(prediction)))))

    if not distances:
        return {
            "mean_error_channels": float("nan"),
            "median_error_channels": float("nan"),
            "rmse_channels": float("nan"),
            "mean_error_m": float("nan"),
            "within_tolerance_rate": float("nan"),
            "tolerance_channels": int(tolerance_channels),
            "distances_channels": np.asarray([], dtype=np.int32),
            "n_evaluated": 0,
        }
    distance_array = np.asarray(distances, dtype=np.float64)
    return {
        "mean_error_channels": float(distance_array.mean()),
        "median_error_channels": float(np.median(distance_array)),
        "rmse_channels": float(np.sqrt(np.mean(np.square(distance_array)))),
        "mean_error_m": float(distance_array.mean() * channel_spacing_m),
        "within_tolerance_rate": float(
            np.mean(distance_array <= max(int(tolerance_channels), 0))
        ),
        "tolerance_channels": int(tolerance_channels),
        "distances_channels": distance_array.astype(np.int32),
        "n_evaluated": int(distance_array.size),
    }


def localization_metrics_from_scores(
    channel_labels: np.ndarray,
    channel_scores: np.ndarray,
    channel_spacing_m: float = 0.5,
    tolerance_channels: int = 2,
) -> dict[str, Any]:
    """Localise anomalies by the arg-max channel score and score them.

    Only passes containing at least one labelled anomalous channel contribute
    to localisation metrics.  Normal passes are evaluated by the detection
    metrics and intentionally excluded here.
    """

    labels = np.asarray(channel_labels)
    scores = np.asarray(channel_scores)
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("channel_labels and channel_scores must share shape (N, C)")
    anomalous = np.flatnonzero(labels.max(axis=1) > 0)
    if anomalous.size == 0:
        return compute_localization_error([], [], channel_spacing_m, tolerance_channels)
    predicted = np.argmax(scores[anomalous], axis=1)
    true_channels = [np.flatnonzero(labels[index] > 0) for index in anomalous]
    return compute_localization_error(
        true_channels,
        predicted,
        channel_spacing_m=channel_spacing_m,
        tolerance_channels=tolerance_channels,
    )


def localization_metrics_by_source(
    channel_labels: np.ndarray,
    channel_scores: np.ndarray,
    sources: Sequence[str] | np.ndarray,
    channel_spacing_m: float = 0.5,
    tolerance_channels: int = 2,
    aggregation: str = "mean",
    sample_scores: Sequence[float] | np.ndarray | None = None,
    threshold: float | None = None,
    detected_only: bool = False,
) -> dict[str, Any]:
    """Localise a persistent anomaly after fusing all windows of one pass.

    Per-window arg-max is noisy because a single ``Tw`` slice contains only a
    fraction of the moving-load response.  Track defects, however, persist over
    many windows.  Aggregating the channel scores of each source recording
    suppresses uncorrelated background residuals while preserving the spatially
    coherent defect response.

    Parameters
    ----------
    channel_labels, channel_scores:
        Arrays shaped ``(N, C)`` with binary labels and channel anomaly scores.
        For ``aggregation="vote"``, pass the *raw* channel residual: each
        window contributes one vote for its strongest channel.
    sources:
        Length-``N`` array identifying the recording/pass of each window.
    aggregation:
        ``vote`` (recommended), ``mean``, ``median`` or ``max`` over the
        windows of a pass.  ``vote`` is the mode of the per-window arg-max and
        is insensitive to absolute residual-scale changes between windows.
    sample_scores, threshold, detected_only:
        When ``detected_only`` is true, only windows whose pass score reaches
        ``threshold`` contribute.  If none do, all windows of that pass are used.
    """

    labels = np.asarray(channel_labels)
    scores = np.asarray(channel_scores)
    source_array = np.asarray(sources).reshape(-1)
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("channel_labels and channel_scores must share shape (N, C)")
    if source_array.size != labels.shape[0]:
        raise ValueError("sources must contain one entry per window")
    aggregation = str(aggregation).lower()
    if aggregation not in {"mean", "median", "max", "vote"}:
        raise ValueError("aggregation must be 'mean', 'median', 'max' or 'vote'")
    if detected_only and (sample_scores is None or threshold is None):
        raise ValueError("detected_only requires sample_scores and threshold")

    score_vector = (
        np.asarray(sample_scores, dtype=np.float64).reshape(-1)
        if sample_scores is not None
        else None
    )
    if score_vector is not None and score_vector.size != labels.shape[0]:
        raise ValueError("sample_scores must contain one entry per window")

    details: list[dict[str, Any]] = []
    true_regions: list[np.ndarray] = []
    predictions: list[int] = []

    for source in np.unique(source_array):
        mask = source_array == source
        source_labels = labels[mask]
        if not source_labels.any():
            continue
        if detected_only and score_vector is not None and threshold is not None:
            selected = mask & (score_vector >= float(threshold))
            if selected.any():
                mask = selected
        true_channels = np.flatnonzero(source_labels.max(axis=0) > 0)
        profile = aggregate_channel_profile(scores[mask], aggregation)
        predicted = int(np.argmax(profile))
        distance = int(np.min(np.abs(true_channels - predicted)))
        details.append(
            {
                "source": str(source),
                "predicted_channel": predicted,
                "true_channels": true_channels.astype(int).tolist(),
                "distance_channels": distance,
                "n_windows": int(np.sum(source_array == source)),
                "n_used": int(mask.sum()),
                "aggregation": aggregation,
            }
        )
        true_regions.append(true_channels)
        predictions.append(predicted)

    summary = compute_localization_error(
        true_regions,
        predictions,
        channel_spacing_m=channel_spacing_m,
        tolerance_channels=tolerance_channels,
    )
    summary["sources"] = details
    summary["n_sources"] = len(details)
    summary["aggregation"] = aggregation
    return summary


def extract_anomaly_intervals(
    channel_scores: np.ndarray,
    threshold_mad: float = 1.0,
    merge_gap: int = 2,
    min_width: int = 1,
    smoothing_sigma: float = 1.0,
) -> dict[str, Any]:
    """Extract the dominant contiguous anomalous channel interval(s).

    A track defect occupies a short, spatially coherent channel range.  The
    profile is smoothed slightly, thresholded relative to its robust baseline,
    and split into runs.  Small gaps are bridged so a weak channel inside the
    defect does not split the output into two fragments.

    Parameters
    ----------
    channel_scores:
        One-dimensional channel anomaly profile.
    threshold_mad:
        Threshold above the median in robust standard deviations (MAD units).
        ``1.0`` is a good default for normalised anomaly scores.
    merge_gap:
        Maximum number of missing channels that may be bridged.
    min_width:
        Minimum interval width in channels.
    smoothing_sigma:
        Gaussian smoothing along channels; set to ``0`` to disable it.

    Returns
    -------
    dict
        ``intervals`` sorted by peak score, plus the robust threshold and
        baseline used to obtain them.
    """

    values = np.asarray(channel_scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("channel_scores must not be empty")
    finite = np.isfinite(values)
    if not finite.all():
        values = values.copy()
        values[~finite] = float(np.median(values[finite])) if finite.any() else 0.0

    sigma = max(float(smoothing_sigma), 0.0)
    smoothed = (
        gaussian_filter1d(values, sigma=sigma, mode="nearest")
        if sigma > 0
        else values.copy()
    )
    baseline = float(np.median(smoothed))
    mad = float(1.4826 * np.median(np.abs(smoothed - baseline)))
    if not np.isfinite(mad) or mad <= 1e-8:
        mad = float(smoothed.std())
    if not np.isfinite(mad) or mad <= 1e-8:
        mad = 1.0
    threshold = baseline + max(float(threshold_mad), 0.0) * mad

    mask = smoothed >= threshold
    if not mask.any():
        # Degenerate case: the robust threshold sits above a narrow peak (for
        # example a two-channel plateau after smoothing).  Fall back to a
        # peak-relative threshold rather than returning a single channel.
        peak = float(smoothed.max())
        if float(np.ptp(smoothed)) <= 1e-8:
            mask = np.zeros_like(mask, dtype=bool)
            mask[int(np.argmax(smoothed))] = True
        else:
            threshold = peak - 0.5 * max(mad, 1e-8)
            mask = smoothed >= threshold

    indices = np.flatnonzero(mask)
    break_after = np.flatnonzero(np.diff(indices) > max(int(merge_gap), 0) + 1) + 1
    groups = np.split(indices, break_after)
    peak_index = int(np.argmax(values))
    intervals: list[dict[str, Any]] = []
    for group in groups:
        if group.size == 0:
            continue
        start, end = int(group[0]), int(group[-1])
        width = end - start + 1
        if width < max(int(min_width), 1) and not (start <= peak_index <= end):
            continue
        intervals.append(
            {
                "start_channel": start,
                "end_channel": end,
                "width_channels": width,
                "max_score": float(values[start : end + 1].max()),
                "mean_score": float(values[start : end + 1].mean()),
            }
        )
    if not intervals:
        intervals.append(
            {
                "start_channel": peak_index,
                "end_channel": peak_index,
                "width_channels": 1,
                "max_score": float(values[peak_index]),
                "mean_score": float(values[peak_index]),
            }
        )
    intervals.sort(key=lambda item: item["max_score"], reverse=True)
    return {
        "intervals": intervals,
        "n_intervals": len(intervals),
        "threshold": float(threshold),
        "baseline": baseline,
        "mad": mad,
        "threshold_mad": float(threshold_mad),
        "merge_gap": int(merge_gap),
        "min_width": int(min_width),
        "smoothing_sigma": sigma,
    }


def interval_metrics_by_source(
    channel_labels: np.ndarray,
    channel_scores: np.ndarray,
    sources: Sequence[str] | np.ndarray,
    channel_spacing_m: float = 0.5,
    tolerance_channels: int = 2,
    aggregation: str = "mean",
    threshold_mad: float = 1.0,
    merge_gap: int = 2,
    min_width: int = 1,
    smoothing_sigma: float = 1.0,
    max_intervals: int = 3,
) -> dict[str, Any]:
    """Evaluate predicted anomalous channel intervals against true regions.

    A real track can carry several defects in one section, so both the ground
    truth and the prediction are treated as *sets of intervals*.  Predictions
    and truths are matched greedily by IoU; unmatched predictions count as
    false alarms and unmatched truths as misses.  This replaces the previous
    behaviour of scoring only the single strongest predicted interval, which
    silently rewarded detectors that collapsed several defects into one.

    Reported quantities
    -------------------
    ``mean_iou``
        Mean IoU of each source's best-matching interval pair, zero for misses.
    ``mean_coverage``
        Fraction of true channels covered by any predicted interval.
    ``interval_precision`` / ``interval_recall`` / ``interval_f1``
        Interval-level scores computed over all matched pairs.
    ``channel_precision``
        Fraction of predicted channels that are truly anomalous.
    """

    labels = np.asarray(channel_labels)
    scores = np.asarray(channel_scores)
    source_array = np.asarray(sources).reshape(-1)
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("channel_labels and channel_scores must share shape (N, C)")
    if source_array.size != labels.shape[0]:
        raise ValueError("sources must contain one entry per window")
    aggregation = str(aggregation).lower()
    if aggregation not in {"mean", "median", "max", "vote"}:
        raise ValueError("aggregation must be 'mean', 'median', 'max' or 'vote'")
    max_intervals = max(int(max_intervals), 1)

    def _runs(channels: np.ndarray) -> list[tuple[int, int]]:
        if channels.size == 0:
            return []
        breaks = np.flatnonzero(np.diff(channels) > 1) + 1
        runs: list[tuple[int, int]] = []
        for group in np.split(channels, breaks):
            if group.size:
                runs.append((int(group[0]), int(group[-1])))
        return runs

    def _match(
        predicted: list[tuple[int, int]], truth: list[tuple[int, int]]
    ) -> list[tuple[int, int, float]]:
        if not predicted or not truth:
            return []
        overlap = np.zeros((len(predicted), len(truth)), dtype=np.float64)
        for i, (p0, p1) in enumerate(predicted):
            for j, (t0, t1) in enumerate(truth):
                intersection = max(0, min(p1, t1) - max(p0, t0) + 1)
                union = (p1 - p0 + 1) + (t1 - t0 + 1) - intersection
                overlap[i, j] = intersection / union if union > 0 else 0.0
        pairs: list[tuple[int, int, float]] = []
        while overlap.size and float(overlap.max()) > 0.0:
            i, j = np.unravel_index(int(np.argmax(overlap)), overlap.shape)
            pairs.append((int(i), int(j), float(overlap[i, j])))
            overlap[i, :] = 0.0
            overlap[:, j] = 0.0
        return pairs

    details: list[dict[str, Any]] = []
    for source in np.unique(source_array):
        mask = source_array == source
        source_labels = labels[mask]
        if not source_labels.any():
            continue
        true_channels = np.flatnonzero(source_labels.max(axis=0) > 0)
        true_runs = _runs(true_channels)
        profile = aggregate_channel_profile(scores[mask], aggregation)
        extraction = extract_anomaly_intervals(
            profile,
            threshold_mad=threshold_mad,
            merge_gap=merge_gap,
            min_width=min_width,
            smoothing_sigma=smoothing_sigma,
        )
        predicted_runs = [
            (int(item["start_channel"]), int(item["end_channel"]))
            for item in extraction["intervals"][:max_intervals]
        ]
        pairs = _match(predicted_runs, true_runs)

        predicted_mask = np.zeros(labels.shape[1], dtype=bool)
        for start, end in predicted_runs:
            predicted_mask[start : end + 1] = True
        true_mask = np.zeros(labels.shape[1], dtype=bool)
        true_mask[true_channels] = True
        intersection = np.logical_and(predicted_mask, true_mask)
        union = np.logical_or(predicted_mask, true_mask)
        channel_iou = float(intersection.sum() / max(union.sum(), 1))
        coverage = float(intersection.sum() / max(true_mask.sum(), 1))
        channel_precision = float(intersection.sum() / max(predicted_mask.sum(), 1))

        best_iou = max((pair[2] for pair in pairs), default=0.0)
        if pairs:
            best_pred, best_true, _ = max(pairs, key=lambda pair: pair[2])
            predicted_interval = list(predicted_runs[best_pred])
            true_interval = list(true_runs[best_true])
        else:
            best_pred, best_true = 0, 0
            predicted_interval = list(predicted_runs[0]) if predicted_runs else [0, 0]
            true_interval = list(true_runs[0]) if true_runs else [0, 0]

        distance = (
            int(np.min(np.abs(true_channels[:, None] - np.flatnonzero(predicted_mask)[None, :])))
            if predicted_mask.any() and true_channels.size
            else int(1e9)
        )
        details.append(
            {
                "source": str(source),
                "predicted_interval": predicted_interval,
                "predicted_intervals": [list(run) for run in predicted_runs],
                "true_intervals": [list(run) for run in true_runs],
                "true_channels": true_channels.astype(int).tolist(),
                "true_interval": true_interval,
                "n_predicted_channels": int(predicted_mask.sum()),
                "n_true_channels": int(true_channels.size),
                "intersection_channels": int(intersection.sum()),
                "iou": float(best_iou),
                "channel_iou": channel_iou,
                "coverage": coverage,
                "channel_precision": channel_precision,
                "hit": bool(intersection.sum() > 0),
                "distance_channels": int(distance),
                "start_error_channels": int(abs(predicted_interval[0] - true_interval[0])),
                "end_error_channels": int(abs(predicted_interval[1] - true_interval[1])),
                "n_windows": int(mask.sum()),
                "n_intervals": int(len(predicted_runs)),
                "n_true_intervals": int(len(true_runs)),
                "n_matched_intervals": int(len(pairs)),
                "max_score": float(
                    extraction["intervals"][0]["max_score"]
                    if extraction["intervals"]
                    else 0.0
                ),
                "threshold": float(extraction["threshold"]),
            }
        )

    if not details:
        return {
            "n_sources": 0,
            "hit_rate": float("nan"),
            "tolerance_hit_rate": float("nan"),
            "mean_iou": float("nan"),
            "median_iou": float("nan"),
            "mean_coverage": float("nan"),
            "median_coverage": float("nan"),
            "mean_start_error_channels": float("nan"),
            "mean_end_error_channels": float("nan"),
            "mean_distance_channels": float("nan"),
            "mean_predicted_width_channels": float("nan"),
            "mean_true_width_channels": float("nan"),
            "exact_interval_rate": float("nan"),
            "interval_precision": float("nan"),
            "interval_recall": float("nan"),
            "interval_f1": float("nan"),
            "channel_precision": float("nan"),
            "distances_channels": [],
            "sources": [],
            "aggregation": aggregation,
            "threshold_mad": float(threshold_mad),
            "merge_gap": int(merge_gap),
            "min_width": int(min_width),
            "smoothing_sigma": float(smoothing_sigma),
        }

    ious = np.asarray([item["iou"] for item in details], dtype=np.float64)
    coverages = np.asarray([item["coverage"] for item in details], dtype=np.float64)
    distances = np.asarray(
        [item["distance_channels"] for item in details], dtype=np.float64
    )
    start_errors = np.asarray(
        [item["start_error_channels"] for item in details], dtype=np.float64
    )
    end_errors = np.asarray(
        [item["end_error_channels"] for item in details], dtype=np.float64
    )
    n_true_intervals = int(sum(item["n_true_intervals"] for item in details))
    n_predicted_intervals = int(sum(item["n_intervals"] for item in details))
    n_matched = int(sum(item["n_matched_intervals"] for item in details))
    interval_precision = n_matched / max(n_predicted_intervals, 1)
    interval_recall = n_matched / max(n_true_intervals, 1)
    interval_f1 = (
        2 * interval_precision * interval_recall
        / max(interval_precision + interval_recall, 1e-12)
    )
    return {
        "n_sources": len(details),
        "hit_rate": float(np.mean([item["hit"] for item in details])),
        "tolerance_hit_rate": float(
            np.mean(distances <= max(int(tolerance_channels), 0))
        ),
        "mean_iou": float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "mean_coverage": float(coverages.mean()),
        "median_coverage": float(np.median(coverages)),
        "mean_start_error_channels": float(start_errors.mean()),
        "mean_end_error_channels": float(end_errors.mean()),
        "mean_distance_channels": float(distances.mean()),
        "mean_predicted_width_channels": float(
            np.mean([item["n_predicted_channels"] for item in details])
        ),
        "mean_true_width_channels": float(
            np.mean([item["n_true_channels"] for item in details])
        ),
        "mean_predicted_intervals": float(
            np.mean([item["n_intervals"] for item in details])
        ),
        "mean_true_intervals": float(
            np.mean([item["n_true_intervals"] for item in details])
        ),
        "interval_precision": float(interval_precision),
        "interval_recall": float(interval_recall),
        "interval_f1": float(interval_f1),
        "channel_precision": float(
            np.mean([item["channel_precision"] for item in details])
        ),
        "exact_interval_rate": float(
            np.mean(
                [
                    item["predicted_interval"] == item["true_interval"]
                    for item in details
                ]
            )
        ),
        "distances_channels": distances.astype(int).tolist(),
        "sources": details,
        "aggregation": aggregation,
        "threshold_mad": float(threshold_mad),
        "merge_gap": int(merge_gap),
        "min_width": int(min_width),
        "smoothing_sigma": float(smoothing_sigma),
        "channel_spacing_m": float(channel_spacing_m),
    }


def pass_profile_scores(
    channel_scores: np.ndarray,
    sources: Sequence[str] | np.ndarray,
    aggregation: str = "mean",
    statistic: str = "topk_mean",
    topk: int = 3,
) -> dict[str, float]:
    """Fuse all windows of a pass into one channel profile, then one score.

    A transient defect is active for only part of a pass, so averaging
    *window scores* (each of which is already a maximum over channels) buries it
    under the windows where it is silent.  Integrating the residual per channel
    over the whole pass first, and only then reducing the channel axis, keeps
    the evidence that repeats across windows.  This is the statistic a
    maintenance decision would actually use.
    """

    scores = np.asarray(channel_scores, dtype=np.float64)
    source_array = np.asarray(sources).reshape(-1)
    if scores.ndim != 2:
        raise ValueError("channel_scores must have shape (N, C)")
    if source_array.size != scores.shape[0]:
        raise ValueError("sources must contain one entry per window")
    statistic = str(statistic).lower()
    if statistic not in {"max", "topk_mean"}:
        raise ValueError("statistic must be 'max' or 'topk_mean'")

    pass_scores: dict[str, float] = {}
    for source in np.unique(source_array):
        mask = source_array == source
        profile = aggregate_channel_profile(scores[mask], aggregation)
        if statistic == "max" or topk <= 1:
            value = float(np.max(profile))
        else:
            k = int(min(max(topk, 1), profile.size))
            value = float(np.mean(np.sort(profile)[-k:]))
        pass_scores[str(source)] = value
    return pass_scores


def detection_metrics_by_source(
    sample_labels: Sequence[int] | np.ndarray,
    sample_scores: Sequence[float],
    sources: Sequence[str] | np.ndarray,
    score_aggregation: str = "mean",
    threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate detection after fusing all windows belonging to one pass.

    Window-level thresholds are sensitive to short-lived background bursts.
    A maintenance decision is usually made per pass, so this function reports
    the pass-level ROC AUC.  When a calibrated pass threshold is supplied, F1
    and false-alarm rate use it; otherwise they use the best-F1 threshold as a
    clearly labelled diagnostic.
    """

    labels = _as_binary_vector(sample_labels, "sample_labels")
    scores = np.asarray(sample_scores, dtype=np.float64).reshape(-1)
    source_array = np.asarray(sources).reshape(-1)
    if scores.size != labels.size or source_array.size != labels.size:
        raise ValueError("sample_labels, sample_scores and sources must align")
    aggregation = str(score_aggregation).lower()
    if aggregation not in {"mean", "median", "max"}:
        raise ValueError("score_aggregation must be 'mean', 'median' or 'max'")
    reducer = {"mean": np.mean, "median": np.median, "max": np.max}[aggregation]

    pass_labels: list[int] = []
    pass_scores: list[float] = []
    details: list[dict[str, Any]] = []
    for source in np.unique(source_array):
        mask = source_array == source
        source_labels = labels[mask]
        label = int(source_labels.max())
        score = float(reducer(scores[mask]))
        pass_labels.append(label)
        pass_scores.append(score)
        details.append(
            {
                "source": str(source),
                "label": label,
                "score": score,
                "n_windows": int(mask.sum()),
                "window_score_mean": float(scores[mask].mean()),
                "window_score_max": float(scores[mask].max()),
            }
        )

    y_true = np.asarray(pass_labels, dtype=np.int8)
    y_score = np.asarray(pass_scores, dtype=np.float64)
    roc_auc = compute_roc_auc(y_true, y_score)
    if threshold is None:
        operating_threshold, f1 = find_best_f1_threshold(y_true, y_score)
        threshold_source = "best_f1_diagnostic"
    else:
        operating_threshold = float(threshold)
        f1 = compute_f1(y_true, (y_score >= operating_threshold).astype(np.int8))
        threshold_source = "provided"
    predictions = (y_score >= operating_threshold).astype(np.int8)
    return {
        "n_sources": len(details),
        "n_anomalous_sources": int(y_true.sum()),
        "n_normal_sources": int((y_true == 0).sum()),
        "roc_auc": roc_auc,
        "threshold": operating_threshold,
        "threshold_source": threshold_source,
        "f1": f1,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "false_alarm_rate": compute_false_alarm_rate(y_true, predictions),
        "score_aggregation": aggregation,
        "sources": details,
    }


def evaluate_all(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float],
    threshold: float,
    channel_labels: np.ndarray | None = None,
    channel_scores: np.ndarray | None = None,
    channel_spacing_m: float = 0.5,
    tolerance_channels: int = 2,
) -> dict[str, Any]:
    """Compute the standard metric bundle used by ``scripts/evaluate.py``.

    Detection metrics (ROC AUC, F1, false alarm rate, precision, recall) are
    always returned.  Localisation metrics are added when both channel labels
    and channel scores are supplied.
    """

    labels = _as_binary_vector(y_true, "y_true")
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape[0] != score_array.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    predictions = (score_array >= float(threshold)).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "n_samples": int(labels.size),
        "n_anomalous": int(labels.sum()),
        "n_normal": int(labels.size - labels.sum()),
        "roc_auc": compute_roc_auc(labels, score_array),
        "f1": compute_f1(labels, predictions),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "false_alarm_rate": compute_false_alarm_rate(labels, predictions),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    if channel_labels is not None and channel_scores is not None:
        localization = localization_metrics_from_scores(
            channel_labels=channel_labels,
            channel_scores=channel_scores,
            channel_spacing_m=channel_spacing_m,
            tolerance_channels=tolerance_channels,
        )
        # Convert numpy arrays to lists for JSON serialisation.
        localization["distances_channels"] = localization[
            "distances_channels"
        ].tolist()
        metrics["localization"] = localization
    return metrics


def find_best_f1_threshold(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float],
) -> tuple[float, float]:
    """Return ``(threshold, f1)`` maximising F1 over observed score values.

    Intended for diagnostics only; selecting a threshold on the test set is
    optimistic.  Production thresholds should come from normal calibration data.
    """

    labels = _as_binary_vector(y_true, "y_true")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape[0] != values.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    candidates = np.unique(values)
    if candidates.size == 0:
        return 0.0, 0.0
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for candidate in candidates:
        predictions = (values >= candidate).astype(np.int8)
        value = compute_f1(labels, predictions)
        if value > best_f1:
            best_f1 = value
            best_threshold = float(candidate)
    return best_threshold, float(best_f1)

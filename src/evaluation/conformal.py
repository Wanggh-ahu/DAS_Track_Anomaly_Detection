"""Split-conformal thresholding with a finite-sample false-alarm guarantee.

The previous calibration took ``max(normal pass score) * (1 + margin)``.  A
maximum is the noisiest statistic available: with ten calibration passes it has
no extrapolation power at all, and a single contaminated pass pushes the
threshold out of reach forever.

Split conformal prediction gives a distribution-free guarantee instead.  With
``n`` exchangeable calibration scores and a target false-alarm rate ``alpha``,
the threshold ``s_(k)`` with ``k = ceil((n + 1) * (1 - alpha))`` satisfies

    P(score_new > threshold) <= alpha

for a new normal sample, provided ``k <= n``.  That condition is exactly the
sample-size requirement ``n >= ceil(1 / alpha) - 1``: with too few calibration
passes no method can promise a 1 % false-alarm rate, and this module says so
instead of silently returning a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def minimum_calibration_size(alpha: float) -> int:
    """Smallest calibration set that can certify false-alarm rate ``alpha``."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return int(math.ceil(1.0 / alpha)) - 1


@dataclass(frozen=True)
class ConformalThreshold:
    """A calibrated threshold plus the diagnostics needed to trust it."""

    threshold: float
    alpha: float
    n_calibration: int
    rank: int
    finite: bool
    empirical_fpr: float
    fallback_used: bool
    reason: str = ""

    @property
    def guaranteed(self) -> bool:
        """Whether the finite-sample guarantee actually applies."""

        return self.finite and not self.fallback_used

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "split_conformal",
            "threshold": float(self.threshold),
            "alpha": float(self.alpha),
            "n_calibration": int(self.n_calibration),
            "rank": int(self.rank),
            "finite": bool(self.finite),
            "guaranteed": bool(self.guaranteed),
            "empirical_fpr": float(self.empirical_fpr),
            "fallback_used": bool(self.fallback_used),
            "reason": self.reason,
            "min_calibration_size": minimum_calibration_size(self.alpha),
        }

    def p_value(self, score: float, scores: np.ndarray) -> float:
        """Conformal p-value of a score against the calibration sample."""

        calibration = np.asarray(scores, dtype=np.float64).reshape(-1)
        if calibration.size == 0:
            return float("nan")
        return float((1 + np.sum(calibration >= float(score))) / (calibration.size + 1))


def split_conformal_quantile(
    scores: Iterable[float] | np.ndarray,
    alpha: float = 0.05,
    fallback_margin: float = 0.05,
) -> ConformalThreshold:
    """Calibrate a threshold with a guaranteed false-alarm rate.

    Parameters
    ----------
    scores:
        Anomaly scores of *normal* calibration samples (windows or passes).
    alpha:
        Target false-alarm rate.  ``0.05`` needs at least 19 calibration
        samples, ``0.01`` needs at least 99.
    fallback_margin:
        Relative margin applied to the maximum when the calibration set is too
        small to certify ``alpha``.

    Returns
    -------
    ConformalThreshold
        ``guaranteed`` is ``False`` when the sample size was insufficient; the
        threshold is still usable but carries no promise.
    """

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        raise ValueError("calibration scores must not be empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    ordered = np.sort(values)
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    needed = minimum_calibration_size(alpha)

    if rank <= n:
        threshold = float(ordered[rank - 1])
        finite = True
        fallback = False
        reason = (
            f"n={n} >= {needed} calibration samples; "
            f"false-alarm rate <= {alpha:.3f} with probability 1-alpha"
        )
    else:
        # Not enough calibration data to certify alpha.  Stay conservative:
        # sit above the observed maximum so the empirical FPR is zero, and say
        # plainly that the guarantee does not hold.
        top = float(ordered[-1])
        threshold = top * (1.0 + max(fallback_margin, 0.0)) if top > 0 else top
        finite = False
        fallback = True
        reason = (
            f"n={n} < {needed} calibration samples required for alpha={alpha:.3f}; "
            "falling back to max * (1 + margin) with no guarantee"
        )

    empirical = float(np.mean(values > threshold))
    return ConformalThreshold(
        threshold=threshold,
        alpha=float(alpha),
        n_calibration=n,
        rank=rank,
        finite=finite,
        empirical_fpr=empirical,
        fallback_used=fallback,
        reason=reason,
    )


def conformal_threshold_report(
    window: ConformalThreshold, pass_level: ConformalThreshold
) -> dict[str, Any]:
    """Bundle window- and pass-level calibrations for the metadata sidecar."""

    return {
        "window": window.to_dict(),
        "pass": pass_level.to_dict(),
        "guaranteed": bool(window.guaranteed and pass_level.guaranteed),
    }

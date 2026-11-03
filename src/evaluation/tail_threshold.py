"""Pass-level thresholds that survive having only ~20 calibration passes.

Split-conformal calibration on the *window* level is well behaved: with ~1000
calibration windows the ``(1 - alpha)`` quantile is an order statistic about
50 samples from the maximum.  The pass level has no such luxury.  A dataset with
60 normal passes gives 21 calibration passes, and at ``alpha = 0.05``

    k = ceil((21 + 1) * 0.95) = 21

i.e. the conformal threshold is the **largest** of the 21 scores.  A threshold
defined by a single observation has no stability at all: it moved by a factor of
two between retrainings of the same model on the same data, once letting the
false-alarm rate reach 10.5 % and once pushing it to 0 % while recall collapsed
to zero.  Neither number was wrong; the estimator was.

There is no way around this with order statistics -- you cannot get a 5 % tail
quantile 2 samples from the top of a 21-sample set.  The only remaining lever is
to *model* the tail instead of counting in it.  This module fits a generalised
Pareto distribution to the exceedances over a high threshold (the standard
peaks-over-threshold construction) and reports its ``1 - alpha`` quantile.

It trades the distribution-free guarantee for stability, which is the right
trade here: the guarantee is vacuous when the threshold is the sample maximum,
because "at most 5 % of normals exceed the largest normal I have seen" is not a
statement about the next pass.  Callers keep the conformal number as a sanity
check and should say which one they used.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable

import numpy as np

from .conformal import ConformalThreshold

LOGGER = logging.getLogger(__name__)

#: Minimum number of exceedances before a two-parameter tail fit is trusted.
MIN_EXCEEDANCES = 8


def _fit_gpd(exceedances: np.ndarray) -> tuple[float, float, float]:
    """Return ``(shape, scale, loglik)`` of a GPD fit to positive exceedances."""

    from scipy.stats import genpareto

    shape, _loc, scale = genpareto.fit(exceedances, floc=0.0)
    loglik = float(np.sum(genpareto.logpdf(exceedances, shape, loc=0.0, scale=scale)))
    return float(shape), float(scale), loglik


def _gpd_quantile(
    threshold: float, shape: float, scale: float, n: int, n_exceed: int, alpha: float
) -> float:
    """Invert the peaks-over-threshold tail to the ``1 - alpha`` quantile."""

    target = n * float(alpha)
    if target <= 0.0:
        return float("inf")
    ratio = n_exceed / target
    if abs(shape) < 1e-6:
        return float(threshold + scale * math.log(ratio))
    return float(threshold + (scale / shape) * (ratio**shape - 1.0))


def gpd_threshold(
    scores: Iterable[float] | np.ndarray,
    alpha: float = 0.05,
    tail_fraction: float = 0.5,
    min_exceedances: int = MIN_EXCEEDANCES,
) -> ConformalThreshold:
    """Peaks-over-threshold estimate of the ``1 - alpha`` pass-score quantile.

    ``tail_fraction`` sets where the tail begins: half of the calibration scores
    is used for the fit by default.  With 21 samples that is 10 exceedances,
    which is thin but far more information than "the single largest score".

    Falls back to the split-conformal maximum (with a relative margin) when the
    tail fit is not identifiable or lands below the observed data, so the
    returned threshold is always usable.  ``reason`` states which path was
    taken; ``finite`` is ``False`` for the fallback because no tail model was
    accepted.
    """

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        raise ValueError("calibration scores must not be empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    ordered = np.sort(values)
    tail_fraction = float(min(max(tail_fraction, 0.1), 0.9))
    n_tail = int(max(math.floor(n * tail_fraction), 1))
    start = max(n - n_tail, 1)
    tail_start = float(ordered[start - 1])

    fallback_top = float(ordered[-1])
    fallback = fallback_top * 1.05 if fallback_top > 0 else fallback_top

    exceedances = values[values > tail_start] - tail_start
    if exceedances.size < max(int(min_exceedances), 1):
        return ConformalThreshold(
            threshold=fallback,
            alpha=float(alpha),
            n_calibration=n,
            rank=int(math.ceil((n + 1) * (1.0 - alpha))),
            finite=False,
            empirical_fpr=float(np.mean(values > fallback)),
            fallback_used=True,
            reason=(
                f"n={n}: only {exceedances.size} exceedances above "
                f"{tail_start:.4g}, need {min_exceedances}; using max * 1.05"
            ),
        )

    try:
        shape, scale, _ = _fit_gpd(exceedances)
    except (ValueError, FloatingPointError) as error:  # pragma: no cover - scipy
        return ConformalThreshold(
            threshold=fallback,
            alpha=float(alpha),
            n_calibration=n,
            rank=int(math.ceil((n + 1) * (1.0 - alpha))),
            finite=False,
            empirical_fpr=float(np.mean(values > fallback)),
            fallback_used=True,
            reason=f"n={n}: GPD fit failed ({error}); using max * 1.05",
        )

    estimate = _gpd_quantile(
        tail_start, shape, scale, n, int(exceedances.size), alpha
    )
    # A tail model that lands inside the bulk was not describing this tail.
    degraded = not np.isfinite(estimate) or estimate <= tail_start
    # A shape that implies a bounded upper endpoint below the target is the
    # same failure in disguise.
    if not degraded and shape < 0.0:
        endpoint = tail_start - scale / shape
        degraded = estimate > endpoint
    if degraded:
        return ConformalThreshold(
            threshold=fallback,
            alpha=float(alpha),
            n_calibration=n,
            rank=int(math.ceil((n + 1) * (1.0 - alpha))),
            finite=False,
            empirical_fpr=float(np.mean(values > fallback)),
            fallback_used=True,
            reason=(
                f"n={n}: GPD estimate {estimate:.4g} not compatible with the "
                f"observed tail above {tail_start:.4g}; using max * 1.05"
            ),
        )

    # Sanity floor: a 5 % threshold cannot be below the 90th percentile of the
    # data it was fitted to.
    floor = float(np.quantile(values, max(1.0 - 2.0 * alpha, 0.5)))
    estimate = float(max(estimate, floor))
    return ConformalThreshold(
        threshold=estimate,
        alpha=float(alpha),
        n_calibration=n,
        rank=int(math.ceil((n + 1) * (1.0 - alpha))),
        finite=True,
        empirical_fpr=float(np.mean(values > estimate)),
        fallback_used=False,
        reason=(
            f"n={n}: GPD tail above {tail_start:.4g} with {exceedances.size} "
            f"exceedances (shape {shape:+.3f}, scale {scale:.4g})"
        ),
    )

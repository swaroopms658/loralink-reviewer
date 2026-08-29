"""Small-sample statistics for reviewer-response tables."""
from __future__ import annotations
from statistics import fmean, stdev
from scipy.stats import t as _t

def mean_std_ci(values, confidence: float = 0.95) -> dict:
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": 0.0, "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": 0}
    m = fmean(vals)
    if n < 2:
        return {"mean": m, "std": 0.0, "ci_lo": m, "ci_hi": m, "n": n}
    s = stdev(vals)                      # ddof=1
    half = _t.ppf(0.5 + confidence / 2, df=n - 1) * s / (n ** 0.5)
    return {"mean": m, "std": s, "ci_lo": m - half, "ci_hi": m + half, "n": n}

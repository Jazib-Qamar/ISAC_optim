"""Statistical summaries for ICC Monte Carlo tables (mean, median, CI, percentiles)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def mean_ci_95(values: Iterable[float]) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` using a Student-t interval.

    For ``n < 2`` the interval collapses to the point estimate.
    """
    arr = finite(values)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    tcrit = float(stats.t.ppf(0.975, arr.size - 1))
    half = tcrit * se
    return mean, mean - half, mean + half


def describe(values: Iterable[float]) -> dict[str, float]:
    arr = finite(values)
    if arr.size == 0:
        return {
            "count": 0, "mean": np.nan, "median": np.nan, "std": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan,
            "p05": np.nan, "p25": np.nan, "p75": np.nan, "p95": np.nan, "max": np.nan, "min": np.nan,
        }
    mean, lo, hi = mean_ci_95(arr)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "count": int(arr.size),
        "mean": mean,
        "median": float(np.median(arr)),
        "std": std,
        "ci95_low": lo,
        "ci95_high": hi,
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def rate_with_ci(flags: Iterable[bool]) -> dict[str, float]:
    arr = np.asarray(list(flags), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "rate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean, lo, hi = mean_ci_95(arr)
    return {"count": int(arr.size), "rate": mean, "ci95_low": lo, "ci95_high": hi}


def summarize_methods(
    frame: pd.DataFrame,
    columns: Iterable[str],
    method_column: str = "method",
    ok_only: bool = True,
) -> pd.DataFrame:
    rows = []
    for method, sub in frame.groupby(method_column, sort=False):
        data = sub[sub["status"] == "ok"] if ok_only and "status" in sub.columns else sub
        for col in columns:
            if col not in data.columns:
                continue
            stats_row = describe(data[col].tolist())
            rows.append({"method": method, "metric": col, **stats_row})
        if "physical_sensing_satisfied" in data.columns:
            phys = rate_with_ci(data["physical_sensing_satisfied"].astype(bool))
            rows.append({"method": method, "metric": "physical_sensing_feasibility_rate", **phys,
                         "mean": phys["rate"], "median": np.nan, "std": np.nan,
                         "p05": np.nan, "p25": np.nan, "p75": np.nan, "p95": np.nan, "max": np.nan, "min": np.nan})
        if "dense_psl_satisfied" in data.columns:
            psl = rate_with_ci(data["dense_psl_satisfied"].astype(bool))
            rows.append({"method": method, "metric": "dense_psl_feasibility_rate", **psl,
                         "mean": psl["rate"], "median": np.nan, "std": np.nan,
                         "p05": np.nan, "p25": np.nan, "p75": np.nan, "p95": np.nan, "max": np.nan, "min": np.nan})
        if "false_sensing_feasibility" in data.columns:
            false = rate_with_ci(data["false_sensing_feasibility"].astype(bool))
            rows.append({"method": method, "metric": "false_sensing_feasibility_rate", **false,
                         "mean": false["rate"], "median": np.nan, "std": np.nan,
                         "p05": np.nan, "p25": np.nan, "p75": np.nan, "p95": np.nan, "max": np.nan, "min": np.nan})
    return pd.DataFrame(rows)


def pearson_spearman(x: Iterable[float], y: Iterable[float]) -> dict[str, float]:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if xa.size < 3:
        return {
            "n": int(xa.size),
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
        }
    pr = stats.pearsonr(xa, ya)
    sr = stats.spearmanr(xa, ya)
    return {
        "n": int(xa.size),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }

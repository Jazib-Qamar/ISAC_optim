"""Console reporting and summary-statistics helpers shared by the experiments."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from configs.default import DefaultConfig

STAGE2_ROOT = Path("results") / "stage2"
STAGE25_ROOT = Path("results") / "stage2_5"


def print_header(title: str, width: int = 78) -> None:
    print("=" * width)
    print(title)
    print("=" * width)


def print_config(cfg: DefaultConfig) -> None:
    """Print every configuration section with derived quantities."""
    print_header("Configuration")
    for section_name in ("ofdm", "channel", "sensing", "energy", "ambiguity", "optimization", "simulation"):
        section = getattr(cfg, section_name)
        print(f"[{section_name}]")
        for field in dataclasses.fields(section):
            print(f"  {field.name:32s} = {getattr(section, field.name)}")
    print("[derived]")
    print(f"  {'noise_psd_w_per_hz':32s} = {cfg.ofdm.noise_psd_w_per_hz:.4e}")
    print(f"  {'noise_power_per_subcarrier_w':32s} = {cfg.ofdm.noise_power_per_subcarrier_w:.4e}")
    print(f"  {'peak_power_w':32s} = {cfg.ofdm.peak_power_w:.5f}")
    print(f"  {'bandwidth_hz':32s} = {cfg.ofdm.bandwidth_hz:.1f}")
    print(f"  {'round_trip_delay_s':32s} = {cfg.sensing.round_trip_delay_s:.4e}")
    print(f"  {'mainlobe_scale_1_over_B_s':32s} = {1.0 / cfg.ofdm.bandwidth_hz:.4e}")
    print()


def experiment_dir(name: str, root: Path | None = None) -> Path:
    """``results/stage2/<name>`` by default (created).  Pass ``STAGE25_ROOT`` for Stage 2.5."""
    path = (root or STAGE2_ROOT) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def distribution_summary(values: Iterable[float]) -> dict[str, float]:
    """mean / std / median / 5th / 95th percentile of finite values."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "p05": np.nan, "p95": np.nan, "count": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "count": int(arr.size),
    }


def summarize_by_method(frame: pd.DataFrame, columns: Iterable[str], method_column: str = "method") -> pd.DataFrame:
    """Long-format table: one row per (method, metric) with distribution statistics."""
    rows = []
    for method, sub in frame.groupby(method_column, sort=False):
        ok = sub[sub["status"] == "ok"] if "status" in sub else sub
        for col in columns:
            if col not in ok:
                continue
            stats = distribution_summary(ok[col].tolist())
            rows.append({"method": method, "metric": col, **stats})
    return pd.DataFrame(rows)


def print_frame(frame: pd.DataFrame, float_format: str = "{:.6g}") -> None:
    with pd.option_context("display.float_format", float_format.format, "display.width", 200, "display.max_columns", 50):
        print(frame.to_string(index=False))
    print()

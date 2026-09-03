"""Shared paths, config snapshots and CLI helpers for ICC experiments."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import pandas as pd

from configs.default import DefaultConfig, default_config
from isac.evaluation.reporting import print_config
from isac.evaluation.scenario import with_num_subcarriers, with_seed

ICC_ROOT = Path("results") / "icc_paper"
ICC_RAW = ICC_ROOT / "raw_data"
ICC_FIG = ICC_ROOT / "figures"
ICC_TAB = ICC_ROOT / "tables"
ICC_CFG = ICC_ROOT / "configs"
ICC_LOG = ICC_ROOT / "logs"
ICC_SUM = ICC_ROOT / "summary"


def icc_dirs() -> None:
    for path in (ICC_ROOT, ICC_RAW, ICC_FIG, ICC_TAB, ICC_CFG, ICC_LOG, ICC_SUM):
        path.mkdir(parents=True, exist_ok=True)


def snapshot_config(cfg: DefaultConfig, name: str, extra: dict[str, Any] | None = None) -> Path:
    icc_dirs()
    payload: dict[str, Any] = {"experiment": name}
    for section_name in ("ofdm", "channel", "tdl", "sensing", "energy", "ambiguity", "optimization", "simulation"):
        section = getattr(cfg, section_name)
        row = {}
        for field in dataclasses.fields(section):
            value = getattr(section, field.name)
            if isinstance(value, tuple):
                value = list(value)
            row[field.name] = value
        payload[section_name] = row
    payload["derived"] = {
        "noise_psd_w_per_hz": cfg.ofdm.noise_psd_w_per_hz,
        "noise_power_per_subcarrier_w": cfg.ofdm.noise_power_per_subcarrier_w,
        "peak_power_w": cfg.ofdm.peak_power_w,
        "bandwidth_hz": cfg.ofdm.bandwidth_hz,
        "symbol_duration_s": cfg.ofdm.symbol_duration_s,
        "round_trip_delay_s": cfg.sensing.round_trip_delay_s,
        "path_gain": cfg.channel.path_gain,
        "tdl_num_taps": cfg.tdl.resolved_num_taps(),
        "mainlobe_scale_s": 1.0 / cfg.ofdm.bandwidth_hz,
    }
    if extra:
        payload["run"] = extra
    path = ICC_CFG / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def icc_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="tiny sample sizes for smoke tests")
    parser.add_argument("--num-realizations", type=int, default=None)
    parser.add_argument("--num-subcarriers", type=int, default=None)
    return parser


def config_from_icc_args(args: argparse.Namespace) -> DefaultConfig:
    cfg = with_seed(default_config(), args.seed)
    if getattr(args, "num_subcarriers", None):
        cfg = with_num_subcarriers(cfg, args.num_subcarriers)
    return cfg


def write_csv(frame: pd.DataFrame, name: str) -> Path:
    icc_dirs()
    path = ICC_RAW / name
    frame.to_csv(path, index=False)
    return path


def write_log(name: str, text: str) -> Path:
    icc_dirs()
    path = ICC_LOG / name
    path.write_text(text)
    return path


def print_and_snapshot(cfg: DefaultConfig, name: str, extra: dict[str, Any] | None = None) -> None:
    print_config(cfg)
    snapshot_config(cfg, name, extra)

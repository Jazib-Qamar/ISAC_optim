"""Scenario construction shared by the experiments (seeded channel draws, argument parsing)."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from configs.default import ChannelConfig, DefaultConfig, SimulationConfig, default_config
from isac.channels.rayleigh import rayleigh_channel
from isac.communication.rate import channel_gain
from isac.system import ISACSystem


def draw_channel_gain(cfg: DefaultConfig, rng: np.random.Generator) -> np.ndarray:
    """One Rayleigh realisation ``|h_k|^2`` including the configured path gain."""
    return channel_gain(rayleigh_channel(cfg.ofdm.num_subcarriers, rng, mean_gain=cfg.channel.path_gain))


def build_system(cfg: DefaultConfig, rng: np.random.Generator, total_power_w: float | None = None) -> tuple[np.ndarray, ISACSystem]:
    """Draw a channel and bundle it with the configuration."""
    gain = draw_channel_gain(cfg, rng)
    return gain, ISACSystem.from_config(cfg, gain, total_power_w=total_power_w)


def with_seed(cfg: DefaultConfig, seed: int | None) -> DefaultConfig:
    if seed is None:
        return cfg
    return dataclasses.replace(cfg, simulation=SimulationConfig(seed=seed, output_dir=cfg.simulation.output_dir))


def with_path_loss(cfg: DefaultConfig, path_loss_db: float) -> DefaultConfig:
    return dataclasses.replace(
        cfg, channel=ChannelConfig(path_loss_db=path_loss_db, temporal_correlation=cfg.channel.temporal_correlation)
    )


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: configuration seed)")
    parser.add_argument("--output-dir", type=Path, default=None, help="output directory (default: results/stage2/<exp>)")
    return parser


def config_from_args(args: argparse.Namespace) -> DefaultConfig:
    return with_seed(default_config(), args.seed)

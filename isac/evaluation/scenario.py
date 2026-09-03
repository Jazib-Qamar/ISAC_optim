"""Scenario construction shared by the experiments (seeded channel draws, argument parsing)."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from configs.default import ChannelConfig, DefaultConfig, SimulationConfig, default_config
from isac.channels.rayleigh import rayleigh_channel
from isac.communication.rate import channel_gain
from isac.sensing.frequencies import centered_subcarrier_frequencies
from isac.system import ISACSystem


def draw_channel_gain(cfg: DefaultConfig, rng: np.random.Generator) -> np.ndarray:
    """One Rayleigh realisation ``|h_k|^2`` including the configured path gain."""
    return channel_gain(rayleigh_channel(cfg.ofdm.num_subcarriers, rng, mean_gain=cfg.channel.path_gain))


def build_system(cfg: DefaultConfig, rng: np.random.Generator, total_power_w: float | None = None) -> tuple[np.ndarray, ISACSystem]:
    """Draw a channel and bundle it with the configuration."""
    gain = draw_channel_gain(cfg, rng)
    return gain, ISACSystem.from_config(cfg, gain, total_power_w=total_power_w)


def draw_asymmetric_channel_gain(
    cfg: DefaultConfig,
    rng: np.random.Generator,
    *,
    tilt: float = 4.0,
    strong_side: str = "negative",
) -> np.ndarray:
    """Rayleigh realisation with a reproducible one-sided spectral tilt.

    The small-scale fading is unmodified; a smooth logistic tilt in frequency
    multiplies ``|h_k|^2`` so that strong communication subcarriers concentrate
    on one side of the OFDM band.  The mean gain is renormalised to the
    the untilted realisation's mean so the path-loss setting is preserved.

    ``tilt`` is the logistic steepness (dimensionless); ``strong_side`` is
    ``"negative"`` or ``"positive"`` frequency.
    """
    if strong_side not in ("negative", "positive"):
        raise ValueError("strong_side must be 'negative' or 'positive'")
    if tilt <= 0.0:
        raise ValueError("tilt must be positive")
    base = draw_channel_gain(cfg, rng)
    freqs = centered_subcarrier_frequencies(cfg.ofdm.num_subcarriers, cfg.ofdm.subcarrier_spacing_hz)
    x = freqs / np.max(np.abs(freqs))
    sign = 1.0 if strong_side == "negative" else -1.0
    logistic = 1.0 / (1.0 + np.exp(sign * tilt * x))
    tilted = base * logistic
    return tilted * (float(np.mean(base)) / float(np.mean(tilted)))


def build_asymmetric_system(
    cfg: DefaultConfig,
    rng: np.random.Generator,
    *,
    tilt: float = 4.0,
    strong_side: str = "negative",
    total_power_w: float | None = None,
) -> tuple[np.ndarray, ISACSystem]:
    """Asymmetric-channel counterpart of :func:`build_system`."""
    gain = draw_asymmetric_channel_gain(cfg, rng, tilt=tilt, strong_side=strong_side)
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

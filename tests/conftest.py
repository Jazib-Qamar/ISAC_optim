"""Shared fixtures for the Stage 2 optimisation tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from configs.default import DefaultConfig, OFDMConfig, default_config
from isac.channels.rayleigh import rayleigh_channel
from isac.communication.rate import channel_gain
from isac.optimization.feasibility import max_sensing_surrogate
from isac.optimization.max_rate import water_filling_capacity
from isac.system import ISACSystem


@pytest.fixture(scope="session")
def cfg() -> DefaultConfig:
    return default_config()


@pytest.fixture(scope="session")
def system64(cfg: DefaultConfig) -> ISACSystem:
    """One deterministic 64-subcarrier Rayleigh realisation with default parameters."""
    rng = np.random.default_rng(2024)
    gain = channel_gain(rayleigh_channel(cfg.ofdm.num_subcarriers, rng, cfg.channel.path_gain))
    return ISACSystem.from_config(cfg, gain)


@pytest.fixture(scope="session")
def thresholds(system64: ISACSystem) -> dict[str, float]:
    """Requirements relative to baseline performance (same rule as the experiments)."""
    c_wf = water_filling_capacity(system64)
    s_max, _ = max_sensing_surrogate(system64.sensing_weights, system64.total_power_w, system64.peak_power_w)
    return {"c_wf": c_wf, "s_max": s_max, "r_min": 0.9 * c_wf, "gamma_s": 0.6 * s_max}


def make_small_system(
    cfg: DefaultConfig,
    num_subcarriers: int,
    gains: np.ndarray,
    total_power_w: float,
    peak_power_factor: float,
    circuit_power_w: float,
) -> ISACSystem:
    """Helper to build tiny systems (K = 2, 3) for brute-force checks."""
    small_cfg = dataclasses.replace(
        cfg,
        ofdm=OFDMConfig(
            num_subcarriers=num_subcarriers,
            total_power_w=total_power_w,
            peak_power_factor=peak_power_factor,
        ),
        energy=dataclasses.replace(cfg.energy, circuit_power_w=circuit_power_w),
    )
    return ISACSystem.from_config(small_cfg, gains)

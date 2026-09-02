"""Tests for the communication rate model."""

from __future__ import annotations

import numpy as np
import pytest

from configs.default import OFDMConfig
from isac.channels.rayleigh import rayleigh_channel
from isac.communication.rate import (
    achievable_rate_bps,
    channel_gain,
    rate_per_subcarrier,
    spectral_efficiency,
    subcarrier_snr,
)

NOISE_POWER_W = 1e-3


def test_zero_power_gives_zero_rate() -> None:
    gain = np.array([1.0, 0.5, 2.0])
    power = np.zeros(3)
    assert spectral_efficiency(power, gain, NOISE_POWER_W) == 0.0
    np.testing.assert_array_equal(rate_per_subcarrier(power, gain, NOISE_POWER_W), 0.0)


def test_unit_snr_gives_one_bit() -> None:
    # SNR = gain * P / sigma^2 = 1  ->  log2(2) = 1 bit/s/Hz on that subcarrier.
    gain = np.array([1.0])
    power = np.array([NOISE_POWER_W])
    np.testing.assert_allclose(subcarrier_snr(power, gain, NOISE_POWER_W), [1.0])
    assert spectral_efficiency(power, gain, NOISE_POWER_W) == pytest.approx(1.0)


def test_spectral_efficiency_increases_with_power() -> None:
    rng = np.random.default_rng(1)
    gain = channel_gain(rayleigh_channel(64, rng))
    scales = np.logspace(-3, 1, 20)
    rates = [spectral_efficiency(np.full(64, s / 64.0), gain, NOISE_POWER_W) for s in scales]
    assert all(np.diff(rates) > 0.0)


def test_rate_is_sum_of_per_subcarrier_rates() -> None:
    rng = np.random.default_rng(2)
    gain = channel_gain(rayleigh_channel(16, rng))
    power = rng.uniform(0.0, 1e-2, size=16)
    per = rate_per_subcarrier(power, gain, NOISE_POWER_W)
    assert spectral_efficiency(power, gain, NOISE_POWER_W) == pytest.approx(per.sum())


def test_achievable_rate_scales_with_subcarrier_spacing() -> None:
    cfg = OFDMConfig()
    gain = np.ones(cfg.num_subcarriers)
    power = np.full(cfg.num_subcarriers, 1e-3)
    se = spectral_efficiency(power, gain, NOISE_POWER_W)
    bps = achievable_rate_bps(power, gain, NOISE_POWER_W, cfg.subcarrier_spacing_hz)
    assert bps == pytest.approx(se * cfg.subcarrier_spacing_hz)


def test_channel_gain_is_squared_magnitude() -> None:
    h = np.array([1 + 1j, 3 - 4j])
    np.testing.assert_allclose(channel_gain(h), [2.0, 25.0])


@pytest.mark.parametrize(
    "power, gain, noise",
    [
        (np.array([-1e-3, 1e-3]), np.ones(2), NOISE_POWER_W),
        (np.ones(2), np.array([1.0, -1.0]), NOISE_POWER_W),
        (np.ones(2), np.ones(3), NOISE_POWER_W),
        (np.ones(2), np.ones(2), 0.0),
        (np.ones(2), np.ones(2), -1.0),
    ],
)
def test_invalid_inputs_raise(power: np.ndarray, gain: np.ndarray, noise: float) -> None:
    with pytest.raises(ValueError):
        subcarrier_snr(power, gain, noise)

"""Tests for classical water-filling and the uniform baseline."""

from __future__ import annotations

import numpy as np
import pytest

from isac.channels.rayleigh import rayleigh_channel
from isac.communication.rate import channel_gain, spectral_efficiency
from isac.communication.water_filling import uniform_power, water_filling

NOISE_POWER_W = 1e-3
TOTAL_POWER_W = 1.0


@pytest.fixture
def gain() -> np.ndarray:
    return channel_gain(rayleigh_channel(64, np.random.default_rng(123)))


def test_uniform_power_sums_to_total() -> None:
    p = uniform_power(64, TOTAL_POWER_W)
    assert p.sum() == pytest.approx(TOTAL_POWER_W)
    assert np.all(p == p[0])


def test_uniform_power_respects_peak_cap() -> None:
    with pytest.raises(ValueError):
        uniform_power(4, 1.0, peak_power_w=0.1)


def test_water_filling_sums_to_total_power(gain: np.ndarray) -> None:
    result = water_filling(gain, NOISE_POWER_W, TOTAL_POWER_W)
    assert result.power_w.sum() == pytest.approx(TOTAL_POWER_W, abs=1e-9)
    assert np.all(result.power_w >= 0.0)
    assert result.num_active >= 1


def test_water_level_matches_active_subcarriers(gain: np.ndarray) -> None:
    result = water_filling(gain, NOISE_POWER_W, TOTAL_POWER_W)
    active = result.power_w > 0.0
    # On active carriers P_k + 1/alpha_k = mu; on inactive carriers 1/alpha_k >= mu.
    np.testing.assert_allclose(
        result.power_w[active] + result.inverse_gain_w[active],
        result.water_level_w,
        rtol=1e-8,
    )
    assert np.all(result.inverse_gain_w[~active] >= result.water_level_w - 1e-9)


def test_water_filling_beats_uniform(gain: np.ndarray) -> None:
    wf = water_filling(gain, NOISE_POWER_W, TOTAL_POWER_W).power_w
    uni = uniform_power(gain.size, TOTAL_POWER_W)
    assert spectral_efficiency(wf, gain, NOISE_POWER_W) >= spectral_efficiency(
        uni, gain, NOISE_POWER_W
    )


def test_flat_channel_reduces_to_uniform() -> None:
    flat = np.full(8, 0.7)
    result = water_filling(flat, NOISE_POWER_W, TOTAL_POWER_W)
    np.testing.assert_allclose(result.power_w, TOTAL_POWER_W / 8, rtol=1e-8)


def test_peak_clipping_is_respected(gain: np.ndarray) -> None:
    unclipped = water_filling(gain, NOISE_POWER_W, TOTAL_POWER_W).power_w
    mean_power = TOTAL_POWER_W / gain.size
    assert unclipped.max() > mean_power
    # A cap strictly between the mean and the unclipped maximum must bind.
    peak = 0.5 * (mean_power + unclipped.max())
    result = water_filling(gain, NOISE_POWER_W, TOTAL_POWER_W, peak_power_w=peak)
    assert result.power_w.sum() == pytest.approx(TOTAL_POWER_W, abs=1e-9)
    assert np.all(result.power_w <= peak + 1e-12)
    assert np.any(np.isclose(result.power_w, peak))


def test_infeasible_peak_raises(gain: np.ndarray) -> None:
    with pytest.raises(ValueError):
        water_filling(gain, NOISE_POWER_W, TOTAL_POWER_W, peak_power_w=0.5 / gain.size)


def test_zero_gain_subcarriers_get_no_power() -> None:
    g = np.array([1.0, 0.0, 2.0, 0.0])
    result = water_filling(g, NOISE_POWER_W, TOTAL_POWER_W)
    assert result.power_w[1] == 0.0 and result.power_w[3] == 0.0
    assert result.power_w.sum() == pytest.approx(TOTAL_POWER_W, abs=1e-9)


def test_low_power_uses_only_best_subcarrier() -> None:
    g = np.array([1.0, 10.0, 2.0])
    tiny = 1e-6
    result = water_filling(g, NOISE_POWER_W, tiny)
    assert result.power_w.argmax() == 1
    assert result.num_active == 1
    assert result.power_w.sum() == pytest.approx(tiny, abs=1e-12)


def test_zero_total_power() -> None:
    result = water_filling(np.ones(4), NOISE_POWER_W, 0.0)
    np.testing.assert_array_equal(result.power_w, 0.0)


def test_all_zero_gain_raises() -> None:
    with pytest.raises(ValueError):
        water_filling(np.zeros(4), NOISE_POWER_W, TOTAL_POWER_W)

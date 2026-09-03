"""Exponential-PDP TDL channel: normalisation, correlation, no constructed tilt."""

from __future__ import annotations

import numpy as np
import pytest

from isac.channels.tdl import (
    draw_tdl_realization,
    exponential_pdp,
    tap_delay_axis,
    tdl_channel,
    tdl_frequency_response,
)
from isac.sensing.frequencies import centered_subcarrier_frequencies


def test_exponential_pdp_normalised() -> None:
    delays = tap_delay_axis(50e-9, 16)
    pdp = exponential_pdp(delays, 300e-9)
    assert pdp.shape == delays.shape
    assert pytest.approx(float(np.sum(pdp)), rel=1e-12) == 1.0
    assert np.all(pdp > 0.0)
    assert np.all(np.diff(pdp) <= 1e-15)  # nonincreasing in delay


def test_average_total_tap_power_is_one() -> None:
    rng = np.random.default_rng(7)
    freqs = centered_subcarrier_frequencies(64, 15e3)
    totals = []
    gains = []
    for _ in range(400):
        real = draw_tdl_realization(
            freqs, rng, rms_delay_spread_s=300e-9, tap_spacing_s=50e-9, num_taps=24,
        )
        totals.append(float(np.sum(np.abs(real.taps) ** 2)))
        gains.append(np.abs(real.frequency_response) ** 2)
    assert float(np.mean(totals)) == pytest.approx(1.0, rel=0.08)
    mean_gain = np.mean(np.stack(gains), axis=0)
    # Frequency-flat average transfer (no constructed spectral tilt).
    assert float(np.std(mean_gain) / np.mean(mean_gain)) < 0.15
    pos = mean_gain[freqs > 0.0]
    neg = mean_gain[freqs < 0.0]
    assert abs(float(np.mean(pos)) - float(np.mean(neg))) / float(np.mean(mean_gain)) < 0.08


def test_smaller_delay_spread_increases_frequency_correlation() -> None:
    freqs = centered_subcarrier_frequencies(64, 15e3)
    lag = 16  # 240 kHz; adjacent 15 kHz tones are correlated for any indoor τ_rms

    def lag_corr(rms: float, n: int = 300) -> float:
        acc = 0.0
        for i in range(n):
            h = tdl_channel(
                freqs, np.random.default_rng(2000 + i),
                rms_delay_spread_s=rms, tap_spacing_s=50e-9, num_taps=40, mean_gain=1.0,
            )
            a, b = h[:-lag], h[lag:]
            acc += float(np.vdot(a, b).real / (np.linalg.norm(a) * np.linalg.norm(b)))
        return acc / n

    rho_small = lag_corr(100e-9)
    rho_large = lag_corr(2e-6)
    assert rho_small > rho_large + 0.15


def test_tdl_frequency_response_matches_direct_sum() -> None:
    freqs = np.array([-15e3, 0.0, 15e3])
    delays = np.array([0.0, 100e-9])
    taps = np.array([0.6 + 0.1j, 0.3 - 0.2j])
    h = tdl_frequency_response(taps, delays, freqs)
    expected = np.array([
        taps[0] + taps[1] * np.exp(-1j * 2 * np.pi * f * delays[1]) for f in freqs
    ])
    np.testing.assert_allclose(h, expected)


def test_tdl_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        exponential_pdp(np.array([0.0, -1e-9]), 300e-9)
    with pytest.raises(ValueError):
        tap_delay_axis(0.0, 4)
    rng = np.random.default_rng(0)
    freqs = centered_subcarrier_frequencies(8, 15e3)
    with pytest.raises(ValueError):
        tdl_channel(freqs, rng, rms_delay_spread_s=1e-7, tap_spacing_s=5e-8, num_taps=4, mean_gain=0.0)

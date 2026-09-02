"""Tests for the delay-domain ambiguity profile, PSL and ISL."""

from __future__ import annotations

import numpy as np
import pytest

from isac.sensing.ambiguity import (
    ambiguity_delay_profile,
    ambiguity_summary,
    default_delay_grid,
    default_mainlobe_exclusion_s,
    dirichlet_delay_profile,
    integrated_sidelobe_level_db,
    peak_sidelobe_level_db,
)
from isac.sensing.frequencies import centered_subcarrier_frequencies

K = 64
DELTA_F = 15e3
B = K * DELTA_F


@pytest.fixture
def freqs() -> np.ndarray:
    return centered_subcarrier_frequencies(K, DELTA_F)


@pytest.fixture
def grid() -> np.ndarray:
    return default_delay_grid(DELTA_F, K, oversampling_factor=16)


def test_grid_contains_zero_and_spans_one_period(grid: np.ndarray) -> None:
    assert np.any(grid == 0.0)
    assert grid.size == K * 16
    assert grid.max() - grid.min() == pytest.approx((1.0 / DELTA_F) * (1 - 1 / grid.size))
    np.testing.assert_allclose(np.diff(grid), grid[1] - grid[0])


def test_normalized_peak_is_one_at_zero_delay(freqs: np.ndarray, grid: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    for _ in range(10):
        power = rng.uniform(0.0, 1.0, K)
        profile = ambiguity_delay_profile(power, freqs, grid)
        zero = int(np.flatnonzero(grid == 0.0)[0])
        assert profile[zero] == pytest.approx(1.0)
        assert profile.max() == pytest.approx(1.0)
        assert np.all(profile <= 1.0 + 1e-12)


def test_profile_is_symmetric_in_magnitude(freqs: np.ndarray) -> None:
    rng = np.random.default_rng(1)
    power = rng.uniform(0.0, 1.0, K)
    taus = np.linspace(-3.0 / B, 3.0 / B, 301)
    profile = ambiguity_delay_profile(power, freqs, taus)
    np.testing.assert_allclose(profile, profile[::-1], atol=1e-12)


def test_uniform_allocation_matches_dirichlet_kernel(freqs: np.ndarray, grid: np.ndarray) -> None:
    profile = ambiguity_delay_profile(np.full(K, 0.01), freqs, grid)
    reference = dirichlet_delay_profile(K, DELTA_F, grid)
    np.testing.assert_allclose(profile, reference, atol=1e-10)
    # First null at |tau| = 1/B and first sidelobe close to the -13.26 dB sinc value.
    exclusion = default_mainlobe_exclusion_s(B, 1.0)
    psl = peak_sidelobe_level_db(np.full(K, 0.01), freqs, grid, exclusion)
    assert psl == pytest.approx(-13.26, abs=0.15)
    null_profile = ambiguity_delay_profile(np.full(K, 0.01), freqs, np.array([1.0 / B, 0.0]))
    assert null_profile[0] == pytest.approx(0.0, abs=1e-12)


def test_psl_and_isl_are_finite_and_negative_for_valid_allocations(freqs: np.ndarray, grid: np.ndarray) -> None:
    rng = np.random.default_rng(2)
    exclusion = default_mainlobe_exclusion_s(B)
    for _ in range(10):
        power = rng.uniform(0.0, 1.0, K)
        psl = peak_sidelobe_level_db(power, freqs, grid, exclusion)
        isl = integrated_sidelobe_level_db(power, freqs, grid, exclusion)
        assert np.isfinite(psl) and psl <= 0.0
        assert np.isfinite(isl)
        summary = ambiguity_summary(power, freqs, grid, exclusion)
        assert summary.psl_db == pytest.approx(psl)
        assert summary.isl_db == pytest.approx(isl)
        assert abs(summary.peak_sidelobe_delay_s) >= exclusion


def test_concentrating_power_changes_sidelobe_structure(freqs: np.ndarray, grid: np.ndarray) -> None:
    exclusion = default_mainlobe_exclusion_s(B)
    uniform = np.full(K, 1.0 / K)
    edges = np.zeros(K)
    edges[[0, K - 1]] = 0.5
    psl_uniform = peak_sidelobe_level_db(uniform, freqs, grid, exclusion)
    psl_edges = peak_sidelobe_level_db(edges, freqs, grid, exclusion)
    # Two tones give a pure cosine |cos(pi (K-1) Delta_f tau)| whose "sidelobes" reach
    # 0 dB (up to the delay-grid quantisation of the cosine peaks).
    assert psl_edges == pytest.approx(0.0, abs=0.05)
    assert psl_edges > psl_uniform
    # A tapered (Hann) spectrum lowers the sidelobes below the uniform case, but it
    # also widens the mainlobe to ~2/B, so both must be compared with the same,
    # wider exclusion region.
    taper = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(K) / K))
    wide_exclusion = default_mainlobe_exclusion_s(B, factor=2.0)
    psl_taper = peak_sidelobe_level_db(taper, freqs, grid, wide_exclusion)
    psl_uniform_wide = peak_sidelobe_level_db(uniform, freqs, grid, wide_exclusion)
    assert psl_taper < psl_uniform_wide
    # With the narrow (1/B) exclusion the widened Hann mainlobe leaks into the
    # search region and is *reported* as a high "sidelobe" - documented caveat.
    assert peak_sidelobe_level_db(taper, freqs, grid, exclusion) > psl_uniform


def test_mainlobe_exclusion_validation(freqs: np.ndarray, grid: np.ndarray) -> None:
    power = np.full(K, 1.0)
    with pytest.raises(ValueError):
        peak_sidelobe_level_db(power, freqs, grid, mainlobe_exclusion_s=0.0)
    with pytest.raises(ValueError):
        peak_sidelobe_level_db(power, freqs, grid, mainlobe_exclusion_s=10.0)  # excludes everything
    with pytest.raises(ValueError):
        ambiguity_delay_profile(np.zeros(K), freqs, grid)
    with pytest.raises(ValueError):
        ambiguity_delay_profile(-power, freqs, grid)

"""Tests for the delay Fisher information, CRB and the linear sensing surrogate."""

from __future__ import annotations

import numpy as np
import pytest

from configs.default import SPEED_OF_LIGHT_M_PER_S
from isac.sensing.crb import (
    crb_from_fisher_information,
    delay_crb,
    delay_crb_summary,
    range_crb_from_delay_crb,
)
from isac.sensing.fim import (
    delay_fisher_information,
    sensing_information_surrogate,
    surrogate_to_fisher_scale,
)
from isac.sensing.frequencies import (
    centered_subcarrier_frequencies,
    centered_subcarrier_indices,
    sensing_weights,
)

K = 64
DELTA_F = 15e3
BETA = 1e-5 * np.exp(1j * 0.3)
NOISE_VAR = 3e-16


@pytest.fixture
def freqs() -> np.ndarray:
    return centered_subcarrier_frequencies(K, DELTA_F)


def test_frequency_grid_is_centered_and_uniform(freqs: np.ndarray) -> None:
    assert freqs.shape == (K,)
    assert freqs.sum() == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(np.diff(freqs), DELTA_F)
    # sum f_k^2 = Delta_f^2 * K (K^2 - 1) / 12
    assert np.sum(freqs**2) == pytest.approx(DELTA_F**2 * K * (K**2 - 1) / 12.0)
    assert centered_subcarrier_indices(4).tolist() == [-1.5, -0.5, 0.5, 1.5]


def test_fisher_information_matches_closed_form(freqs: np.ndarray) -> None:
    power = np.full(K, 1.0 / K)
    expected = 2.0 * abs(BETA) ** 2 / NOISE_VAR * np.sum((2 * np.pi * freqs) ** 2 * power)
    assert delay_fisher_information(power, freqs, BETA, NOISE_VAR) == pytest.approx(expected)


def test_fisher_information_matches_real_gaussian_finite_difference(freqs: np.ndarray) -> None:
    """Independent check of the 2/sigma^2 factor via the real-valued Gaussian FIM.

    Stack [Re y, Im y] with per-component variance sigma^2/2 and evaluate
    J = (1/(sigma^2/2)) * sum ||d mu / d tau||^2 with a central finite difference
    of the noiseless mean mu_k(tau) = beta sqrt(P_k) exp(-j 2 pi f_k tau).
    """
    rng = np.random.default_rng(5)
    power = rng.uniform(0.0, 1.0, size=K)
    tau0 = 3.3e-7
    step = 1e-12

    def mean(tau: float) -> np.ndarray:
        return BETA * np.sqrt(power) * np.exp(-2j * np.pi * freqs * tau)

    derivative = (mean(tau0 + step) - mean(tau0 - step)) / (2.0 * step)
    real_stack = np.concatenate([derivative.real, derivative.imag])
    j_numeric = np.sum(real_stack**2) / (NOISE_VAR / 2.0)
    j_closed = delay_fisher_information(power, freqs, BETA, NOISE_VAR)
    assert j_numeric == pytest.approx(j_closed, rel=1e-6)


def test_fisher_information_increases_when_power_moves_to_edges(freqs: np.ndarray) -> None:
    uniform = np.full(K, 1.0 / K)
    edge = np.zeros(K)
    edge[[0, K - 1]] = 0.5  # same total power concentrated at the band edges
    center = np.zeros(K)
    center[[K // 2 - 1, K // 2]] = 0.5
    j_uniform = delay_fisher_information(uniform, freqs, BETA, NOISE_VAR)
    j_edge = delay_fisher_information(edge, freqs, BETA, NOISE_VAR)
    j_center = delay_fisher_information(center, freqs, BETA, NOISE_VAR)
    assert j_center < j_uniform < j_edge


def test_crb_decreases_as_fisher_information_increases(freqs: np.ndarray) -> None:
    j_values = np.logspace(5, 15, 11)
    crbs = [crb_from_fisher_information(j) for j in j_values]
    assert all(np.diff(crbs) < 0.0)
    power = np.full(K, 1.0 / K)
    j = delay_fisher_information(power, freqs, BETA, NOISE_VAR)
    assert delay_crb(power, freqs, BETA, NOISE_VAR) == pytest.approx(1.0 / j)


def test_crb_scales_linearly_with_noise_and_inversely_with_power(freqs: np.ndarray) -> None:
    power = np.full(K, 1.0 / K)
    base = delay_crb(power, freqs, BETA, NOISE_VAR)
    assert delay_crb(power, freqs, BETA, 2.0 * NOISE_VAR) == pytest.approx(2.0 * base)
    assert delay_crb(2.0 * power, freqs, BETA, NOISE_VAR) == pytest.approx(0.5 * base)
    assert delay_crb(power, freqs, BETA, NOISE_VAR, num_symbols=4) == pytest.approx(base / 4.0)


def test_zero_power_gives_infinite_crb(freqs: np.ndarray) -> None:
    assert delay_fisher_information(np.zeros(K), freqs, BETA, NOISE_VAR) == 0.0
    assert delay_crb(np.zeros(K), freqs, BETA, NOISE_VAR) == np.inf


def test_unknown_amplitude_never_has_more_information(freqs: np.ndarray) -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        power = rng.uniform(0.0, 1.0, size=K)
        j_known = delay_fisher_information(power, freqs, BETA, NOISE_VAR, known_amplitude=True)
        j_unknown = delay_fisher_information(power, freqs, BETA, NOISE_VAR, known_amplitude=False)
        assert j_unknown <= j_known * (1.0 + 1e-12)
    # Symmetric allocations: power-weighted mean frequency is zero -> identical information.
    symmetric = np.full(K, 1.0 / K)
    assert delay_fisher_information(
        symmetric, freqs, BETA, NOISE_VAR, known_amplitude=False
    ) == pytest.approx(delay_fisher_information(symmetric, freqs, BETA, NOISE_VAR))


def test_one_sided_allocation_has_zero_unknown_amplitude_information(freqs: np.ndarray) -> None:
    single = np.zeros(K)
    single[0] = 1.0
    assert delay_fisher_information(single, freqs, BETA, NOISE_VAR, known_amplitude=False) == 0.0
    assert delay_fisher_information(single, freqs, BETA, NOISE_VAR, known_amplitude=True) > 0.0


def test_surrogate_is_proportional_to_known_amplitude_fisher_information(freqs: np.ndarray) -> None:
    rng = np.random.default_rng(11)
    weights = sensing_weights(K, DELTA_F, weighting="squared_frequency")
    scale = surrogate_to_fisher_scale(BETA, NOISE_VAR)
    for _ in range(10):
        power = rng.uniform(0.0, 1.0, size=K)
        surrogate = sensing_information_surrogate(power, weights)
        fisher = delay_fisher_information(power, freqs, BETA, NOISE_VAR)
        assert scale * surrogate == pytest.approx(fisher, rel=1e-12)


def test_normalized_index_weights_are_in_unit_interval() -> None:
    w = sensing_weights(K, DELTA_F, weighting="normalized_index")
    assert w.min() >= 0.0 and w.max() == pytest.approx(1.0)
    ratio = sensing_weights(K, DELTA_F, "squared_frequency") / np.where(w > 0, w, np.nan)
    finite = ratio[np.isfinite(ratio)]
    np.testing.assert_allclose(finite, finite[0])  # weightings are proportional


def test_range_crb_conversion() -> None:
    crb_tau = 4e-18
    assert range_crb_from_delay_crb(crb_tau) == pytest.approx(
        (SPEED_OF_LIGHT_M_PER_S / 2.0) ** 2 * crb_tau
    )


def test_summary_fields_are_consistent(freqs: np.ndarray) -> None:
    power = np.full(K, 1.0 / K)
    s = delay_crb_summary(power, freqs, BETA, NOISE_VAR)
    assert s.delay_crb_s2 == pytest.approx(1.0 / s.fisher_information)
    assert s.delay_rmse_bound_s == pytest.approx(np.sqrt(s.delay_crb_s2))
    assert s.range_rmse_bound_m == pytest.approx(SPEED_OF_LIGHT_M_PER_S / 2.0 * s.delay_rmse_bound_s)


@pytest.mark.parametrize(
    "power, beta, noise",
    [
        (np.full(K, -1.0), BETA, NOISE_VAR),
        (np.ones(K + 1), BETA, NOISE_VAR),
        (np.ones(K), 0.0, NOISE_VAR),
        (np.ones(K), BETA, 0.0),
    ],
)
def test_invalid_inputs_raise(freqs: np.ndarray, power: np.ndarray, beta: complex, noise: float) -> None:
    with pytest.raises(ValueError):
        delay_fisher_information(power, freqs, beta, noise)

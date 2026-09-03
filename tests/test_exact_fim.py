"""Stage 2.5 tests: unknown-amplitude kernel, exact FIM constraint, DCP, RSOC equivalence."""

from __future__ import annotations

import numpy as np
import pytest

from isac.optimization import InfeasibleProblemError
from isac.optimization.dinkelbach import solve_dinkelbach_ee
from isac.optimization.feasibility import max_unknown_amplitude_fim
from isac.optimization.fim_constraints import unknown_fim_problem_is_dcp
from isac.optimization.kkt_analysis import finite_difference_kernel_gradient
from isac.optimization.max_rate import solve_max_rate
from isac.optimization.min_power import solve_min_power
from isac.optimization.sensing_spec import QUAD_OVER_LIN, RSOC, UNKNOWN_AMPLITUDE_EXACT
from isac.optimization.solver import ACCEPTED_STATUSES
from isac.sensing.fim import (
    delay_fisher_information,
    power_moments,
    power_weighted_spectral_variance,
    spectral_centroid_hz,
    unknown_amplitude_fim_scale,
    unknown_amplitude_kernel,
    unknown_amplitude_kernel_gradient,
    unknown_amplitude_sensing_information,
)
from isac.sensing.frequencies import centered_subcarrier_frequencies
from isac.system import ISACSystem

K = 64
DELTA_F = 15e3
BETA = 1e-5 * np.exp(1j * 0.3)
NOISE_VAR = 3e-16


@pytest.fixture
def freqs() -> np.ndarray:
    return centered_subcarrier_frequencies(K, DELTA_F)


def test_unknown_information_agrees_with_stage1_fim(freqs: np.ndarray) -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        power = rng.uniform(0.05, 1.0, size=K)
        j_stage1 = delay_fisher_information(power, freqs, BETA, NOISE_VAR, known_amplitude=False)
        j_named = unknown_amplitude_sensing_information(power, freqs, BETA, NOISE_VAR)
        assert j_named == pytest.approx(j_stage1)
        c_beta = unknown_amplitude_fim_scale(BETA, NOISE_VAR)
        assert j_named == pytest.approx(c_beta * unknown_amplitude_kernel(power, freqs), rel=1e-12)


def test_spectral_variance_identity(freqs: np.ndarray) -> None:
    rng = np.random.default_rng(9)
    for _ in range(15):
        power = rng.uniform(0.02, 1.0, size=K)
        s0, s1, s2 = power_moments(power, freqs)
        f_bar = s1 / s0
        g_from_moments = s2 - s1**2 / s0
        g_from_sum = float(np.sum(power * (freqs - f_bar) ** 2))
        assert unknown_amplitude_kernel(power, freqs) == pytest.approx(g_from_moments, rel=1e-12)
        assert unknown_amplitude_kernel(power, freqs) == pytest.approx(g_from_sum, rel=1e-12)
        assert power_weighted_spectral_variance(power, freqs) == pytest.approx(g_from_sum / s0, rel=1e-12)


def test_exact_fim_never_exceeds_known_amplitude(freqs: np.ndarray) -> None:
    rng = np.random.default_rng(11)
    for _ in range(20):
        power = rng.uniform(0.0, 1.0, size=K)
        j_known = delay_fisher_information(power, freqs, BETA, NOISE_VAR, known_amplitude=True)
        j_unknown = unknown_amplitude_sensing_information(power, freqs, BETA, NOISE_VAR)
        assert j_unknown <= j_known * (1.0 + 1e-12)


def test_symmetric_spectrum_has_near_zero_centroid(freqs: np.ndarray) -> None:
    symmetric = np.full(K, 1.0 / K)
    assert spectral_centroid_hz(symmetric, freqs) == pytest.approx(0.0, abs=1e-9)
    w_eff = unknown_amplitude_kernel_gradient(symmetric, freqs)
    np.testing.assert_allclose(w_eff, freqs**2, atol=1e-18)


def test_one_sided_tone_has_zero_unknown_fim(freqs: np.ndarray) -> None:
    single = np.zeros(K)
    single[0] = 1.0
    assert unknown_amplitude_kernel(single, freqs) == pytest.approx(0.0, abs=1e-18)
    assert unknown_amplitude_sensing_information(single, freqs, BETA, NOISE_VAR) == pytest.approx(0.0, abs=1e-18)


def test_finite_difference_gradient_matches_centroid_formula(freqs: np.ndarray) -> None:
    rng = np.random.default_rng(13)
    power = rng.uniform(0.2, 1.0, size=K)  # strictly positive
    analytic = unknown_amplitude_kernel_gradient(power, freqs)
    numeric = finite_difference_kernel_gradient(power, freqs, relative_step=1e-6)
    np.testing.assert_allclose(numeric, analytic, rtol=1e-5, atol=1e-2 * np.max(analytic))


def test_unknown_fim_constraint_is_dcp(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    assert unknown_fim_problem_is_dcp(system64, 0.4 * j_max, QUAD_OVER_LIN)
    assert unknown_fim_problem_is_dcp(system64, 0.4 * j_max, RSOC)


def test_exact_fim_min_power_meets_requested_fim(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    gamma_j = 0.4 * j_max
    res = solve_min_power(
        system64,
        0.5 * thresholds["c_wf"],
        sensing_model=UNKNOWN_AMPLITUDE_EXACT,
        min_unknown_fim=gamma_j,
    )
    assert res.solver_status in ACCEPTED_STATUSES
    assert res.feasible, res.feasibility.violated
    assert res.metrics.delay_fim_unknown_amplitude_per_s2 >= gamma_j * (1 - 1e-6)


def test_exact_fim_max_rate_meets_requested_fim(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    gamma_j = 0.45 * j_max
    res = solve_max_rate(
        system64,
        sensing_model=UNKNOWN_AMPLITUDE_EXACT,
        min_unknown_fim=gamma_j,
    )
    assert res.feasible, res.feasibility.violated
    assert res.metrics.delay_fim_unknown_amplitude_per_s2 >= gamma_j * (1 - 1e-6)
    assert res.tx_power_w == pytest.approx(system64.total_power_w, rel=1e-5)


def test_exact_fim_dinkelbach_meets_requested_fim(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    gamma_j = 0.35 * j_max
    dk = solve_dinkelbach_ee(
        system64,
        sensing_model=UNKNOWN_AMPLITUDE_EXACT,
        min_unknown_fim=gamma_j,
        rel_tolerance=1e-8,
    )
    assert dk.converged
    assert dk.result.feasible, dk.result.feasibility.violated
    assert dk.result.metrics.delay_fim_unknown_amplitude_per_s2 >= gamma_j * (1 - 1e-6)


def test_quad_over_lin_and_rsoc_agree_on_max_rate(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    gamma_j = 0.4 * j_max
    a = solve_max_rate(
        system64, sensing_model=UNKNOWN_AMPLITUDE_EXACT, min_unknown_fim=gamma_j,
        unknown_fim_representation=QUAD_OVER_LIN,
    )
    b = solve_max_rate(
        system64, sensing_model=UNKNOWN_AMPLITUDE_EXACT, min_unknown_fim=gamma_j,
        unknown_fim_representation=RSOC,
    )
    np.testing.assert_allclose(a.power_allocation, b.power_allocation, atol=5e-4 * system64.total_power_w)
    assert a.rate_spectral_efficiency == pytest.approx(b.rate_spectral_efficiency, rel=1e-4)


def test_exact_fim_infeasible_crb_detected(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    with pytest.raises(InfeasibleProblemError):
        solve_max_rate(
            system64,
            sensing_model=UNKNOWN_AMPLITUDE_EXACT,
            min_unknown_fim=1.2 * j_max,
        )


def test_crb_and_rmse_interfaces_match_min_fim(system64: ISACSystem) -> None:
    from isac.sensing.crb import crb_from_fisher_information, min_fim_from_delay_crb, min_fim_from_range_rmse, range_crb_from_delay_crb

    j_max, _ = max_unknown_amplitude_fim(system64)
    gamma_j = 0.3 * j_max
    crb_max = 1.0 / gamma_j
    via_crb = solve_max_rate(
        system64, sensing_model=UNKNOWN_AMPLITUDE_EXACT, max_delay_crb_s2=crb_max,
    )
    via_j = solve_max_rate(
        system64, sensing_model=UNKNOWN_AMPLITUDE_EXACT, min_unknown_fim=gamma_j,
    )
    np.testing.assert_allclose(via_crb.power_allocation, via_j.power_allocation, atol=2e-5 * system64.total_power_w)
    assert min_fim_from_delay_crb(crb_max) == pytest.approx(gamma_j)
    j_u = via_j.metrics.delay_fim_unknown_amplitude_per_s2
    rmse_u = float(np.sqrt(range_crb_from_delay_crb(crb_from_fisher_information(j_u))))
    assert min_fim_from_range_rmse(rmse_u) == pytest.approx(j_u, rel=1e-10)
    via_rmse = solve_max_rate(
        system64, sensing_model=UNKNOWN_AMPLITUDE_EXACT, max_range_rmse_m=rmse_u * 1.05,
    )
    assert via_rmse.metrics.delay_fim_unknown_amplitude_per_s2 >= min_fim_from_range_rmse(rmse_u * 1.05) * (1 - 1e-6)

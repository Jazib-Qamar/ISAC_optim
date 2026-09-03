"""Stage 2.5 tests: KKT gradient identity, symmetric-weight reduction, stationarity."""

from __future__ import annotations

import numpy as np
import pytest

from isac.optimization.feasibility import max_unknown_amplitude_fim
from isac.optimization.kkt_analysis import analyze_exact_fim_kkt, finite_difference_kernel_gradient
from isac.optimization.max_rate import solve_max_rate
from isac.optimization.sensing_spec import UNKNOWN_AMPLITUDE_EXACT
from isac.sensing.fim import unknown_amplitude_kernel_gradient
from isac.system import ISACSystem


def test_kkt_symmetric_allocation_reduces_to_fk2(system64: ISACSystem) -> None:
    power = np.full(system64.num_subcarriers, system64.total_power_w / system64.num_subcarriers)
    w_eff = unknown_amplitude_kernel_gradient(power, system64.frequencies_hz)
    np.testing.assert_allclose(w_eff, system64.frequencies_hz**2, atol=1e-18)
    report = analyze_exact_fim_kkt(power, system64, min_unknown_fim=1.0)
    assert report.spectral_centroid_hz == pytest.approx(0.0, abs=1e-9)


def test_kkt_finite_difference_on_interior_allocation(system64: ISACSystem) -> None:
    rng = np.random.default_rng(4)
    power = rng.uniform(0.3 * system64.peak_power_w, 0.7 * system64.peak_power_w, system64.num_subcarriers)
    power *= system64.total_power_w / power.sum()
    analytic = unknown_amplitude_kernel_gradient(power, system64.frequencies_hz)
    numeric = finite_difference_kernel_gradient(power, system64.frequencies_hz, relative_step=1e-6)
    np.testing.assert_allclose(numeric, analytic, rtol=2e-5, atol=1e-2 * np.max(analytic))


def test_kkt_stationarity_and_complementary_slackness_on_max_rate(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    gamma_j = 0.45 * j_max
    res = solve_max_rate(
        system64,
        sensing_model=UNKNOWN_AMPLITUDE_EXACT,
        min_unknown_fim=gamma_j,
    )
    report = analyze_exact_fim_kkt(
        res.power_allocation, system64, min_unknown_fim=gamma_j, q_bit_per_j=0.0, psl_socs_active=False,
    )
    assert int(np.count_nonzero(report.interior_mask)) >= 2
    # Stationarity residual is in bit/s/Hz per watt; a tight LS fit should be small
    # relative to the communication gradient scale.
    comm_scale = float(np.max(np.abs(system64.gain_to_noise_per_w / np.log(2.0))))
    assert report.max_stationarity_residual < 0.05 * comm_scale
    # FIM constraint should be essentially active at this relatively high Gamma_J.
    slack = res.metrics.delay_fim_unknown_amplitude_per_s2 - gamma_j
    assert abs(slack) / gamma_j < 5e-3 or report.fim_complementary_slackness < 1e-3 * gamma_j
    # Total-power complementary slackness: either nu~0 or the budget is used.
    budget_slack = system64.total_power_w - res.tx_power_w
    assert budget_slack < 1e-4 * system64.total_power_w or abs(report.nu_total_power) < 1e-6


def test_kkt_notes_psl_socs_break_simple_structure(system64: ISACSystem) -> None:
    report = analyze_exact_fim_kkt(
        np.full(system64.num_subcarriers, system64.total_power_w / system64.num_subcarriers),
        system64,
        min_unknown_fim=1.0,
        psl_socs_active=True,
    )
    assert report.psl_socs_active
    assert "SOC" in report.note

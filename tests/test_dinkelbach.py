"""Tests for the Dinkelbach energy-efficiency optimiser, including a K = 2 brute-force check."""

from __future__ import annotations

import numpy as np
import pytest

from configs.default import DefaultConfig
from isac.communication.rate import spectral_efficiency
from isac.optimization import InfeasibleProblemError
from isac.optimization.dinkelbach import solve_dinkelbach_ee
from isac.optimization.max_rate import solve_max_rate
from isac.optimization.solver import ACCEPTED_STATUSES
from isac.system import ISACSystem
from tests.conftest import make_small_system


def test_dinkelbach_residual_converges_to_zero(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    res = solve_dinkelbach_ee(system64, thresholds["gamma_s"], rel_tolerance=1e-8)
    assert res.converged
    assert abs(res.final_residual_bps) <= 1e-8 * res.result.rate_bps
    residuals = np.abs(res.history_frame()["residual_bps"].to_numpy())
    # The residual sequence is non-negative (F(q_n) >= 0 for q_n <= q*) and decreasing.
    assert np.all(res.history_frame()["residual_bps"].to_numpy() >= -1e-6 * res.result.rate_bps)
    assert residuals[-1] < residuals[0]


def test_dinkelbach_q_equals_achieved_ee(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    res = solve_dinkelbach_ee(system64, thresholds["gamma_s"])
    assert res.q_final_bit_per_j == pytest.approx(res.result.energy_efficiency_bit_per_j, rel=1e-7)
    assert all(step.solver_status in ACCEPTED_STATUSES for step in res.history)


def test_dinkelbach_solution_satisfies_constraints(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    res = solve_dinkelbach_ee(system64, thresholds["gamma_s"], min_rate_se=0.5 * thresholds["c_wf"])
    assert res.result.feasible, res.result.feasibility.violated
    assert res.result.sensing_surrogate >= thresholds["gamma_s"] * (1 - 1e-6)
    assert res.result.rate_spectral_efficiency >= 0.5 * thresholds["c_wf"] * (1 - 1e-6)


def test_dinkelbach_does_not_necessarily_use_all_power(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    gamma = 0.3 * thresholds["s_max"]
    small = solve_dinkelbach_ee(system64, gamma, max_power_w=system64.total_power_w)
    large = solve_dinkelbach_ee(system64, gamma, max_power_w=10.0 * system64.total_power_w)
    # A larger feasible set cannot reduce the optimum EE ...
    assert large.result.energy_efficiency_bit_per_j >= small.result.energy_efficiency_bit_per_j * (1 - 1e-6)
    # ... and the EE-optimal allocation leaves most of the enlarged budget unused.
    assert large.result.tx_power_w < 0.5 * 10.0 * system64.total_power_w


def test_dinkelbach_uses_no_more_power_than_max_rate(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    gamma = 0.3 * thresholds["s_max"]
    ee = solve_dinkelbach_ee(system64, gamma)
    mr = solve_max_rate(system64, gamma)
    assert ee.result.tx_power_w <= mr.tx_power_w * (1 + 1e-6)
    assert ee.result.energy_efficiency_bit_per_j >= mr.energy_efficiency_bit_per_j * (1 - 1e-8)


def test_dinkelbach_infeasible_set_detected(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    with pytest.raises(InfeasibleProblemError):
        solve_dinkelbach_ee(system64, 1.1 * thresholds["s_max"])


def _grid_search_ee(
    system: ISACSystem,
    gamma_s: float,
    lo: np.ndarray,
    hi: np.ndarray,
    grid_points: int,
) -> tuple[float, np.ndarray]:
    """EE grid search over the box ``[lo, hi]`` intersected with the feasible set (K = 2)."""
    axis1 = np.linspace(lo[0], hi[0], grid_points)
    axis2 = np.linspace(lo[1], hi[1], grid_points)
    p1, p2 = np.meshgrid(axis1, axis2, indexing="ij")
    total = p1 + p2
    surrogate = system.sensing_weights[0] * p1 + system.sensing_weights[1] * p2
    feasible = (total <= system.total_power_w * (1 + 1e-12)) & (surrogate >= gamma_s * (1 - 1e-12))
    alpha = system.gain_to_noise_per_w
    rate_se = np.log2(1.0 + alpha[0] * p1) + np.log2(1.0 + alpha[1] * p2)
    rate_bps = system.subcarrier_spacing_hz * rate_se
    p_sys = system.circuit_power_w + total / system.pa_efficiency
    ee = np.where(feasible, rate_bps / p_sys, -np.inf)
    idx = np.unravel_index(int(np.argmax(ee)), ee.shape)
    return float(ee[idx]), np.array([p1[idx], p2[idx]])


def _brute_force_ee(system: ISACSystem, gamma_s: float, grid_points: int, refinements: int = 3) -> tuple[float, np.ndarray, float]:
    """Dense grid search over the whole box, then zoom in around the best cell.

    Returns ``(best_ee, best_point, final_step)``.  Every evaluated point is
    feasible, so the returned EE is a lower bound on the true optimum.
    """
    lo = np.zeros(2)
    hi = np.full(2, system.peak_power_w)
    best_ee, best_p = _grid_search_ee(system, gamma_s, lo, hi, grid_points)
    step = (hi - lo) / (grid_points - 1)
    for _ in range(refinements):
        lo = np.maximum(best_p - 2.0 * step, 0.0)
        hi = np.minimum(best_p + 2.0 * step, system.peak_power_w)
        ee, p = _grid_search_ee(system, gamma_s, lo, hi, grid_points)
        if ee > best_ee:
            best_ee, best_p = ee, p
        step = (hi - lo) / (grid_points - 1)
    return best_ee, best_p, float(np.max(step))


@pytest.mark.parametrize(
    "gains, total_power_w, peak_factor, circuit_power_w, gamma_fraction",
    [
        # interior EE optimum (small circuit power -> EE peaks well below the budget)
        (np.array([3.0e-12, 0.8e-12]), 0.2, 1.8, 0.002, 0.0),
        # budget-limited optimum (large circuit power -> use (almost) all power)
        (np.array([1.0e-12, 2.5e-12]), 0.05, 1.5, 0.5, 0.0),
        # active sensing constraint with asymmetric weights emulated via unequal gains
        (np.array([4.0e-12, 0.3e-12]), 0.1, 1.9, 0.01, 0.7),
    ],
)
def test_dinkelbach_matches_brute_force_for_two_subcarriers(
    cfg: DefaultConfig,
    gains: np.ndarray,
    total_power_w: float,
    peak_factor: float,
    circuit_power_w: float,
    gamma_fraction: float,
) -> None:
    system = make_small_system(cfg, 2, gains, total_power_w, peak_factor, circuit_power_w)
    s_max = float(np.dot(system.sensing_weights, np.minimum(system.peak_power_w, [total_power_w / 2] * 2)))
    gamma_s = gamma_fraction * s_max

    ee_grid, p_grid, step = _brute_force_ee(system, gamma_s, grid_points=401)
    res = solve_dinkelbach_ee(system, gamma_s, rel_tolerance=1e-10)

    assert res.converged
    assert res.result.feasible
    ee_dink = res.result.energy_efficiency_bit_per_j
    # The grid optimum is a feasible point, so it can never exceed the true optimum
    # (up to solver tolerance); and the refined grid is fine enough to get within 1e-6.
    assert ee_dink >= ee_grid * (1 - 1e-7)
    assert ee_dink == pytest.approx(ee_grid, rel=1e-6)
    # The maximiser is unique (strictly concave numerator over an affine denominator),
    # so the allocations must agree up to the larger of the final grid resolution and
    # the conic solver's primal accuracy on a flat objective (~1e-4 relative).
    np.testing.assert_allclose(res.result.power_allocation, p_grid, atol=max(20.0 * step, 1e-4 * system.peak_power_w))
    # Cross-check EE with the Stage 1 rate function on the returned allocation.
    se = spectral_efficiency(res.result.power_allocation, system.channel_gain, system.noise_power_w)
    assert ee_dink == pytest.approx(system.subcarrier_spacing_hz * se / res.result.system_power_w)

"""Tests for the min-power and max-rate OFDM-ISAC optimisers."""

from __future__ import annotations

import numpy as np
import pytest

from isac.communication.rate import spectral_efficiency
from isac.communication.water_filling import uniform_power, water_filling
from isac.evaluation.metrics import evaluate_allocation
from isac.optimization import InfeasibleProblemError
from isac.optimization.feasibility import max_sensing_surrogate
from isac.optimization.heuristics import edge_weighted_allocation, sensing_optimal_allocation
from isac.optimization.max_rate import solve_max_rate, water_filling_capacity
from isac.optimization.min_power import solve_min_power
from isac.optimization.solver import ACCEPTED_STATUSES
from isac.system import ISACSystem

POWER_TOL_W = 1e-6


# --------------------------------------------------------------------------- min power
def test_min_power_constraints_satisfied(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    res = solve_min_power(system64, thresholds["r_min"], thresholds["gamma_s"])
    assert res.solver_status in ACCEPTED_STATUSES
    assert res.feasible, res.feasibility.violated
    assert res.rate_spectral_efficiency >= thresholds["r_min"] * (1 - 1e-6)
    assert res.sensing_surrogate >= thresholds["gamma_s"] * (1 - 1e-6)
    assert np.all(res.power_allocation >= 0.0)
    assert np.max(res.power_allocation) <= system64.peak_power_w * (1 + 1e-6)
    assert res.tx_power_w <= system64.total_power_w * (1 + 1e-6)


def test_min_power_metrics_are_recomputed_consistently(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    res = solve_min_power(system64, thresholds["r_min"], thresholds["gamma_s"])
    independent = evaluate_allocation(res.power_allocation, system64)
    assert res.metrics == independent
    assert res.tx_power_w == pytest.approx(res.power_allocation.sum())
    # objective (sum P) agrees with the recomputed transmit power
    assert res.objective_value == pytest.approx(res.tx_power_w, rel=1e-5, abs=1e-7)


def test_min_power_uses_less_than_full_budget_when_requirements_are_loose(
    system64: ISACSystem, thresholds: dict[str, float]
) -> None:
    res = solve_min_power(system64, 0.5 * thresholds["c_wf"], 0.2 * thresholds["s_max"])
    assert res.tx_power_w < system64.total_power_w * 0.9
    active = [c.name for c in res.feasibility.checks if c.active]
    assert "min_rate" in active or "min_sensing" in active


def test_min_power_infeasible_rate_detected(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    with pytest.raises(InfeasibleProblemError):
        solve_min_power(system64, 1.2 * thresholds["c_wf"], 0.0)


def test_min_power_infeasible_sensing_detected(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    with pytest.raises(InfeasibleProblemError):
        solve_min_power(system64, 0.0, 1.2 * thresholds["s_max"])


def test_min_power_monotone_in_rate_requirement(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    gamma = 0.3 * thresholds["s_max"]
    fractions = [0.5, 0.7, 0.85, 0.95]
    powers = [solve_min_power(system64, f * thresholds["c_wf"], gamma).tx_power_w for f in fractions]
    assert all(np.diff(powers) >= -POWER_TOL_W)


def test_min_power_monotone_in_sensing_requirement(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    r_min = 0.6 * thresholds["c_wf"]
    fractions = [0.2, 0.5, 0.7, 0.9]
    powers = [solve_min_power(system64, r_min, f * thresholds["s_max"]).tx_power_w for f in fractions]
    assert all(np.diff(powers) >= -POWER_TOL_W)


def test_min_power_rejects_negative_requirements(system64: ISACSystem) -> None:
    with pytest.raises(ValueError):
        solve_min_power(system64, -1.0, 0.0)


# --------------------------------------------------------------------------- max rate
def test_max_rate_constraints_satisfied(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    res = solve_max_rate(system64, thresholds["gamma_s"])
    assert res.solver_status in ACCEPTED_STATUSES
    assert res.feasible, res.feasibility.violated
    assert res.sensing_surrogate >= thresholds["gamma_s"] * (1 - 1e-6)
    assert res.tx_power_w == pytest.approx(system64.total_power_w, rel=1e-6)


def test_max_rate_without_sensing_matches_water_filling(system64: ISACSystem) -> None:
    res = solve_max_rate(system64, 0.0)
    wf = water_filling(system64.channel_gain, system64.noise_power_w, system64.total_power_w, system64.peak_power_w)
    np.testing.assert_allclose(res.power_allocation, wf.power_w, atol=1e-5 * system64.total_power_w)
    assert res.rate_spectral_efficiency == pytest.approx(water_filling_capacity(system64), rel=1e-6)


def test_max_rate_beats_uniform_when_uniform_is_feasible(system64: ISACSystem) -> None:
    uni = uniform_power(system64.num_subcarriers, system64.total_power_w, system64.peak_power_w)
    uni_metrics = evaluate_allocation(uni, system64)
    res = solve_max_rate(system64, uni_metrics.sensing_surrogate)  # uniform is feasible by construction
    assert res.rate_spectral_efficiency >= uni_metrics.rate_spectral_efficiency * (1 - 1e-8)


def test_max_rate_stricter_sensing_respected_and_rate_not_increasing(
    system64: ISACSystem, thresholds: dict[str, float]
) -> None:
    fractions = [0.0, 0.3, 0.6, 0.9]
    rates = []
    for f in fractions:
        res = solve_max_rate(system64, f * thresholds["s_max"])
        assert res.sensing_surrogate >= f * thresholds["s_max"] * (1 - 1e-6)
        rates.append(res.rate_spectral_efficiency)
    # Shrinking the feasible set cannot increase the optimum.
    assert all(np.diff(rates) <= 1e-6 * rates[0])


def test_max_rate_infeasible_sensing_detected(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    with pytest.raises(InfeasibleProblemError):
        solve_max_rate(system64, 1.05 * thresholds["s_max"])


def test_max_rate_optional_water_filling_fraction_constraint(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    c_wf = thresholds["c_wf"]
    res = solve_max_rate(system64, 0.3 * thresholds["s_max"], rate_loss_fraction=0.05)
    assert res.extra["water_filling_capacity_se"] == pytest.approx(c_wf)
    assert res.rate_spectral_efficiency >= 0.95 * c_wf * (1 - 1e-8)


# --------------------------------------------------------------------------- heuristics
def test_sensing_optimal_allocation_is_greedy_edge_fill(system64: ISACSystem) -> None:
    p = sensing_optimal_allocation(system64.sensing_weights, system64.total_power_w, system64.peak_power_w)
    s_max, _ = max_sensing_surrogate(system64.sensing_weights, system64.total_power_w, system64.peak_power_w)
    assert p.sum() == pytest.approx(system64.total_power_w)
    assert np.max(p) <= system64.peak_power_w * (1 + 1e-12)
    assert np.dot(system64.sensing_weights, p) == pytest.approx(s_max)
    # Random feasible points never beat the greedy optimum.
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(0, system64.peak_power_w, system64.num_subcarriers)
        q *= min(1.0, system64.total_power_w / q.sum())
        assert np.dot(system64.sensing_weights, q) <= s_max * (1 + 1e-12)


def test_edge_weighted_allocation_respects_limits(system64: ISACSystem) -> None:
    p = edge_weighted_allocation(system64.sensing_weights, system64.total_power_w, system64.peak_power_w)
    assert p.sum() == pytest.approx(system64.total_power_w, rel=1e-9)
    assert np.all(p >= 0.0) and np.max(p) <= system64.peak_power_w * (1 + 1e-12)
    # Edge tones carry at least as much as central tones.
    assert p[0] >= p[system64.num_subcarriers // 2]
    assert spectral_efficiency(p, system64.channel_gain, system64.noise_power_w) > 0.0

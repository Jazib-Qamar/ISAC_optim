"""Tests for the independent constraint verification and the solver helper."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from isac.optimization import InfeasibleProblemError
from isac.optimization.feasibility import check_static_constraints, max_sensing_surrogate
from isac.optimization.min_power import solve_min_power
from isac.optimization.solver import solve_with_fallback
from isac.system import ISACSystem


def test_valid_solution_passes(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    res = solve_min_power(system64, thresholds["r_min"], thresholds["gamma_s"])
    report = check_static_constraints(
        res.power_allocation, system64, thresholds["r_min"], thresholds["gamma_s"]
    )
    assert report.feasible
    assert report.max_violation == 0.0
    assert set(report.slack_dict()) == {
        "slack_power_nonnegative",
        "slack_peak_power",
        "slack_total_power",
        "slack_min_rate",
        "slack_min_sensing",
    }


def test_corrupted_solutions_are_detected(system64: ISACSystem, thresholds: dict[str, float]) -> None:
    res = solve_min_power(system64, thresholds["r_min"], thresholds["gamma_s"])
    p = res.power_allocation.copy()
    r_min, gamma_s = thresholds["r_min"], thresholds["gamma_s"]

    negative = p.copy()
    negative[3] = -1e-3
    assert "power_nonnegative" in check_static_constraints(negative, system64, r_min, gamma_s).violated

    over_peak = p.copy()
    over_peak[0] = 1.5 * system64.peak_power_w
    assert "peak_power" in check_static_constraints(over_peak, system64, r_min, gamma_s).violated

    over_budget = p * (system64.total_power_w / p.sum()) * 1.1
    assert "total_power" in check_static_constraints(over_budget, system64, r_min, gamma_s).violated

    scaled_down = 0.5 * p  # both requirements were active, so halving power breaks both
    report = check_static_constraints(scaled_down, system64, r_min, gamma_s)
    assert "min_rate" in report.violated and "min_sensing" in report.violated
    assert report.max_violation > 0.0

    # Moving power from edges to the centre keeps the budget but breaks sensing.
    centred = np.zeros_like(p)
    centred[system64.num_subcarriers // 2 - 4 : system64.num_subcarriers // 2 + 4] = p.sum() / 8
    centred = np.minimum(centred, system64.peak_power_w)
    assert "min_sensing" in check_static_constraints(centred, system64, None, gamma_s).violated


def test_tolerance_semantics() -> None:
    weights = np.array([1.0, 2.0])
    s_max, alloc = max_sensing_surrogate(weights, total_power_w=1.0, peak_power_w=0.75)
    np.testing.assert_allclose(alloc, [0.25, 0.75])
    assert s_max == pytest.approx(0.25 + 1.5)
    with pytest.raises(ValueError):
        max_sensing_surrogate(np.array([-1.0, 1.0]), 1.0, 1.0)


def test_solver_helper_reports_infeasible_and_rejects_non_dcp() -> None:
    x = cp.Variable(2)
    infeasible = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1.0, cp.sum(x) <= 1.0])
    with pytest.raises(InfeasibleProblemError):
        solve_with_fallback(infeasible, "toy")

    non_dcp = cp.Problem(cp.Minimize(-cp.sum_squares(x)), [x >= 0.0, x <= 1.0])
    with pytest.raises(ValueError):
        solve_with_fallback(non_dcp, "toy")

    feasible = cp.Problem(cp.Minimize(cp.sum_squares(x - 1.0)), [x >= 0.0])
    info = solve_with_fallback(feasible, "toy")
    assert info.status == cp.OPTIMAL
    assert info.solver_name == "CLARABEL"
    assert info.attempts[0][0] == "CLARABEL"

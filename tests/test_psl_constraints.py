"""Stage 2.5 tests: sampled-grid PSL SOCs and cutting-plane generation."""

from __future__ import annotations

import numpy as np
import pytest

from isac.optimization import InfeasibleProblemError
from isac.optimization.ambiguity_constraints import (
    dual_grid_psl_report,
    optimization_sidelobe_delays,
    psl_db_to_linear_amplitude,
    validation_delay_grid,
)
from isac.optimization.cutting_plane import solve_with_psl_cutting_plane
from isac.optimization.feasibility import max_unknown_amplitude_fim
from isac.optimization.max_rate import solve_max_rate
from isac.optimization.sensing_spec import UNKNOWN_AMPLITUDE_EXACT
from isac.sensing.ambiguity import peak_normalized_response_db
from isac.system import ISACSystem


def test_psl_db_to_linear_rejects_positive_db() -> None:
    with pytest.raises(ValueError):
        psl_db_to_linear_amplitude(1.0)
    assert psl_db_to_linear_amplitude(-20.0) == pytest.approx(0.1)


def test_sampled_psl_max_rate_satisfies_optimization_grid(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    delays = optimization_sidelobe_delays(system64, oversampling_factor=4)
    psl_max = -6.0
    res = solve_max_rate(
        system64,
        sensing_model=UNKNOWN_AMPLITUDE_EXACT,
        min_unknown_fim=0.25 * j_max,
        psl_max_db=psl_max,
        psl_delays_s=delays,
    )
    assert res.feasible, res.feasibility.violated
    opt_psl = peak_normalized_response_db(res.power_allocation, system64.frequencies_hz, delays)
    assert opt_psl <= psl_max + 0.15  # small numerical slack in dB
    val_grid = validation_delay_grid(system64, oversampling_factor=32)
    report = dual_grid_psl_report(
        res.power_allocation, system64, np.concatenate([[0.0], delays]), val_grid, psl_max,
        system64.mainlobe_exclusion_s, 4, 32,
    )
    assert "validation_grid_psl_db" in report
    assert "optimization_grid_psl_db" in report
    assert "worst_delay_s" in report


def test_impossible_psl_is_infeasible_not_relaxed(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    delays = optimization_sidelobe_delays(system64, oversampling_factor=4)
    with pytest.raises(InfeasibleProblemError):
        solve_max_rate(
            system64,
            sensing_model=UNKNOWN_AMPLITUDE_EXACT,
            min_unknown_fim=0.3 * j_max,
            psl_max_db=-80.0,
            psl_delays_s=delays,
        )


def test_cutting_plane_adds_constraints_on_violation_and_can_converge(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    # Start with a very coarse optimisation grid so a denser validation grid can violate.
    result = solve_with_psl_cutting_plane(
        solve_max_rate,
        system64,
        psl_max_db=-8.0,
        optimization_oversampling=2,
        validation_oversampling=16,
        max_iterations=8,
        psl_tolerance_db=0.5,
        solve_kwargs={
            "sensing_model": UNKNOWN_AMPLITUDE_EXACT,
            "min_unknown_fim": 0.2 * j_max,
        },
    )
    assert result.history[0].num_psl_constraints >= 1
    if not result.converged:
        # Still a valid outcome: report the remaining violation rather than relaxing.
        assert result.history[-1].violation_margin_db < 0.0
    else:
        assert result.history[-1].worst_psl_db <= -8.0 + 0.5
    # Constraint count is non-decreasing.
    counts = [step.num_psl_constraints for step in result.history]
    assert all(np.diff(counts) >= 0)


def test_cutting_plane_impossible_psl_raises(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    with pytest.raises(InfeasibleProblemError):
        solve_with_psl_cutting_plane(
            solve_max_rate,
            system64,
            psl_max_db=-80.0,
            optimization_oversampling=2,
            validation_oversampling=8,
            max_iterations=3,
            solve_kwargs={
                "sensing_model": UNKNOWN_AMPLITUDE_EXACT,
                "min_unknown_fim": 0.25 * j_max,
            },
        )

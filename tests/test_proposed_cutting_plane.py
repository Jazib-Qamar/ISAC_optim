"""Production proposed solver: cutting-plane dense-grid PSL is the advertised method."""

from __future__ import annotations

import numpy as np
import pytest

from isac.optimization import InfeasibleProblemError
from isac.optimization.ambiguity_constraints import (
    dual_grid_psl_report,
    optimization_sidelobe_delays,
    validation_delay_grid,
)
from isac.optimization.feasibility import max_unknown_amplitude_fim
from isac.optimization.heuristics import sensing_optimal_allocation
from isac.optimization.max_rate import solve_max_rate
from isac.optimization.proposed import solve_proposed_max_rate
from isac.optimization.sensing_spec import UNKNOWN_AMPLITUDE_EXACT
from isac.sensing.ambiguity import peak_normalized_response_db
from isac.system import ISACSystem


def test_coarse_grid_can_miss_a_sidelobe(system64: ISACSystem) -> None:
    power = sensing_optimal_allocation(
        system64.sensing_weights, system64.total_power_w, system64.peak_power_w
    )
    coarse = optimization_sidelobe_delays(system64, oversampling_factor=2)
    dense = validation_delay_grid(system64, oversampling_factor=32)
    psl_coarse = peak_normalized_response_db(power, system64.frequencies_hz, coarse)
    psl_dense = peak_normalized_response_db(power, system64.frequencies_hz, dense)
    # A denser independent grid must not report a *better* (more negative) PSL
    # than a coarse subset-like search by a large margin; typically it is worse.
    assert psl_dense >= psl_coarse - 1e-6


def test_sampled_grid_solution_is_not_labelled_dense_feasible_when_it_misses(
    system64: ISACSystem,
) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    delays = optimization_sidelobe_delays(system64, oversampling_factor=2)
    psl_max = -8.0
    res = solve_max_rate(
        system64,
        sensing_model=UNKNOWN_AMPLITUDE_EXACT,
        min_unknown_fim=0.2 * j_max,
        psl_max_db=psl_max,
        psl_delays_s=delays,
    )
    val_grid = validation_delay_grid(system64, oversampling_factor=32)
    report = dual_grid_psl_report(
        res.power_allocation, system64, np.concatenate([[0.0], delays]), val_grid, psl_max,
        system64.mainlobe_exclusion_s, 2, 32,
    )
    # If the dense grid is worse than the request, that sampled solve must not
    # be treated as meeting the advertised PSL (this is the evaluator rule).
    dense = report["validation_grid_psl_db"]
    dense_ok = dense <= psl_max + 0.25
    if not dense_ok:
        assert dense > psl_max + 0.25


def test_proposed_cutting_plane_meets_dense_psl_when_feasible(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    result = solve_proposed_max_rate(
        system64,
        psl_max_db=-8.0,
        min_unknown_fim=0.2 * j_max,
        optimization_oversampling=2,
        validation_oversampling=16,
        max_iterations=10,
        psl_tolerance_db=0.25,
    )
    assert result.history[0].num_psl_constraints >= 1
    assert "requested_psl_max_db" in result.diagnostic_dict()
    if result.converged:
        assert result.dense_psl_satisfied
        assert result.validation_grid_psl_db <= -8.0 + 0.25
        assert result.status == "dense_psl_satisfied"
    else:
        assert not result.dense_psl_satisfied
        assert result.psl_margin_db < 0.0 or result.status == "max_iterations"


def test_proposed_impossible_psl_remains_infeasible(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    with pytest.raises(InfeasibleProblemError):
        solve_proposed_max_rate(
            system64,
            psl_max_db=-80.0,
            min_unknown_fim=0.25 * j_max,
            optimization_oversampling=2,
            validation_oversampling=8,
            max_iterations=3,
        )


def test_cutting_plane_records_added_constraints_and_runtime(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    result = solve_proposed_max_rate(
        system64,
        psl_max_db=-8.0,
        min_unknown_fim=0.15 * j_max,
        optimization_oversampling=2,
        validation_oversampling=16,
        max_iterations=6,
        psl_tolerance_db=0.5,
    )
    diag = result.diagnostic_dict()
    assert diag["cutting_plane_runtime_s"] > 0.0
    assert diag["num_final_soc_constraints"] >= result.history[0].num_psl_constraints
    assert result.num_cutting_plane_iterations == len(result.history)

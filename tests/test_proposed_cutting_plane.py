"""Production proposed solver: cutting-plane dense-grid PSL is the advertised method."""

from __future__ import annotations

import numpy as np
import pytest

from isac.communication.water_filling import uniform_power
from isac.evaluation.physical import evaluate_independent, target_from_unknown_fim
from isac.optimization import InfeasibleProblemError
from isac.optimization.ambiguity_constraints import (
    optimization_sidelobe_delays,
    validation_delay_grid,
)
from isac.optimization.feasibility import max_sensing_surrogate, max_unknown_amplitude_fim
from isac.optimization.heuristics import sensing_optimal_allocation
from isac.optimization.max_rate import solve_max_rate
from isac.optimization.proposed import solve_proposed_max_rate
from isac.optimization.sensing_spec import UNKNOWN_AMPLITUDE_EXACT
from isac.sensing.ambiguity import peak_sidelobe_level_db
from isac.system import ISACSystem


def _edge_heavy(system: ISACSystem) -> np.ndarray:
    return sensing_optimal_allocation(
        system.sensing_weights, system.total_power_w, system.peak_power_w
    )


def test_coarse_grid_can_miss_a_sidelobe(system64: ISACSystem) -> None:
    """A 2× optimisation grid can under-report PSL versus an independent 32× grid."""
    power = _edge_heavy(system64)
    coarse = optimization_sidelobe_delays(system64, oversampling_factor=2)
    dense = validation_delay_grid(system64, oversampling_factor=32)
    coarse_grid = np.concatenate([[0.0], coarse])
    psl_coarse = peak_sidelobe_level_db(
        power, system64.frequencies_hz, coarse_grid, system64.mainlobe_exclusion_s
    )
    psl_dense = peak_sidelobe_level_db(
        power, system64.frequencies_hz, dense, system64.mainlobe_exclusion_s
    )
    assert psl_dense >= psl_coarse - 1e-6
    assert psl_dense >= psl_coarse + 0.4, (
        f"expected a coarse-grid miss of ≥0.4 dB, got coarse={psl_coarse:.3f} "
        f"dense={psl_dense:.3f}"
    )


def test_sampled_grid_solution_is_not_labelled_dense_feasible_when_it_misses(
    system64: ISACSystem,
) -> None:
    """Sampled-SOC success on a 2× grid is not a dense-PSL certificate."""
    j_max, _ = max_unknown_amplitude_fim(system64)
    s_max, _ = max_sensing_surrogate(
        system64.sensing_weights, system64.total_power_w, system64.peak_power_w
    )
    delays = optimization_sidelobe_delays(system64, oversampling_factor=2)
    psl_max = -0.6
    tol = 0.05
    res = solve_max_rate(
        system64,
        sensing_model=UNKNOWN_AMPLITUDE_EXACT,
        min_unknown_fim=0.99 * j_max,
        psl_max_db=psl_max,
        psl_delays_s=delays,
    )
    coarse_grid = np.concatenate([[0.0], delays])
    dense = validation_delay_grid(system64, oversampling_factor=32)
    opt_psl = peak_sidelobe_level_db(
        res.power_allocation, system64.frequencies_hz, coarse_grid, system64.mainlobe_exclusion_s
    )
    dense_psl = peak_sidelobe_level_db(
        res.power_allocation, system64.frequencies_hz, dense, system64.mainlobe_exclusion_s
    )
    assert opt_psl <= psl_max + tol
    assert dense_psl > psl_max + tol

    target = target_from_unknown_fim(
        system64,
        0.99 * j_max,
        j_max=j_max,
        s_max=s_max,
        psl_max_db=psl_max,
        fim_fraction=0.99,
        psl_tolerance_db=tol,
        optimization_oversampling=2,
        validation_oversampling=32,
    )
    row = evaluate_independent(
        res.power_allocation,
        system64,
        target,
        method="exact_efim_sampled_psl",
        label="Exact EFIM+sampled PSL",
        optimizer_model="exact_efim",
        extra={"dense_psl_satisfied": True},
    )
    assert row["dense_psl_satisfied"] is False
    assert row["validation_grid_psl_db"] > psl_max + tol


def test_proposed_cutting_plane_meets_dense_psl_when_feasible(system64: ISACSystem) -> None:
    """A feasible uniform-spectrum PSL request is met on the independent dense grid."""
    j_max, _ = max_unknown_amplitude_fim(system64)
    uniform = uniform_power(system64.num_subcarriers, system64.total_power_w, system64.peak_power_w)
    val_grid = validation_delay_grid(system64, oversampling_factor=16)
    psl_uniform = peak_sidelobe_level_db(
        uniform, system64.frequencies_hz, val_grid, system64.mainlobe_exclusion_s
    )
    result = solve_proposed_max_rate(
        system64,
        psl_max_db=psl_uniform,
        min_unknown_fim=0.2 * j_max,
        optimization_oversampling=2,
        validation_oversampling=16,
        max_iterations=10,
        psl_tolerance_db=0.25,
    )
    assert result.history[0].num_psl_constraints >= 1
    assert "requested_psl_max_db" in result.diagnostic_dict()
    assert result.converged
    assert result.dense_psl_satisfied
    assert result.validation_grid_psl_db <= psl_uniform + 0.25
    assert result.status == "dense_psl_satisfied"


def test_cutting_plane_detects_sampled_miss_and_adds_dense_cuts(system64: ISACSystem) -> None:
    """When the first sampled solve misses the dense grid, cuts are added until it is met."""
    j_max, _ = max_unknown_amplitude_fim(system64)
    psl_max = -0.8
    tol = 0.05
    result = solve_proposed_max_rate(
        system64,
        psl_max_db=psl_max,
        min_unknown_fim=0.98 * j_max,
        optimization_oversampling=2,
        validation_oversampling=32,
        max_iterations=10,
        psl_tolerance_db=tol,
    )
    assert result.history[0].added_delay_s is not None
    assert result.num_added_soc_constraints >= 1
    assert result.num_cutting_plane_iterations >= 2
    assert result.converged
    assert result.dense_psl_satisfied
    assert result.validation_grid_psl_db <= psl_max + tol
    assert result.initial_validation_grid_psl_db > psl_max + tol


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
    assert result.converged
    assert result.dense_psl_satisfied

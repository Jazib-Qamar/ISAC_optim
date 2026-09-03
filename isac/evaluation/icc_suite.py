"""Fair ICC method catalog: same physical target, independent evaluator.

Method identifiers (stable, used in CSVs)
-----------------------------------------
uniform
water_filling
conventional_s2
conventional_s2_psl
exact_efim
exact_efim_sampled_psl
exact_efim_cutting_plane_psl
exact_efim_cutting_plane_psl_ee

Display names are IEEE-caption friendly.  Conventional S2 still optimises
``S(P)=sum f_k^2 P_k``; exact EFIM optimises ``G(P)``.  After every solve the
independent evaluator scores unknown-amplitude FIM and dense-grid PSL.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import numpy as np

from configs.default import DefaultConfig
from isac.baselines.internal import run_conventional_s2, run_exact_efim, run_uniform, run_water_filling
from isac.communication.water_filling import uniform_power
from isac.evaluation.physical import (
    PhysicalSensingTarget,
    evaluate_independent,
    failed_independent_row,
    target_from_unknown_fim,
)
from isac.optimization.ambiguity_constraints import optimization_sidelobe_delays, validation_delay_grid
from isac.optimization.exceptions import InfeasibleProblemError, SolverFailureError
from isac.optimization.feasibility import max_sensing_surrogate, max_unknown_amplitude_fim
from isac.optimization.proposed import solve_proposed_ee, solve_proposed_max_rate
from isac.sensing.ambiguity import peak_sidelobe_level_db
from isac.system import ISACSystem

METHOD_ORDER: tuple[str, ...] = (
    "uniform",
    "water_filling",
    "conventional_s2",
    "conventional_s2_psl",
    "exact_efim",
    "exact_efim_sampled_psl",
    "exact_efim_cutting_plane_psl",
    "exact_efim_cutting_plane_psl_ee",
)

DISPLAY_NAME: dict[str, str] = {
    "uniform": "Uniform",
    "water_filling": "Water-filling",
    "conventional_s2": r"Conventional $S_2$",
    "conventional_s2_psl": r"Conventional $S_2$+PSL",
    "exact_efim": "Exact EFIM",
    "exact_efim_sampled_psl": "Exact EFIM+sampled PSL",
    "exact_efim_cutting_plane_psl": "Exact EFIM+cutting-plane PSL",
    "exact_efim_cutting_plane_psl_ee": "Exact EFIM+cutting-plane PSL+EE",
}

OPTIMIZER_MODEL: dict[str, str] = {
    "uniform": "none",
    "water_filling": "none",
    "conventional_s2": "conventional_s2",
    "conventional_s2_psl": "conventional_s2",
    "exact_efim": "exact_efim",
    "exact_efim_sampled_psl": "exact_efim",
    "exact_efim_cutting_plane_psl": "exact_efim",
    "exact_efim_cutting_plane_psl_ee": "exact_efim",
}

CHEAP_METHODS: tuple[str, ...] = ("uniform", "water_filling", "conventional_s2", "exact_efim")
SAMPLED_PSL_METHODS: tuple[str, ...] = ("conventional_s2_psl", "exact_efim_sampled_psl")
CUTTING_PLANE_METHODS: tuple[str, ...] = (
    "exact_efim_cutting_plane_psl",
    "exact_efim_cutting_plane_psl_ee",
)


def build_physical_target(
    system: ISACSystem,
    cfg: DefaultConfig,
    *,
    fim_fraction: float | None = None,
    psl_max_db: float | None | object = ...,
    use_uniform_psl: bool = True,
) -> PhysicalSensingTarget:
    """Common unknown-amplitude FIM target, with optional uniform-spectrum PSL.

    ``psl_max_db=...`` (ellipsis) means: if ``use_uniform_psl``, set the request
    to the dense-grid PSL of the uniform allocation; if ``psl_max_db`` is a
    float, use that; if ``None``, no PSL requirement is advertised.
    """
    fraction = cfg.optimization.fim_fraction_of_maximum if fim_fraction is None else float(fim_fraction)
    j_max, _ = max_unknown_amplitude_fim(system)
    s_max, _ = max_sensing_surrogate(system.sensing_weights, system.total_power_w, system.peak_power_w)
    gamma_j = fraction * j_max
    amb = cfg.ambiguity
    if psl_max_db is ...:
        if use_uniform_psl:
            uniform = uniform_power(system.num_subcarriers, system.total_power_w, system.peak_power_w)
            val_grid = validation_delay_grid(system, amb.validation_oversampling_factor)
            psl = peak_sidelobe_level_db(uniform, system.frequencies_hz, val_grid, system.mainlobe_exclusion_s)
        else:
            psl = None
    else:
        psl = psl_max_db  # type: ignore[assignment]
    return target_from_unknown_fim(
        system,
        gamma_j,
        j_max=j_max,
        s_max=s_max,
        psl_max_db=psl,
        fim_fraction=fraction,
        psl_tolerance_db=amb.psl_tolerance_db,
        optimization_oversampling=amb.optimization_oversampling_factor,
        validation_oversampling=amb.validation_oversampling_factor,
    )


def _try_row(
    method: str,
    system: ISACSystem,
    target: PhysicalSensingTarget,
    runner,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        result = runner()
    except InfeasibleProblemError as exc:
        return failed_independent_row(
            method, DISPLAY_NAME[method],
            status="infeasible", error=str(exc),
            solve_time_s=time.perf_counter() - t0,
            optimizer_model=OPTIMIZER_MODEL[method], target=target,
        )
    except SolverFailureError as exc:
        return failed_independent_row(
            method, DISPLAY_NAME[method],
            status="solver_failure", error=str(exc),
            solve_time_s=time.perf_counter() - t0,
            optimizer_model=OPTIMIZER_MODEL[method], target=target,
        )
    extra = dict(result.metadata.get("extra", {})) if result.metadata else {}
    extra.update({k: v for k, v in result.metadata.items() if k != "extra"})
    return evaluate_independent(
        result.power_allocation,
        system,
        target,
        method=method,
        label=DISPLAY_NAME[method],
        status="ok" if result.convergence_status == "ok" else result.convergence_status,
        solver_status=str(result.metadata.get("solver_status", result.convergence_status)),
        solve_time_s=result.runtime_s,
        solver_inaccurate=bool(result.metadata.get("solver_inaccurate", False)),
        optimizer_model=OPTIMIZER_MODEL[method],
        extra=extra,
    )


def run_method(
    method: str,
    system: ISACSystem,
    target: PhysicalSensingTarget,
    cfg: DefaultConfig,
) -> dict[str, Any]:
    """Run one catalog method and return an independent-evaluator row."""
    delays = optimization_sidelobe_delays(system, target.optimization_oversampling)
    psl = target.psl_max_db
    amb = cfg.ambiguity

    if method == "uniform":
        return _try_row(method, system, target, lambda: run_uniform(system))
    if method == "water_filling":
        return _try_row(method, system, target, lambda: run_water_filling(system))
    if method == "conventional_s2":
        return _try_row(
            method, system, target,
            lambda: run_conventional_s2(system, min_sensing_surrogate=target.gamma_conventional_s),
        )
    if method == "conventional_s2_psl":
        if psl is None:
            raise ValueError("conventional_s2_psl requires a PSL target")
        return _try_row(
            method, system, target,
            lambda: run_conventional_s2(
                system,
                min_sensing_surrogate=target.gamma_conventional_s,
                psl_max_db=psl,
                psl_delays_s=delays,
            ),
        )
    if method == "exact_efim":
        return _try_row(
            method, system, target,
            lambda: run_exact_efim(system, min_unknown_fim=target.gamma_unknown_fim),
        )
    if method == "exact_efim_sampled_psl":
        if psl is None:
            raise ValueError("exact_efim_sampled_psl requires a PSL target")
        return _try_row(
            method, system, target,
            lambda: run_exact_efim(
                system,
                min_unknown_fim=target.gamma_unknown_fim,
                psl_max_db=psl,
                psl_delays_s=delays,
            ),
        )
    if method == "exact_efim_cutting_plane_psl":
        if psl is None:
            raise ValueError("cutting-plane PSL requires a PSL target")
        t0 = time.perf_counter()
        try:
            cp_res = solve_proposed_max_rate(
                system,
                psl,
                min_unknown_fim=target.gamma_unknown_fim,
                optimization_oversampling=amb.optimization_oversampling_factor,
                validation_oversampling=amb.validation_oversampling_factor,
                max_iterations=amb.cutting_plane_max_iterations,
                psl_tolerance_db=amb.psl_tolerance_db,
            )
        except InfeasibleProblemError as exc:
            return failed_independent_row(
                method, DISPLAY_NAME[method],
                status="infeasible", error=str(exc),
                solve_time_s=time.perf_counter() - t0,
                optimizer_model="exact_efim", target=target,
            )
        extra = dict(cp_res.result.extra)
        extra.update(cp_res.diagnostic_dict())
        extra["dense_psl_satisfied"] = cp_res.dense_psl_satisfied
        status = "ok" if not cp_res.infeasible else "infeasible"
        if status == "ok" and not cp_res.dense_psl_satisfied:
            extra["psl_unresolved"] = True
        return evaluate_independent(
            cp_res.result.power_allocation, system, target,
            method=method, label=DISPLAY_NAME[method],
            status=status,
            solver_status=cp_res.result.solver_status,
            solve_time_s=cp_res.total_runtime_s,
            solver_inaccurate=cp_res.result.solver_inaccurate,
            optimizer_model="exact_efim", extra=extra,
        )
    if method == "exact_efim_cutting_plane_psl_ee":
        if psl is None:
            raise ValueError("cutting-plane PSL+EE requires a PSL target")
        t0 = time.perf_counter()
        try:
            cp_res = solve_proposed_ee(
                system,
                psl,
                min_unknown_fim=target.gamma_unknown_fim,
                optimization_oversampling=amb.optimization_oversampling_factor,
                validation_oversampling=amb.validation_oversampling_factor,
                max_iterations=amb.cutting_plane_max_iterations,
                psl_tolerance_db=amb.psl_tolerance_db,
                dinkelbach_max_iterations=cfg.optimization.dinkelbach_max_iterations,
                dinkelbach_rel_tolerance=cfg.optimization.dinkelbach_rel_tolerance,
            )
        except InfeasibleProblemError as exc:
            return failed_independent_row(
                method, DISPLAY_NAME[method],
                status="infeasible", error=str(exc),
                solve_time_s=time.perf_counter() - t0,
                optimizer_model="exact_efim", target=target,
            )
        extra = dict(cp_res.result.extra)
        extra.update(cp_res.diagnostic_dict())
        extra["dense_psl_satisfied"] = cp_res.dense_psl_satisfied
        status = "ok" if not cp_res.infeasible else "infeasible"
        return evaluate_independent(
            cp_res.result.power_allocation, system, target,
            method=method, label=DISPLAY_NAME[method],
            status=status,
            solver_status=cp_res.result.solver_status,
            solve_time_s=cp_res.total_runtime_s,
            solver_inaccurate=cp_res.result.solver_inaccurate,
            optimizer_model="exact_efim", extra=extra,
        )
    raise ValueError(f"unknown method {method!r}")


def run_methods(
    system: ISACSystem,
    target: PhysicalSensingTarget,
    cfg: DefaultConfig,
    methods: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    chosen: Iterable[str] = METHOD_ORDER if methods is None else methods
    rows = []
    for method in chosen:
        row = run_method(method, system, target, cfg)
        rows.append(row)
    return rows

"""Production proposed-method wrappers: exact EFIM + cutting-plane dense PSL.

The ICC *proposed* allocation is **not** a single sampled-grid PSL solve.
It is:

1. exact unknown-amplitude delay FIM (or EE under that constraint);
2. sampled-grid PSL SOCs on a moderate delay grid;
3. independent dense-grid validation;
4. cutting-plane generation of the worst dense-grid violator until the dense
   PSL request is met, the iteration cap is hit, or the program is infeasible.

A sampled-grid primal that fails independent dense validation is never labelled
as satisfying the PSL requirement (``dense_psl_satisfied=False``).
Impossible PSL / FIM requests are not relaxed.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from isac.optimization.common import OptimizationResult
from isac.optimization.cutting_plane import CuttingPlaneResult, solve_with_psl_cutting_plane
from isac.optimization.dinkelbach import DinkelbachResult, solve_dinkelbach_ee
from isac.optimization.max_rate import solve_max_rate
from isac.optimization.sensing_spec import UNKNOWN_AMPLITUDE_EXACT
from isac.optimization.solver import DEFAULT_SOLVER_PREFERENCE
from isac.system import ISACSystem

PROPOSED_MAX_RATE = "exact_efim_cutting_plane_psl"
PROPOSED_EE = "exact_efim_cutting_plane_psl_ee"


def _base_kwargs(
    *,
    min_unknown_fim: float | None,
    max_delay_crb_s2: float | None,
    max_range_rmse_m: float | None,
    unknown_fim_representation: str,
    s0_epsilon_w: float,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "sensing_model": UNKNOWN_AMPLITUDE_EXACT,
        "min_unknown_fim": min_unknown_fim,
        "max_delay_crb_s2": max_delay_crb_s2,
        "max_range_rmse_m": max_range_rmse_m,
        "unknown_fim_representation": unknown_fim_representation,
        "s0_epsilon_w": s0_epsilon_w,
    }
    if extra:
        kwargs.update(extra)
    return kwargs


def solve_proposed_max_rate(
    system: ISACSystem,
    psl_max_db: float,
    *,
    min_unknown_fim: float | None = None,
    max_delay_crb_s2: float | None = None,
    max_range_rmse_m: float | None = None,
    optimization_oversampling: int = 4,
    validation_oversampling: int = 32,
    max_iterations: int = 20,
    psl_tolerance_db: float = 0.25,
    initial_delays_s: NDArray[np.float64] | None = None,
    unknown_fim_representation: str = "quad_over_lin",
    s0_epsilon_w: float = 1e-12,
    solver_preference: Sequence[str] = DEFAULT_SOLVER_PREFERENCE,
) -> CuttingPlaneResult:
    """Max-rate proposed method: exact EFIM + cutting-plane dense PSL."""
    extra = {"solver_preference": tuple(solver_preference)}
    return solve_with_psl_cutting_plane(
        solve_max_rate,
        system,
        psl_max_db,
        optimization_oversampling=optimization_oversampling,
        validation_oversampling=validation_oversampling,
        max_iterations=max_iterations,
        psl_tolerance_db=psl_tolerance_db,
        initial_delays_s=initial_delays_s,
        solve_kwargs=_base_kwargs(
            min_unknown_fim=min_unknown_fim,
            max_delay_crb_s2=max_delay_crb_s2,
            max_range_rmse_m=max_range_rmse_m,
            unknown_fim_representation=unknown_fim_representation,
            s0_epsilon_w=s0_epsilon_w,
            extra=extra,
        ),
    )


def _dinkelbach_solve_fn(system: ISACSystem, **kwargs: Any) -> OptimizationResult:
    dres: DinkelbachResult = solve_dinkelbach_ee(system, **kwargs)
    dres.result.extra["dinkelbach_iterations"] = dres.num_iterations
    dres.result.extra["dinkelbach_converged"] = dres.converged
    dres.result.extra["dinkelbach_q_final_bit_per_j"] = dres.q_final_bit_per_j
    dres.result.extra["dinkelbach_final_residual_bps"] = dres.final_residual_bps
    return dres.result


def solve_proposed_ee(
    system: ISACSystem,
    psl_max_db: float,
    *,
    min_unknown_fim: float | None = None,
    max_delay_crb_s2: float | None = None,
    max_range_rmse_m: float | None = None,
    min_rate_se: float | None = None,
    optimization_oversampling: int = 4,
    validation_oversampling: int = 32,
    max_iterations: int = 20,
    psl_tolerance_db: float = 0.25,
    initial_delays_s: NDArray[np.float64] | None = None,
    unknown_fim_representation: str = "quad_over_lin",
    s0_epsilon_w: float = 1e-12,
    solver_preference: Sequence[str] = DEFAULT_SOLVER_PREFERENCE,
    dinkelbach_max_iterations: int = 50,
    dinkelbach_rel_tolerance: float = 1e-8,
) -> CuttingPlaneResult:
    """EE proposed method: Dinkelbach + exact EFIM + cutting-plane dense PSL.

    Dinkelbach's algorithm is the *solver* for the fractional EE program; it is
    not advertised as a contribution.
    """
    extra = {
        "solver_preference": tuple(solver_preference),
        "min_rate_se": min_rate_se,
        "max_iterations": dinkelbach_max_iterations,
        "rel_tolerance": dinkelbach_rel_tolerance,
    }
    return solve_with_psl_cutting_plane(
        _dinkelbach_solve_fn,
        system,
        psl_max_db,
        optimization_oversampling=optimization_oversampling,
        validation_oversampling=validation_oversampling,
        max_iterations=max_iterations,
        psl_tolerance_db=psl_tolerance_db,
        initial_delays_s=initial_delays_s,
        solve_kwargs=_base_kwargs(
            min_unknown_fim=min_unknown_fim,
            max_delay_crb_s2=max_delay_crb_s2,
            max_range_rmse_m=max_range_rmse_m,
            unknown_fim_representation=unknown_fim_representation,
            s0_epsilon_w=s0_epsilon_w,
            extra=extra,
        ),
    )

"""Problem B - maximum communication rate subject to a sensing requirement.

Stage 2 (default ``sensing_model="known_amplitude_linear"``)::

    maximize    R(P, h)
    subject to  S(P) = sum_k w_k P_k >= Gamma_s
                0 <= P_k <= P_peak
                sum_k P_k <= P_total
    optional    R(P, h) >= (1 - gamma_c) * C_WF

Stage 2.5 replaces ``S(P) >= Gamma_s`` with the unknown-amplitude delay FIM
``J_tau^eff(P) >= Gamma_J`` and optionally adds sampled-grid PSL cones.

``C_WF`` is the classical water-filling sum spectral efficiency computed with
the *same* channel, noise, ``P_total`` and ``P_peak`` (Stage 1 implementation).
The optional rate-loss constraint is off by default.

Maximising a concave function over a convex set is a convex program.
"""

from __future__ import annotations

from typing import Sequence

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from isac.communication.rate import spectral_efficiency
from isac.communication.water_filling import water_filling
from isac.optimization.assemble import assemble_static_constraints
from isac.optimization.common import OptimizationResult, build_scaled_model, finalize_solution
from isac.optimization.feasibility import DEFAULT_ABS_TOL, DEFAULT_REL_TOL
from isac.optimization.sensing_spec import (
    DEFAULT_S0_EPSILON_W,
    KNOWN_AMPLITUDE_LINEAR,
    QUAD_OVER_LIN,
    spec_from_legacy,
)
from isac.optimization.solver import DEFAULT_SOLVER_PREFERENCE, solve_with_fallback
from isac.system import ISACSystem

METHOD_NAME = "max_rate_isac"


def water_filling_capacity(system: ISACSystem, total_power_w: float | None = None) -> float:
    """``C_WF``: water-filling sum spectral efficiency [bit/s/Hz] under the same limits."""
    budget = system.total_power_w if total_power_w is None else float(total_power_w)
    result = water_filling(system.channel_gain, system.noise_power_w, budget, peak_power_w=system.peak_power_w)
    return spectral_efficiency(result.power_w, system.channel_gain, system.noise_power_w)


def solve_max_rate(
    system: ISACSystem,
    min_sensing_surrogate: float | None = None,
    total_power_w: float | None = None,
    rate_loss_fraction: float | None = None,
    solver_preference: Sequence[str] = DEFAULT_SOLVER_PREFERENCE,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    sensing_model: str = KNOWN_AMPLITUDE_LINEAR,
    min_unknown_fim: float | None = None,
    max_delay_crb_s2: float | None = None,
    max_range_rmse_m: float | None = None,
    psl_max_db: float | None = None,
    psl_delays_s: NDArray[np.float64] | None = None,
    psl_mainlobe_exclusion_s: float | None = None,
    unknown_fim_representation: str = QUAD_OVER_LIN,
    s0_epsilon_w: float = DEFAULT_S0_EPSILON_W,
) -> OptimizationResult:
    """Solve the maximum-rate OFDM-ISAC problem.

    Parameters
    ----------
    min_sensing_surrogate:
        ``Gamma_s`` for ``S(P) = sum_k w_k P_k``.  Required for the linear model.
    total_power_w:
        ``P_total``; defaults to ``system.total_power_w``.
    rate_loss_fraction:
        Optional ``gamma_c``: adds ``R >= (1 - gamma_c) C_WF``.  ``None`` (default)
        disables the constraint.
    sensing_model:
        ``known_amplitude_linear`` or ``unknown_amplitude_exact``.

    Raises
    ------
    InfeasibleProblemError, SolverFailureError
    """
    if rate_loss_fraction is not None and not 0.0 <= rate_loss_fraction <= 1.0:
        raise ValueError("rate_loss_fraction must lie in [0, 1]")
    spec = spec_from_legacy(
        min_sensing_surrogate,
        sensing_model=sensing_model,
        min_unknown_fim=min_unknown_fim,
        max_delay_crb_s2=max_delay_crb_s2,
        max_range_rmse_m=max_range_rmse_m,
        psl_max_db=psl_max_db,
        psl_delays_s=psl_delays_s,
        psl_mainlobe_exclusion_s=psl_mainlobe_exclusion_s,
        unknown_fim_representation=unknown_fim_representation,
        s0_epsilon_w=s0_epsilon_w,
    )

    model = build_scaled_model(system, total_power_w)
    min_rate_se: float | None = None
    extra: dict[str, float] = {}
    if rate_loss_fraction is not None:
        c_wf = water_filling_capacity(system, total_power_w)
        min_rate_se = (1.0 - rate_loss_fraction) * c_wf
        extra = {"water_filling_capacity_se": c_wf}

    constraints, sensing_extras = assemble_static_constraints(model, system, spec, min_rate_se=min_rate_se)
    extra.update(sensing_extras)
    problem = cp.Problem(cp.Maximize(model.rate_se), constraints)
    info = solve_with_fallback(problem, METHOD_NAME, solver_preference)
    return finalize_solution(
        METHOD_NAME,
        model,
        info,
        system,
        min_rate_se=min_rate_se,
        min_sensing_surrogate=spec.min_sensing_surrogate,
        max_power_w=total_power_w,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        extra=extra,
        min_unknown_fim=spec.resolved_min_unknown_fim(),
        max_psl_db=spec.psl_max_db,
        psl_delay_grid_s=spec.psl_delays_s,
        psl_mainlobe_exclusion_s=spec.resolved_psl_exclusion_s(system) if spec.psl_max_db is not None else None,
    )

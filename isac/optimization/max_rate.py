"""Problem B - maximum communication rate subject to a sensing requirement.

    maximize    R(P, h) = (1/ln 2) sum_k log(1 + alpha_k P_k)
    subject to  S(P) = sum_k w_k P_k >= Gamma_s
                0 <= P_k <= P_peak
                sum_k P_k <= P_total
    optional    R(P, h) >= (1 - gamma_c) * C_WF

``C_WF`` is the classical water-filling sum spectral efficiency computed with
the *same* channel, noise, ``P_total`` and ``P_peak`` (Stage 1 implementation).
The optional constraint is off by default.

Maximising a concave function over a convex set is a convex program.  The total
power constraint is an inequality; because the rate is strictly increasing in
every ``P_k`` it is active at the optimum whenever the peak caps do not already
absorb the whole budget.
"""

from __future__ import annotations

from typing import Sequence

import cvxpy as cp

from isac.communication.water_filling import water_filling
from isac.communication.rate import spectral_efficiency
from isac.optimization.common import OptimizationResult, build_scaled_model, finalize_solution
from isac.optimization.feasibility import DEFAULT_ABS_TOL, DEFAULT_REL_TOL
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
    min_sensing_surrogate: float,
    total_power_w: float | None = None,
    rate_loss_fraction: float | None = None,
    solver_preference: Sequence[str] = DEFAULT_SOLVER_PREFERENCE,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> OptimizationResult:
    """Solve the maximum-rate OFDM-ISAC problem.

    Parameters
    ----------
    min_sensing_surrogate:
        ``Gamma_s`` for ``S(P) = sum_k w_k P_k``.
    total_power_w:
        ``P_total``; defaults to ``system.total_power_w``.
    rate_loss_fraction:
        Optional ``gamma_c``: adds ``R >= (1 - gamma_c) C_WF``.  ``None`` (default)
        disables the constraint.

    Raises
    ------
    InfeasibleProblemError, SolverFailureError
    """
    if min_sensing_surrogate < 0.0:
        raise ValueError("min_sensing_surrogate must be non-negative")
    if rate_loss_fraction is not None and not 0.0 <= rate_loss_fraction <= 1.0:
        raise ValueError("rate_loss_fraction must lie in [0, 1]")

    model = build_scaled_model(system, total_power_w)
    constraints = list(model.box_constraints)
    constraints.append(model.sensing_scaled >= min_sensing_surrogate / model.sensing_scale)

    min_rate_se: float | None = None
    extra: dict[str, float] = {}
    if rate_loss_fraction is not None:
        c_wf = water_filling_capacity(system, total_power_w)
        min_rate_se = (1.0 - rate_loss_fraction) * c_wf
        constraints.append(model.rate_se >= min_rate_se)
        extra = {"water_filling_capacity_se": c_wf}

    problem = cp.Problem(cp.Maximize(model.rate_se), constraints)
    info = solve_with_fallback(problem, METHOD_NAME, solver_preference)
    return finalize_solution(
        METHOD_NAME,
        model,
        info,
        system,
        min_rate_se=min_rate_se,
        min_sensing_surrogate=min_sensing_surrogate,
        max_power_w=total_power_w,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        extra=extra,
    )

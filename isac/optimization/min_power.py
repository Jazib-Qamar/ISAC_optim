"""Problem A - minimum transmit power subject to communication and sensing requirements.

    minimize    sum_k P_k
    subject to  R(P, h) = (1/ln 2) sum_k log(1 + alpha_k P_k) >= R_min
                S(P)    = sum_k w_k P_k                      >= Gamma_s
                0 <= P_k <= P_peak
                sum_k P_k <= P_max

The objective is linear, the rate constraint is the superlevel set of a concave
function (convex), and all other constraints are linear, so this is a convex
program; the solver's ``OPTIMAL`` status certifies a global optimum up to solver
tolerance.  The total-power constraint is an inequality: the solution uses only
the power needed to meet both requirements.
"""

from __future__ import annotations

from typing import Sequence

import cvxpy as cp

from isac.optimization.common import OptimizationResult, build_scaled_model, finalize_solution
from isac.optimization.feasibility import DEFAULT_ABS_TOL, DEFAULT_REL_TOL
from isac.optimization.solver import DEFAULT_SOLVER_PREFERENCE, solve_with_fallback
from isac.system import ISACSystem

METHOD_NAME = "min_power_isac"


def solve_min_power(
    system: ISACSystem,
    min_rate_se: float,
    min_sensing_surrogate: float,
    max_power_w: float | None = None,
    solver_preference: Sequence[str] = DEFAULT_SOLVER_PREFERENCE,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> OptimizationResult:
    """Solve the minimum-power OFDM-ISAC problem.

    Parameters
    ----------
    system:
        Physical parameters (channel, noise, weights, ``P_peak``, power model).
    min_rate_se:
        ``R_min`` in summed spectral efficiency [bit/s/Hz] (``>= 0``).
    min_sensing_surrogate:
        ``Gamma_s`` in the units of ``sum_k w_k P_k`` (``>= 0``).
    max_power_w:
        ``P_max``; defaults to ``system.total_power_w``.

    Raises
    ------
    InfeasibleProblemError, SolverFailureError
    """
    if min_rate_se < 0.0 or min_sensing_surrogate < 0.0:
        raise ValueError("min_rate_se and min_sensing_surrogate must be non-negative")

    model = build_scaled_model(system, max_power_w)
    constraints = list(model.box_constraints)
    constraints.append(model.rate_se >= min_rate_se)
    constraints.append(model.sensing_scaled >= min_sensing_surrogate / model.sensing_scale)
    problem = cp.Problem(cp.Minimize(model.tx_power_w), constraints)

    info = solve_with_fallback(problem, METHOD_NAME, solver_preference)
    return finalize_solution(
        METHOD_NAME,
        model,
        info,
        system,
        min_rate_se=min_rate_se,
        min_sensing_surrogate=min_sensing_surrogate,
        max_power_w=max_power_w,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )

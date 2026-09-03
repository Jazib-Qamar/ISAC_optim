"""Problem A - minimum transmit power subject to communication and sensing requirements.

Stage 2 (default ``sensing_model="known_amplitude_linear"``)::

    minimize    sum_k P_k
    subject to  R(P, h) >= R_min
                S(P)    = sum_k w_k P_k  >= Gamma_s
                0 <= P_k <= P_peak
                sum_k P_k <= P_max

Stage 2.5 (``sensing_model="unknown_amplitude_exact"``) replaces the linear
surrogate with the unknown-complex-amplitude delay FIM

    J_tau^eff(P) = C_beta * (S2 - S1^2/S0)  >= Gamma_J

which is a convex superlevel-set constraint (quadratic-over-linear).  Optional
sampled-grid PSL second-order cones may be added; ``P_k <= P_peak`` remains a
peak spectral power cap, not a PSL constraint.

The total-power constraint is an inequality: the solution uses only the power
needed to meet the requirements.
"""

from __future__ import annotations

from typing import Sequence

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

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

METHOD_NAME = "min_power_isac"


def solve_min_power(
    system: ISACSystem,
    min_rate_se: float,
    min_sensing_surrogate: float | None = None,
    max_power_w: float | None = None,
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
    """Solve the minimum-power OFDM-ISAC problem.

    Parameters
    ----------
    system:
        Physical parameters (channel, noise, weights, ``P_peak``, power model).
    min_rate_se:
        ``R_min`` in summed spectral efficiency [bit/s/Hz] (``>= 0``).
    min_sensing_surrogate:
        ``Gamma_s`` in the units of ``sum_k w_k P_k`` (``>= 0``).  Required when
        ``sensing_model="known_amplitude_linear"``.
    max_power_w:
        ``P_max``; defaults to ``system.total_power_w``.
    sensing_model:
        ``known_amplitude_linear`` (Stage 2 surrogate) or
        ``unknown_amplitude_exact`` (Stage 2.5 FIM).
    min_unknown_fim, max_delay_crb_s2, max_range_rmse_m:
        Equivalent unknown-amplitude FIM requirements (tightest implied ``Gamma_J``
        is used).  Required for ``unknown_amplitude_exact``.
    psl_max_db, psl_delays_s:
        Optional sampled-grid PSL SOC family.  ``psl_delays_s`` must be provided
        whenever ``psl_max_db`` is set.

    Raises
    ------
    InfeasibleProblemError, SolverFailureError
    """
    if min_rate_se < 0.0:
        raise ValueError("min_rate_se must be non-negative")
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

    model = build_scaled_model(system, max_power_w)
    constraints, extras = assemble_static_constraints(model, system, spec, min_rate_se=min_rate_se)
    problem = cp.Problem(cp.Minimize(model.tx_power_w), constraints)

    info = solve_with_fallback(problem, METHOD_NAME, solver_preference)
    return finalize_solution(
        METHOD_NAME,
        model,
        info,
        system,
        min_rate_se=min_rate_se,
        min_sensing_surrogate=spec.min_sensing_surrogate,
        max_power_w=max_power_w,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        extra=extras,
        min_unknown_fim=spec.resolved_min_unknown_fim(),
        max_psl_db=spec.psl_max_db,
        psl_delay_grid_s=spec.psl_delays_s,
        psl_mainlobe_exclusion_s=spec.resolved_psl_exclusion_s(system) if spec.psl_max_db is not None else None,
    )

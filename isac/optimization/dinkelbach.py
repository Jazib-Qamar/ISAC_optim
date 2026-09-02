"""Problem C - energy-efficiency maximisation by Dinkelbach's parametric method.

    maximize    EE(P) = R_bps(P, h) / P_sys(P)
    subject to  S(P) >= Gamma_s,  0 <= P_k <= P_peak,  sum_k P_k <= P_max,
                (optional) R(P, h) >= R_min

with ``R_bps = Delta_f * (1/ln 2) sum_k log(1 + alpha_k P_k)`` [bit/s] and
``P_sys = P_circuit + sum_k P_k / eta_PA`` [W], so ``EE`` is in bit/J.

Dinkelbach's method
-------------------
``EE`` is a ratio of a concave, non-negative numerator and an affine, positive
denominator over a convex compact set (a concave-convex fractional program).
Define, for a parameter ``q >= 0`` [bit/J],

    F(q) = max_{P feasible} [ R_bps(P) - q * P_sys(P) ].

``F`` is convex, strictly decreasing, and ``F(q) = 0`` exactly at ``q = q* = max EE``.
The iteration

    P_n   = argmax_P [ R_bps(P) - q_n P_sys(P) ]        (convex program)
    q_{n+1} = R_bps(P_n) / P_sys(P_n)

starts at ``q_0 = 0`` (a pure rate maximisation) and converges superlinearly to
``q*``.  The residual ``F(q_n) = R_bps(P_n) - q_n P_sys(P_n)`` tends to **zero**
(not to ``q*``) and is used as the stopping criterion:

    |F(q_n)| <= tol_rel * R_bps(P_n).

Numerical scaling: the sub-problem is solved in spectral-efficiency units, i.e.
the objective ``R_se(P) - (q / Delta_f) * P_sys(P)`` is used internally; because
``Delta_f > 0`` this does not change the maximiser.  ``q`` is a CVXPY parameter,
so the problem is canonicalised once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

from isac.communication.rate import spectral_efficiency
from isac.energy.power_model import system_power
from isac.optimization.common import OptimizationResult, build_scaled_model, finalize_solution
from isac.optimization.feasibility import DEFAULT_ABS_TOL, DEFAULT_REL_TOL
from isac.optimization.solver import DEFAULT_SOLVER_PREFERENCE, solve_with_fallback
from isac.system import ISACSystem

METHOD_NAME = "dinkelbach_ee_isac"


@dataclass(frozen=True)
class DinkelbachIteration:
    """One Dinkelbach step (all quantities recomputed with the NumPy model)."""

    iteration: int
    q_bit_per_j: float  # parameter used in this sub-problem
    rate_bps: float
    tx_power_w: float
    system_power_w: float
    energy_efficiency_bit_per_j: float  # = q_new
    residual_bps: float  # F(q) = R - q P_sys
    residual_relative: float  # |F(q)| / R
    solver_status: str
    solver_name: str
    solve_time_s: float


@dataclass(frozen=True)
class DinkelbachResult:
    """Final solution plus the complete convergence history."""

    result: OptimizationResult
    history: tuple[DinkelbachIteration, ...]
    converged: bool

    @property
    def num_iterations(self) -> int:
        return len(self.history)

    @property
    def q_final_bit_per_j(self) -> float:
        """Last parameter value used (``q_n`` of the final sub-problem)."""
        return self.history[-1].q_bit_per_j

    @property
    def final_residual_bps(self) -> float:
        return self.history[-1].residual_bps

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(step) for step in self.history])


def solve_dinkelbach_ee(
    system: ISACSystem,
    min_sensing_surrogate: float,
    max_power_w: float | None = None,
    min_rate_se: float | None = None,
    q_initial_bit_per_j: float = 0.0,
    rel_tolerance: float = 1e-8,
    max_iterations: int = 50,
    solver_preference: Sequence[str] = DEFAULT_SOLVER_PREFERENCE,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> DinkelbachResult:
    """Maximise energy efficiency [bit/J] with Dinkelbach's algorithm.

    Parameters
    ----------
    min_sensing_surrogate:
        ``Gamma_s`` for ``S(P) = sum_k w_k P_k``.
    max_power_w:
        ``P_max``; defaults to ``system.total_power_w``.
    min_rate_se:
        Optional ``R_min`` [bit/s/Hz summed]; ``None`` disables the constraint.
    q_initial_bit_per_j:
        Starting parameter ``q_0`` (default 0 -> first step is rate maximisation).
    rel_tolerance:
        Stop when ``|F(q)| <= rel_tolerance * R_bps(P*)``.
    max_iterations:
        Iteration cap; ``converged=False`` is reported if it is reached.

    Raises
    ------
    InfeasibleProblemError, SolverFailureError
        Raised on the first sub-problem if the feasible set is empty (the
        feasible set does not depend on ``q``).
    """
    if min_sensing_surrogate < 0.0:
        raise ValueError("min_sensing_surrogate must be non-negative")
    if min_rate_se is not None and min_rate_se < 0.0:
        raise ValueError("min_rate_se must be non-negative")
    if q_initial_bit_per_j < 0.0:
        raise ValueError("q_initial_bit_per_j must be non-negative")
    if rel_tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("rel_tolerance must be positive and max_iterations >= 1")

    model = build_scaled_model(system, max_power_w)
    constraints = list(model.box_constraints)
    constraints.append(model.sensing_scaled >= min_sensing_surrogate / model.sensing_scale)
    if min_rate_se is not None:
        constraints.append(model.rate_se >= min_rate_se)

    delta_f = system.subcarrier_spacing_hz
    q_scaled = cp.Parameter(nonneg=True, name="q_over_delta_f")
    objective = cp.Maximize(model.rate_se - q_scaled * model.system_power_w)
    problem = cp.Problem(objective, constraints)

    history: list[DinkelbachIteration] = []
    q = float(q_initial_bit_per_j)
    converged = False
    info = None
    for iteration in range(1, max_iterations + 1):
        q_scaled.value = q / delta_f
        info = solve_with_fallback(problem, f"{METHOD_NAME}[iter {iteration}]", solver_preference)
        power = np.maximum(np.asarray(model.x.value, dtype=np.float64) * system.peak_power_w, 0.0)

        rate_bps = delta_f * spectral_efficiency(power, system.channel_gain, system.noise_power_w)
        p_sys = system_power(power, system.circuit_power_w, system.pa_efficiency)
        residual = rate_bps - q * p_sys
        q_new = rate_bps / p_sys
        history.append(
            DinkelbachIteration(
                iteration=iteration,
                q_bit_per_j=q,
                rate_bps=rate_bps,
                tx_power_w=float(np.sum(power)),
                system_power_w=p_sys,
                energy_efficiency_bit_per_j=q_new,
                residual_bps=residual,
                residual_relative=abs(residual) / rate_bps if rate_bps > 0.0 else float("inf"),
                solver_status=info.status,
                solver_name=info.solver_name,
                solve_time_s=info.solve_time_s,
            )
        )
        if rate_bps > 0.0 and abs(residual) <= rel_tolerance * rate_bps:
            converged = True
            break
        q = q_new

    assert info is not None
    result = finalize_solution(
        METHOD_NAME,
        model,
        info,
        system,
        min_rate_se=min_rate_se,
        min_sensing_surrogate=min_sensing_surrogate,
        max_power_w=max_power_w,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        extra={
            "dinkelbach_iterations": len(history),
            "dinkelbach_converged": converged,
            "dinkelbach_q_final_bit_per_j": history[-1].q_bit_per_j,
            "dinkelbach_final_residual_bps": history[-1].residual_bps,
            "total_solve_time_s": float(sum(step.solve_time_s for step in history)),
        },
    )
    return DinkelbachResult(result=result, history=tuple(history), converged=converged)

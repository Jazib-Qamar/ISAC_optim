"""Shared CVXPY building blocks and the common result container of the static optimisers.

Scaling
-------
The decision variable is ``x = P / P_peak`` in ``[0, 1]^K`` so that the solver
sees O(1) quantities.  The sensing weights are divided by ``max_k w_k``; the
sensing threshold is divided by the same constant.  Objective and constraints
are otherwise the physical expressions:

    R(P)   = (1 / ln 2) * sum_k log(1 + alpha_k P_k)                 [bit/s/Hz summed]
    S(P)   = sum_k w_k P_k
    P_sys  = P_circuit + sum_k P_k / eta_PA                          [W]

with ``alpha_k = |h_k|^2 / sigma_n^2``.  ``log1p`` is concave and increasing, so
``R`` is concave, ``R >= R_min`` is a convex constraint and every problem in this
package is a DCP-compliant convex program (solved with exponential-cone solvers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from isac.evaluation.metrics import AllocationMetrics, evaluate_allocation
from isac.optimization.feasibility import FeasibilityReport, check_static_constraints
from isac.optimization.solver import SolveInfo
from isac.system import ISACSystem

LN2: float = float(np.log(2.0))


@dataclass(frozen=True)
class ScaledModel:
    """CVXPY expressions for one system with the scaled variable ``x = P / P_peak``."""

    x: cp.Variable
    power_w: cp.Expression  # P = P_peak * x           [W]
    rate_se: cp.Expression  # R(P)                     [bit/s/Hz summed]
    sensing_scaled: cp.Expression  # S(P) / w_max
    sensing_scale: float  # w_max (divide thresholds by this)
    tx_power_w: cp.Expression  # sum_k P_k              [W]
    system_power_w: cp.Expression  # P_circuit + sum P / eta [W]
    box_constraints: list[cp.Constraint]


def build_scaled_model(system: ISACSystem, max_power_w: float | None = None) -> ScaledModel:
    """Create the variable, physical expressions and box/budget constraints.

    Parameters
    ----------
    max_power_w:
        Total power limit ``sum_k P_k <= P_max``; defaults to ``system.total_power_w``.
    """
    budget = system.total_power_w if max_power_w is None else float(max_power_w)
    if budget <= 0.0:
        raise ValueError("max_power_w must be positive")
    k = system.num_subcarriers
    peak = system.peak_power_w
    x = cp.Variable(k, name="x_power_over_peak")
    power = peak * x
    alpha_scaled = system.gain_to_noise_per_w * peak  # SNR_k = alpha_scaled_k * x_k
    rate = cp.sum(cp.log1p(cp.multiply(alpha_scaled, x))) / LN2
    w_max = float(np.max(system.sensing_weights))
    sensing_scale = w_max if w_max > 0.0 else 1.0
    sensing = (system.sensing_weights / sensing_scale) @ power
    tx = cp.sum(power)
    p_sys = system.circuit_power_w + tx / system.pa_efficiency
    box = [x >= 0.0, x <= 1.0, tx <= budget]
    return ScaledModel(
        x=x,
        power_w=power,
        rate_se=rate,
        sensing_scaled=sensing,
        sensing_scale=sensing_scale,
        tx_power_w=tx,
        system_power_w=p_sys,
        box_constraints=box,
    )


@dataclass(frozen=True)
class OptimizationResult:
    """Solution of one static optimiser, scored with the NumPy model.

    ``power_allocation`` is the solver output clipped at zero (negative
    round-off removed); ``raw_min_power_w`` records the smallest pre-clip entry.
    ``metrics`` are recomputed from ``power_allocation`` and ``feasibility``
    verifies every requested constraint independently.
    """

    method: str
    power_allocation: NDArray[np.float64]
    metrics: AllocationMetrics
    feasibility: FeasibilityReport
    solver_status: str
    solver_name: str
    solve_time_s: float
    objective_value: float
    solver_inaccurate: bool
    raw_min_power_w: float
    requirements: dict[str, float | None] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    # Flattened accessors -------------------------------------------------
    @property
    def tx_power_w(self) -> float:
        return self.metrics.tx_power_w

    @property
    def system_power_w(self) -> float:
        return self.metrics.system_power_w

    @property
    def rate_spectral_efficiency(self) -> float:
        return self.metrics.rate_spectral_efficiency

    @property
    def rate_bps(self) -> float:
        return self.metrics.rate_bps

    @property
    def sensing_surrogate(self) -> float:
        return self.metrics.sensing_surrogate

    @property
    def delay_fim_per_s2(self) -> float:
        return self.metrics.delay_fim_per_s2

    @property
    def delay_crb_s2(self) -> float:
        return self.metrics.delay_crb_s2

    @property
    def range_rmse_bound_m(self) -> float:
        return self.metrics.range_rmse_bound_m

    @property
    def energy_efficiency_bit_per_j(self) -> float:
        return self.metrics.energy_efficiency_bit_per_j

    @property
    def constraint_slacks(self) -> dict[str, float]:
        return self.feasibility.slack_dict()

    @property
    def feasible(self) -> bool:
        return self.feasibility.feasible

    def to_row(self) -> dict[str, Any]:
        """Flat record for tabular reporting."""
        row: dict[str, Any] = {"method": self.method}
        row.update(self.metrics.as_dict())
        row.update(
            {
                "solver_status": self.solver_status,
                "solver_name": self.solver_name,
                "solve_time_s": self.solve_time_s,
                "objective_value": self.objective_value,
                "solver_inaccurate": self.solver_inaccurate,
                "raw_min_power_w": self.raw_min_power_w,
                "feasible": self.feasible,
                "max_violation": self.feasibility.max_violation,
            }
        )
        row.update({f"req_{k}": v for k, v in self.requirements.items()})
        row.update(self.constraint_slacks)
        return row


def finalize_solution(
    method: str,
    model: ScaledModel,
    info: SolveInfo,
    system: ISACSystem,
    min_rate_se: float | None,
    min_sensing_surrogate: float | None,
    max_power_w: float | None,
    abs_tol: float,
    rel_tol: float,
    extra: dict[str, Any] | None = None,
) -> OptimizationResult:
    """Extract the power vector, verify constraints and recompute physical metrics."""
    if model.x.value is None:
        raise RuntimeError(f"{method}: solver returned no primal value")
    raw_power = np.asarray(model.x.value, dtype=np.float64) * system.peak_power_w
    report = check_static_constraints(
        raw_power,
        system,
        min_rate_se=min_rate_se,
        min_sensing_surrogate=min_sensing_surrogate,
        max_power_w=max_power_w,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    power = np.maximum(raw_power, 0.0)
    metrics = evaluate_allocation(power, system)
    return OptimizationResult(
        method=method,
        power_allocation=power,
        metrics=metrics,
        feasibility=report,
        solver_status=info.status,
        solver_name=info.solver_name,
        solve_time_s=info.solve_time_s,
        objective_value=info.objective_value,
        solver_inaccurate=info.inaccurate,
        raw_min_power_w=float(np.min(raw_power)),
        requirements={
            "min_rate_se": min_rate_se,
            "min_sensing_surrogate": min_sensing_surrogate,
            "max_power_w": system.total_power_w if max_power_w is None else max_power_w,
        },
        extra=dict(extra or {}),
    )

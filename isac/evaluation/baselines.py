"""Common evaluator for the static OFDM-ISAC baselines.

Every method is scored with :func:`isac.evaluation.metrics.evaluate_allocation`
and checked against the *same* requirements with
:func:`isac.optimization.feasibility.check_static_constraints`, so all numbers
in a comparison table come from one physical model.

Methods
-------
A. ``uniform``            - ``P_k = P_total / K``
B. ``water_filling``      - classical communication-only water-filling (Stage 1)
C. ``min_power_isac``     - Problem A (min power s.t. ``R >= R_min``, ``S >= Gamma_s``)
D. ``max_rate_isac``      - Problem B (max rate s.t. ``S >= Gamma_s``)
E. ``dinkelbach_ee_isac`` - Problem C (max EE s.t. ``S >= Gamma_s`` only)
E'. ``dinkelbach_ee_rmin_isac`` - Problem C with the additional ``R >= R_min``
F. ``edge_weighted``      - ``P_k ∝ w_k`` (heuristic sensing-oriented allocation)
G. ``sensing_optimal``    - maximiser of the linear surrogate (sensing-only)

Heuristic methods A, B, F, G ignore the requirements when constructing their
allocation; their feasibility report therefore states whether they *happen* to
satisfy ``R_min`` / ``Gamma_s``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from isac.communication.water_filling import uniform_power, water_filling
from isac.evaluation.metrics import AllocationMetrics, evaluate_allocation
from isac.optimization.dinkelbach import solve_dinkelbach_ee
from isac.optimization.exceptions import InfeasibleProblemError, SolverFailureError
from isac.optimization.feasibility import FeasibilityReport, check_static_constraints, max_sensing_surrogate
from isac.optimization.heuristics import edge_weighted_allocation, sensing_optimal_allocation
from isac.optimization.max_rate import solve_max_rate, water_filling_capacity
from isac.optimization.min_power import solve_min_power
from isac.optimization.solver import DEFAULT_SOLVER_PREFERENCE
from isac.system import ISACSystem

HEURISTIC_METHODS: tuple[str, ...] = ("uniform", "water_filling", "edge_weighted", "sensing_optimal")
OPTIMIZER_METHODS: tuple[str, ...] = ("min_power_isac", "max_rate_isac", "dinkelbach_ee_isac", "dinkelbach_ee_rmin_isac")
ALL_METHODS: tuple[str, ...] = HEURISTIC_METHODS + OPTIMIZER_METHODS
CORE_METHODS: tuple[str, ...] = ("uniform", "water_filling") + OPTIMIZER_METHODS

METHOD_LABELS: dict[str, str] = {
    "uniform": "Uniform",
    "water_filling": "Water-filling (comm. only)",
    "edge_weighted": "Edge-weighted (∝ w_k)",
    "sensing_optimal": "Sensing-optimal (max S)",
    "min_power_isac": "Min-power ISAC",
    "max_rate_isac": "Max-rate ISAC",
    "dinkelbach_ee_isac": "Dinkelbach EE-ISAC",
    "dinkelbach_ee_rmin_isac": "Dinkelbach EE-ISAC (+R_min)",
}


@dataclass(frozen=True)
class StaticRequirements:
    """Communication and sensing requirements derived from baseline performance.

    ``R_min = rate_fraction * C_WF`` and ``Gamma_s = sensing_fraction * S_max``,
    where ``C_WF`` is the water-filling sum spectral efficiency at ``P_total``
    (with the peak cap) and ``S_max`` the largest surrogate value attainable
    under the peak and total power limits.
    """

    min_rate_se: float
    min_sensing_surrogate: float
    water_filling_capacity_se: float
    max_sensing_surrogate: float
    uniform_sensing_surrogate: float
    rate_fraction: float
    sensing_fraction: float

    @classmethod
    def from_system(cls, system: ISACSystem, rate_fraction: float, sensing_fraction: float) -> "StaticRequirements":
        if not 0.0 <= rate_fraction <= 1.0 or not 0.0 <= sensing_fraction <= 1.0:
            raise ValueError("fractions must lie in [0, 1]")
        c_wf = water_filling_capacity(system)
        s_max, _ = max_sensing_surrogate(system.sensing_weights, system.total_power_w, system.peak_power_w)
        uniform = uniform_power(system.num_subcarriers, system.total_power_w, system.peak_power_w)
        s_uniform = float(np.dot(system.sensing_weights, uniform))
        return cls(
            min_rate_se=rate_fraction * c_wf,
            min_sensing_surrogate=sensing_fraction * s_max,
            water_filling_capacity_se=c_wf,
            max_sensing_surrogate=s_max,
            uniform_sensing_surrogate=s_uniform,
            rate_fraction=rate_fraction,
            sensing_fraction=sensing_fraction,
        )

    def describe(self) -> str:
        lines = [
            "Requirements (constructed relative to baseline performance):",
            f"  C_WF  (water-filling sum SE at P_total, with peak cap) = {self.water_filling_capacity_se:.4f} bit/s/Hz",
            f"  R_min = {self.rate_fraction:.2f} * C_WF                              = {self.min_rate_se:.4f} bit/s/Hz",
            f"  S_max (greedy edge fill under P_peak, P_total)          = {self.max_sensing_surrogate:.4e}",
            f"  S_uniform / S_max                                       = {self.uniform_sensing_surrogate / self.max_sensing_surrogate:.4f}",
            f"  Gamma_s = {self.sensing_fraction:.2f} * S_max                          = {self.min_sensing_surrogate:.4e}",
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, float]:
        return {
            "min_rate_se": self.min_rate_se,
            "min_sensing_surrogate": self.min_sensing_surrogate,
            "water_filling_capacity_se": self.water_filling_capacity_se,
            "max_sensing_surrogate": self.max_sensing_surrogate,
            "uniform_sensing_surrogate": self.uniform_sensing_surrogate,
            "rate_fraction": self.rate_fraction,
            "sensing_fraction": self.sensing_fraction,
        }


@dataclass(frozen=True)
class BaselineOutcome:
    """Result of one method on one system (``status`` in ``ok``/``infeasible``/``solver_failure``)."""

    method: str
    status: str
    power_allocation: NDArray[np.float64] | None
    metrics: AllocationMetrics | None
    feasibility: FeasibilityReport | None
    solver_status: str
    solver_name: str
    solve_time_s: float
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "method": self.method,
            "label": METHOD_LABELS.get(self.method, self.method),
            "status": self.status,
            "solver_status": self.solver_status,
            "solver_name": self.solver_name,
            "solve_time_s": self.solve_time_s,
            "error": self.error,
        }
        if self.metrics is not None:
            row.update(self.metrics.as_dict())
        if self.feasibility is not None:
            row["meets_requirements"] = self.feasibility.feasible
            row["max_violation"] = self.feasibility.max_violation
            row["violated_constraints"] = ";".join(self.feasibility.violated)
            row.update(self.feasibility.slack_dict())
        else:
            row["meets_requirements"] = False
        row.update(self.extra)
        return row


def _heuristic_outcome(
    method: str,
    power: NDArray[np.float64],
    system: ISACSystem,
    requirements: StaticRequirements,
    elapsed: float,
    abs_tol: float,
    rel_tol: float,
    extra: dict[str, Any] | None = None,
) -> BaselineOutcome:
    metrics = evaluate_allocation(power, system)
    report = check_static_constraints(
        power,
        system,
        min_rate_se=requirements.min_rate_se,
        min_sensing_surrogate=requirements.min_sensing_surrogate,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    return BaselineOutcome(
        method=method,
        status="ok",
        power_allocation=power,
        metrics=metrics,
        feasibility=report,
        solver_status="closed_form",
        solver_name="numpy",
        solve_time_s=elapsed,
        extra=dict(extra or {}),
    )


def run_static_baselines(
    system: ISACSystem,
    requirements: StaticRequirements,
    methods: Sequence[str] = ALL_METHODS,
    dinkelbach_min_rate: bool = False,
    solver_preference: Sequence[str] = DEFAULT_SOLVER_PREFERENCE,
    dinkelbach_rel_tolerance: float = 1e-8,
    dinkelbach_max_iterations: int = 50,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-6,
) -> list[BaselineOutcome]:
    """Evaluate the requested methods on one system with shared requirements.

    Parameters
    ----------
    dinkelbach_min_rate:
        If ``True`` the EE optimiser additionally enforces ``R >= R_min``; by
        default it only enforces the sensing requirement (the EE objective
        already values rate).

    Infeasible or failed optimiser runs are *recorded*, not discarded.
    """
    outcomes: list[BaselineOutcome] = []
    k = system.num_subcarriers
    for method in methods:
        start = time.perf_counter()
        try:
            if method == "uniform":
                power = uniform_power(k, system.total_power_w, system.peak_power_w)
                outcomes.append(_heuristic_outcome(method, power, system, requirements, time.perf_counter() - start, abs_tol, rel_tol))
            elif method == "water_filling":
                wf = water_filling(system.channel_gain, system.noise_power_w, system.total_power_w, system.peak_power_w)
                outcomes.append(
                    _heuristic_outcome(
                        method, wf.power_w, system, requirements, time.perf_counter() - start, abs_tol, rel_tol,
                        extra={"water_level_w": wf.water_level_w},
                    )
                )
            elif method == "edge_weighted":
                power = edge_weighted_allocation(system.sensing_weights, system.total_power_w, system.peak_power_w)
                outcomes.append(_heuristic_outcome(method, power, system, requirements, time.perf_counter() - start, abs_tol, rel_tol))
            elif method == "sensing_optimal":
                power = sensing_optimal_allocation(system.sensing_weights, system.total_power_w, system.peak_power_w)
                outcomes.append(_heuristic_outcome(method, power, system, requirements, time.perf_counter() - start, abs_tol, rel_tol))
            elif method == "min_power_isac":
                res = solve_min_power(
                    system, requirements.min_rate_se, requirements.min_sensing_surrogate,
                    solver_preference=solver_preference, abs_tol=abs_tol, rel_tol=rel_tol,
                )
                outcomes.append(
                    BaselineOutcome(
                        method, "ok", res.power_allocation, res.metrics, res.feasibility,
                        res.solver_status, res.solver_name, res.solve_time_s,
                        extra={"solver_inaccurate": res.solver_inaccurate},
                    )
                )
            elif method == "max_rate_isac":
                res = solve_max_rate(
                    system, requirements.min_sensing_surrogate,
                    solver_preference=solver_preference, abs_tol=abs_tol, rel_tol=rel_tol,
                )
                # Report the slack w.r.t. R_min too (informative; not a constraint of this problem).
                report = check_static_constraints(
                    res.power_allocation, system, requirements.min_rate_se, requirements.min_sensing_surrogate,
                    abs_tol=abs_tol, rel_tol=rel_tol,
                )
                outcomes.append(
                    BaselineOutcome(
                        method, "ok", res.power_allocation, res.metrics, report,
                        res.solver_status, res.solver_name, res.solve_time_s,
                        extra={"solver_inaccurate": res.solver_inaccurate},
                    )
                )
            elif method in ("dinkelbach_ee_isac", "dinkelbach_ee_rmin_isac"):
                enforce_rate = dinkelbach_min_rate or method == "dinkelbach_ee_rmin_isac"
                dk = solve_dinkelbach_ee(
                    system, requirements.min_sensing_surrogate,
                    min_rate_se=requirements.min_rate_se if enforce_rate else None,
                    rel_tolerance=dinkelbach_rel_tolerance, max_iterations=dinkelbach_max_iterations,
                    solver_preference=solver_preference, abs_tol=abs_tol, rel_tol=rel_tol,
                )
                res = dk.result
                report = check_static_constraints(
                    res.power_allocation, system, requirements.min_rate_se, requirements.min_sensing_surrogate,
                    abs_tol=abs_tol, rel_tol=rel_tol,
                )
                outcomes.append(
                    BaselineOutcome(
                        method, "ok", res.power_allocation, res.metrics, report,
                        res.solver_status, res.solver_name, res.extra["total_solve_time_s"],
                        extra={
                            "solver_inaccurate": res.solver_inaccurate,
                            "dinkelbach_iterations": dk.num_iterations,
                            "dinkelbach_converged": dk.converged,
                            "dinkelbach_q_final_bit_per_j": dk.q_final_bit_per_j,
                            "dinkelbach_final_residual_bps": dk.final_residual_bps,
                            "dinkelbach_enforces_min_rate": enforce_rate,
                        },
                    )
                )
            else:
                raise ValueError(f"unknown method {method!r}")
        except InfeasibleProblemError as exc:
            outcomes.append(
                BaselineOutcome(method, "infeasible", None, None, None, exc.status, exc.solver_name,
                                time.perf_counter() - start, error=str(exc))
            )
        except SolverFailureError as exc:
            outcomes.append(
                BaselineOutcome(method, "solver_failure", None, None, None, "failed", "none",
                                time.perf_counter() - start, error=str(exc))
            )
    return outcomes


def outcomes_to_frame(outcomes: Sequence[BaselineOutcome]) -> pd.DataFrame:
    """Tabulate outcomes (one row per method)."""
    return pd.DataFrame([o.to_row() for o in outcomes])

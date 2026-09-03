"""Optional cutting-plane generation for sampled-grid PSL constraints.

Loop
----
1. Solve the convex program on the current (moderate) delay-SOC set.
2. Evaluate actual PSL on a denser independent validation grid.
3. If ``PSL_validation <= PSL_target + tolerance``, stop.
4. Otherwise add the worst-violating delay ``tau*`` as a new SOC and re-solve.

This tightens a *sampled-grid* constraint; it is not a continuous-delay certificate.
Impossible PSL requests remain infeasible and are not silently relaxed.

A sampled-grid primal that still violates the independent dense grid is
**not** reported as PSL-feasible (``dense_psl_satisfied=False``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from isac.optimization.ambiguity_constraints import (
    dual_grid_psl_report,
    evaluate_psl_on_grid,
    optimization_sidelobe_delays,
    validation_delay_grid,
)
from isac.optimization.common import OptimizationResult
from isac.optimization.exceptions import InfeasibleProblemError
from isac.system import ISACSystem

SolveFn = Callable[..., OptimizationResult]

STATUS_CONVERGED = "dense_psl_satisfied"
STATUS_MAX_ITERATIONS = "max_iterations"
STATUS_INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class CuttingPlaneStep:
    """One cutting-plane iteration."""

    iteration: int
    num_psl_constraints: int
    worst_psl_db: float
    worst_delay_s: float
    violation_margin_db: float
    solver_time_s: float
    solver_status: str
    added_delay_s: float | None


@dataclass(frozen=True)
class CuttingPlaneResult:
    """Final solve plus the cutting-plane history and dense-grid diagnostics."""

    result: OptimizationResult
    history: tuple[CuttingPlaneStep, ...]
    converged: bool
    validation_oversampling: int
    optimization_oversampling: int
    status: str = STATUS_MAX_ITERATIONS
    requested_psl_max_db: float = float("nan")
    optimization_grid_psl_db: float = float("nan")
    validation_grid_psl_db: float = float("nan")
    psl_margin_db: float = float("nan")
    initial_validation_grid_psl_db: float = float("nan")
    initial_optimization_grid_psl_db: float = float("nan")
    num_cutting_plane_iterations: int = 0
    num_added_soc_constraints: int = 0
    num_final_soc_constraints: int = 0
    worst_violating_delays_s: tuple[float, ...] = ()
    total_runtime_s: float = 0.0
    dense_psl_satisfied: bool = False
    infeasible: bool = False

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(step) for step in self.history])

    def diagnostic_dict(self) -> dict[str, Any]:
        """Flat diagnostics for paper tables (no power vector)."""
        return {
            "requested_psl_max_db": self.requested_psl_max_db,
            "optimization_grid_psl_db": self.optimization_grid_psl_db,
            "validation_grid_psl_db": self.validation_grid_psl_db,
            "psl_margin_db": self.psl_margin_db,
            "initial_optimization_grid_psl_db": self.initial_optimization_grid_psl_db,
            "initial_validation_grid_psl_db": self.initial_validation_grid_psl_db,
            "cutting_plane_iterations": self.num_cutting_plane_iterations,
            "num_added_soc_constraints": self.num_added_soc_constraints,
            "num_final_soc_constraints": self.num_final_soc_constraints,
            "cutting_plane_status": self.status,
            "cutting_plane_converged": self.converged,
            "dense_psl_satisfied": self.dense_psl_satisfied,
            "cutting_plane_infeasible": self.infeasible,
            "cutting_plane_runtime_s": self.total_runtime_s,
            "worst_violating_delay_s": self.worst_violating_delays_s[-1] if self.worst_violating_delays_s else float("nan"),
            "num_worst_violating_delays": len(self.worst_violating_delays_s),
            "validation_oversampling": self.validation_oversampling,
            "optimization_oversampling": self.optimization_oversampling,
        }


def _unique_delays(delays: NDArray[np.float64], new_tau: float, atol: float = 1e-16) -> NDArray[np.float64]:
    if delays.size == 0:
        return np.asarray([new_tau], dtype=np.float64)
    if np.any(np.abs(delays - new_tau) <= atol):
        return delays
    return np.append(delays, new_tau)


def _attach_report(
    result: OptimizationResult,
    system: ISACSystem,
    opt_grid: NDArray[np.float64],
    val_grid: NDArray[np.float64],
    psl_max_db: float,
    exclusion: float,
    optimization_oversampling: int,
    validation_oversampling: int,
    delays: NDArray[np.float64],
    iteration: int,
) -> dict[str, float | int]:
    report = dual_grid_psl_report(
        result.power_allocation,
        system,
        opt_grid,
        val_grid,
        psl_max_db,
        exclusion,
        optimization_oversampling,
        validation_oversampling,
    )
    result.extra.update(report)
    result.extra["cutting_plane_iteration"] = iteration
    result.extra["cutting_plane_num_constraints"] = int(delays.size)
    return report


def solve_with_psl_cutting_plane(
    solve_fn: SolveFn,
    system: ISACSystem,
    psl_max_db: float,
    *,
    optimization_oversampling: int = 4,
    validation_oversampling: int = 32,
    max_iterations: int = 20,
    psl_tolerance_db: float = 0.25,
    initial_delays_s: NDArray[np.float64] | None = None,
    mainlobe_exclusion_s: float | None = None,
    solve_kwargs: dict[str, Any] | None = None,
) -> CuttingPlaneResult:
    """Wrap a static solver with PSL cutting-plane generation.

    ``solve_fn`` must accept the keyword arguments ``psl_max_db``,
    ``psl_delays_s`` and ``psl_mainlobe_exclusion_s`` (the Stage 2.5 optimiser
    APIs do).  Other arguments are forwarded via ``solve_kwargs``.

    Raises
    ------
    InfeasibleProblemError
        If the *first* sampled-grid program is infeasible.  Later infeasibility
        after adding a dense-grid cut is returned as ``status="infeasible"``
        without relaxing the PSL target.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    exclusion = system.mainlobe_exclusion_s if mainlobe_exclusion_s is None else float(mainlobe_exclusion_s)
    if initial_delays_s is None:
        delays = optimization_sidelobe_delays(system, optimization_oversampling, exclusion)
    else:
        delays = np.asarray(initial_delays_s, dtype=np.float64).copy()
    initial_num = int(delays.size)
    opt_grid = np.concatenate([[0.0], delays]) if delays.size else np.array([0.0])
    val_grid = validation_delay_grid(system, validation_oversampling)
    kwargs = dict(solve_kwargs or {})
    kwargs["psl_max_db"] = psl_max_db
    kwargs["psl_mainlobe_exclusion_s"] = exclusion

    history: list[CuttingPlaneStep] = []
    added_delays: list[float] = []
    last_result: OptimizationResult | None = None
    last_report: dict[str, float | int] = {}
    initial_val_psl = float("nan")
    initial_opt_psl = float("nan")
    status = STATUS_MAX_ITERATIONS
    infeasible = False
    t0 = time.perf_counter()

    for iteration in range(1, max_iterations + 1):
        kwargs["psl_delays_s"] = delays
        try:
            last_result = solve_fn(system, **kwargs)
        except InfeasibleProblemError:
            infeasible = True
            status = STATUS_INFEASIBLE
            if last_result is None:
                raise
            last_result.extra["cutting_plane_status"] = status
            last_result.extra["cutting_plane_infeasible"] = True
            last_result.extra["cutting_plane_converged"] = False
            last_result.extra["cutting_plane_iterations"] = len(history)
            break

        last_report = _attach_report(
            last_result, system, opt_grid, val_grid, psl_max_db, exclusion,
            optimization_oversampling, validation_oversampling, delays, iteration,
        )
        val = evaluate_psl_on_grid(last_result.power_allocation, system, val_grid, exclusion, validation_oversampling)
        margin = psl_max_db - val.psl_db
        if iteration == 1:
            initial_val_psl = val.psl_db
            initial_opt_psl = float(last_report["optimization_grid_psl_db"])
        added: float | None = None
        satisfied = val.psl_db <= psl_max_db + psl_tolerance_db
        if not satisfied:
            added = val.peak_sidelobe_delay_s
            added_delays.append(float(added))
        history.append(
            CuttingPlaneStep(
                iteration=iteration,
                num_psl_constraints=int(delays.size),
                worst_psl_db=val.psl_db,
                worst_delay_s=val.peak_sidelobe_delay_s,
                violation_margin_db=margin,
                solver_time_s=last_result.solve_time_s,
                solver_status=last_result.solver_status,
                added_delay_s=added,
            )
        )
        if satisfied:
            status = STATUS_CONVERGED
            break
        delays = _unique_delays(delays, val.peak_sidelobe_delay_s)
        delays = _unique_delays(delays, -val.peak_sidelobe_delay_s)

    assert last_result is not None
    runtime = time.perf_counter() - t0
    converged = status == STATUS_CONVERGED
    dense_ok = bool(converged)
    last_result.extra["cutting_plane_converged"] = converged
    last_result.extra["cutting_plane_iterations"] = len(history)
    last_result.extra["cutting_plane_status"] = status
    last_result.extra["cutting_plane_infeasible"] = infeasible
    last_result.extra["dense_psl_satisfied"] = dense_ok
    last_result.extra["cutting_plane_runtime_s"] = runtime
    last_result.extra["num_added_soc_constraints"] = int(delays.size - initial_num) if not infeasible else int(max(delays.size - initial_num, 0))
    last_result.extra["num_final_soc_constraints"] = int(delays.size)
    last_result.extra["initial_validation_grid_psl_db"] = initial_val_psl
    last_result.extra["initial_optimization_grid_psl_db"] = initial_opt_psl

    val_psl = float(last_report.get("validation_grid_psl_db", float("nan"))) if last_report else float("nan")
    opt_psl = float(last_report.get("optimization_grid_psl_db", float("nan"))) if last_report else float("nan")
    margin = float(last_report.get("violation_margin_db", float("nan"))) if last_report else float("nan")

    return CuttingPlaneResult(
        result=last_result,
        history=tuple(history),
        converged=converged,
        validation_oversampling=validation_oversampling,
        optimization_oversampling=optimization_oversampling,
        status=status,
        requested_psl_max_db=float(psl_max_db),
        optimization_grid_psl_db=opt_psl,
        validation_grid_psl_db=val_psl,
        psl_margin_db=margin,
        initial_validation_grid_psl_db=initial_val_psl,
        initial_optimization_grid_psl_db=initial_opt_psl,
        num_cutting_plane_iterations=len(history),
        num_added_soc_constraints=max(int(delays.size) - initial_num, 0),
        num_final_soc_constraints=int(delays.size),
        worst_violating_delays_s=tuple(added_delays),
        total_runtime_s=runtime,
        dense_psl_satisfied=dense_ok,
        infeasible=infeasible,
    )

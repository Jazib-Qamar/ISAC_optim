"""Optional cutting-plane generation for sampled-grid PSL constraints.

Loop
----
1. Solve the convex program on the current (moderate) delay-SOC set.
2. Evaluate actual PSL on a denser independent validation grid.
3. If ``PSL_validation <= PSL_target + tolerance``, stop.
4. Otherwise add the worst-violating delay ``tau*`` as a new SOC and re-solve.

This tightens a *sampled-grid* constraint; it is not a continuous-delay certificate.
Impossible PSL requests remain infeasible and are not silently relaxed.
"""

from __future__ import annotations

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
from isac.system import ISACSystem

SolveFn = Callable[..., OptimizationResult]


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
    """Final solve plus the cutting-plane history."""

    result: OptimizationResult
    history: tuple[CuttingPlaneStep, ...]
    converged: bool
    validation_oversampling: int
    optimization_oversampling: int

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(step) for step in self.history])


def _unique_delays(delays: NDArray[np.float64], new_tau: float, atol: float = 1e-16) -> NDArray[np.float64]:
    if delays.size == 0:
        return np.asarray([new_tau], dtype=np.float64)
    if np.any(np.abs(delays - new_tau) <= atol):
        return delays
    return np.append(delays, new_tau)


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
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    exclusion = system.mainlobe_exclusion_s if mainlobe_exclusion_s is None else float(mainlobe_exclusion_s)
    if initial_delays_s is None:
        delays = optimization_sidelobe_delays(system, optimization_oversampling, exclusion)
    else:
        delays = np.asarray(initial_delays_s, dtype=np.float64).copy()
    opt_grid = np.concatenate([[0.0], delays]) if delays.size else np.array([0.0])
    val_grid = validation_delay_grid(system, validation_oversampling)
    kwargs = dict(solve_kwargs or {})
    kwargs["psl_max_db"] = psl_max_db
    kwargs["psl_mainlobe_exclusion_s"] = exclusion

    history: list[CuttingPlaneStep] = []
    last_result: OptimizationResult | None = None
    converged = False
    for iteration in range(1, max_iterations + 1):
        kwargs["psl_delays_s"] = delays
        last_result = solve_fn(system, **kwargs)
        val = evaluate_psl_on_grid(last_result.power_allocation, system, val_grid, exclusion, validation_oversampling)
        margin = psl_max_db - val.psl_db
        added: float | None = None
        satisfied = val.psl_db <= psl_max_db + psl_tolerance_db
        if not satisfied:
            added = val.peak_sidelobe_delay_s
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
        # Attach dual-grid diagnostics to the result extras (mutable copy).
        report = dual_grid_psl_report(
            last_result.power_allocation,
            system,
            opt_grid,
            val_grid,
            psl_max_db,
            exclusion,
            optimization_oversampling,
            validation_oversampling,
        )
        last_result.extra.update(report)
        last_result.extra["cutting_plane_iteration"] = iteration
        last_result.extra["cutting_plane_num_constraints"] = int(delays.size)
        if satisfied:
            converged = True
            break
        delays = _unique_delays(delays, val.peak_sidelobe_delay_s)
        # Also constrain the even counterpart -tau ( |A| is even in tau ).
        delays = _unique_delays(delays, -val.peak_sidelobe_delay_s)

    assert last_result is not None
    last_result.extra["cutting_plane_converged"] = converged
    last_result.extra["cutting_plane_iterations"] = len(history)
    return CuttingPlaneResult(
        result=last_result,
        history=tuple(history),
        converged=converged,
        validation_oversampling=validation_oversampling,
        optimization_oversampling=optimization_oversampling,
    )

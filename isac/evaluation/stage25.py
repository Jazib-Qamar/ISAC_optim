"""Shared helpers for Stage 2.5 static-oracle experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from isac.communication.water_filling import water_filling
from isac.evaluation.metrics import evaluate_allocation
from isac.optimization.ambiguity_constraints import (
    dual_grid_psl_report,
    optimization_sidelobe_delays,
    validation_delay_grid,
)
from isac.optimization.exceptions import InfeasibleProblemError, SolverFailureError
from isac.optimization.feasibility import max_sensing_surrogate, max_unknown_amplitude_fim
from isac.optimization.max_rate import solve_max_rate, water_filling_capacity
from isac.optimization.sensing_spec import KNOWN_AMPLITUDE_LINEAR, UNKNOWN_AMPLITUDE_EXACT
from isac.system import ISACSystem


def psl_grids(system: ISACSystem, opt_oversampling: int, val_oversampling: int):
    delays = optimization_sidelobe_delays(system, opt_oversampling)
    opt_grid = np.concatenate([[0.0], delays]) if delays.size else np.array([0.0])
    val_grid = validation_delay_grid(system, val_oversampling)
    return delays, opt_grid, val_grid


def allocation_row(
    method: str,
    power: np.ndarray,
    system: ISACSystem,
    *,
    status: str = "ok",
    solve_time_s: float = 0.0,
    solver_status: str = "ok",
    solver_inaccurate: bool = False,
    extra: dict[str, Any] | None = None,
    psl_max_db: float | None = None,
    opt_grid: np.ndarray | None = None,
    val_grid: np.ndarray | None = None,
    opt_oversampling: int | None = None,
    val_oversampling: int | None = None,
) -> dict[str, Any]:
    metrics = evaluate_allocation(power, system)
    row: dict[str, Any] = {
        "method": method,
        "status": status,
        "solver_status": solver_status,
        "solve_time_s": solve_time_s,
        "solver_inaccurate": solver_inaccurate,
        "error": "",
        **metrics.as_dict(),
        "mainlobe_exclusion_s": system.mainlobe_exclusion_s,
    }
    if psl_max_db is not None and opt_grid is not None and val_grid is not None:
        row.update(
            dual_grid_psl_report(
                power, system, opt_grid, val_grid, psl_max_db,
                system.mainlobe_exclusion_s, opt_oversampling, val_oversampling,
            )
        )
    if extra:
        row.update(extra)
    return row


def failed_row(method: str, status: str, error: str, solve_time_s: float) -> dict[str, Any]:
    return {
        "method": method,
        "status": status,
        "solver_status": status,
        "solve_time_s": solve_time_s,
        "error": error,
        "solver_inaccurate": False,
    }


def try_max_rate(system: ISACSystem, method: str, **kwargs) -> dict[str, Any]:
    import time

    start = time.perf_counter()
    try:
        res = solve_max_rate(system, **kwargs)
        extras = {
            "solve_time_s": res.solve_time_s,
            "solver_status": res.solver_status,
            "solver_inaccurate": res.solver_inaccurate,
            "extra": dict(res.extra),
        }
        psl_kwargs = {}
        if kwargs.get("psl_max_db") is not None and kwargs.get("psl_delays_s") is not None:
            delays = kwargs["psl_delays_s"]
            psl_kwargs = {
                "psl_max_db": kwargs["psl_max_db"],
                "opt_grid": np.concatenate([[0.0], delays]),
                "val_grid": validation_delay_grid(system, 32),
                "opt_oversampling": None,
                "val_oversampling": 32,
            }
        return allocation_row(method, res.power_allocation, system, **extras, **psl_kwargs)
    except InfeasibleProblemError as exc:
        return failed_row(method, "infeasible", str(exc), time.perf_counter() - start)
    except SolverFailureError as exc:
        return failed_row(method, "solver_failure", str(exc), time.perf_counter() - start)


def reference_levels(system: ISACSystem) -> dict[str, float]:
    c_wf = water_filling_capacity(system)
    s_max, _ = max_sensing_surrogate(system.sensing_weights, system.total_power_w, system.peak_power_w)
    j_max, _ = max_unknown_amplitude_fim(system)
    return {"c_wf": c_wf, "s_max": s_max, "j_max": j_max}


def four_formulation_rows(
    system: ISACSystem,
    *,
    sensing_fraction: float = 0.6,
    fim_fraction: float = 0.6,
    psl_max_db: float | None = None,
    opt_oversampling: int = 4,
) -> list[dict[str, Any]]:
    """Water-filling, linear-S max-rate, exact-FIM max-rate, exact-FIM+PSL max-rate."""
    refs = reference_levels(system)
    delays, _, _ = psl_grids(system, opt_oversampling, 32)
    rows: list[dict[str, Any]] = []

    wf = water_filling(system.channel_gain, system.noise_power_w, system.total_power_w, system.peak_power_w)
    rows.append(allocation_row("water_filling", wf.power_w, system, extra={"label": "Water-filling (comm. only)"}))

    rows.append(
        try_max_rate(
            system, "linear_sensing",
            min_sensing_surrogate=sensing_fraction * refs["s_max"],
            sensing_model=KNOWN_AMPLITUDE_LINEAR,
        )
    )
    rows[-1]["label"] = "Linear S(P) (known-amp. surrogate)"
    rows[-1]["sensing_fraction"] = sensing_fraction
    rows[-1]["s_max"] = refs["s_max"]

    rows.append(
        try_max_rate(
            system, "exact_fim",
            sensing_model=UNKNOWN_AMPLITUDE_EXACT,
            min_unknown_fim=fim_fraction * refs["j_max"],
        )
    )
    rows[-1]["label"] = "Exact unknown-amplitude FIM"
    rows[-1]["fim_fraction"] = fim_fraction
    rows[-1]["j_max"] = refs["j_max"]

    if psl_max_db is not None:
        row = try_max_rate(
            system, "exact_fim_psl",
            sensing_model=UNKNOWN_AMPLITUDE_EXACT,
            min_unknown_fim=fim_fraction * refs["j_max"],
            psl_max_db=psl_max_db,
            psl_delays_s=delays,
        )
        row["label"] = "Exact FIM + sampled PSL"
        row["fim_fraction"] = fim_fraction
        row["j_max"] = refs["j_max"]
        row["psl_max_db_request"] = psl_max_db
        rows.append(row)
    return rows


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)

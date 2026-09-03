"""Independent physical evaluator for ICC comparisons.

Every allocation — regardless of which *optimizer* produced it — is scored with
the Stage 1 NumPy model.  The conventional optimizer's claimed sensing metric
``S(P) = sum_k f_k^2 P_k`` (equivalently known-amplitude FIM) is stored
separately from the exact unknown-amplitude EFIM ``J_tau^eff = C_beta G(P)``.

A method that only constrained ``S(P)`` may *claim* sensing feasibility while
the independent evaluator reports a physical violation.  Dense-grid PSL is
never inferred from the optimisation grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from isac.evaluation.metrics import evaluate_allocation
from isac.optimization.ambiguity_constraints import (
    dual_grid_psl_report,
    optimization_sidelobe_delays,
    validation_delay_grid,
)
from isac.optimization.feasibility import DEFAULT_ABS_TOL, DEFAULT_REL_TOL
from isac.sensing.crb import min_fim_from_range_rmse
from isac.sensing.fim import unknown_amplitude_fim_scale
from isac.system import ISACSystem


@dataclass(frozen=True)
class PhysicalSensingTarget:
    """One common physical sensing request used by every compared method.

    ``gamma_unknown_fim`` is the requirement imposed on the *independent*
    unknown-amplitude evaluator.  ``gamma_conventional_s`` is the S2 threshold
    the *conventional* optimizer is allowed to use (``J_known = C_beta S``),
    never the exact EFIM.
    """

    gamma_unknown_fim: float
    gamma_conventional_s: float
    c_beta: float
    j_max: float
    s_max: float
    max_range_rmse_m: float
    fim_fraction: float
    psl_max_db: float | None
    psl_tolerance_db: float
    optimization_oversampling: int
    validation_oversampling: int
    sensing_abs_tol: float = DEFAULT_ABS_TOL
    sensing_rel_tol: float = DEFAULT_REL_TOL

    def unknown_fim_tolerance(self) -> float:
        return self.sensing_abs_tol + self.sensing_rel_tol * abs(self.gamma_unknown_fim)

    def conventional_s_tolerance(self) -> float:
        return self.sensing_abs_tol + self.sensing_rel_tol * abs(self.gamma_conventional_s)


def conventional_s_from_unknown_fim(gamma_unknown_fim: float, c_beta: float) -> float:
    """S2 threshold such that ``J_known = C_beta S`` equals ``Gamma_J``.

    The conventional optimizer still maximises rate (or EE) subject to
    ``S(P) >= this value``.  It does **not** receive the exact EFIM constraint.
    """
    if c_beta <= 0.0:
        raise ValueError("c_beta must be positive")
    if gamma_unknown_fim < 0.0:
        raise ValueError("gamma_unknown_fim must be non-negative")
    return float(gamma_unknown_fim) / float(c_beta)


def c_beta_of(system: ISACSystem) -> float:
    return unknown_amplitude_fim_scale(
        system.reflection_coefficient,
        system.noise_power_w,
        num_symbols=system.num_symbols,
    )


def range_rmse_of_fim(gamma_unknown_fim: float) -> float:
    """Range RMSE bound implied by ``J_tau^eff = Gamma_J`` (monostatic)."""
    if gamma_unknown_fim <= 0.0:
        return float("inf")
    from configs.default import SPEED_OF_LIGHT_M_PER_S

    delay_crb = 1.0 / gamma_unknown_fim
    return float(np.sqrt((SPEED_OF_LIGHT_M_PER_S / 2.0) ** 2 * delay_crb))


def target_from_unknown_fim(
    system: ISACSystem,
    gamma_unknown_fim: float,
    *,
    j_max: float,
    s_max: float,
    psl_max_db: float | None,
    fim_fraction: float,
    psl_tolerance_db: float,
    optimization_oversampling: int,
    validation_oversampling: int,
) -> PhysicalSensingTarget:
    c_beta = c_beta_of(system)
    return PhysicalSensingTarget(
        gamma_unknown_fim=float(gamma_unknown_fim),
        gamma_conventional_s=conventional_s_from_unknown_fim(gamma_unknown_fim, c_beta),
        c_beta=c_beta,
        j_max=float(j_max),
        s_max=float(s_max),
        max_range_rmse_m=range_rmse_of_fim(gamma_unknown_fim),
        fim_fraction=float(fim_fraction),
        psl_max_db=None if psl_max_db is None else float(psl_max_db),
        psl_tolerance_db=float(psl_tolerance_db),
        optimization_oversampling=int(optimization_oversampling),
        validation_oversampling=int(validation_oversampling),
    )


def target_from_range_rmse(
    system: ISACSystem,
    max_range_rmse_m: float,
    *,
    j_max: float,
    s_max: float,
    psl_max_db: float | None,
    psl_tolerance_db: float,
    optimization_oversampling: int,
    validation_oversampling: int,
) -> PhysicalSensingTarget:
    gamma_j = min_fim_from_range_rmse(max_range_rmse_m)
    return target_from_unknown_fim(
        system,
        gamma_j,
        j_max=j_max,
        s_max=s_max,
        psl_max_db=psl_max_db,
        fim_fraction=gamma_j / j_max if j_max > 0.0 else float("nan"),
        psl_tolerance_db=psl_tolerance_db,
        optimization_oversampling=optimization_oversampling,
        validation_oversampling=validation_oversampling,
    )


def fim_mismatch_percent(j_known: float, j_unknown: float) -> float:
    """``ε_J = 100 (J_known - J_unknown) / J_known`` [%].  NaN if ``J_known=0``."""
    if not np.isfinite(j_known) or j_known <= 0.0:
        return float("nan")
    return 100.0 * (j_known - j_unknown) / j_known


def evaluate_independent(
    power_w: ArrayLike,
    system: ISACSystem,
    target: PhysicalSensingTarget,
    *,
    method: str,
    label: str,
    status: str = "ok",
    solver_status: str = "ok",
    solve_time_s: float = 0.0,
    solver_inaccurate: bool = False,
    error: str = "",
    optimizer_model: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one allocation with the shared physical model and the common target.

    ``optimizer_model``:
        * ``none`` — heuristic / communication-only (does not claim a sensing constraint)
        * ``conventional_s2`` — constrained ``S(P) >= Gamma_S``
        * ``exact_efim`` — constrained ``J_unknown >= Gamma_J``
    """
    power: NDArray[np.float64] = np.asarray(power_w, dtype=np.float64)
    metrics = evaluate_allocation(power, system)
    j_known = metrics.delay_fim_known_amplitude_per_s2
    j_unknown = metrics.delay_fim_unknown_amplitude_per_s2
    claimed_s = metrics.sensing_surrogate
    claimed_j_known = target.c_beta * claimed_s

    physical_ok = bool(j_unknown + target.unknown_fim_tolerance() >= target.gamma_unknown_fim)
    physical_ok_1pct = bool(j_unknown >= 0.99 * target.gamma_unknown_fim) if target.gamma_unknown_fim > 0 else physical_ok
    physical_ok_2pct = bool(j_unknown >= 0.98 * target.gamma_unknown_fim) if target.gamma_unknown_fim > 0 else physical_ok
    if optimizer_model == "conventional_s2":
        claimed_ok = bool(claimed_s + target.conventional_s_tolerance() >= target.gamma_conventional_s)
    elif optimizer_model == "exact_efim":
        claimed_ok = physical_ok
    else:
        claimed_ok = False

    false_feasibility = bool(
        optimizer_model == "conventional_s2" and status == "ok" and claimed_ok and not physical_ok
    )
    material_false_1pct = bool(
        optimizer_model == "conventional_s2" and status == "ok" and claimed_ok and not physical_ok_1pct
    )
    relative_unknown_gap = (
        (j_unknown - target.gamma_unknown_fim) / target.gamma_unknown_fim
        if target.gamma_unknown_fim > 0.0 else float("nan")
    )

    delays = optimization_sidelobe_delays(system, target.optimization_oversampling)
    opt_grid = np.concatenate([[0.0], delays]) if delays.size else np.array([0.0])
    val_grid = validation_delay_grid(system, target.validation_oversampling)
    psl_request = target.psl_max_db if target.psl_max_db is not None else float("nan")
    psl_report = dual_grid_psl_report(
        power,
        system,
        opt_grid,
        val_grid,
        psl_request if np.isfinite(psl_request) else 0.0,
        system.mainlobe_exclusion_s,
        target.optimization_oversampling,
        target.validation_oversampling,
    )
    dense_psl = float(psl_report["validation_grid_psl_db"])
    opt_psl = float(psl_report["optimization_grid_psl_db"])
    if target.psl_max_db is None:
        dense_psl_ok = False
        psl_applicable = False
        psl_margin = float("nan")
    else:
        psl_applicable = True
        psl_margin = float(target.psl_max_db) - dense_psl
        extra_flag = (extra or {}).get("dense_psl_satisfied")
        if extra_flag is True:
            dense_psl_ok = True
        elif extra_flag is False:
            dense_psl_ok = False
        else:
            dense_psl_ok = bool(dense_psl <= target.psl_max_db + target.psl_tolerance_db)

    row: dict[str, Any] = {
        "method": method,
        "label": label,
        "status": status,
        "solver_status": solver_status,
        "solve_time_s": float(solve_time_s),
        "solver_inaccurate": bool(solver_inaccurate),
        "error": error,
        "optimizer_model": optimizer_model,
        **metrics.as_dict(),
        "rate_mbps": metrics.rate_bps / 1e6,
        "ee_mbit_per_j": metrics.energy_efficiency_bit_per_j / 1e6,
        "optimizer_claimed_sensing_metric": claimed_s,
        "optimizer_claimed_known_fim": claimed_j_known,
        "independently_evaluated_unknown_fim": j_unknown,
        "independently_evaluated_known_fim": j_known,
        "fim_mismatch_percent": fim_mismatch_percent(j_known, j_unknown),
        "centroid_magnitude_hz": abs(metrics.spectral_centroid_hz),
        "gamma_unknown_fim": target.gamma_unknown_fim,
        "gamma_conventional_s": target.gamma_conventional_s,
        "c_beta": target.c_beta,
        "j_max": target.j_max,
        "s_max": target.s_max,
        "fim_fraction": target.fim_fraction,
        "max_range_rmse_m_target": target.max_range_rmse_m,
        "optimizer_claimed_sensing_satisfied": claimed_ok if optimizer_model != "none" else False,
        "physical_sensing_satisfied": physical_ok if status == "ok" else False,
        "physical_sensing_satisfied_1pct": physical_ok_1pct if status == "ok" else False,
        "physical_sensing_satisfied_2pct": physical_ok_2pct if status == "ok" else False,
        "relative_unknown_fim_gap": relative_unknown_gap,
        "false_sensing_feasibility": false_feasibility,
        "material_false_sensing_feasibility_1pct": material_false_1pct,
        "psl_requirement_applicable": psl_applicable,
        "requested_psl_max_db": target.psl_max_db if target.psl_max_db is not None else float("nan"),
        "optimization_grid_psl_db": opt_psl,
        "validation_grid_psl_db": dense_psl,
        "dense_psl_db": dense_psl,
        "psl_margin_db": psl_margin,
        "dense_psl_satisfied": bool(psl_applicable and status == "ok" and dense_psl_ok),
        "psl_tolerance_db": target.psl_tolerance_db,
        "optimization_oversampling": target.optimization_oversampling,
        "validation_oversampling": target.validation_oversampling,
        "worst_delay_s": psl_report["worst_delay_s"],
        "mainlobe_exclusion_s": system.mainlobe_exclusion_s,
    }
    if extra:
        for key, value in extra.items():
            if key not in row:
                row[key] = value
    return row


def failed_independent_row(
    method: str,
    label: str,
    *,
    status: str,
    error: str,
    solve_time_s: float,
    optimizer_model: str,
    target: PhysicalSensingTarget,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "label": label,
        "status": status,
        "solver_status": status,
        "solve_time_s": float(solve_time_s),
        "solver_inaccurate": False,
        "error": error,
        "optimizer_model": optimizer_model,
        "optimizer_claimed_sensing_satisfied": False,
        "physical_sensing_satisfied": False,
        "false_sensing_feasibility": False,
        "dense_psl_satisfied": False,
        "psl_requirement_applicable": target.psl_max_db is not None,
        "requested_psl_max_db": target.psl_max_db if target.psl_max_db is not None else float("nan"),
        "gamma_unknown_fim": target.gamma_unknown_fim,
        "gamma_conventional_s": target.gamma_conventional_s,
        "c_beta": target.c_beta,
        "j_max": target.j_max,
        "s_max": target.s_max,
        "fim_fraction": target.fim_fraction,
        "max_range_rmse_m_target": target.max_range_rmse_m,
    }
    if extra:
        row.update(extra)
    return row

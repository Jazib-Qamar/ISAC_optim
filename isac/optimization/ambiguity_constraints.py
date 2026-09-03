"""Sampled-grid delay-ambiguity (PSL) constraints as second-order cones.

For the power-spectrum delay response

    A(tau) = sum_k P_k exp(j 2 pi f_k tau),     A(0) = sum_k P_k,

a threshold ``PSL <= PSL_max_db`` with ``rho = 10^(PSL_max_db / 20)`` is the
family of inequalities

    |A(tau_m)|  <=  rho * sum_k P_k

on a *finite* delay grid ``{tau_m}`` outside the excluded mainlobe.  Writing

    C_m(P) = sum_k P_k cos(2 pi f_k tau_m),
    S_m(P) = sum_k P_k sin(2 pi f_k tau_m),

each inequality is the second-order cone

    || [C_m(P), S_m(P)] ||_2  <=  rho * sum(P).

This is a *sampled-grid* PSL constraint, not a guarantee over continuous delay.
A denser independent validation grid must be used after every solve.
``P_k <= P_peak`` is a peak spectral power cap / PSL *proxy*, not this constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from isac.optimization.common import ScaledModel
from isac.sensing.ambiguity import (
    ambiguity_summary,
    default_delay_grid,
    psl_db_to_linear_amplitude,
    sidelobe_delay_samples,
)
from isac.system import ISACSystem


@dataclass(frozen=True)
class PslGridReport:
    """PSL / ISL on one delay grid, plus the worst-sidelobe location."""

    psl_db: float
    isl_db: float
    peak_sidelobe_delay_s: float
    num_sidelobe_samples: int
    mainlobe_exclusion_s: float
    oversampling_factor: int | None = None


def build_delay_grid(
    subcarrier_spacing_hz: float,
    num_subcarriers: int,
    oversampling_factor: int,
) -> NDArray[np.float64]:
    """Uniform unambiguous delay grid; see :func:`isac.sensing.ambiguity.default_delay_grid`."""
    return default_delay_grid(subcarrier_spacing_hz, num_subcarriers, oversampling_factor)


def optimization_sidelobe_delays(
    system: ISACSystem,
    oversampling_factor: int,
    mainlobe_exclusion_s: float | None = None,
) -> NDArray[np.float64]:
    """Sidelobe delays of a moderately dense optimisation grid."""
    grid = build_delay_grid(system.subcarrier_spacing_hz, system.num_subcarriers, oversampling_factor)
    exclusion = system.mainlobe_exclusion_s if mainlobe_exclusion_s is None else float(mainlobe_exclusion_s)
    return sidelobe_delay_samples(grid, exclusion)


def validation_delay_grid(
    system: ISACSystem,
    oversampling_factor: int,
) -> NDArray[np.float64]:
    """Independent (typically 4x–8x denser) delay grid for post-solve PSL checks."""
    return build_delay_grid(system.subcarrier_spacing_hz, system.num_subcarriers, oversampling_factor)


def cosine_sine_matrices(
    frequencies_hz: ArrayLike,
    delays_s: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Matrices ``C[m,k] = cos(2 pi f_k tau_m)`` and ``S[m,k] = sin(2 pi f_k tau_m)``."""
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    delays = np.asarray(delays_s, dtype=np.float64)
    if freqs.ndim != 1 or delays.ndim != 1:
        raise ValueError("frequencies_hz and delays_s must be 1-D")
    phase = 2.0 * np.pi * np.outer(delays, freqs)
    return np.cos(phase), np.sin(phase)


def build_sampled_psl_constraints(
    power_w: cp.Expression,
    frequencies_hz: ArrayLike,
    delays_s: ArrayLike,
    psl_max_db: float,
) -> tuple[list[cp.Constraint], dict[str, float | int]]:
    """SOC family ``||[C_m, S_m]||_2 <= rho * sum(P)`` on the given delay samples.

    Returns the constraint list and a small diagnostic dict (``rho``, number of
    cones).  An empty delay list yields no PSL cones.
    """
    delays = np.asarray(delays_s, dtype=np.float64)
    rho = psl_db_to_linear_amplitude(psl_max_db)
    extras: dict[str, float | int] = {
        "psl_max_db": float(psl_max_db),
        "psl_rho": rho,
        "num_psl_soc_constraints": int(delays.size),
    }
    if delays.size == 0:
        return [], extras
    cos_mat, sin_mat = cosine_sine_matrices(frequencies_hz, delays)
    s0 = cp.sum(power_w)
    cosine = cos_mat @ power_w
    sine = sin_mat @ power_w
    constraints: list[cp.Constraint] = []
    for idx in range(delays.size):
        constraints.append(cp.norm(cp.hstack([cosine[idx], sine[idx]]), 2) <= rho * s0)
    return constraints, extras


def build_psl_constraints_for_model(
    model: ScaledModel,
    system: ISACSystem,
    psl_max_db: float,
    delays_s: ArrayLike,
) -> tuple[list[cp.Constraint], dict[str, float | int]]:
    """Convenience wrapper around :func:`build_sampled_psl_constraints`."""
    return build_sampled_psl_constraints(model.power_w, system.frequencies_hz, delays_s, psl_max_db)


def evaluate_psl_on_grid(
    power_w: ArrayLike,
    system: ISACSystem,
    delay_grid_s: ArrayLike,
    mainlobe_exclusion_s: float | None = None,
    oversampling_factor: int | None = None,
) -> PslGridReport:
    """Actual delay-domain PSL/ISL on an independent grid (NumPy evaluator)."""
    exclusion = system.mainlobe_exclusion_s if mainlobe_exclusion_s is None else float(mainlobe_exclusion_s)
    result = ambiguity_summary(power_w, system.frequencies_hz, delay_grid_s, exclusion)
    return PslGridReport(
        psl_db=result.psl_db,
        isl_db=result.isl_db,
        peak_sidelobe_delay_s=result.peak_sidelobe_delay_s,
        num_sidelobe_samples=result.num_sidelobe_samples,
        mainlobe_exclusion_s=result.mainlobe_exclusion_s,
        oversampling_factor=oversampling_factor,
    )


def dual_grid_psl_report(
    power_w: ArrayLike,
    system: ISACSystem,
    optimization_grid_s: ArrayLike,
    validation_grid_s: ArrayLike,
    psl_max_db: float,
    mainlobe_exclusion_s: float | None = None,
    optimization_oversampling: int | None = None,
    validation_oversampling: int | None = None,
) -> dict[str, float | int]:
    """Compare requested / optimisation-grid / validation-grid PSL.

    ``violation_margin_db`` is ``psl_max_db - validation_psl_db``: positive means
    the validation-grid PSL is *better* (more negative) than the request.
    """
    exclusion = system.mainlobe_exclusion_s if mainlobe_exclusion_s is None else float(mainlobe_exclusion_s)
    opt = evaluate_psl_on_grid(
        power_w, system, optimization_grid_s, exclusion, optimization_oversampling
    )
    val = evaluate_psl_on_grid(
        power_w, system, validation_grid_s, exclusion, validation_oversampling
    )
    return {
        "requested_psl_max_db": float(psl_max_db),
        "optimization_grid_psl_db": opt.psl_db,
        "validation_grid_psl_db": val.psl_db,
        "validation_grid_isl_db": val.isl_db,
        "worst_delay_s": val.peak_sidelobe_delay_s,
        "violation_margin_db": float(psl_max_db) - val.psl_db,
        "optimization_grid_sidelobe_samples": opt.num_sidelobe_samples,
        "validation_grid_sidelobe_samples": val.num_sidelobe_samples,
        "mainlobe_exclusion_s": exclusion,
    }

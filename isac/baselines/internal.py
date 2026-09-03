"""Internal baselines that already exist in this repository.

These wrap Stage 1/2/2.5 allocators into :class:`BaselineResult`.  They are
**not** published-paper reproductions.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from isac.baselines.base import BaselineResult
from isac.communication.water_filling import uniform_power, water_filling
from isac.optimization.exceptions import InfeasibleProblemError, SolverFailureError
from isac.optimization.max_rate import solve_max_rate
from isac.optimization.sensing_spec import KNOWN_AMPLITUDE_LINEAR, UNKNOWN_AMPLITUDE_EXACT
from isac.sensing.fim import sensing_information_surrogate
from isac.system import ISACSystem


def _ok(name: str, power: np.ndarray, runtime: float, claimed: float | None, **meta: Any) -> BaselineResult:
    return BaselineResult(
        method_name=name,
        power_allocation=np.asarray(power, dtype=np.float64),
        runtime_s=float(runtime),
        convergence_status="ok",
        optimizer_claimed_sensing_metric=claimed,
        metadata=dict(meta),
    )


def run_uniform(system: ISACSystem, **_kwargs: Any) -> BaselineResult:
    t0 = time.perf_counter()
    power = uniform_power(system.num_subcarriers, system.total_power_w, system.peak_power_w)
    claimed = sensing_information_surrogate(power, system.sensing_weights)
    return _ok("uniform", power, time.perf_counter() - t0, claimed)


def run_water_filling(system: ISACSystem, **_kwargs: Any) -> BaselineResult:
    t0 = time.perf_counter()
    res = water_filling(system.channel_gain, system.noise_power_w, system.total_power_w, system.peak_power_w)
    claimed = sensing_information_surrogate(res.power_w, system.sensing_weights)
    return _ok("water_filling", res.power_w, time.perf_counter() - t0, claimed, water_level_w=res.water_level_w)


def run_conventional_s2(
    system: ISACSystem,
    *,
    min_sensing_surrogate: float,
    psl_max_db: float | None = None,
    psl_delays_s: np.ndarray | None = None,
    **kwargs: Any,
) -> BaselineResult:
    """Max-rate subject to conventional ``S(P) >= Gamma_S`` (not exact EFIM)."""
    t0 = time.perf_counter()
    try:
        res = solve_max_rate(
            system,
            min_sensing_surrogate=min_sensing_surrogate,
            sensing_model=KNOWN_AMPLITUDE_LINEAR,
            psl_max_db=psl_max_db,
            psl_delays_s=psl_delays_s,
            **kwargs,
        )
    except (InfeasibleProblemError, SolverFailureError) as exc:
        raise
    claimed = sensing_information_surrogate(res.power_allocation, system.sensing_weights)
    name = "conventional_s2_psl" if psl_max_db is not None else "conventional_s2"
    return _ok(
        name,
        res.power_allocation,
        time.perf_counter() - t0,
        claimed,
        solver_status=res.solver_status,
        solver_inaccurate=res.solver_inaccurate,
        extra=dict(res.extra),
    )


def run_exact_efim(
    system: ISACSystem,
    *,
    min_unknown_fim: float,
    psl_max_db: float | None = None,
    psl_delays_s: np.ndarray | None = None,
    **kwargs: Any,
) -> BaselineResult:
    """Max-rate subject to exact unknown-amplitude EFIM (sampled PSL optional)."""
    t0 = time.perf_counter()
    res = solve_max_rate(
        system,
        sensing_model=UNKNOWN_AMPLITUDE_EXACT,
        min_unknown_fim=min_unknown_fim,
        psl_max_db=psl_max_db,
        psl_delays_s=psl_delays_s,
        **kwargs,
    )
    claimed = float(res.metrics.delay_fim_unknown_amplitude_per_s2)
    name = "exact_efim_sampled_psl" if psl_max_db is not None else "exact_efim"
    return _ok(
        name,
        res.power_allocation,
        time.perf_counter() - t0,
        claimed,
        solver_status=res.solver_status,
        solver_inaccurate=res.solver_inaccurate,
        extra=dict(res.extra),
    )

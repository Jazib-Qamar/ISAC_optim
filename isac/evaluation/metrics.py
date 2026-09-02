"""Consistent physical-metric evaluation for any power allocation.

Every baseline and every optimiser result is scored through
:func:`evaluate_allocation`, which calls only the validated Stage 1 NumPy
functions (rate, FIM, CRB, power model) plus the delay-domain ambiguity
evaluator.  CVXPY expressions are never used for reporting.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from isac.communication.rate import spectral_efficiency
from isac.energy.power_model import energy_efficiency, system_power, tx_power
from isac.sensing.ambiguity import ambiguity_summary
from isac.sensing.crb import delay_crb_summary
from isac.sensing.fim import delay_fisher_information, sensing_information_surrogate
from isac.system import ISACSystem


@dataclass(frozen=True)
class AllocationMetrics:
    """All physical metrics of one power allocation, computed from one :class:`ISACSystem`.

    Units are given in the attribute names.  ``psl_db``/``isl_db`` are the
    *actual* delay-domain sidelobe levels, not the peak-power proxy.
    """

    rate_spectral_efficiency: float  # bit/s/Hz summed over subcarriers
    rate_bps: float  # bit/s
    tx_power_w: float
    system_power_w: float
    energy_efficiency_bit_per_j: float
    sensing_surrogate: float  # sum_k w_k P_k, units follow w_k
    delay_fim_per_s2: float  # configured amplitude model
    delay_fim_known_amplitude_per_s2: float
    delay_fim_unknown_amplitude_per_s2: float
    delay_crb_s2: float
    range_rmse_bound_m: float
    psl_db: float
    isl_db: float
    peak_power_w: float
    power_variance_w2: float
    num_active_subcarriers: int

    def as_dict(self) -> dict[str, float]:
        """Flat dictionary (for DataFrame rows)."""
        return dataclasses.asdict(self)


def evaluate_allocation(power_w: ArrayLike, system: ISACSystem) -> AllocationMetrics:
    """Score a power allocation with the shared physical model.

    Parameters
    ----------
    power_w:
        Per-subcarrier transmit power ``P_k`` [W], shape ``(K,)``.  Must be
        non-negative; a tiny negative solver round-off should be clipped by the
        caller before evaluation.
    system:
        Fixed physical parameters of the scenario.
    """
    power: NDArray[np.float64] = np.asarray(power_w, dtype=np.float64)
    if power.shape != system.channel_gain.shape:
        raise ValueError(
            f"power_w has shape {power.shape} but the system has {system.num_subcarriers} subcarriers"
        )
    if np.any(power < 0.0):
        raise ValueError("power_w must be non-negative")

    se = spectral_efficiency(power, system.channel_gain, system.noise_power_w)
    rate_bps = system.subcarrier_spacing_hz * se
    p_tx = tx_power(power)
    p_sys = system_power(power, system.circuit_power_w, system.pa_efficiency)

    crb = delay_crb_summary(
        power,
        system.frequencies_hz,
        system.reflection_coefficient,
        system.noise_power_w,
        num_symbols=system.num_symbols,
        known_amplitude=system.known_amplitude,
    )
    fim_known = delay_fisher_information(
        power, system.frequencies_hz, system.reflection_coefficient, system.noise_power_w,
        num_symbols=system.num_symbols, known_amplitude=True,
    )
    fim_unknown = delay_fisher_information(
        power, system.frequencies_hz, system.reflection_coefficient, system.noise_power_w,
        num_symbols=system.num_symbols, known_amplitude=False,
    )

    if p_tx > 0.0:
        amb = ambiguity_summary(power, system.frequencies_hz, system.delay_grid_s, system.mainlobe_exclusion_s)
        psl_db, isl_db = amb.psl_db, amb.isl_db
    else:
        psl_db = isl_db = float("nan")

    return AllocationMetrics(
        rate_spectral_efficiency=se,
        rate_bps=rate_bps,
        tx_power_w=p_tx,
        system_power_w=p_sys,
        energy_efficiency_bit_per_j=energy_efficiency(rate_bps, p_sys),
        sensing_surrogate=sensing_information_surrogate(power, system.sensing_weights),
        delay_fim_per_s2=crb.fisher_information,
        delay_fim_known_amplitude_per_s2=fim_known,
        delay_fim_unknown_amplitude_per_s2=fim_unknown,
        delay_crb_s2=crb.delay_crb_s2,
        range_rmse_bound_m=crb.range_rmse_bound_m,
        psl_db=psl_db,
        isl_db=isl_db,
        peak_power_w=float(np.max(power)),
        power_variance_w2=float(np.var(power)),
        num_active_subcarriers=int(np.count_nonzero(power > 0.0)),
    )

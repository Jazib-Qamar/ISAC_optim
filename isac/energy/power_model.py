"""Transmitter power-consumption model and energy efficiency.

Model
-----
    P_tx  = sum_k P_k                                   [W]  radiated power
    P_sys = P_circuit + P_tx / eta_PA                   [W]  consumed power
    E_slot = T_slot * P_sys                             [J]  energy per slot
    EE    = R_bps / P_sys                               [bit/J]

with ``0 < eta_PA <= 1`` the power-amplifier efficiency and ``P_circuit >= 0``
the static circuit power.  Because ``P_circuit > 0``, the energy efficiency is
*not* maximised by transmitting vanishing power - this is what makes the
Dinkelbach EE problem non-trivial.

Units
-----
Energy efficiency is returned in bit/J when the rate is given in bit/s.  If a
spectral efficiency (bit/s/Hz summed over subcarriers) is passed instead, the
result is in (bit/s/Hz)/W; the helper :func:`energy_efficiency` therefore
takes the rate in bit/s explicitly to avoid hidden unit mixing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _validate_power_vector(power_w: ArrayLike) -> np.ndarray:
    power = np.asarray(power_w, dtype=np.float64)
    if power.ndim != 1:
        raise ValueError("power_w must be a 1-D array of per-subcarrier powers")
    if np.any(power < 0.0):
        raise ValueError("power_w must be non-negative")
    return power


def _validate_power_model(circuit_power_w: float, pa_efficiency: float) -> None:
    if circuit_power_w < 0.0:
        raise ValueError("circuit_power_w must be non-negative")
    if not 0.0 < pa_efficiency <= 1.0:
        raise ValueError("pa_efficiency must lie in (0, 1]")


def tx_power(power_w: ArrayLike) -> float:
    """Total radiated transmit power ``sum_k P_k`` [W]."""
    return float(np.sum(_validate_power_vector(power_w)))


def system_power(
    power_w: ArrayLike,
    circuit_power_w: float,
    pa_efficiency: float,
) -> float:
    """Total consumed power ``P_sys = P_circuit + sum_k P_k / eta_PA`` [W].

    Parameters
    ----------
    power_w:
        Per-subcarrier transmit power ``P_k`` [W], shape ``(K,)``.
    circuit_power_w:
        Static circuit power ``P_circuit`` [W].
    pa_efficiency:
        Power-amplifier efficiency ``eta_PA`` in ``(0, 1]``.
    """
    _validate_power_model(circuit_power_w, pa_efficiency)
    return circuit_power_w + tx_power(power_w) / pa_efficiency


def energy_per_slot(
    power_w: ArrayLike,
    circuit_power_w: float,
    pa_efficiency: float,
    slot_duration_s: float,
) -> float:
    """Energy consumed in one slot ``T_slot * P_sys`` [J]."""
    if slot_duration_s <= 0.0:
        raise ValueError("slot_duration_s must be positive")
    return slot_duration_s * system_power(power_w, circuit_power_w, pa_efficiency)


def energy_efficiency(rate_bps: float, system_power_w: float) -> float:
    """Energy efficiency ``EE = R / P_sys`` [bit/J].

    Parameters
    ----------
    rate_bps:
        Achievable rate ``R`` [bit/s].
    system_power_w:
        Consumed system power ``P_sys`` [W]; must be strictly positive.
    """
    if rate_bps < 0.0:
        raise ValueError("rate_bps must be non-negative")
    if not np.isfinite(system_power_w) or system_power_w <= 0.0:
        raise ValueError("system_power_w must be a positive finite number")
    return rate_bps / system_power_w

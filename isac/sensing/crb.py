"""Cramér-Rao bounds for delay and range estimation in OFDM sensing.

    CRB_tau = 1 / J_tau                                  [s^2]
    CRB_R   = (c / 2)^2 * CRB_tau                        [m^2]   (monostatic, tau = 2R/c)

``J_tau`` is computed by :func:`isac.sensing.fim.delay_fisher_information`.
When no power is transmitted, ``J_tau = 0`` and the bounds are ``+inf``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from configs.default import SPEED_OF_LIGHT_M_PER_S
from isac.sensing.fim import delay_fisher_information


@dataclass(frozen=True)
class DelayCRBResult:
    """Delay/range estimation bounds for one power allocation.

    Attributes
    ----------
    fisher_information:
        ``J_tau`` [1/s^2].
    delay_crb_s2:
        ``CRB_tau`` [s^2].
    delay_rmse_bound_s:
        ``sqrt(CRB_tau)`` [s].
    range_crb_m2:
        ``(c/2)^2 CRB_tau`` [m^2].
    range_rmse_bound_m:
        ``sqrt(range_crb_m2)`` [m].
    """

    fisher_information: float
    delay_crb_s2: float
    delay_rmse_bound_s: float
    range_crb_m2: float
    range_rmse_bound_m: float


def crb_from_fisher_information(fisher_information: float) -> float:
    """``CRB = 1 / J``; returns ``+inf`` for ``J = 0``.

    Raises
    ------
    ValueError
        If ``J`` is negative or not finite.
    """
    if not np.isfinite(fisher_information) or fisher_information < 0.0:
        raise ValueError("fisher_information must be finite and non-negative")
    if fisher_information == 0.0:
        return float("inf")
    return 1.0 / fisher_information


def delay_crb(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
    reflection_coefficient: complex,
    noise_variance_w: float,
    num_symbols: int = 1,
    known_amplitude: bool = True,
) -> float:
    """Delay CRB ``1 / J_tau`` [s^2].

    Parameters are identical to
    :func:`isac.sensing.fim.delay_fisher_information`.
    """
    fisher = delay_fisher_information(
        power_w,
        frequencies_hz,
        reflection_coefficient,
        noise_variance_w,
        num_symbols=num_symbols,
        known_amplitude=known_amplitude,
    )
    return crb_from_fisher_information(fisher)


def range_crb_from_delay_crb(delay_crb_s2: float) -> float:
    """Convert a round-trip delay CRB [s^2] into a range CRB [m^2] via ``R = c tau / 2``."""
    if delay_crb_s2 < 0.0:
        raise ValueError("delay_crb_s2 must be non-negative")
    return (SPEED_OF_LIGHT_M_PER_S / 2.0) ** 2 * delay_crb_s2


def min_fim_from_delay_crb(max_delay_crb_s2: float) -> float:
    """Convert ``CRB_tau <= CRB_tau_max`` [s^2] into ``J_tau >= 1 / CRB_tau_max`` [1/s^2]."""
    if not np.isfinite(max_delay_crb_s2) or max_delay_crb_s2 <= 0.0:
        raise ValueError("max_delay_crb_s2 must be a positive finite number")
    return 1.0 / max_delay_crb_s2


def min_fim_from_range_rmse(max_range_rmse_m: float) -> float:
    """Convert ``RMSE_R <= RMSE_max`` [m] into a minimum delay FIM [1/s^2].

    Monostatic geometry ``R = c tau / 2`` gives ``CRB_R = (c/2)^2 CRB_tau`` and
    ``RMSE_R = sqrt(CRB_R)``, so

        CRB_tau <= (2 RMSE_max / c)^2
        J_tau   >= 1 / CRB_tau.
    """
    if not np.isfinite(max_range_rmse_m) or max_range_rmse_m <= 0.0:
        raise ValueError("max_range_rmse_m must be a positive finite number")
    max_delay_crb_s2 = (2.0 * max_range_rmse_m / SPEED_OF_LIGHT_M_PER_S) ** 2
    return min_fim_from_delay_crb(max_delay_crb_s2)


def delay_crb_summary(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
    reflection_coefficient: complex,
    noise_variance_w: float,
    num_symbols: int = 1,
    known_amplitude: bool = True,
) -> DelayCRBResult:
    """Compute Fisher information, delay CRB and range CRB in one call."""
    fisher = delay_fisher_information(
        power_w,
        frequencies_hz,
        reflection_coefficient,
        noise_variance_w,
        num_symbols=num_symbols,
        known_amplitude=known_amplitude,
    )
    crb_tau = crb_from_fisher_information(fisher)
    crb_range = range_crb_from_delay_crb(crb_tau) if np.isfinite(crb_tau) else float("inf")
    return DelayCRBResult(
        fisher_information=fisher,
        delay_crb_s2=crb_tau,
        delay_rmse_bound_s=float(np.sqrt(crb_tau)),
        range_crb_m2=crb_range,
        range_rmse_bound_m=float(np.sqrt(crb_range)),
    )

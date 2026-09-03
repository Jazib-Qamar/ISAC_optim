"""Exponential-PDP tapped-delay-line small-scale fading.

Model
-----
Discrete taps ``α_l ~ CN(0, P_l)`` at delays ``τ_l = l Δτ``, ``l = 0, ..., L-1``,
with a *normalised* exponential power-delay profile

    P_l  ∝  exp(-τ_l / τ_rms),     sum_l P_l = 1.

The baseband frequency response on the OFDM grid is

    H(f_k) = sum_l α_l exp(-j 2 π f_k τ_l),

and the communication channel including large-scale path gain ``G`` is
``h_k = sqrt(G) H(f_k)``.  Uncorrelated taps imply

    E[|h_k|^2] = G    for every k

so there is **no** constructed positive/negative-frequency gain tilt.  Frequency
selectivity of a *realisation* is a consequence of the delay spread, not of a
sign-dependent multiplier.  Keep this module separate from the controlled
logistic tilt in :mod:`isac.evaluation.scenario`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from isac.channels.rayleigh import complex_gaussian


@dataclass(frozen=True)
class TDLRealization:
    """One TDL draw: taps, delays, PDP and frequency response (before path gain)."""

    delays_s: NDArray[np.float64]
    pdp: NDArray[np.float64]
    taps: NDArray[np.complex128]
    frequency_response: NDArray[np.complex128]
    rms_delay_spread_s: float
    tap_spacing_s: float


def exponential_pdp(delays_s: NDArray[np.float64], rms_delay_spread_s: float) -> NDArray[np.float64]:
    """Normalised exponential PDP ``P_l ∝ exp(-τ_l / τ_rms)``, ``sum P = 1``."""
    delays = np.asarray(delays_s, dtype=np.float64)
    if delays.ndim != 1 or delays.size < 1:
        raise ValueError("delays_s must be a non-empty 1-D array")
    if rms_delay_spread_s <= 0.0:
        raise ValueError("rms_delay_spread_s must be positive")
    if np.any(delays < 0.0):
        raise ValueError("delays_s must be non-negative")
    raw = np.exp(-delays / rms_delay_spread_s)
    total = float(np.sum(raw))
    if total <= 0.0:
        raise ValueError("exponential PDP degenerated to zero")
    return raw / total


def tap_delay_axis(
    tap_spacing_s: float,
    num_taps: int,
) -> NDArray[np.float64]:
    """Uniform tap delays ``τ_l = l Δτ`` [s], shape ``(L,)``."""
    if tap_spacing_s <= 0.0:
        raise ValueError("tap_spacing_s must be positive")
    if num_taps < 1:
        raise ValueError("num_taps must be at least 1")
    return np.arange(num_taps, dtype=np.float64) * float(tap_spacing_s)


def tdl_frequency_response(
    taps: NDArray[np.complex128],
    delays_s: NDArray[np.float64],
    frequencies_hz: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """``H(f_k) = sum_l α_l exp(-j 2 π f_k τ_l)``, shape ``(K,)``."""
    alpha = np.asarray(taps, dtype=np.complex128)
    delays = np.asarray(delays_s, dtype=np.float64)
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    if alpha.ndim != 1 or delays.ndim != 1 or freqs.ndim != 1:
        raise ValueError("taps, delays_s and frequencies_hz must be 1-D")
    if alpha.shape != delays.shape:
        raise ValueError("taps and delays_s must have the same shape")
    phase = 2.0 * np.pi * np.outer(freqs, delays)
    return np.exp(-1j * phase) @ alpha


def draw_tdl_realization(
    frequencies_hz: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    rms_delay_spread_s: float,
    tap_spacing_s: float,
    num_taps: int,
) -> TDLRealization:
    """Draw uncorrelated CN(0, P_l) taps and the corresponding ``H(f_k)``."""
    delays = tap_delay_axis(tap_spacing_s, num_taps)
    pdp = exponential_pdp(delays, rms_delay_spread_s)
    taps = complex_gaussian(num_taps, rng, variance=1.0) * np.sqrt(pdp)
    response = tdl_frequency_response(taps, delays, np.asarray(frequencies_hz, dtype=np.float64))
    return TDLRealization(
        delays_s=delays,
        pdp=pdp,
        taps=taps.astype(np.complex128),
        frequency_response=response.astype(np.complex128),
        rms_delay_spread_s=float(rms_delay_spread_s),
        tap_spacing_s=float(tap_spacing_s),
    )


def tdl_channel(
    frequencies_hz: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    rms_delay_spread_s: float,
    tap_spacing_s: float,
    num_taps: int,
    mean_gain: float = 1.0,
) -> NDArray[np.complex128]:
    """One TDL channel realisation ``h_k = sqrt(G) H(f_k)``, shape ``(K,)``.

    Parameters
    ----------
    frequencies_hz:
        Centred baseband subcarrier frequencies ``f_k`` [Hz].
    mean_gain:
        Large-scale power gain ``G = E[|h_k|^2]``.
    """
    if mean_gain <= 0.0:
        raise ValueError("mean_gain must be positive")
    realisation = draw_tdl_realization(
        frequencies_hz,
        rng,
        rms_delay_spread_s=rms_delay_spread_s,
        tap_spacing_s=tap_spacing_s,
        num_taps=num_taps,
    )
    return np.sqrt(mean_gain) * realisation.frequency_response

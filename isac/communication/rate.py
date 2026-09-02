"""Achievable-rate model for a single-user OFDM link.

Model
-----
For subcarrier ``k`` with transmit power ``P_k`` [W], complex channel
coefficient ``h_k`` and noise power ``sigma^2 = N0 * Delta_f`` [W]:

    SNR_k = |h_k|^2 * P_k / sigma^2

    r_k   = log2(1 + SNR_k)                    [bit/s/Hz]
    R     = sum_k r_k                          [bit/s/Hz summed over subcarriers]

``R`` is the sum of per-subcarrier spectral efficiencies.  Multiplying by the
subcarrier spacing ``Delta_f`` gives the achievable rate in bit/s
(:func:`achievable_rate_bps`).

All functions accept array-like inputs and are fully vectorised.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_inputs(
    power_w: ArrayLike,
    channel_gain: ArrayLike,
    noise_power_w: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert inputs to float arrays and check shapes and signs."""
    power = np.asarray(power_w, dtype=np.float64)
    gain = np.asarray(channel_gain, dtype=np.float64)
    if power.shape != gain.shape:
        raise ValueError(
            f"power_w and channel_gain must have the same shape, got {power.shape} "
            f"and {gain.shape}"
        )
    if np.any(power < 0.0):
        raise ValueError("power_w must be non-negative")
    if np.any(gain < 0.0):
        raise ValueError("channel_gain must be non-negative")
    if not np.isfinite(noise_power_w) or noise_power_w <= 0.0:
        raise ValueError("noise_power_w must be a positive finite number")
    return power, gain


def channel_gain(channel: ArrayLike) -> NDArray[np.float64]:
    """Return the channel power gain ``|h_k|^2`` (dimensionless) from complex ``h_k``."""
    h = np.asarray(channel)
    return np.abs(h).astype(np.float64) ** 2


def subcarrier_snr(
    power_w: ArrayLike,
    channel_gain: ArrayLike,
    noise_power_w: float,
) -> NDArray[np.float64]:
    """Per-subcarrier signal-to-noise ratio ``SNR_k = |h_k|^2 P_k / sigma^2``.

    Parameters
    ----------
    power_w:
        Transmit power per subcarrier ``P_k`` [W], shape ``(K,)``.
    channel_gain:
        Channel power gain ``|h_k|^2`` (dimensionless), shape ``(K,)``.
    noise_power_w:
        Noise power per subcarrier ``sigma^2 = N0 * Delta_f`` [W].

    Returns
    -------
    ndarray
        Linear SNR per subcarrier (dimensionless), shape ``(K,)``.
    """
    power, gain = _validate_inputs(power_w, channel_gain, noise_power_w)
    return gain * power / noise_power_w


def rate_per_subcarrier(
    power_w: ArrayLike,
    channel_gain: ArrayLike,
    noise_power_w: float,
) -> NDArray[np.float64]:
    """Per-subcarrier spectral efficiency ``r_k = log2(1 + SNR_k)`` [bit/s/Hz]."""
    snr = subcarrier_snr(power_w, channel_gain, noise_power_w)
    return np.log2(1.0 + snr)


def spectral_efficiency(
    power_w: ArrayLike,
    channel_gain: ArrayLike,
    noise_power_w: float,
) -> float:
    """Sum spectral efficiency ``R = sum_k log2(1 + SNR_k)``.

    Returns
    -------
    float
        Sum of per-subcarrier spectral efficiencies [bit/s/Hz summed over the
        ``K`` subcarriers].  Equivalently, bits per OFDM channel use.
    """
    return float(np.sum(rate_per_subcarrier(power_w, channel_gain, noise_power_w)))


def achievable_rate_bps(
    power_w: ArrayLike,
    channel_gain: ArrayLike,
    noise_power_w: float,
    subcarrier_spacing_hz: float,
) -> float:
    """Achievable rate in bit/s: ``Delta_f * sum_k log2(1 + SNR_k)``.

    Parameters
    ----------
    subcarrier_spacing_hz:
        Subcarrier spacing ``Delta_f`` [Hz].
    """
    if subcarrier_spacing_hz <= 0.0:
        raise ValueError("subcarrier_spacing_hz must be positive")
    return subcarrier_spacing_hz * spectral_efficiency(power_w, channel_gain, noise_power_w)

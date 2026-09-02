"""Classical communication-only water-filling power allocation.

Problem
-------
    maximize   sum_k log2(1 + alpha_k P_k)
    subject to sum_k P_k = P_total,   0 <= P_k <= P_peak

with the channel-gain-to-noise ratio ``alpha_k = |h_k|^2 / sigma^2`` [1/W].

Solution
--------
The KKT conditions yield the water-filling form

    P_k = clip(mu - 1/alpha_k, 0, P_peak)

where the water level ``mu`` [W] is the unique value for which
``sum_k P_k = P_total``.  ``mu`` is found by bisection because the total power
``sum_k clip(mu - 1/alpha_k, 0, P_peak)`` is continuous and non-decreasing in
``mu``.

Without the peak cap (``peak_power_w=None``) this is the textbook water-filling
solution.  With a finite cap the problem remains convex and the clipped form is
still the exact KKT solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class WaterFillingResult:
    """Output of :func:`water_filling`.

    Attributes
    ----------
    power_w:
        Optimal per-subcarrier power ``P_k`` [W], shape ``(K,)``.
    water_level_w:
        Water level ``mu`` [W].
    inverse_gain_w:
        Noise-to-gain ratio ``1/alpha_k`` [W] (the "floor" of the water-filling
        picture), shape ``(K,)``.  ``inf`` where ``alpha_k = 0``.
    num_active:
        Number of subcarriers with strictly positive power.
    iterations:
        Number of bisection iterations used.
    """

    power_w: NDArray[np.float64]
    water_level_w: float
    inverse_gain_w: NDArray[np.float64]
    num_active: int
    iterations: int


def uniform_power(
    num_subcarriers: int,
    total_power_w: float,
    peak_power_w: float | None = None,
) -> NDArray[np.float64]:
    """Equal power ``P_k = P_total / K`` on every subcarrier [W].

    Raises
    ------
    ValueError
        If ``P_total / K`` exceeds the peak cap.
    """
    if num_subcarriers < 1:
        raise ValueError("num_subcarriers must be at least 1")
    if total_power_w < 0.0:
        raise ValueError("total_power_w must be non-negative")
    per_subcarrier = total_power_w / num_subcarriers
    if peak_power_w is not None and per_subcarrier > peak_power_w * (1.0 + 1e-12):
        raise ValueError(
            f"uniform allocation {per_subcarrier:.3e} W exceeds peak cap {peak_power_w:.3e} W"
        )
    return np.full(num_subcarriers, per_subcarrier, dtype=np.float64)


def _clipped_allocation(
    water_level_w: float,
    inverse_gain_w: NDArray[np.float64],
    peak_power_w: float | None,
) -> NDArray[np.float64]:
    """Evaluate ``clip(mu - 1/alpha_k, 0, P_peak)``."""
    power = np.maximum(water_level_w - inverse_gain_w, 0.0)
    if peak_power_w is not None:
        power = np.minimum(power, peak_power_w)
    return power


def water_filling(
    channel_gain: ArrayLike,
    noise_power_w: float,
    total_power_w: float,
    peak_power_w: float | None = None,
    tolerance_w: float = 1e-12,
    max_iterations: int = 200,
) -> WaterFillingResult:
    """Solve the water-filling problem by bisection on the water level.

    Parameters
    ----------
    channel_gain:
        Channel power gain ``|h_k|^2`` (dimensionless), shape ``(K,)``.
    noise_power_w:
        Noise power per subcarrier ``sigma^2`` [W].
    total_power_w:
        Total power budget ``P_total`` [W]; the solution satisfies
        ``sum_k P_k = P_total`` to within ``tolerance_w``.
    peak_power_w:
        Optional per-subcarrier cap ``P_peak`` [W].  ``None`` disables clipping.
    tolerance_w:
        Absolute tolerance on ``|sum_k P_k - P_total|`` [W].
    max_iterations:
        Maximum number of bisection iterations.

    Returns
    -------
    WaterFillingResult

    Raises
    ------
    ValueError
        On invalid inputs or if ``K * P_peak < P_total`` (infeasible budget), or
        if every channel gain is zero.
    RuntimeError
        If bisection does not reach the requested tolerance.
    """
    gain = np.asarray(channel_gain, dtype=np.float64)
    if gain.ndim != 1 or gain.size == 0:
        raise ValueError("channel_gain must be a non-empty 1-D array")
    if np.any(gain < 0.0):
        raise ValueError("channel_gain must be non-negative")
    if not np.isfinite(noise_power_w) or noise_power_w <= 0.0:
        raise ValueError("noise_power_w must be a positive finite number")
    if total_power_w < 0.0:
        raise ValueError("total_power_w must be non-negative")
    if peak_power_w is not None:
        if peak_power_w <= 0.0:
            raise ValueError("peak_power_w must be positive when given")
        if gain.size * peak_power_w < total_power_w * (1.0 - 1e-12):
            raise ValueError(
                f"infeasible: K * P_peak = {gain.size * peak_power_w:.3e} W is below "
                f"P_total = {total_power_w:.3e} W"
            )
    if tolerance_w <= 0.0:
        raise ValueError("tolerance_w must be positive")

    usable = gain > 0.0
    if not np.any(usable):
        raise ValueError("at least one channel gain must be strictly positive")

    with np.errstate(divide="ignore"):
        inverse_gain = np.where(usable, noise_power_w / gain, np.inf)

    if total_power_w == 0.0:
        return WaterFillingResult(
            power_w=np.zeros_like(gain),
            water_level_w=float(np.min(inverse_gain)),
            inverse_gain_w=inverse_gain,
            num_active=0,
            iterations=0,
        )

    finite_floor = inverse_gain[usable]
    # At mu = min floor no power is allocated; at mu = max floor + P_total every
    # usable subcarrier could carry the whole budget, so the sum exceeds P_total.
    low = float(np.min(finite_floor))
    high = float(np.max(finite_floor)) + total_power_w
    if peak_power_w is not None:
        high = max(high, low + peak_power_w + total_power_w)

    iterations = 0
    power = _clipped_allocation(high, inverse_gain, peak_power_w)
    for iterations in range(1, max_iterations + 1):
        mid = 0.5 * (low + high)
        power = _clipped_allocation(mid, inverse_gain, peak_power_w)
        excess = float(np.sum(power)) - total_power_w
        if abs(excess) <= tolerance_w:
            low = high = mid
            break
        if excess > 0.0:
            high = mid
        else:
            low = mid
    else:
        # Fall back to the midpoint after exhausting iterations.
        mid = 0.5 * (low + high)
        power = _clipped_allocation(mid, inverse_gain, peak_power_w)
        if abs(float(np.sum(power)) - total_power_w) > 1e3 * tolerance_w:
            raise RuntimeError("water-filling bisection failed to converge")
        low = high = mid

    water_level = 0.5 * (low + high)
    return WaterFillingResult(
        power_w=power,
        water_level_w=float(water_level),
        inverse_gain_w=inverse_gain,
        num_active=int(np.count_nonzero(power > 0.0)),
        iterations=iterations,
    )

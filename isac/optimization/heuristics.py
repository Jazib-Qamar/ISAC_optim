"""Non-iterative heuristic allocations used as static baselines.

* :func:`edge_weighted_allocation` - power proportional to the sensing weights
  ``w_k`` (more power at the band edges), redistributed to respect the peak cap.
* :func:`sensing_optimal_allocation` - the allocation that maximises the linear
  sensing surrogate under the box and budget constraints (greedy LP solution,
  see :func:`isac.optimization.feasibility.max_sensing_surrogate`).

Both spend the full budget ``P_total`` (subject to ``K P_peak >= P_total``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from isac.optimization.feasibility import max_sensing_surrogate


def edge_weighted_allocation(
    weights: ArrayLike,
    total_power_w: float,
    peak_power_w: float,
    max_passes: int = 100,
) -> NDArray[np.float64]:
    """``P_k ∝ w_k`` with iterative clipping at ``P_peak`` and budget redistribution.

    Subcarriers whose proportional share exceeds ``P_peak`` are fixed at the cap
    and the remaining budget is redistributed proportionally over the others
    until no cap is exceeded.  Zero-weight subcarriers receive no power.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or np.any(w < 0.0) or not np.any(w > 0.0):
        raise ValueError("weights must be a 1-D non-negative array with a positive entry")
    if total_power_w < 0.0 or peak_power_w <= 0.0:
        raise ValueError("total_power_w must be >= 0 and peak_power_w > 0")
    if w.size * peak_power_w < total_power_w * (1.0 - 1e-12):
        raise ValueError("infeasible: K * P_peak < P_total")

    power = np.zeros_like(w)
    capped = np.zeros(w.size, dtype=bool)
    for _ in range(max_passes):
        free = (~capped) & (w > 0.0)
        remaining = total_power_w - float(np.sum(power[capped]))
        if not np.any(free) or remaining <= 0.0:
            break
        share = remaining * w[free] / float(np.sum(w[free]))
        power[free] = share
        newly_capped = free & (power > peak_power_w)
        if not np.any(newly_capped):
            break
        power[newly_capped] = peak_power_w
        capped |= newly_capped
    else:
        raise RuntimeError("edge-weighted redistribution did not settle")
    if np.sum(power) < total_power_w * (1.0 - 1e-9) and np.all(capped | (w == 0.0)):
        # Every positive-weight tone is capped; spread the remainder over zero-weight tones.
        zero_w = w == 0.0
        remaining = total_power_w - float(np.sum(power))
        if np.any(zero_w) and remaining > 0.0:
            power[zero_w] = min(peak_power_w, remaining / np.count_nonzero(zero_w))
    return power


def sensing_optimal_allocation(
    weights: ArrayLike,
    total_power_w: float,
    peak_power_w: float,
) -> NDArray[np.float64]:
    """Allocation maximising ``sum_k w_k P_k`` under box and budget constraints (sensing-only)."""
    _, power = max_sensing_surrogate(weights, total_power_w, peak_power_w)
    return power

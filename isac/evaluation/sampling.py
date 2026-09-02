"""Random feasible power allocations for surrogate / proxy validation experiments.

All samples satisfy ``0 <= P_k <= P_peak`` and ``sum_k P_k <= P_total``.  Several
structurally different families are mixed so that the validation covers
symmetric, asymmetric, sparse, and smoothly tapered spectra rather than only
i.i.d. noise-like allocations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from isac.system import ISACSystem

FAMILIES: tuple[str, ...] = ("uniform_box", "dirichlet", "sparse", "one_sided_band", "tapered", "edge_heavy")


def _clip_to_feasible(power: NDArray[np.float64], system: ISACSystem, budget_fraction: float) -> NDArray[np.float64]:
    power = np.maximum(power, 0.0)
    total = float(np.sum(power))
    if total <= 0.0:
        power = np.full(system.num_subcarriers, system.total_power_w / system.num_subcarriers)
        total = float(np.sum(power))
    power = power * (budget_fraction * system.total_power_w / total)
    power = np.minimum(power, system.peak_power_w)  # clipping only lowers the total
    return power


def sample_feasible_allocations(
    system: ISACSystem,
    num_samples: int,
    rng: np.random.Generator,
    families: tuple[str, ...] = FAMILIES,
    min_budget_fraction: float = 0.2,
) -> tuple[NDArray[np.float64], list[str]]:
    """Draw ``num_samples`` feasible allocations, cycling through the families.

    Returns ``(powers, family_labels)`` with ``powers`` of shape ``(N, K)``.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    k = system.num_subcarriers
    idx = np.arange(k)
    powers = np.empty((num_samples, k))
    labels: list[str] = []
    for i in range(num_samples):
        family = families[i % len(families)]
        budget_fraction = rng.uniform(min_budget_fraction, 1.0)
        if family == "uniform_box":
            raw = rng.uniform(0.0, 1.0, k)
        elif family == "dirichlet":
            concentration = rng.choice([0.1, 0.3, 1.0, 3.0, 10.0])
            raw = rng.dirichlet(np.full(k, concentration))
        elif family == "sparse":
            m = int(rng.integers(2, max(3, k // 2)))
            raw = np.zeros(k)
            raw[rng.choice(k, size=m, replace=False)] = rng.uniform(0.5, 1.0, m)
        elif family == "one_sided_band":
            width = int(rng.integers(max(2, k // 8), max(3, k // 2)))
            start = int(rng.integers(0, k - width + 1))
            raw = np.zeros(k)
            raw[start : start + width] = rng.uniform(0.5, 1.0, width)
        elif family == "tapered":
            exponent = rng.uniform(0.2, 4.0)
            raw = np.sin(np.pi * (idx + 0.5) / k) ** exponent
        elif family == "edge_heavy":
            exponent = rng.uniform(0.5, 6.0)
            centred = np.abs(idx - (k - 1) / 2.0) / ((k - 1) / 2.0)
            raw = centred**exponent + rng.uniform(0.0, 0.05, k)
        else:
            raise ValueError(f"unknown family {family!r}")
        powers[i] = _clip_to_feasible(raw, system, budget_fraction)
        labels.append(family)
    return powers, labels

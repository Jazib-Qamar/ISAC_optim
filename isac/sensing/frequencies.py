"""Centred OFDM subcarrier frequency grid and derived sensing weights.

Definitions
-----------
    n_k = k - (K - 1) / 2                      centred subcarrier index (dimensionless)
    f_k = n_k * Delta_f                        baseband subcarrier frequency [Hz]

for ``k = 0, ..., K-1``.  The grid is symmetric about 0 Hz so that
``sum_k f_k = 0`` and ``sum_k f_k^2 = Delta_f^2 * K (K^2 - 1) / 12``.

Sensing surrogate weights
-------------------------
The linear sensing-information surrogate used by the convex optimisers is
``S(P) = sum_k w_k P_k`` with either

* ``w_k = f_k^2``                          [Hz^2]        (``"squared_frequency"``), or
* ``w_k = (n_k / n_max)^2``                (dimensionless, ``"normalized_index"``),

where ``n_max = (K - 1) / 2``.  Both are proportional to each other; the first
keeps the exact link to the delay Fisher information (see ``fim.py``), the
second is numerically better scaled for optimisation solvers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

SURROGATE_WEIGHTINGS: tuple[str, ...] = ("squared_frequency", "normalized_index")


def centered_subcarrier_indices(num_subcarriers: int) -> NDArray[np.float64]:
    """Centred indices ``n_k = k - (K-1)/2`` (dimensionless), shape ``(K,)``."""
    if num_subcarriers < 1:
        raise ValueError("num_subcarriers must be at least 1")
    return np.arange(num_subcarriers, dtype=np.float64) - (num_subcarriers - 1) / 2.0


def centered_subcarrier_frequencies(
    num_subcarriers: int,
    subcarrier_spacing_hz: float,
) -> NDArray[np.float64]:
    """Baseband subcarrier frequencies ``f_k = n_k * Delta_f`` [Hz], shape ``(K,)``."""
    if subcarrier_spacing_hz <= 0.0:
        raise ValueError("subcarrier_spacing_hz must be positive")
    return centered_subcarrier_indices(num_subcarriers) * subcarrier_spacing_hz


def sensing_weights(
    num_subcarriers: int,
    subcarrier_spacing_hz: float,
    weighting: str = "squared_frequency",
) -> NDArray[np.float64]:
    """Weights ``w_k`` of the linear sensing surrogate ``S(P) = sum_k w_k P_k``.

    Parameters
    ----------
    weighting:
        ``"squared_frequency"`` returns ``f_k^2`` [Hz^2];
        ``"normalized_index"`` returns ``(n_k / n_max)^2`` (dimensionless, in [0, 1]).

    Returns
    -------
    ndarray
        Non-negative weights, shape ``(K,)``.
    """
    if weighting not in SURROGATE_WEIGHTINGS:
        raise ValueError(f"weighting must be one of {SURROGATE_WEIGHTINGS}, got {weighting!r}")
    if weighting == "squared_frequency":
        return centered_subcarrier_frequencies(num_subcarriers, subcarrier_spacing_hz) ** 2
    indices = centered_subcarrier_indices(num_subcarriers)
    if num_subcarriers == 1:
        return np.zeros(1, dtype=np.float64)
    return (indices / ((num_subcarriers - 1) / 2.0)) ** 2

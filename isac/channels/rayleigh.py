"""Frequency-selective Rayleigh fading with independent subcarriers.

Model
-----
    h_k = sqrt(G) * g_k,    g_k ~ CN(0, 1) i.i.d. over k = 0, ..., K-1

where ``G`` is the large-scale (path-loss) power gain.  With ``G = 1`` the
coefficients are exactly ``CN(0, 1)`` as in the base system model; the
default configuration applies ``G = 10^(-PL_dB/10)`` so that per-subcarrier
SNRs are physically meaningful with a thermal noise floor.

The independence across subcarriers is a deliberate first-stage simplification
(equivalent to a channel whose delay spread exceeds the OFDM symbol duration).
A correlated tapped-delay-line model can be added later without changing the
interface.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def complex_gaussian(
    shape: int | tuple[int, ...],
    rng: np.random.Generator,
    variance: float = 1.0,
) -> NDArray[np.complex128]:
    """Draw circularly-symmetric complex Gaussian samples ``CN(0, variance)``.

    Real and imaginary parts are independent ``N(0, variance/2)`` so that
    ``E[|x|^2] = variance``.
    """
    if variance < 0.0:
        raise ValueError("variance must be non-negative")
    scale = np.sqrt(variance / 2.0)
    real = rng.standard_normal(shape) * scale
    imag = rng.standard_normal(shape) * scale
    return (real + 1j * imag).astype(np.complex128)


def rayleigh_channel(
    num_subcarriers: int,
    rng: np.random.Generator,
    mean_gain: float = 1.0,
) -> NDArray[np.complex128]:
    """Draw one frequency-selective Rayleigh channel realisation.

    Parameters
    ----------
    num_subcarriers:
        Number of subcarriers ``K``.
    rng:
        NumPy random generator (pass ``np.random.default_rng(seed)`` for
        reproducibility).
    mean_gain:
        Large-scale power gain ``G = E[|h_k|^2]`` (dimensionless).  Use ``1.0``
        for the normalised ``CN(0, 1)`` model.

    Returns
    -------
    ndarray of complex128
        Channel coefficients ``h_k``, shape ``(K,)``.
    """
    if num_subcarriers < 1:
        raise ValueError("num_subcarriers must be at least 1")
    if mean_gain <= 0.0:
        raise ValueError("mean_gain must be positive")
    return complex_gaussian(num_subcarriers, rng, variance=mean_gain)


def rayleigh_channel_batch(
    num_realizations: int,
    num_subcarriers: int,
    rng: np.random.Generator,
    mean_gain: float = 1.0,
) -> NDArray[np.complex128]:
    """Draw ``num_realizations`` independent channels, shape ``(N, K)``."""
    if num_realizations < 1:
        raise ValueError("num_realizations must be at least 1")
    if num_subcarriers < 1:
        raise ValueError("num_subcarriers must be at least 1")
    if mean_gain <= 0.0:
        raise ValueError("mean_gain must be positive")
    return complex_gaussian((num_realizations, num_subcarriers), rng, variance=mean_gain)

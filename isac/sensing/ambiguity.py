"""Delay-domain ambiguity (autocorrelation) response of a power-shaped OFDM spectrum.

Model (first version: power-spectrum-induced delay response)
------------------------------------------------------------
    A(tau)      = sum_k P_k exp(j 2 pi f_k tau)                 [W]
    A_norm(tau) = |A(tau)| / |A(0)| = |A(tau)| / sum_k P_k      (dimensionless)

This is the inverse Fourier transform of the discrete power spectrum, i.e. the
expected matched-filter autocorrelation when the per-subcarrier symbols are
uncorrelated with ``E|x_k|^2 = P_k`` (phase-code effects average out).  It is
therefore the delay ambiguity that the *power allocation alone* controls.

Properties used by the implementation and tests
-----------------------------------------------
* For ``P_k >= 0``, ``|A(tau)| <= sum_k P_k = A(0)`` (triangle inequality), so the
  main peak is at ``tau = 0`` and ``A_norm(0) = 1``.
* ``|A(-tau)| = |A(tau)|`` because ``P_k`` is real.
* Because ``f_k = n_k Delta_f`` with ``n_k`` on a half-integer (even ``K``) or
  integer (odd ``K``) lattice, ``|A(tau)|`` is periodic with period ``1 / Delta_f``.
  The unambiguous delay interval is ``[-1/(2 Delta_f), 1/(2 Delta_f)]``.
* A uniform allocation gives the Dirichlet kernel
  ``A_norm(tau) = |sin(pi K Delta_f tau) / (K sin(pi Delta_f tau))|`` whose first
  null is at ``|tau| = 1 / B`` with ``B = K Delta_f``.

Mainlobe exclusion
------------------
The mainlobe scale is ``1 / B``.  Sidelobes are searched only over
``|tau| >= mainlobe_exclusion_s``; the default is ``1.0 / B`` (first null of the
uniform-spectrum response).  For strongly non-uniform spectra (e.g. edge-only
allocations) the mainlobe can be narrower and the "sidelobe" structure becomes
grating-lobe-like; the exclusion width is therefore an explicit, configurable
parameter that is reported with every PSL value.

Metrics
-------
    PSL [dB] = 20 log10( max_{|tau| >= tau_ex} A_norm(tau) )
    ISL [dB] = 10 log10( sum_{|tau| >= tau_ex} A_norm(tau)^2 / sum_{|tau| < tau_ex} A_norm(tau)^2 )

Both are evaluated on a uniform delay grid (the ``d tau`` factors cancel in
the ISL ratio).  The peak spectral power constraint ``P_k <= P_peak`` used by
the optimisers is only a *proxy* for the PSL; its validity is tested empirically
in ``experiments/exp07_psl_proxy_validation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class AmbiguityResult:
    """Delay-domain sidelobe metrics for one power allocation."""

    psl_db: float
    isl_db: float
    mainlobe_exclusion_s: float
    peak_sidelobe_delay_s: float
    num_sidelobe_samples: int


def default_delay_grid(
    subcarrier_spacing_hz: float,
    num_subcarriers: int,
    oversampling_factor: int = 16,
) -> NDArray[np.float64]:
    """Uniform delay grid over one unambiguous period ``[-1/(2 Delta_f), 1/(2 Delta_f))``.

    The grid has ``K * oversampling_factor`` points, i.e. ``oversampling_factor``
    samples per mainlobe scale ``1 / B``, and always contains ``tau = 0``.
    """
    if subcarrier_spacing_hz <= 0.0:
        raise ValueError("subcarrier_spacing_hz must be positive")
    if num_subcarriers < 1 or oversampling_factor < 2:
        raise ValueError("num_subcarriers must be >= 1 and oversampling_factor >= 2")
    num_points = num_subcarriers * oversampling_factor
    period = 1.0 / subcarrier_spacing_hz
    step = period / num_points
    indices = np.arange(num_points) - num_points // 2
    return indices * step


def default_mainlobe_exclusion_s(bandwidth_hz: float, factor: float = 1.0) -> float:
    """Mainlobe half-width ``factor / B`` [s] excluded from the sidelobe search."""
    if bandwidth_hz <= 0.0 or factor <= 0.0:
        raise ValueError("bandwidth_hz and factor must be positive")
    return factor / bandwidth_hz


def _validate(power_w: ArrayLike, frequencies_hz: ArrayLike, delay_grid_s: ArrayLike):
    power = np.asarray(power_w, dtype=np.float64)
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    grid = np.asarray(delay_grid_s, dtype=np.float64)
    if power.ndim != 1 or power.shape != freqs.shape:
        raise ValueError("power_w and frequencies_hz must be 1-D arrays of equal length")
    if np.any(power < 0.0):
        raise ValueError("power_w must be non-negative")
    if not np.any(power > 0.0):
        raise ValueError("at least one subcarrier must carry positive power")
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("delay_grid_s must be a 1-D array with at least two points")
    return power, freqs, grid


def ambiguity_delay_profile(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
    delay_grid_s: ArrayLike,
) -> NDArray[np.float64]:
    """Normalised delay response ``A_norm(tau) = |sum_k P_k e^{j 2 pi f_k tau}| / sum_k P_k``.

    Returns
    -------
    ndarray
        Dimensionless magnitude in ``[0, 1]``, shape ``(M,)`` matching the delay grid.
    """
    power, freqs, grid = _validate(power_w, frequencies_hz, delay_grid_s)
    phases = np.exp(2j * np.pi * np.outer(grid, freqs))
    response = phases @ power
    return np.abs(response) / float(np.sum(power))


def _sidelobe_mask(delay_grid_s: NDArray[np.float64], mainlobe_exclusion_s: float) -> NDArray[np.bool_]:
    if mainlobe_exclusion_s <= 0.0:
        raise ValueError("mainlobe_exclusion_s must be positive")
    mask = np.abs(delay_grid_s) >= mainlobe_exclusion_s
    if not np.any(mask):
        raise ValueError("mainlobe exclusion covers the whole delay grid; no sidelobe region left")
    if np.all(mask):
        raise ValueError("mainlobe exclusion excludes nothing; the grid must contain |tau| < exclusion")
    return mask


def peak_normalized_response_db(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
    delays_s: ArrayLike,
) -> float:
    """``20 log10(max_m |A(tau_m)| / A(0))`` [dB] on an explicit delay list.

    Unlike :func:`peak_sidelobe_level_db` this does *not* apply a mainlobe mask:
    the caller is responsible for passing only the delays that were constrained
    (or the delays that should be checked).
    """
    power, freqs, grid = _validate(power_w, frequencies_hz, delays_s)
    phases = np.exp(2j * np.pi * np.outer(grid, freqs))
    peak = float(np.max(np.abs(phases @ power) / float(np.sum(power))))
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(peak)


def peak_sidelobe_level_db(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
    delay_grid_s: ArrayLike,
    mainlobe_exclusion_s: float,
) -> float:
    """Peak sidelobe level ``20 log10(max_{|tau| >= tau_ex} A_norm(tau))`` [dB] (<= 0)."""
    profile = ambiguity_delay_profile(power_w, frequencies_hz, delay_grid_s)
    grid = np.asarray(delay_grid_s, dtype=np.float64)
    mask = _sidelobe_mask(grid, mainlobe_exclusion_s)
    peak = float(np.max(profile[mask]))
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(peak)


def integrated_sidelobe_level_db(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
    delay_grid_s: ArrayLike,
    mainlobe_exclusion_s: float,
) -> float:
    """Integrated sidelobe level ``10 log10(sidelobe energy / mainlobe energy)`` [dB]."""
    profile = ambiguity_delay_profile(power_w, frequencies_hz, delay_grid_s)
    grid = np.asarray(delay_grid_s, dtype=np.float64)
    mask = _sidelobe_mask(grid, mainlobe_exclusion_s)
    sidelobe_energy = float(np.sum(profile[mask] ** 2))
    mainlobe_energy = float(np.sum(profile[~mask] ** 2))
    if sidelobe_energy <= 0.0:
        return float("-inf")
    return 10.0 * np.log10(sidelobe_energy / mainlobe_energy)


def ambiguity_summary(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
    delay_grid_s: ArrayLike,
    mainlobe_exclusion_s: float,
) -> AmbiguityResult:
    """Compute PSL, ISL and the delay of the peak sidelobe in one pass."""
    profile = ambiguity_delay_profile(power_w, frequencies_hz, delay_grid_s)
    grid = np.asarray(delay_grid_s, dtype=np.float64)
    mask = _sidelobe_mask(grid, mainlobe_exclusion_s)
    side = profile[mask]
    peak_index = int(np.argmax(side))
    peak = float(side[peak_index])
    sidelobe_energy = float(np.sum(side**2))
    mainlobe_energy = float(np.sum(profile[~mask] ** 2))
    return AmbiguityResult(
        psl_db=20.0 * np.log10(peak) if peak > 0.0 else float("-inf"),
        isl_db=10.0 * np.log10(sidelobe_energy / mainlobe_energy) if sidelobe_energy > 0.0 else float("-inf"),
        mainlobe_exclusion_s=float(mainlobe_exclusion_s),
        peak_sidelobe_delay_s=float(grid[mask][peak_index]),
        num_sidelobe_samples=int(np.count_nonzero(mask)),
    )


def psl_db_to_linear_amplitude(psl_max_db: float) -> float:
    """Convert a PSL threshold [dB] into a linear amplitude ratio ``rho = 10^(PSL/20)``.

    ``PSL <= PSL_max_db`` means ``max |A(tau_m)| / A(0) <= rho``.  ``PSL_max_db``
    must be ``<= 0`` because ``|A(tau)| <= A(0)`` for non-negative spectra.
    """
    if not np.isfinite(psl_max_db):
        raise ValueError("psl_max_db must be finite")
    if psl_max_db > 0.0:
        raise ValueError("psl_max_db must be <= 0 dB (sidelobes cannot exceed the main peak)")
    return float(10.0 ** (psl_max_db / 20.0))


def sidelobe_delay_samples(
    delay_grid_s: ArrayLike,
    mainlobe_exclusion_s: float,
) -> NDArray[np.float64]:
    """Delay samples with ``|tau| >= mainlobe_exclusion_s`` (the sampled-sidelobe set)."""
    grid = np.asarray(delay_grid_s, dtype=np.float64)
    mask = _sidelobe_mask(grid, mainlobe_exclusion_s)
    return grid[mask]


def dirichlet_delay_profile(
    num_subcarriers: int,
    subcarrier_spacing_hz: float,
    delay_grid_s: ArrayLike,
) -> NDArray[np.float64]:
    """Closed-form ``|sin(pi K Delta_f tau) / (K sin(pi Delta_f tau))|`` for a uniform allocation.

    Used as an analytical reference in tests.
    """
    grid = np.asarray(delay_grid_s, dtype=np.float64)
    x = np.pi * subcarrier_spacing_hz * grid
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.sin(num_subcarriers * x) / (num_subcarriers * np.sin(x))
    # sin(x) = 0 at multiples of 1/Delta_f where the kernel is +/-1.
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    return np.abs(ratio)

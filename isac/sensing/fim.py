"""Fisher information for round-trip delay in single-target OFDM sensing.

Signal model (one OFDM symbol, monostatic, single point target)
----------------------------------------------------------------
    y_k = beta * x_k * exp(-j 2 pi f_k tau) + n_k,      k = 0, ..., K-1

* ``beta``  complex reflection coefficient (two-way path loss, RCS, gains),
* ``x_k``   transmitted symbol on subcarrier ``k`` with ``|x_k|^2 = P_k`` [W],
* ``f_k``   centred baseband subcarrier frequency [Hz],
* ``tau``   round-trip delay [s],
* ``n_k ~ CN(0, sigma^2)`` i.i.d. with ``sigma^2 = N0 * Delta_f`` [W].

Because ``|x_k|^2 = P_k`` is a *power* and ``sigma^2`` is a noise *power* in the
subcarrier bandwidth, the ratio ``P_k / sigma^2`` equals the per-subcarrier
energy-to-noise-PSD ratio ``E_k / N0`` for a rectangular OFDM symbol of
duration ``T = 1 / Delta_f``.  Coherent integration over ``N`` symbols scales the
Fisher information by ``N``.

Noise convention and the factor of 2
------------------------------------
``n_k ~ CN(0, sigma^2)`` denotes a circularly-symmetric complex Gaussian whose
real and imaginary parts are independent ``N(0, sigma^2 / 2)``, so that
``E[|n_k|^2] = sigma^2`` is the noise *power per complex observation*.  Writing
the observation as the real vector ``[Re y_k, Im y_k]`` with covariance
``(sigma^2 / 2) I_2`` and applying the real Gaussian FIM formula gives

    J_ij = (1 / (sigma^2 / 2)) * sum_k [ d(Re mu_k)/d theta_i d(Re mu_k)/d theta_j
                                         + d(Im mu_k)/d theta_i d(Im mu_k)/d theta_j ]
         = (2 / sigma^2) * Re{ sum_k (d mu_k / d theta_i)^* (d mu_k / d theta_j) }.

The factor ``2 / sigma^2`` therefore comes from the per-component variance
``sigma^2 / 2``; it is *not* an extra bandwidth or two-sided-PSD factor.

Fisher information (complex Gaussian observations with parameter-dependent mean)
-------------------------------------------------------------------------------
    J_ij = (2 / sigma^2) * Re{ sum_k (d mu_k / d theta_i)^* (d mu_k / d theta_j) }

with ``mu_k = beta x_k exp(-j 2 pi f_k tau)`` and ``d mu_k / d tau = -j 2 pi f_k mu_k``.

**Known amplitude** (``beta`` known, ``theta = tau``):

    J_tau = N * (2 |beta|^2 / sigma^2) * sum_k (2 pi f_k)^2 P_k          [1/s^2]

**Unknown complex amplitude** (``theta = [tau, Re beta, Im beta]``), after
eliminating the nuisance parameters by the Schur complement:

    J_tau^eff = N * (2 |beta|^2 / sigma^2) * (2 pi)^2 *
                [ sum_k f_k^2 P_k - (sum_k f_k P_k)^2 / sum_k P_k ]     [1/s^2]

i.e. the power-weighted *variance* of the subcarrier frequencies replaces the
power-weighted second moment.  For allocations symmetric about 0 Hz the two
expressions coincide; in general ``J_tau^eff <= J_tau``.

Linear surrogate
----------------
    S(P) = sum_k w_k P_k

is *proportional* to the known-amplitude ``J_tau`` when ``w_k = f_k^2``:
``J_tau = N (2|beta|^2 / sigma^2) (2 pi)^2 S(P)``.  It is *not* the CRB, and it
is not exactly proportional to the unknown-amplitude information for
asymmetric allocations.  Names are kept separate deliberately.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_power_and_frequencies(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    power = np.asarray(power_w, dtype=np.float64)
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    if power.ndim != 1 or freqs.ndim != 1:
        raise ValueError("power_w and frequencies_hz must be 1-D arrays")
    if power.shape != freqs.shape:
        raise ValueError(
            f"power_w and frequencies_hz must have the same shape, got {power.shape} "
            f"and {freqs.shape}"
        )
    if np.any(power < 0.0):
        raise ValueError("power_w must be non-negative")
    return power, freqs


def _validate_scalars(reflection_coefficient: complex, noise_variance_w: float, num_symbols: int) -> float:
    if not np.isfinite(noise_variance_w) or noise_variance_w <= 0.0:
        raise ValueError("noise_variance_w must be a positive finite number")
    if num_symbols < 1:
        raise ValueError("num_symbols must be at least 1")
    beta_power = float(abs(complex(reflection_coefficient)) ** 2)
    if beta_power <= 0.0:
        raise ValueError("reflection_coefficient must be non-zero")
    return beta_power


def delay_fisher_information(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
    reflection_coefficient: complex,
    noise_variance_w: float,
    num_symbols: int = 1,
    known_amplitude: bool = True,
) -> float:
    """Fisher information ``J_tau`` for the round-trip delay [1/s^2].

    Parameters
    ----------
    power_w:
        Per-subcarrier transmit power ``P_k`` [W], shape ``(K,)``.
    frequencies_hz:
        Centred baseband subcarrier frequencies ``f_k`` [Hz], shape ``(K,)``.
    reflection_coefficient:
        Complex reflection coefficient ``beta`` (dimensionless).
    noise_variance_w:
        Noise power per subcarrier ``sigma^2 = N0 * Delta_f`` [W].
    num_symbols:
        Number of coherently integrated OFDM symbols ``N``.
    known_amplitude:
        ``True`` for the known-``beta`` model, ``False`` to treat ``beta`` as an
        unknown complex nuisance parameter (Schur-complement information).

    Returns
    -------
    float
        ``J_tau`` [1/s^2].  Zero if no power is transmitted.
    """
    power, freqs = _validate_power_and_frequencies(power_w, frequencies_hz)
    beta_power = _validate_scalars(reflection_coefficient, noise_variance_w, num_symbols)

    total_power = float(np.sum(power))
    if total_power == 0.0:
        return 0.0

    second_moment = float(np.sum(freqs**2 * power))
    if known_amplitude:
        weighted_moment = second_moment
    else:
        first_moment = float(np.sum(freqs * power))
        weighted_moment = second_moment - first_moment**2 / total_power
        weighted_moment = max(weighted_moment, 0.0)  # guard against round-off

    scale = num_symbols * 2.0 * beta_power / noise_variance_w
    return scale * (2.0 * np.pi) ** 2 * weighted_moment


def sensing_information_surrogate(
    power_w: ArrayLike,
    weights: ArrayLike,
) -> float:
    """Linear sensing-information surrogate ``S(P) = sum_k w_k P_k``.

    Units follow the weights: ``[W * Hz^2]`` for ``w_k = f_k^2`` and ``[W]`` for
    normalised-index weights.  This is a *surrogate*, not the CRB.
    """
    power = np.asarray(power_w, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if power.shape != w.shape:
        raise ValueError(
            f"power_w and weights must have the same shape, got {power.shape} and {w.shape}"
        )
    if np.any(power < 0.0):
        raise ValueError("power_w must be non-negative")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    return float(np.dot(w, power))


def surrogate_to_fisher_scale(
    reflection_coefficient: complex,
    noise_variance_w: float,
    num_symbols: int = 1,
) -> float:
    """Constant ``c`` such that ``J_tau = c * S(P)`` for ``w_k = f_k^2`` (known amplitude).

    ``c = N * (2 |beta|^2 / sigma^2) * (2 pi)^2 = 8 pi^2 N |beta|^2 / sigma^2``
    with units ``[1/(W s^2 Hz^2)]``.  The same constant converts the unknown-amplitude
    kernel ``G(P)`` into ``J_tau^eff`` (see :func:`unknown_amplitude_fim_scale`).
    """
    beta_power = _validate_scalars(reflection_coefficient, noise_variance_w, num_symbols)
    return num_symbols * 2.0 * beta_power / noise_variance_w * (2.0 * np.pi) ** 2


def unknown_amplitude_fim_scale(
    reflection_coefficient: complex,
    noise_variance_w: float,
    num_symbols: int = 1,
) -> float:
    """Scale ``C_beta`` such that ``J_tau^eff = C_beta * G(P)`` [1/(W s^2 Hz^2)].

    ``C_beta = 8 pi^2 N |beta|^2 / sigma_n^2``.  This is identical to
    :func:`surrogate_to_fisher_scale`; the name is kept so the unknown-amplitude
    constraint is not confused with the linear surrogate ``S(P)``.
    """
    return surrogate_to_fisher_scale(reflection_coefficient, noise_variance_w, num_symbols)


def power_moments(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
) -> tuple[float, float, float]:
    """Power-spectrum moments ``(S0, S1, S2)``.

    ``S0 = sum P_k`` [W], ``S1 = sum f_k P_k`` [W Hz], ``S2 = sum f_k^2 P_k`` [W Hz^2].
    """
    power, freqs = _validate_power_and_frequencies(power_w, frequencies_hz)
    s0 = float(np.sum(power))
    s1 = float(np.dot(freqs, power))
    s2 = float(np.dot(freqs**2, power))
    return s0, s1, s2


def spectral_centroid_hz(power_w: ArrayLike, frequencies_hz: ArrayLike) -> float:
    """Power-weighted spectral centroid ``f_bar_P = S1 / S0`` [Hz].

    Returns ``0`` when no power is transmitted (the centroid is undefined).
    """
    s0, s1, _ = power_moments(power_w, frequencies_hz)
    if s0 == 0.0:
        return 0.0
    return s1 / s0


def unknown_amplitude_kernel(power_w: ArrayLike, frequencies_hz: ArrayLike) -> float:
    """Concave kernel ``G(P) = S2 - S1^2 / S0`` [W Hz^2] of the unknown-amplitude delay FIM.

    Identity (verified in tests): ``G(P) = sum_k P_k (f_k - f_bar_P)^2`` for ``S0 > 0``.
    This is *not* a surrogate: ``J_tau^eff = C_beta * G(P)`` under the Stage 1 model.
    Returns ``0`` when ``S0 = 0``.
    """
    s0, s1, s2 = power_moments(power_w, frequencies_hz)
    if s0 == 0.0:
        return 0.0
    return max(s2 - s1**2 / s0, 0.0)


def power_weighted_spectral_variance(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
) -> float:
    """Power-weighted spectral variance ``G(P) / S0 = sum_k (P_k/S0) (f_k - f_bar_P)^2`` [Hz^2].

    Returns ``0`` when no power is transmitted.
    """
    s0, _, _ = power_moments(power_w, frequencies_hz)
    if s0 == 0.0:
        return 0.0
    return unknown_amplitude_kernel(power_w, frequencies_hz) / s0


def unknown_amplitude_sensing_information(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
    reflection_coefficient: complex,
    noise_variance_w: float,
    num_symbols: int = 1,
) -> float:
    """Effective delay Fisher information for unknown complex ``beta`` [1/s^2].

    Delegates to :func:`delay_fisher_information` with ``known_amplitude=False``
    so there is a single physical model.  Equivalent closed form:

        J_tau^eff = C_beta * G(P),   G(P) = S2 - S1^2/S0.
    """
    return delay_fisher_information(
        power_w,
        frequencies_hz,
        reflection_coefficient,
        noise_variance_w,
        num_symbols=num_symbols,
        known_amplitude=False,
    )


def unknown_amplitude_kernel_gradient(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
) -> NDArray[np.float64]:
    """Exact gradient ``dG/dP_k = (f_k - f_bar_P)^2`` [Hz^2] for ``S0 > 0``.

    ``f_bar_P`` depends on the whole vector ``P``; the envelope terms from
    ``d f_bar / d P_k`` cancel, leaving this simple expression.  It is *not* an
    independent per-tone weight: changing any ``P_j`` moves the centroid.
    """
    power, freqs = _validate_power_and_frequencies(power_w, frequencies_hz)
    s0 = float(np.sum(power))
    if s0 <= 0.0:
        raise ValueError("unknown_amplitude_kernel_gradient requires strictly positive total power")
    f_bar = float(np.dot(freqs, power)) / s0
    return (freqs - f_bar) ** 2

"""KKT structure of the exact unknown-amplitude delay-FIM formulation.

Kernel and gradient
-------------------
    G(P)      = S2 - S1^2 / S0  =  sum_k P_k (f_k - f_bar_P)^2
    dG / dP_k = (f_k - f_bar_P)^2

``f_bar_P = S1/S0`` depends on the *whole* allocation, so the effective sensing
weight ``w_eff,k = (f_k - f_bar_P)^2`` is coupled across subcarriers.  It is a
structural characterisation, not an independent per-tone closed form.

When the spectrum is symmetric about 0 Hz, ``f_bar_P ≈ 0`` and
``w_eff,k ≈ f_k^2`` — the Stage 2 linear surrogate is that special case.

Stationarity (no active sampled-PSL SOCs)
-----------------------------------------
For the Dinkelbach sub-problem in spectral-efficiency units

    maximize  R_se(P) - (q / Delta_f) P_sys(P)
    s.t.      G(P) >= gamma_fim,  sum P <= P_max,  0 <= P_k <= P_peak

an interior carrier (``0 < P_k < P_peak``) satisfies

    alpha_k / [(1 + alpha_k P_k) ln 2]  -  lambda_E  +  mu_G (f_k - f_bar_P)^2  -  nu  = 0

with ``lambda_E = (q / Delta_f) / eta_PA`` and ``mu_G >= 0``, ``nu >= 0``.
If the constraint is written on ``J = C_beta G``, then ``mu_J = mu_G / C_beta``.

When sampled-PSL SOCs are active, additional coupled dual terms appear; this
module does **not** claim the simple water-filling structure in that case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from isac.sensing.fim import (
    spectral_centroid_hz,
    unknown_amplitude_fim_scale,
    unknown_amplitude_kernel,
    unknown_amplitude_kernel_gradient,
    unknown_amplitude_sensing_information,
)
from isac.system import ISACSystem

LN2: float = float(np.log(2.0))
INTERIOR_FRAC: float = 1e-4


@dataclass(frozen=True)
class KKTReport:
    """Stationarity diagnostics for one allocation under the exact-FIM KKT."""

    spectral_centroid_hz: float
    effective_weights_hz2: NDArray[np.float64]
    squared_frequency_weights_hz2: NDArray[np.float64]
    interior_mask: NDArray[np.bool_]
    mu_kernel: float  # multiplier on G(P) >= gamma
    nu_total_power: float
    lambda_energy: float
    max_stationarity_residual: float
    mean_stationarity_residual: float
    fim_complementary_slackness: float
    power_complementary_slackness: float
    gradient_max_abs_error: float
    psl_socs_active: bool
    note: str

    def as_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "mu_kernel": self.mu_kernel,
            "nu_total_power": self.nu_total_power,
            "lambda_energy": self.lambda_energy,
            "max_stationarity_residual": self.max_stationarity_residual,
            "mean_stationarity_residual": self.mean_stationarity_residual,
            "fim_complementary_slackness": self.fim_complementary_slackness,
            "power_complementary_slackness": self.power_complementary_slackness,
            "gradient_max_abs_error": self.gradient_max_abs_error,
            "num_interior_carriers": int(np.count_nonzero(self.interior_mask)),
            "psl_socs_active": self.psl_socs_active,
            "note": self.note,
        }


def finite_difference_kernel_gradient(
    power_w: ArrayLike,
    frequencies_hz: ArrayLike,
    relative_step: float = 1e-6,
) -> NDArray[np.float64]:
    """Central finite-difference approximation of ``dG/dP`` [Hz^2]."""
    power = np.asarray(power_w, dtype=np.float64)
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    grad = np.empty_like(power)
    for k in range(power.size):
        step = relative_step * max(power[k], np.max(power) * 1e-3, 1e-12)
        plus = power.copy()
        minus = power.copy()
        plus[k] += step
        minus[k] = max(minus[k] - step, 0.0)
        # If clipping hit zero, fall back to forward difference.
        if minus[k] == 0.0 and power[k] > 0.0:
            grad[k] = (unknown_amplitude_kernel(plus, freqs) - unknown_amplitude_kernel(power, freqs)) / step
        else:
            grad[k] = (unknown_amplitude_kernel(plus, freqs) - unknown_amplitude_kernel(minus, freqs)) / (plus[k] - minus[k])
    return grad


def communication_rate_gradient_se(power_w: NDArray[np.float64], alpha_per_w: NDArray[np.float64]) -> NDArray[np.float64]:
    """``d R_se / d P_k = alpha_k / ((1 + alpha_k P_k) ln 2)`` [bit/s/Hz / W]."""
    return alpha_per_w / ((1.0 + alpha_per_w * power_w) * LN2)


def interior_mask(power_w: NDArray[np.float64], peak_power_w: float, frac: float = INTERIOR_FRAC) -> NDArray[np.bool_]:
    """Carriers bounded away from ``0`` and ``P_peak``."""
    lo = frac * peak_power_w
    hi = (1.0 - frac) * peak_power_w
    return (power_w > lo) & (power_w < hi)


def analyze_exact_fim_kkt(
    power_w: ArrayLike,
    system: ISACSystem,
    *,
    min_unknown_fim: float,
    q_bit_per_j: float = 0.0,
    psl_socs_active: bool = False,
    fd_relative_step: float = 1e-6,
) -> KKTReport:
    """Fit ``(mu_G, nu)`` by least squares on interior carriers and report residuals.

    The energy term is ``lambda_E = (q / Delta_f) / eta_PA`` corresponding to an
    objective written in spectral-efficiency units (as in the Dinkelbach
    sub-problem).  For a pure max-rate problem pass ``q_bit_per_j=0``.

    If ``psl_socs_active`` is true the simple stationarity claim is *not* made;
    residuals are still computed (they will generally be large) and the note
    records that SOC duals are missing.
    """
    power = np.asarray(power_w, dtype=np.float64)
    freqs = system.frequencies_hz
    f_bar = spectral_centroid_hz(power, freqs)
    w_eff = unknown_amplitude_kernel_gradient(power, freqs)
    w_old = freqs**2
    fd = finite_difference_kernel_gradient(power, freqs, relative_step=fd_relative_step)
    grad_err = float(np.max(np.abs(fd - w_eff)))

    alpha = system.gain_to_noise_per_w
    comm_grad = communication_rate_gradient_se(power, alpha)
    lambda_e = (q_bit_per_j / system.subcarrier_spacing_hz) / system.pa_efficiency
    mask = interior_mask(power, system.peak_power_w)
    note = (
        "sampled-PSL SOCs are active: the simple (mu, nu) stationarity does not apply"
        if psl_socs_active
        else "exact-FIM / no-active-PSL special case"
    )

    mu = nu = 0.0
    residuals = np.full(power.size, np.nan)
    max_res = mean_res = float("nan")
    if int(np.count_nonzero(mask)) >= 2:
        # comm_grad - lambda_E + mu * w_eff - nu = 0
        # columns: [w_eff, -1]
        a = np.column_stack([w_eff[mask], -np.ones(int(np.count_nonzero(mask)))])
        b = -(comm_grad[mask] - lambda_e)
        theta, *_ = np.linalg.lstsq(a, b, rcond=None)
        mu, nu = float(theta[0]), float(theta[1])
        residuals = comm_grad - lambda_e + mu * w_eff - nu
        max_res = float(np.max(np.abs(residuals[mask])))
        mean_res = float(np.mean(np.abs(residuals[mask])))

    j_unknown = unknown_amplitude_sensing_information(
        power,
        freqs,
        system.reflection_coefficient,
        system.noise_power_w,
        num_symbols=system.num_symbols,
    )
    c_beta = unknown_amplitude_fim_scale(
        system.reflection_coefficient, system.noise_power_w, system.num_symbols
    )
    # mu is on G; complementary slackness mu_G * (G - gamma_fim) = mu_G * (J - Gamma_J)/C_beta
    gamma_j = float(min_unknown_fim)
    fim_cs = abs(mu) * abs(j_unknown - gamma_j) / c_beta
    slack_power = system.total_power_w - float(np.sum(power))
    power_cs = abs(nu) * abs(slack_power)

    return KKTReport(
        spectral_centroid_hz=f_bar,
        effective_weights_hz2=w_eff,
        squared_frequency_weights_hz2=w_old,
        interior_mask=mask,
        mu_kernel=mu,
        nu_total_power=nu,
        lambda_energy=lambda_e,
        max_stationarity_residual=max_res,
        mean_stationarity_residual=mean_res,
        fim_complementary_slackness=float(fim_cs),
        power_complementary_slackness=float(power_cs),
        gradient_max_abs_error=grad_err,
        psl_socs_active=psl_socs_active,
        note=note,
    )

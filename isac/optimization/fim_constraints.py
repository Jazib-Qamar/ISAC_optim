"""CVXPY encodings of the unknown-complex-amplitude delay-FIM constraint.

The effective delay information after eliminating ``beta`` is

    J_tau^eff(P) = C_beta * G(P),
    G(P)         = S2 - S1^2 / S0,
    S0 = sum P_k,  S1 = f^T P,  S2 = (f^2)^T P.

``S1^2 / S0`` is quadratic-over-linear and convex for ``S0 > 0``, so ``G`` is
concave and ``G >= gamma_fim`` is a convex (DCP) constraint.  The equivalent
rotated-SOC form

    S1^2  <=  S0 * (S2 - gamma_fim),   S0 >= 0,  S2 - gamma_fim >= 0

is provided for numerical comparison; the primary encoding is
``cp.quad_over_lin``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from isac.optimization.common import ScaledModel
from isac.optimization.sensing_spec import QUAD_OVER_LIN, RSOC, SensingSpec
from isac.sensing.fim import unknown_amplitude_fim_scale
from isac.system import ISACSystem


@dataclass(frozen=True)
class UnknownFimEncoding:
    """CVXPY pieces of one unknown-amplitude FIM constraint."""

    constraints: list[cp.Constraint]
    s0: cp.Expression
    s1: cp.Expression
    s2: cp.Expression
    gamma_fim: float  # G-threshold [W Hz^2]
    gamma_j: float  # J-threshold [1/s^2]
    scale_c_beta: float
    representation: str


def unknown_fim_moments(power_w: cp.Expression, frequencies_hz: NDArray[np.float64]) -> tuple[cp.Expression, cp.Expression, cp.Expression]:
    """Affine CVXPY expressions ``(S0, S1, S2)`` for a power vector."""
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    s0 = cp.sum(power_w)
    s1 = freqs @ power_w
    s2 = (freqs**2) @ power_w
    return s0, s1, s2


def encode_unknown_fim_constraint(
    power_w: cp.Expression,
    frequencies_hz: NDArray[np.float64],
    gamma_fim: float,
    representation: str = QUAD_OVER_LIN,
    s0_epsilon_w: float = 1e-12,
) -> UnknownFimEncoding:
    """Return DCP constraints enforcing ``G(P) >= gamma_fim``.

    ``gamma_fim`` is the kernel threshold [W Hz^2], i.e. ``Gamma_J / C_beta``.
    Frequencies are scaled by ``max |f_k|`` so that ``S1``/``S2`` stay O(1) in
    the solver; the constraint is algebraically identical.
    """
    if gamma_fim < 0.0:
        raise ValueError("gamma_fim must be non-negative")
    if representation not in (QUAD_OVER_LIN, RSOC):
        raise ValueError(f"unknown representation {representation!r}")
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    f_scale = float(np.max(np.abs(freqs)))
    if f_scale <= 0.0:
        raise ValueError("frequencies_hz must contain a non-zero entry")
    freqs_n = freqs / f_scale
    gamma_n = gamma_fim / (f_scale**2)
    s0, s1, s2 = unknown_fim_moments(power_w, freqs_n)
    constraints: list[cp.Constraint] = [s0 >= s0_epsilon_w]
    if representation == QUAD_OVER_LIN:
        # G_n = S2_n - quad_over_lin(S1_n, S0) is concave; G_n >= gamma_n is DCP.
        constraints.append(s2 - cp.quad_over_lin(s1, s0) >= gamma_n)
    else:
        # Rotated SOC: ||[2 S1, S0 - t]||_2 <= S0 + t  with t = S2_n - gamma_n >= 0.
        t = s2 - gamma_n
        constraints.append(t >= 0.0)
        constraints.append(cp.SOC(s0 + t, cp.hstack([2.0 * s1, s0 - t])))
    return UnknownFimEncoding(
        constraints=constraints,
        s0=s0,
        s1=s1,
        s2=s2,
        gamma_fim=float(gamma_fim),
        gamma_j=float("nan"),
        scale_c_beta=float("nan"),
        representation=representation,
    )


def build_unknown_fim_constraints(
    model: ScaledModel,
    system: ISACSystem,
    spec: SensingSpec,
) -> tuple[list[cp.Constraint], dict[str, float]]:
    """Build the unknown-amplitude FIM constraint for one scaled model.

    Returns ``(constraints, extras)`` where ``extras`` records ``Gamma_J``,
    ``gamma_fim`` and ``C_beta`` for logging.
    """
    gamma_j = spec.resolved_min_unknown_fim()
    if gamma_j is None:
        raise ValueError("unknown-amplitude FIM constraint requested without a Gamma_J")
    c_beta = unknown_amplitude_fim_scale(
        system.reflection_coefficient, system.noise_power_w, system.num_symbols
    )
    gamma_fim = gamma_j / c_beta
    encoding = encode_unknown_fim_constraint(
        model.power_w,
        system.frequencies_hz,
        gamma_fim,
        representation=spec.unknown_fim_representation,
        s0_epsilon_w=spec.s0_epsilon_w,
    )
    extras = {
        "min_unknown_fim": gamma_j,
        "gamma_fim_kernel": gamma_fim,
        "c_beta": c_beta,
        "unknown_fim_representation": spec.unknown_fim_representation,
        "s0_epsilon_w": spec.s0_epsilon_w,
    }
    return encoding.constraints, extras


def unknown_fim_problem_is_dcp(
    system: ISACSystem,
    gamma_j: float,
    representation: str = QUAD_OVER_LIN,
) -> bool:
    """Build a dummy min-power problem and report CVXPY DCP compliance."""
    from isac.optimization.common import build_scaled_model

    model = build_scaled_model(system)
    spec = SensingSpec(
        model="unknown_amplitude_exact",
        min_unknown_fim=gamma_j,
        unknown_fim_representation=representation,
    )
    constraints = list(model.box_constraints)
    extra_cons, _ = build_unknown_fim_constraints(model, system, spec)
    constraints.extend(extra_cons)
    problem = cp.Problem(cp.Minimize(model.tx_power_w), constraints)
    return bool(problem.is_dcp())

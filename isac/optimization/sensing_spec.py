"""Sensing-constraint specification for the Stage 2.5 static oracle.

Two *sensing models* are supported; they are not interchangeable:

* ``known_amplitude_linear`` — the Stage 2 linear surrogate ``S(P) = sum_k w_k P_k``,
  which is exactly proportional to the *known-amplitude* delay FIM when
  ``w_k = f_k^2``.
* ``unknown_amplitude_exact`` — the Stage 1 unknown-complex-``beta`` delay FIM
  ``J_tau^eff = C_beta * (S2 - S1^2/S0)``.  This is a concave function of ``P``
  (quadratic-over-linear) and its superlevel set is a convex constraint.

The per-subcarrier cap ``P_k <= P_peak`` is a *peak spectral power* constraint,
not a PSL constraint.  Direct sampled-grid PSL control is a separate optional
SOC family (see :mod:`isac.optimization.ambiguity_constraints`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from isac.sensing.crb import min_fim_from_delay_crb, min_fim_from_range_rmse
from isac.system import ISACSystem

KNOWN_AMPLITUDE_LINEAR: str = "known_amplitude_linear"
UNKNOWN_AMPLITUDE_EXACT: str = "unknown_amplitude_exact"
SENSING_MODELS: tuple[str, ...] = (KNOWN_AMPLITUDE_LINEAR, UNKNOWN_AMPLITUDE_EXACT)
QUAD_OVER_LIN: str = "quad_over_lin"
RSOC: str = "rsoc"
FIM_REPRESENTATIONS: tuple[str, ...] = (QUAD_OVER_LIN, RSOC)

# Small positive floor on total power so ``quad_over_lin(S1, S0)`` stays defined.
DEFAULT_S0_EPSILON_W: float = 1e-12


@dataclass(frozen=True)
class SensingSpec:
    """Requirements attached to one static solve.

    Exactly one of the unknown-amplitude FIM inputs may be used to set
    ``Gamma_J`` (the tightest implied bound wins if several are given).
    """

    model: str = KNOWN_AMPLITUDE_LINEAR
    min_sensing_surrogate: float | None = None
    min_unknown_fim: float | None = None
    max_delay_crb_s2: float | None = None
    max_range_rmse_m: float | None = None
    psl_max_db: float | None = None
    psl_delays_s: NDArray[np.float64] | None = None
    psl_mainlobe_exclusion_s: float | None = None
    unknown_fim_representation: str = QUAD_OVER_LIN
    s0_epsilon_w: float = DEFAULT_S0_EPSILON_W

    def __post_init__(self) -> None:
        if self.model not in SENSING_MODELS:
            raise ValueError(f"model must be one of {SENSING_MODELS}, got {self.model!r}")
        if self.unknown_fim_representation not in FIM_REPRESENTATIONS:
            raise ValueError(
                f"unknown_fim_representation must be one of {FIM_REPRESENTATIONS}, "
                f"got {self.unknown_fim_representation!r}"
            )
        if self.s0_epsilon_w <= 0.0:
            raise ValueError("s0_epsilon_w must be positive")
        if self.min_sensing_surrogate is not None and self.min_sensing_surrogate < 0.0:
            raise ValueError("min_sensing_surrogate must be non-negative")
        if self.min_unknown_fim is not None and self.min_unknown_fim < 0.0:
            raise ValueError("min_unknown_fim must be non-negative")
        if self.psl_max_db is not None and self.psl_max_db > 0.0:
            raise ValueError("psl_max_db must be <= 0")
        if self.psl_delays_s is not None:
            delays = np.asarray(self.psl_delays_s, dtype=np.float64)
            object.__setattr__(self, "psl_delays_s", delays)

    def resolved_min_unknown_fim(self) -> float | None:
        """Tightest ``Gamma_J`` [1/s^2] implied by the unknown-amplitude inputs, or ``None``."""
        candidates: list[float] = []
        if self.min_unknown_fim is not None:
            candidates.append(float(self.min_unknown_fim))
        if self.max_delay_crb_s2 is not None:
            candidates.append(min_fim_from_delay_crb(self.max_delay_crb_s2))
        if self.max_range_rmse_m is not None:
            candidates.append(min_fim_from_range_rmse(self.max_range_rmse_m))
        if not candidates:
            return None
        return max(candidates)

    def validate_for_solve(self) -> None:
        """Raise if the spec is missing the requirement required by ``model``."""
        if self.model == KNOWN_AMPLITUDE_LINEAR:
            if self.min_sensing_surrogate is None:
                raise ValueError("known_amplitude_linear requires min_sensing_surrogate")
        elif self.resolved_min_unknown_fim() is None:
            raise ValueError(
                "unknown_amplitude_exact requires min_unknown_fim, max_delay_crb_s2, "
                "or max_range_rmse_m"
            )

    def resolved_psl_exclusion_s(self, system: ISACSystem) -> float:
        if self.psl_mainlobe_exclusion_s is not None:
            return float(self.psl_mainlobe_exclusion_s)
        return float(system.mainlobe_exclusion_s)


def spec_from_legacy(
    min_sensing_surrogate: float | None,
    *,
    sensing_model: str = KNOWN_AMPLITUDE_LINEAR,
    min_unknown_fim: float | None = None,
    max_delay_crb_s2: float | None = None,
    max_range_rmse_m: float | None = None,
    psl_max_db: float | None = None,
    psl_delays_s: NDArray[np.float64] | None = None,
    psl_mainlobe_exclusion_s: float | None = None,
    unknown_fim_representation: str = QUAD_OVER_LIN,
    s0_epsilon_w: float = DEFAULT_S0_EPSILON_W,
) -> SensingSpec:
    """Build a :class:`SensingSpec` from the optimiser keyword arguments."""
    spec = SensingSpec(
        model=sensing_model,
        min_sensing_surrogate=min_sensing_surrogate,
        min_unknown_fim=min_unknown_fim,
        max_delay_crb_s2=max_delay_crb_s2,
        max_range_rmse_m=max_range_rmse_m,
        psl_max_db=psl_max_db,
        psl_delays_s=psl_delays_s,
        psl_mainlobe_exclusion_s=psl_mainlobe_exclusion_s,
        unknown_fim_representation=unknown_fim_representation,
        s0_epsilon_w=s0_epsilon_w,
    )
    spec.validate_for_solve()
    return spec

"""Independent numerical verification of the static OFDM-ISAC constraints.

The optimisers never trust the solver's own feasibility report: every returned
allocation is re-checked here with the NumPy model.  A constraint

    value >= bound   (or value <= bound)

is accepted if its slack ``value - bound`` (resp. ``bound - value``) is at least
``-(abs_tol + rel_tol * |bound|)``.  Slacks are reported for every constraint
so that experiments can see which requirements are active (slack ~ 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from isac.communication.rate import spectral_efficiency
from isac.sensing.fim import sensing_information_surrogate
from isac.system import ISACSystem

DEFAULT_ABS_TOL: float = 1e-9
DEFAULT_REL_TOL: float = 1e-6


@dataclass(frozen=True)
class ConstraintCheck:
    """Result of one constraint verification.

    ``slack >= 0`` means satisfied exactly; ``satisfied`` additionally allows the
    tolerance.  ``active`` flags constraints with ``|slack| <= tolerance``.
    """

    name: str
    relation: str  # ">=" or "<="
    value: float
    bound: float
    slack: float
    tolerance: float
    satisfied: bool

    @property
    def active(self) -> bool:
        return abs(self.slack) <= self.tolerance

    @property
    def violation(self) -> float:
        """Positive amount by which the constraint is violated (0 if satisfied)."""
        return max(0.0, -self.slack)


@dataclass(frozen=True)
class FeasibilityReport:
    """Collection of constraint checks for one allocation."""

    checks: tuple[ConstraintCheck, ...]

    @property
    def feasible(self) -> bool:
        return all(c.satisfied for c in self.checks)

    @property
    def max_violation(self) -> float:
        return max((c.violation for c in self.checks), default=0.0)

    @property
    def violated(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if not c.satisfied)

    def slack_dict(self) -> dict[str, float]:
        return {f"slack_{c.name}": c.slack for c in self.checks}

    def get(self, name: str) -> ConstraintCheck:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)


def _check(name: str, relation: str, value: float, bound: float, abs_tol: float, rel_tol: float) -> ConstraintCheck:
    if relation == ">=":
        slack = value - bound
    elif relation == "<=":
        slack = bound - value
    else:
        raise ValueError("relation must be '>=' or '<='")
    tolerance = abs_tol + rel_tol * abs(bound)
    return ConstraintCheck(
        name=name,
        relation=relation,
        value=float(value),
        bound=float(bound),
        slack=float(slack),
        tolerance=float(tolerance),
        satisfied=bool(slack >= -tolerance),
    )


def check_static_constraints(
    power_w: ArrayLike,
    system: ISACSystem,
    min_rate_se: float | None = None,
    min_sensing_surrogate: float | None = None,
    max_power_w: float | None = None,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> FeasibilityReport:
    """Verify all hard constraints of the static ISAC problems for ``power_w``.

    Parameters
    ----------
    power_w:
        Candidate allocation (may contain tiny negative solver round-off).
    system:
        Physical parameters (``peak_power_w`` is used as the per-tone cap).
    min_rate_se:
        ``R_min`` in summed spectral-efficiency units [bit/s/Hz]; ``None`` skips.
    min_sensing_surrogate:
        ``Gamma_s`` for ``S(P) = sum_k w_k P_k``; ``None`` skips.
    max_power_w:
        Total power limit; defaults to ``system.total_power_w``.
    """
    power: NDArray[np.float64] = np.asarray(power_w, dtype=np.float64)
    if power.shape != system.channel_gain.shape:
        raise ValueError("power_w must match the number of subcarriers")
    budget = system.total_power_w if max_power_w is None else float(max_power_w)

    checks = [
        _check("power_nonnegative", ">=", float(np.min(power)), 0.0, abs_tol, rel_tol),
        _check("peak_power", "<=", float(np.max(power)), system.peak_power_w, abs_tol, rel_tol),
        _check("total_power", "<=", float(np.sum(power)), budget, abs_tol, rel_tol),
    ]
    clipped = np.maximum(power, 0.0)
    if min_rate_se is not None:
        rate = spectral_efficiency(clipped, system.channel_gain, system.noise_power_w)
        checks.append(_check("min_rate", ">=", rate, float(min_rate_se), abs_tol, rel_tol))
    if min_sensing_surrogate is not None:
        surrogate = sensing_information_surrogate(clipped, system.sensing_weights)
        checks.append(_check("min_sensing", ">=", surrogate, float(min_sensing_surrogate), abs_tol, rel_tol))
    return FeasibilityReport(tuple(checks))


def max_sensing_surrogate(
    weights: ArrayLike,
    total_power_w: float,
    peak_power_w: float,
) -> tuple[float, NDArray[np.float64]]:
    """Largest ``S(P) = sum_k w_k P_k`` under ``0 <= P_k <= P_peak``, ``sum P_k <= P_total``.

    This is a linear program with box and budget constraints; the optimum is
    greedy: fill subcarriers in decreasing order of ``w_k`` up to ``P_peak``
    until the budget is exhausted.  Returns ``(S_max, P_max_allocation)``.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or np.any(w < 0.0):
        raise ValueError("weights must be a 1-D non-negative array")
    if total_power_w < 0.0 or peak_power_w <= 0.0:
        raise ValueError("total_power_w must be >= 0 and peak_power_w > 0")
    order = np.argsort(-w, kind="stable")
    power = np.zeros_like(w)
    remaining = float(total_power_w)
    for idx in order:
        if remaining <= 0.0:
            break
        allocation = min(peak_power_w, remaining)
        power[idx] = allocation
        remaining -= allocation
    return float(np.dot(w, power)), power

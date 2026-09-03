"""Baseline interface for ICC comparisons.

Internal methods (uniform, water-filling, conventional S2, exact EFIM, …)
implement this interface.  A future published-method reproduction must return
the same :class:`BaselineResult` and be scored by the independent physical
evaluator.  Do **not** attach a published-paper label unless that method has
actually been implemented under this system model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from isac.system import ISACSystem


@dataclass(frozen=True)
class BaselineResult:
    """Standardised output of one baseline / proposed allocation method."""

    method_name: str
    power_allocation: NDArray[np.float64]
    runtime_s: float
    convergence_status: str
    optimizer_claimed_sensing_metric: float | None
    subcarrier_assignment: NDArray[np.int64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AllocationBaseline(Protocol):
    """Callable baseline: ``(system, **kwargs) -> BaselineResult``."""

    name: str

    def run(self, system: ISACSystem, **kwargs: Any) -> BaselineResult: ...

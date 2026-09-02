"""Static convex OFDM-ISAC power-allocation optimisers (CVXPY, CLARABEL preferred, SCS fallback).

All problems are formulated directly as DCP-compliant convex programs:

* :mod:`isac.optimization.min_power`  - minimum transmit power subject to rate and sensing requirements,
* :mod:`isac.optimization.max_rate`   - maximum rate subject to a sensing requirement,
* :mod:`isac.optimization.dinkelbach` - maximum energy efficiency via Dinkelbach's parametric method.

Physical metrics of every solution are recomputed with the NumPy model; the
CVXPY objective value is stored only as a diagnostic.
"""

from isac.optimization.exceptions import (
    InfeasibleProblemError,
    OptimizationError,
    SolverFailureError,
)

__all__ = ["InfeasibleProblemError", "OptimizationError", "SolverFailureError"]

"""Reusable CVXPY solve helper with solver fallback and strict status checking.

Accepted statuses are ``OPTIMAL`` and ``OPTIMAL_INACCURATE``; they are reported
separately in :class:`SolveInfo` so that experiments can count inaccurate
solves.  Statuses ``INFEASIBLE`` / ``UNBOUNDED`` raise
:class:`InfeasibleProblemError`; anything else triggers the next solver in the
preference list and finally :class:`SolverFailureError`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import cvxpy as cp

from isac.optimization.exceptions import InfeasibleProblemError, SolverFailureError

DEFAULT_SOLVER_PREFERENCE: tuple[str, ...] = ("CLARABEL", "SCS")
ACCEPTED_STATUSES: frozenset[str] = frozenset({cp.OPTIMAL, cp.OPTIMAL_INACCURATE})
INFEASIBLE_STATUSES: frozenset[str] = frozenset({cp.INFEASIBLE, cp.UNBOUNDED})
INACCURATE_INFEASIBLE_STATUSES: frozenset[str] = frozenset({cp.INFEASIBLE_INACCURATE, cp.UNBOUNDED_INACCURATE})


@dataclass(frozen=True)
class SolveInfo:
    """Outcome of :func:`solve_with_fallback`.

    Attributes
    ----------
    status:
        CVXPY status string of the accepted solve.
    solver_name:
        Solver that produced the accepted solution.
    solve_time_s:
        Wall-clock time of the accepted solve [s] (including CVXPY compilation).
    solver_time_s:
        Solver-reported time [s] if available, else ``nan``.
    objective_value:
        CVXPY objective value (diagnostic only; physical metrics are recomputed).
    inaccurate:
        ``True`` if the status was ``OPTIMAL_INACCURATE``.
    attempts:
        ``(solver_name, status)`` for every solver tried, in order.
    """

    status: str
    solver_name: str
    solve_time_s: float
    solver_time_s: float
    objective_value: float
    inaccurate: bool
    attempts: tuple[tuple[str, str], ...]


def solve_with_fallback(
    problem: cp.Problem,
    problem_name: str,
    solver_preference: Sequence[str] = DEFAULT_SOLVER_PREFERENCE,
    solver_options: dict[str, dict] | None = None,
) -> SolveInfo:
    """Solve ``problem`` with the first solver that returns an accepted status.

    Parameters
    ----------
    problem:
        A DCP-compliant CVXPY problem.
    problem_name:
        Used in exception messages.
    solver_preference:
        Ordered solver names (must be installed).
    solver_options:
        Optional per-solver keyword arguments, e.g. ``{"SCS": {"eps": 1e-9}}``.

    Raises
    ------
    InfeasibleProblemError
        If a solver proves the problem infeasible or unbounded (a definitive
        ``INFEASIBLE``/``UNBOUNDED`` status stops the fallback chain; an
        ``*_INACCURATE`` infeasibility is re-checked with the next solver).
    SolverFailureError
        If no solver returns an accepted status.
    """
    if not problem.is_dcp():
        raise ValueError(f"{problem_name}: problem is not DCP-compliant; refusing to solve")
    if not solver_preference:
        raise ValueError("solver_preference must not be empty")
    options = solver_options or {}
    attempts: list[tuple[str, str]] = []
    last_infeasible: tuple[str, str] | None = None

    for solver_name in solver_preference:
        start = time.perf_counter()
        try:
            problem.solve(solver=solver_name, **options.get(solver_name, {}))
        except cp.error.SolverError as exc:  # solver crashed or is unavailable
            attempts.append((solver_name, f"SolverError: {exc}"))
            continue
        elapsed = time.perf_counter() - start
        status = str(problem.status)
        attempts.append((solver_name, status))

        if status in ACCEPTED_STATUSES:
            stats = problem.solver_stats
            solver_time = float(stats.solve_time) if stats is not None and stats.solve_time is not None else float("nan")
            return SolveInfo(
                status=status,
                solver_name=solver_name,
                solve_time_s=elapsed,
                solver_time_s=solver_time,
                objective_value=float(problem.value),
                inaccurate=(status == cp.OPTIMAL_INACCURATE),
                attempts=tuple(attempts),
            )
        if status in INFEASIBLE_STATUSES:
            raise InfeasibleProblemError(problem_name, status, solver_name)
        if status in INACCURATE_INFEASIBLE_STATUSES:
            last_infeasible = (solver_name, status)
            continue

    if last_infeasible is not None:
        solver_name, status = last_infeasible
        raise InfeasibleProblemError(
            problem_name, status, solver_name, detail="only an inaccurate infeasibility certificate was obtained"
        )
    raise SolverFailureError(problem_name, attempts)

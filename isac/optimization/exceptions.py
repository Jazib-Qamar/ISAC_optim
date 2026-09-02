"""Custom exceptions raised by the static optimisers."""

from __future__ import annotations


class OptimizationError(RuntimeError):
    """Base class for optimiser failures."""


class InfeasibleProblemError(OptimizationError):
    """The problem was reported infeasible (or unbounded) by every solver tried.

    Attributes
    ----------
    status:
        Final CVXPY status string (e.g. ``"infeasible"``).
    solver_name:
        Name of the last solver that was tried.
    problem_name:
        Human-readable name of the optimisation problem.
    """

    def __init__(self, problem_name: str, status: str, solver_name: str, detail: str = "") -> None:
        self.problem_name = problem_name
        self.status = status
        self.solver_name = solver_name
        message = f"{problem_name}: problem is {status} (solver {solver_name})"
        if detail:
            message = f"{message}. {detail}"
        super().__init__(message)


class SolverFailureError(OptimizationError):
    """No solver returned an accepted status and the problem was not proven infeasible."""

    def __init__(self, problem_name: str, attempts: list[tuple[str, str]]) -> None:
        self.problem_name = problem_name
        self.attempts = attempts
        summary = ", ".join(f"{name}: {status}" for name, status in attempts) or "no solver attempted"
        super().__init__(f"{problem_name}: all solvers failed ({summary})")

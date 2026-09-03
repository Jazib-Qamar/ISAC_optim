"""Shared assembly of rate / sensing / sampled-PSL constraints on a scaled model."""

from __future__ import annotations

from typing import Any

import cvxpy as cp

from isac.optimization.ambiguity_constraints import build_psl_constraints_for_model
from isac.optimization.common import ScaledModel
from isac.optimization.fim_constraints import build_unknown_fim_constraints
from isac.optimization.sensing_spec import KNOWN_AMPLITUDE_LINEAR, UNKNOWN_AMPLITUDE_EXACT, SensingSpec
from isac.system import ISACSystem


def assemble_static_constraints(
    model: ScaledModel,
    system: ISACSystem,
    spec: SensingSpec,
    min_rate_se: float | None = None,
) -> tuple[list[cp.Constraint], dict[str, Any]]:
    """Box + optional rate + sensing-model + optional sampled-PSL constraints.

    The box/budget constraints already live on ``model.box_constraints``.
    """
    spec.validate_for_solve()
    constraints: list[cp.Constraint] = list(model.box_constraints)
    extras: dict[str, Any] = {"sensing_model": spec.model}

    if min_rate_se is not None:
        constraints.append(model.rate_se >= min_rate_se)

    if spec.model == KNOWN_AMPLITUDE_LINEAR:
        assert spec.min_sensing_surrogate is not None
        constraints.append(model.sensing_scaled >= spec.min_sensing_surrogate / model.sensing_scale)
        extras["min_sensing_surrogate"] = spec.min_sensing_surrogate
    elif spec.model == UNKNOWN_AMPLITUDE_EXACT:
        fim_cons, fim_extras = build_unknown_fim_constraints(model, system, spec)
        constraints.extend(fim_cons)
        extras.update(fim_extras)
    else:
        raise ValueError(f"unsupported sensing model {spec.model!r}")

    if spec.psl_max_db is not None:
        delays = spec.psl_delays_s
        if delays is None or delays.size == 0:
            raise ValueError("psl_max_db is set but psl_delays_s is empty; refusing to skip PSL control")
        psl_cons, psl_extras = build_psl_constraints_for_model(model, system, spec.psl_max_db, delays)
        constraints.extend(psl_cons)
        extras.update(psl_extras)
        extras["psl_mainlobe_exclusion_s"] = spec.resolved_psl_exclusion_s(system)

    return constraints, extras

"""Independent physical evaluator: claimed vs actual sensing / dense PSL."""

from __future__ import annotations

import numpy as np

from isac.evaluation.physical import evaluate_independent, target_from_unknown_fim
from isac.optimization.feasibility import max_sensing_surrogate, max_unknown_amplitude_fim
from isac.communication.water_filling import uniform_power
from isac.system import ISACSystem


def _target(system: ISACSystem, fraction: float = 0.5):
    j_max, _ = max_unknown_amplitude_fim(system)
    s_max, _ = max_sensing_surrogate(system.sensing_weights, system.total_power_w, system.peak_power_w)
    return target_from_unknown_fim(
        system,
        fraction * j_max,
        j_max=j_max,
        s_max=s_max,
        psl_max_db=-13.0,
        fim_fraction=fraction,
        psl_tolerance_db=0.25,
        optimization_oversampling=4,
        validation_oversampling=16,
    )


def test_uniform_does_not_claim_sensing_constraint(system64: ISACSystem) -> None:
    target = _target(system64)
    power = uniform_power(system64.num_subcarriers, system64.total_power_w, system64.peak_power_w)
    row = evaluate_independent(
        power, system64, target,
        method="uniform", label="Uniform",
        optimizer_model="none",
    )
    assert row["optimizer_claimed_sensing_satisfied"] is False
    assert row["false_sensing_feasibility"] is False
    assert "independently_evaluated_unknown_fim" in row
    assert "validation_grid_psl_db" in row
    assert row["dense_psl_db"] == row["validation_grid_psl_db"]


def test_dense_psl_flag_uses_validation_grid_not_opt_grid(system64: ISACSystem) -> None:
    target = _target(system64, 0.2)
    power = uniform_power(system64.num_subcarriers, system64.total_power_w, system64.peak_power_w)
    row = evaluate_independent(
        power, system64, target,
        method="uniform", label="Uniform",
        optimizer_model="none",
        extra={"dense_psl_satisfied": False},
    )
    # Explicit extra False must win: sampled-grid success is not dense success.
    assert row["dense_psl_satisfied"] is False

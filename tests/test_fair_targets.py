"""Fair physical targets: conventional S2 does not receive exact EFIM."""

from __future__ import annotations

import numpy as np
import pytest

from isac.evaluation.icc_suite import build_physical_target, run_method
from isac.evaluation.physical import (
    conventional_s_from_unknown_fim,
    fim_mismatch_percent,
)
from isac.optimization.feasibility import max_unknown_amplitude_fim
from isac.optimization.max_rate import solve_max_rate
from isac.optimization.sensing_spec import KNOWN_AMPLITUDE_LINEAR, UNKNOWN_AMPLITUDE_EXACT
from isac.sensing.fim import unknown_amplitude_fim_scale
from isac.system import ISACSystem
from configs.default import default_config


def test_conventional_s_threshold_matches_known_amplitude_fim(system64: ISACSystem) -> None:
    c_beta = unknown_amplitude_fim_scale(
        system64.reflection_coefficient, system64.noise_power_w, system64.num_symbols,
    )
    gamma_j = 1.0e18
    gamma_s = conventional_s_from_unknown_fim(gamma_j, c_beta)
    assert gamma_s * c_beta == pytest.approx(gamma_j, rel=1e-12)


def test_mismatch_zero_for_symmetric_spectrum(system64: ISACSystem) -> None:
    power = np.ones(system64.num_subcarriers) * system64.total_power_w / system64.num_subcarriers
    from isac.evaluation.metrics import evaluate_allocation

    m = evaluate_allocation(power, system64)
    eps = fim_mismatch_percent(m.delay_fim_known_amplitude_per_s2, m.delay_fim_unknown_amplitude_per_s2)
    assert eps == pytest.approx(0.0, abs=1e-6)


def test_fair_targets_use_same_gamma_j_for_s2_and_efim(system64: ISACSystem) -> None:
    cfg = default_config()
    target = build_physical_target(system64, cfg, fim_fraction=0.4, use_uniform_psl=True)
    assert target.gamma_unknown_fim > 0.0
    assert target.gamma_conventional_s == pytest.approx(target.gamma_unknown_fim / target.c_beta, rel=1e-12)
    s2 = run_method("conventional_s2", system64, target, cfg)
    efim = run_method("exact_efim", system64, target, cfg)
    assert s2["status"] == "ok"
    assert efim["status"] == "ok"
    # Conventional still reports S as the claimed metric, not G.
    assert s2["optimizer_model"] == "conventional_s2"
    assert efim["optimizer_model"] == "exact_efim"
    assert s2["optimizer_claimed_sensing_metric"] == pytest.approx(s2["sensing_surrogate"], rel=1e-9)
    # Independent unknown FIM is always populated.
    assert s2["independently_evaluated_unknown_fim"] > 0.0
    assert efim["independently_evaluated_unknown_fim"] + 1e6 >= target.gamma_unknown_fim


def test_conventional_optimizer_does_not_receive_exact_efim(system64: ISACSystem) -> None:
    j_max, _ = max_unknown_amplitude_fim(system64)
    gamma_j = 0.45 * j_max
    c_beta = unknown_amplitude_fim_scale(
        system64.reflection_coefficient, system64.noise_power_w, system64.num_symbols,
    )
    gamma_s = conventional_s_from_unknown_fim(gamma_j, c_beta)
    res = solve_max_rate(
        system64,
        min_sensing_surrogate=gamma_s,
        sensing_model=KNOWN_AMPLITUDE_LINEAR,
    )
    # The linear-S program has no unknown-FIM requirement attached.
    assert res.requirements.get("min_unknown_fim") in (None, 0.0) or res.requirements["min_unknown_fim"] is None
    assert res.extra.get("sensing_model") == KNOWN_AMPLITUDE_LINEAR
    # Contrast: exact EFIM solve does attach Gamma_J.
    exact = solve_max_rate(
        system64,
        sensing_model=UNKNOWN_AMPLITUDE_EXACT,
        min_unknown_fim=gamma_j,
    )
    assert exact.requirements["min_unknown_fim"] == pytest.approx(gamma_j)
    assert exact.extra.get("sensing_model") == UNKNOWN_AMPLITUDE_EXACT

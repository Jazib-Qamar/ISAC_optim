"""Tests for the transmitter power-consumption model."""

from __future__ import annotations

import numpy as np
import pytest

from isac.energy.power_model import energy_efficiency, energy_per_slot, system_power, tx_power


def test_tx_power_is_sum() -> None:
    assert tx_power(np.array([0.1, 0.2, 0.3])) == pytest.approx(0.6)


def test_system_power_formula() -> None:
    p = np.array([0.25, 0.25])
    assert system_power(p, circuit_power_w=0.5, pa_efficiency=0.4) == pytest.approx(0.5 + 0.5 / 0.4)


def test_system_power_with_zero_transmit_power_is_circuit_power() -> None:
    assert system_power(np.zeros(8), circuit_power_w=0.3, pa_efficiency=0.5) == pytest.approx(0.3)


def test_energy_per_slot() -> None:
    p = np.array([1.0])
    assert energy_per_slot(p, 0.0, 1.0, slot_duration_s=2e-3) == pytest.approx(2e-3)


def test_energy_efficiency_units() -> None:
    assert energy_efficiency(rate_bps=2e6, system_power_w=4.0) == pytest.approx(5e5)


@pytest.mark.parametrize("pa_efficiency", [0.0, -0.1, 1.5])
def test_invalid_pa_efficiency(pa_efficiency: float) -> None:
    with pytest.raises(ValueError):
        system_power(np.ones(2), 0.1, pa_efficiency)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        system_power(np.array([-1.0]), 0.1, 0.5)
    with pytest.raises(ValueError):
        system_power(np.ones(2), -0.1, 0.5)
    with pytest.raises(ValueError):
        energy_efficiency(1.0, 0.0)
    with pytest.raises(ValueError):
        energy_per_slot(np.ones(2), 0.1, 0.5, slot_duration_s=0.0)

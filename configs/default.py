"""Default configuration for the OFDM-ISAC energy-efficiency framework.

All physical quantities carry explicit SI units in their field names or
docstrings.  The configuration is built from small, frozen dataclasses so that
every experiment can be reproduced from a single ``DefaultConfig`` instance.

Conventions
-----------
* Powers are in watt [W], energies in joule [J], durations in second [s].
* Noise power spectral density is specified in dBm/Hz and converted to W/Hz.
* Large-scale path loss is specified in dB and applied as a multiplicative
  gain to the CN(0, 1) small-scale Rayleigh coefficients.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass, field

SPEED_OF_LIGHT_M_PER_S: float = 299_792_458.0
"""Speed of light in vacuum [m/s]."""


def dbm_to_watt(value_dbm: float) -> float:
    """Convert a power level from dBm to watt."""
    return 10.0 ** ((value_dbm - 30.0) / 10.0)


def db_to_linear(value_db: float) -> float:
    """Convert a power ratio from dB to a linear ratio."""
    return 10.0 ** (value_db / 10.0)


@dataclass(frozen=True)
class OFDMConfig:
    """OFDM transmitter and receiver front-end parameters.

    Attributes
    ----------
    num_subcarriers:
        Number of subcarriers ``K``.
    subcarrier_spacing_hz:
        Subcarrier spacing ``Delta_f`` [Hz].
    carrier_frequency_hz:
        Carrier frequency ``f_c`` [Hz].  Not used by the baseband model in this
        stage but kept for later Doppler/range extensions.
    noise_psd_dbm_per_hz:
        Thermal noise power spectral density ``N0`` [dBm/Hz] at the receiver
        input (``-174`` dBm/Hz corresponds to 290 K).
    noise_figure_db:
        Receiver noise figure [dB], added to ``N0``.
    total_power_w:
        Total transmit power budget ``P_total`` [W].
    peak_power_factor:
        Per-subcarrier peak-power cap expressed as a multiple of the uniform
        allocation ``P_total / K``.  ``P_peak = peak_power_factor * P_total / K``.
    """

    num_subcarriers: int = 64
    subcarrier_spacing_hz: float = 15e3
    carrier_frequency_hz: float = 3.5e9
    noise_psd_dbm_per_hz: float = -174.0
    noise_figure_db: float = 7.0
    total_power_w: float = 1.0
    peak_power_factor: float = 4.0

    def __post_init__(self) -> None:
        if self.num_subcarriers < 2:
            raise ValueError("num_subcarriers must be at least 2")
        if self.subcarrier_spacing_hz <= 0.0:
            raise ValueError("subcarrier_spacing_hz must be positive")
        if self.carrier_frequency_hz <= 0.0:
            raise ValueError("carrier_frequency_hz must be positive")
        if self.noise_figure_db < 0.0:
            raise ValueError("noise_figure_db must be non-negative")
        if self.total_power_w <= 0.0:
            raise ValueError("total_power_w must be positive")
        if self.peak_power_factor < 1.0:
            raise ValueError(
                "peak_power_factor must be >= 1, otherwise the total power budget "
                "cannot be spent even with a uniform allocation"
            )

    @property
    def noise_psd_w_per_hz(self) -> float:
        """Effective noise PSD including the noise figure [W/Hz]."""
        return dbm_to_watt(self.noise_psd_dbm_per_hz + self.noise_figure_db)

    @property
    def noise_power_per_subcarrier_w(self) -> float:
        """Noise power in one subcarrier bandwidth ``N0 * Delta_f`` [W]."""
        return self.noise_psd_w_per_hz * self.subcarrier_spacing_hz

    @property
    def bandwidth_hz(self) -> float:
        """Occupied bandwidth ``K * Delta_f`` [Hz]."""
        return self.num_subcarriers * self.subcarrier_spacing_hz

    @property
    def symbol_duration_s(self) -> float:
        """Useful OFDM symbol duration ``1 / Delta_f`` [s] (no cyclic prefix)."""
        return 1.0 / self.subcarrier_spacing_hz

    @property
    def peak_power_w(self) -> float:
        """Per-subcarrier peak transmit power cap ``P_peak`` [W]."""
        return self.peak_power_factor * self.total_power_w / self.num_subcarriers


@dataclass(frozen=True)
class ChannelConfig:
    """Communication channel parameters.

    Attributes
    ----------
    path_loss_db:
        Large-scale path loss [dB] between transmitter and communication user.
        The effective channel is ``h_k = sqrt(path_gain) * g_k`` with
        ``g_k ~ CN(0, 1)``.
    temporal_correlation:
        Gauss-Markov correlation coefficient ``rho`` in ``[0, 1]`` used by the
        time-varying channel model (later stage).
    """

    path_loss_db: float = 120.0
    temporal_correlation: float = 0.99

    def __post_init__(self) -> None:
        if self.path_loss_db < 0.0:
            raise ValueError("path_loss_db must be non-negative")
        if not 0.0 <= self.temporal_correlation <= 1.0:
            raise ValueError("temporal_correlation must lie in [0, 1]")

    @property
    def path_gain(self) -> float:
        """Linear large-scale power gain ``10^(-PL/10)`` (dimensionless)."""
        return db_to_linear(-self.path_loss_db)


@dataclass(frozen=True)
class SensingConfig:
    """Monostatic single-target sensing parameters.

    Attributes
    ----------
    target_range_m:
        One-way range to the point target [m].  The round-trip delay is
        ``tau = 2 * range / c``.
    reflection_gain_db:
        Composite two-way power gain ``|beta|^2`` [dB] of the sensing return
        (two-way path loss, radar cross-section, antenna gains).
    reflection_phase_rad:
        Phase of the complex reflection coefficient ``beta`` [rad].
    num_symbols:
        Number of coherently integrated OFDM symbols ``N``.  Fisher
        information scales linearly with ``N``.
    known_amplitude:
        If ``True`` the reflection coefficient is treated as known when
        computing the delay Fisher information; otherwise it is treated as an
        unknown complex nuisance parameter.
    surrogate_weighting:
        Weighting rule for the linear sensing surrogate ``S(P) = sum w_k P_k``.
        One of ``"squared_frequency"`` (``w_k = f_k^2`` [Hz^2]) or
        ``"normalized_index"`` (``w_k = ((k - (K-1)/2) / ((K-1)/2))^2``, dimensionless).
    """

    target_range_m: float = 50.0
    reflection_gain_db: float = -100.0
    reflection_phase_rad: float = 0.0
    num_symbols: int = 1
    known_amplitude: bool = True
    surrogate_weighting: str = "squared_frequency"

    def __post_init__(self) -> None:
        if self.target_range_m <= 0.0:
            raise ValueError("target_range_m must be positive")
        if self.num_symbols < 1:
            raise ValueError("num_symbols must be at least 1")
        if self.surrogate_weighting not in ("squared_frequency", "normalized_index"):
            raise ValueError(
                "surrogate_weighting must be 'squared_frequency' or 'normalized_index'"
            )

    @property
    def round_trip_delay_s(self) -> float:
        """Round-trip propagation delay ``tau = 2 R / c`` [s]."""
        return 2.0 * self.target_range_m / SPEED_OF_LIGHT_M_PER_S

    @property
    def reflection_coefficient(self) -> complex:
        """Complex reflection coefficient ``beta`` (dimensionless amplitude)."""
        magnitude = db_to_linear(self.reflection_gain_db) ** 0.5
        return cmath.rect(magnitude, self.reflection_phase_rad)


@dataclass(frozen=True)
class EnergyConfig:
    """Transmitter power-consumption and energy-harvesting parameters.

    Attributes
    ----------
    circuit_power_w:
        Static circuit power ``P_circuit`` [W] (baseband, oscillators, ...).
    pa_efficiency:
        Power-amplifier drain efficiency ``eta_PA`` in ``(0, 1]``.
    slot_duration_s:
        Duration of one scheduling slot ``T_slot`` [s].
    """

    circuit_power_w: float = 0.5
    pa_efficiency: float = 0.35
    slot_duration_s: float = 1e-3

    def __post_init__(self) -> None:
        if self.circuit_power_w < 0.0:
            raise ValueError("circuit_power_w must be non-negative")
        if not 0.0 < self.pa_efficiency <= 1.0:
            raise ValueError("pa_efficiency must lie in (0, 1]")
        if self.slot_duration_s <= 0.0:
            raise ValueError("slot_duration_s must be positive")


@dataclass(frozen=True)
class SimulationConfig:
    """Reproducibility and output settings."""

    seed: int = 0
    output_dir: str = "results"


@dataclass(frozen=True)
class DefaultConfig:
    """Top-level container bundling all sub-configurations."""

    ofdm: OFDMConfig = field(default_factory=OFDMConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    sensing: SensingConfig = field(default_factory=SensingConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def default_config() -> DefaultConfig:
    """Return the default configuration used throughout the experiments."""
    return DefaultConfig()

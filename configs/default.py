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
class AmbiguityConfig:
    """Delay-domain ambiguity / sidelobe evaluation settings.

    The power-spectrum-induced delay response ``A(tau) = sum_k P_k exp(j 2 pi f_k tau)``
    is periodic with period ``1 / Delta_f``; the unambiguous delay interval is
    ``[-1/(2 Delta_f), 1/(2 Delta_f)]``.  The mainlobe of a rectangular
    (uniform) spectrum of bandwidth ``B = K Delta_f`` has its first null at
    ``|tau| = 1 / B``.

    Attributes
    ----------
    oversampling_factor:
        Number of delay-grid samples per mainlobe scale ``1 / B``.  The grid
        therefore has ``K * oversampling_factor`` points over one period.
        Used by Stage 2 metric evaluation.
    mainlobe_exclusion_factor:
        Half-width of the excluded mainlobe region in units of ``1 / B``.  ``1.0``
        excludes exactly up to the first null of the uniform-spectrum response.
    optimization_oversampling_factor:
        Moderately dense grid used to *build* sampled-PSL SOC constraints
        (Stage 2.5).  Independent of ``oversampling_factor``.
    validation_oversampling_factor:
        Denser independent grid used to *verify* PSL after optimisation
        (typically 4x–8x the optimisation grid).
    """

    oversampling_factor: int = 16
    mainlobe_exclusion_factor: float = 1.0
    optimization_oversampling_factor: int = 4
    validation_oversampling_factor: int = 32

    def __post_init__(self) -> None:
        if self.oversampling_factor < 2:
            raise ValueError("oversampling_factor must be at least 2")
        if self.mainlobe_exclusion_factor <= 0.0:
            raise ValueError("mainlobe_exclusion_factor must be positive")
        if self.optimization_oversampling_factor < 2:
            raise ValueError("optimization_oversampling_factor must be at least 2")
        if self.validation_oversampling_factor < self.optimization_oversampling_factor:
            raise ValueError("validation_oversampling_factor must be >= optimization_oversampling_factor")


@dataclass(frozen=True)
class OptimizationConfig:
    """Static convex-optimisation settings.

    Attributes
    ----------
    solver_preference:
        Ordered CVXPY solver names; the first that returns an accepted status
        is used (``CLARABEL`` preferred, ``SCS`` fallback).
    feasibility_abs_tol:
        Absolute tolerance used when verifying constraints on power quantities
        [W] and on dimensionless quantities.
    feasibility_rel_tol:
        Relative tolerance (fraction of the constraint bound) used when
        verifying constraints.
    dinkelbach_max_iterations:
        Iteration cap of the Dinkelbach loop.
    dinkelbach_rel_tolerance:
        Stop when ``|F(q)| <= tol * R_bps(P*)``.  The residual ``F(q)`` is in
        bit/s, so the tolerance is expressed relative to the achieved rate.
    rate_fraction_of_water_filling:
        ``R_min = fraction * C_WF`` used by the static experiments, where
        ``C_WF`` is the water-filling sum spectral efficiency at ``P_total``.
    sensing_fraction_of_maximum:
        ``Gamma_s = fraction * S_max`` where ``S_max`` is the largest surrogate
        value attainable under the peak and total power limits.
    """

    solver_preference: tuple[str, ...] = ("CLARABEL", "SCS")
    feasibility_abs_tol: float = 1e-9
    feasibility_rel_tol: float = 1e-6
    dinkelbach_max_iterations: int = 50
    dinkelbach_rel_tolerance: float = 1e-8
    rate_fraction_of_water_filling: float = 0.9
    sensing_fraction_of_maximum: float = 0.6

    def __post_init__(self) -> None:
        if not self.solver_preference:
            raise ValueError("solver_preference must contain at least one solver")
        if self.feasibility_abs_tol <= 0.0 or self.feasibility_rel_tol < 0.0:
            raise ValueError("feasibility tolerances must be positive")
        if self.dinkelbach_max_iterations < 1:
            raise ValueError("dinkelbach_max_iterations must be at least 1")
        if self.dinkelbach_rel_tolerance <= 0.0:
            raise ValueError("dinkelbach_rel_tolerance must be positive")
        if not 0.0 <= self.rate_fraction_of_water_filling <= 1.0:
            raise ValueError("rate_fraction_of_water_filling must lie in [0, 1]")
        if not 0.0 <= self.sensing_fraction_of_maximum <= 1.0:
            raise ValueError("sensing_fraction_of_maximum must lie in [0, 1]")


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
    ambiguity: AmbiguityConfig = field(default_factory=AmbiguityConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def default_config() -> DefaultConfig:
    """Return the default configuration used throughout the experiments."""
    return DefaultConfig()

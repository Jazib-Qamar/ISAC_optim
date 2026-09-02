"""Per-realisation bundle of the physical parameters shared by every metric and optimiser.

An :class:`ISACSystem` fixes one channel realisation together with the OFDM,
sensing, energy and ambiguity parameters so that *all* baselines and optimisers
evaluate exactly the same physical model.  It contains no decision variables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from configs.default import DefaultConfig
from isac.sensing.frequencies import centered_subcarrier_frequencies, sensing_weights


@dataclass(frozen=True)
class ISACSystem:
    """Fixed physical parameters of one OFDM-ISAC scenario.

    Attributes
    ----------
    channel_gain:
        Communication channel power gain ``|h_k|^2`` (dimensionless), shape ``(K,)``.
    noise_power_w:
        Noise power per subcarrier ``sigma_n^2 = N0 Delta_f`` [W] (shared by the
        communication and sensing receivers).
    frequencies_hz:
        Centred baseband subcarrier frequencies ``f_k`` [Hz], shape ``(K,)``.
    sensing_weights:
        Surrogate weights ``w_k`` (``f_k^2`` [Hz^2] or normalised index), shape ``(K,)``.
    subcarrier_spacing_hz:
        ``Delta_f`` [Hz].
    peak_power_w:
        Per-subcarrier peak spectral power cap ``P_peak`` [W] (a PSL *proxy*).
    total_power_w:
        Total transmit power budget ``P_total`` / ``P_max`` [W].
    circuit_power_w, pa_efficiency:
        Power-model parameters ``P_circuit`` [W] and ``eta_PA``.
    reflection_coefficient, num_symbols, known_amplitude:
        Sensing FIM parameters (see :mod:`isac.sensing.fim`).
    delay_grid_s:
        Delay grid [s] for the ambiguity evaluation, shape ``(M,)``.
    mainlobe_exclusion_s:
        Half-width [s] of the mainlobe region excluded from sidelobe search.
    """

    channel_gain: NDArray[np.float64]
    noise_power_w: float
    frequencies_hz: NDArray[np.float64]
    sensing_weights: NDArray[np.float64]
    subcarrier_spacing_hz: float
    peak_power_w: float
    total_power_w: float
    circuit_power_w: float
    pa_efficiency: float
    reflection_coefficient: complex
    num_symbols: int
    known_amplitude: bool
    delay_grid_s: NDArray[np.float64]
    mainlobe_exclusion_s: float

    def __post_init__(self) -> None:
        gain = np.asarray(self.channel_gain, dtype=np.float64)
        if gain.ndim != 1 or gain.size < 1:
            raise ValueError("channel_gain must be a non-empty 1-D array")
        if np.any(gain < 0.0):
            raise ValueError("channel_gain must be non-negative")
        if self.frequencies_hz.shape != gain.shape or self.sensing_weights.shape != gain.shape:
            raise ValueError("channel_gain, frequencies_hz and sensing_weights must share one shape")
        if self.noise_power_w <= 0.0:
            raise ValueError("noise_power_w must be positive")
        if self.peak_power_w <= 0.0 or self.total_power_w <= 0.0:
            raise ValueError("peak_power_w and total_power_w must be positive")
        if self.subcarrier_spacing_hz <= 0.0:
            raise ValueError("subcarrier_spacing_hz must be positive")
        if self.mainlobe_exclusion_s <= 0.0:
            raise ValueError("mainlobe_exclusion_s must be positive")
        object.__setattr__(self, "channel_gain", gain)

    @property
    def num_subcarriers(self) -> int:
        """Number of subcarriers ``K``."""
        return int(self.channel_gain.size)

    @property
    def bandwidth_hz(self) -> float:
        """Occupied bandwidth ``K Delta_f`` [Hz]."""
        return self.num_subcarriers * self.subcarrier_spacing_hz

    @property
    def gain_to_noise_per_w(self) -> NDArray[np.float64]:
        """``alpha_k = |h_k|^2 / sigma_n^2`` [1/W]; ``SNR_k = alpha_k P_k``."""
        return self.channel_gain / self.noise_power_w

    @classmethod
    def from_config(
        cls,
        cfg: DefaultConfig,
        channel_gain: ArrayLike,
        total_power_w: float | None = None,
    ) -> "ISACSystem":
        """Build the bundle from a :class:`DefaultConfig` and one channel realisation.

        Parameters
        ----------
        total_power_w:
            Optional override of the power budget (used by power sweeps).  The
            peak cap is *not* rescaled: it stays ``cfg.ofdm.peak_power_w``.
        """
        from isac.sensing.ambiguity import default_delay_grid, default_mainlobe_exclusion_s

        gain = np.asarray(channel_gain, dtype=np.float64)
        ofdm = cfg.ofdm
        k = gain.size
        if k != ofdm.num_subcarriers:
            raise ValueError(
                f"channel_gain has {k} entries but the configuration has {ofdm.num_subcarriers} subcarriers"
            )
        bandwidth = ofdm.bandwidth_hz
        return cls(
            channel_gain=gain,
            noise_power_w=ofdm.noise_power_per_subcarrier_w,
            frequencies_hz=centered_subcarrier_frequencies(k, ofdm.subcarrier_spacing_hz),
            sensing_weights=sensing_weights(k, ofdm.subcarrier_spacing_hz, cfg.sensing.surrogate_weighting),
            subcarrier_spacing_hz=ofdm.subcarrier_spacing_hz,
            peak_power_w=ofdm.peak_power_w,
            total_power_w=ofdm.total_power_w if total_power_w is None else float(total_power_w),
            circuit_power_w=cfg.energy.circuit_power_w,
            pa_efficiency=cfg.energy.pa_efficiency,
            reflection_coefficient=cfg.sensing.reflection_coefficient,
            num_symbols=cfg.sensing.num_symbols,
            known_amplitude=cfg.sensing.known_amplitude,
            delay_grid_s=default_delay_grid(ofdm.subcarrier_spacing_hz, k, cfg.ambiguity.oversampling_factor),
            mainlobe_exclusion_s=default_mainlobe_exclusion_s(bandwidth, cfg.ambiguity.mainlobe_exclusion_factor),
        )

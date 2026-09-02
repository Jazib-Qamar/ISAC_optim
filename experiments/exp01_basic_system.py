"""Experiment 01 - basic OFDM-ISAC system sanity check.

Steps
-----
1. Draw one frequency-selective Rayleigh channel (K = 64 subcarriers).
2. Build the uniform power allocation.
3. Build the communication-only water-filling allocation (same budget).
4. Evaluate for both allocations:
   - sum spectral efficiency and achievable rate,
   - delay Fisher information and delay/range CRB,
   - linear sensing surrogate S(P) = sum_k w_k P_k,
   - transmit power, system power and energy efficiency.
5. Print a comparison table, save it to CSV, and plot
   - channel power gain vs subcarrier index,
   - uniform vs water-filling power allocation (with the water level).

Usage
-----
    python experiments/exp01_basic_system.py [--seed 0] [--output-dir results/exp01] [--show]

No result is hard-coded: every number is computed from the drawn channel.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from configs.default import DefaultConfig, SimulationConfig, default_config
from isac.channels.rayleigh import rayleigh_channel
from isac.communication.rate import (
    achievable_rate_bps,
    channel_gain,
    spectral_efficiency,
    subcarrier_snr,
)
from isac.communication.water_filling import WaterFillingResult, uniform_power, water_filling
from isac.energy.power_model import energy_efficiency, system_power, tx_power
from isac.sensing.crb import delay_crb_summary
from isac.sensing.fim import sensing_information_surrogate
from isac.sensing.frequencies import centered_subcarrier_frequencies, sensing_weights


def evaluate_allocation(
    name: str,
    power_w: np.ndarray,
    gain: np.ndarray,
    frequencies_hz: np.ndarray,
    weights: np.ndarray,
    cfg: DefaultConfig,
) -> dict[str, float | str]:
    """Compute every communication, sensing and energy metric for one allocation."""
    noise_w = cfg.ofdm.noise_power_per_subcarrier_w
    se = spectral_efficiency(power_w, gain, noise_w)
    rate_bps = achievable_rate_bps(power_w, gain, noise_w, cfg.ofdm.subcarrier_spacing_hz)
    p_tx = tx_power(power_w)
    p_sys = system_power(power_w, cfg.energy.circuit_power_w, cfg.energy.pa_efficiency)
    crb = delay_crb_summary(
        power_w,
        frequencies_hz,
        cfg.sensing.reflection_coefficient,
        noise_w,
        num_symbols=cfg.sensing.num_symbols,
        known_amplitude=cfg.sensing.known_amplitude,
    )
    return {
        "allocation": name,
        "spectral_efficiency_bit_per_s_per_hz": se,
        "rate_mbps": rate_bps / 1e6,
        "tx_power_w": p_tx,
        "system_power_w": p_sys,
        "energy_efficiency_mbit_per_j": energy_efficiency(rate_bps, p_sys) / 1e6,
        "fisher_information_per_s2": crb.fisher_information,
        "delay_crb_s2": crb.delay_crb_s2,
        "delay_rmse_bound_ns": crb.delay_rmse_bound_s * 1e9,
        "range_rmse_bound_m": crb.range_rmse_bound_m,
        "sensing_surrogate": sensing_information_surrogate(power_w, weights),
        "peak_power_w": float(np.max(power_w)),
        "num_active_subcarriers": int(np.count_nonzero(power_w > 0.0)),
    }


def print_config(cfg: DefaultConfig) -> None:
    """Print the configuration used by the run."""
    print("=" * 72)
    print("Configuration")
    print("=" * 72)
    for section_name in ("ofdm", "channel", "sensing", "energy", "simulation"):
        section = getattr(cfg, section_name)
        print(f"[{section_name}]")
        for field in dataclasses.fields(section):
            print(f"  {field.name:28s} = {getattr(section, field.name)}")
    print(f"  derived noise PSD            = {cfg.ofdm.noise_psd_w_per_hz:.3e} W/Hz")
    print(f"  derived noise / subcarrier   = {cfg.ofdm.noise_power_per_subcarrier_w:.3e} W")
    print(f"  derived P_peak               = {cfg.ofdm.peak_power_w:.4f} W")
    print(f"  derived round-trip delay     = {cfg.sensing.round_trip_delay_s * 1e9:.3f} ns")
    print()


def plot_channel_gain(gain: np.ndarray, snr_uniform: np.ndarray, out: Path) -> None:
    """Plot |h_k|^2 (dB) and the corresponding uniform-power SNR per subcarrier."""
    import matplotlib.pyplot as plt

    k = np.arange(gain.size)
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(k, 10 * np.log10(gain), "o-", color="tab:blue", label=r"channel gain $|h_k|^2$")
    ax1.set_xlabel("subcarrier index k")
    ax1.set_ylabel("channel power gain [dB]", color="tab:blue")
    ax1.grid(True, which="both", alpha=0.4)
    ax2 = ax1.twinx()
    ax2.plot(k, 10 * np.log10(snr_uniform), "s--", color="tab:red", label="SNR with uniform power")
    ax2.set_ylabel("SNR [dB]", color="tab:red")
    lines = ax1.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="best")
    ax1.set_title("Frequency-selective Rayleigh channel realisation")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_power_allocations(
    uniform_w: np.ndarray,
    wf: WaterFillingResult,
    out: Path,
) -> None:
    """Plot uniform vs water-filling powers together with the water-filling floor and level."""
    import matplotlib.pyplot as plt

    k = np.arange(uniform_w.size)
    width = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(k - width / 2, uniform_w * 1e3, width, label="uniform power", color="tab:gray")
    ax.bar(k + width / 2, wf.power_w * 1e3, width, label="water-filling power", color="tab:green")
    floor = np.where(np.isfinite(wf.inverse_gain_w), wf.inverse_gain_w, np.nan)
    ax.plot(k, floor * 1e3, "k.", label=r"noise-to-gain floor $1/\alpha_k$")
    ax.axhline(wf.water_level_w * 1e3, color="tab:blue", ls="--", label=r"water level $\mu$")
    ax.set_xlabel("subcarrier index k")
    ax.set_ylabel("power [mW]")
    ax.set_title("Uniform vs communication water-filling power allocation")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(cfg: DefaultConfig, output_dir: Path, show: bool) -> pd.DataFrame:
    """Execute the experiment and return the comparison table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.simulation.seed)
    ofdm = cfg.ofdm

    h = rayleigh_channel(ofdm.num_subcarriers, rng, mean_gain=cfg.channel.path_gain)
    gain = channel_gain(h)
    freqs = centered_subcarrier_frequencies(ofdm.num_subcarriers, ofdm.subcarrier_spacing_hz)
    weights = sensing_weights(ofdm.num_subcarriers, ofdm.subcarrier_spacing_hz, cfg.sensing.surrogate_weighting)
    noise_w = ofdm.noise_power_per_subcarrier_w

    p_uniform = uniform_power(ofdm.num_subcarriers, ofdm.total_power_w, ofdm.peak_power_w)
    wf = water_filling(gain, noise_w, ofdm.total_power_w, peak_power_w=ofdm.peak_power_w)

    rows = [
        evaluate_allocation("uniform", p_uniform, gain, freqs, weights, cfg),
        evaluate_allocation("water_filling", wf.power_w, gain, freqs, weights, cfg),
    ]
    table = pd.DataFrame(rows).set_index("allocation")

    print_config(cfg)
    print("=" * 72)
    print("Channel realisation")
    print("=" * 72)
    print(f"  mean |h_k|^2                 = {gain.mean():.3e}  (expected ~ {cfg.channel.path_gain:.3e})")
    print(f"  min / max |h_k|^2            = {gain.min():.3e} / {gain.max():.3e}")
    snr_uniform = subcarrier_snr(p_uniform, gain, noise_w)
    print(f"  uniform-power SNR range      = {10*np.log10(snr_uniform.min()):.2f} .. {10*np.log10(snr_uniform.max()):.2f} dB")
    print(f"  water level mu               = {wf.water_level_w*1e3:.4f} mW  ({wf.iterations} bisection iterations)")
    print(f"  sum of water-filling powers  = {wf.power_w.sum():.10f} W  (budget {ofdm.total_power_w} W)")
    print()
    print("=" * 72)
    print("Metric comparison (units in column names)")
    print("=" * 72)
    with pd.option_context("display.float_format", "{:.6g}".format, "display.width", 120):
        print(table.T.to_string())
    print()

    table.to_csv(output_dir / "exp01_metrics.csv")
    per_subcarrier = pd.DataFrame(
        {
            "subcarrier": np.arange(ofdm.num_subcarriers),
            "frequency_hz": freqs,
            "channel_gain": gain,
            "snr_uniform_db": 10 * np.log10(snr_uniform),
            "power_uniform_w": p_uniform,
            "power_water_filling_w": wf.power_w,
            "inverse_gain_w": wf.inverse_gain_w,
            "sensing_weight": weights,
        }
    )
    per_subcarrier.to_csv(output_dir / "exp01_per_subcarrier.csv", index=False)

    plot_channel_gain(gain, snr_uniform, output_dir / "exp01_channel_gain.png")
    plot_power_allocations(p_uniform, wf, output_dir / "exp01_power_allocation.png")
    print(f"Saved CSV tables and figures to {output_dir.resolve()}")

    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="random seed (default from config)")
    parser.add_argument("--output-dir", type=Path, default=None, help="directory for CSV and PNG outputs")
    parser.add_argument("--show", action="store_true", help="open interactive figure windows")
    args = parser.parse_args()

    cfg = default_config()
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, simulation=SimulationConfig(seed=args.seed, output_dir=cfg.simulation.output_dir))
    if not args.show:
        matplotlib.use("Agg")
    output_dir = args.output_dir or Path(cfg.simulation.output_dir) / "exp01"
    run(cfg, output_dir, show=args.show)


if __name__ == "__main__":
    main()

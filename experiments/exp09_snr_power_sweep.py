"""Experiment 2H - SNR sweep via path loss: why the water-filling gain was small in Stage 1.

The path loss is swept (default 100 ... 150 dB) at fixed ``P_total``; the
average per-subcarrier SNR of the uniform allocation is

    SNR_avg = G * (P_total / K) / (N0 Delta_f),   G = 10^(-PL/10).

For each path loss, ``num_channels`` Rayleigh realisations are drawn and the
methods uniform, water-filling, max-rate ISAC (Gamma_s = f_S S_max) and
Dinkelbach EE-ISAC (same Gamma_s) are evaluated.  Means over realisations are
plotted against the average SNR.  The default configuration is *not* changed.

Outputs (results/stage2/exp09_snr_power_sweep/): exp09_raw.csv, exp09_summary.csv, figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from isac.evaluation.baselines import METHOD_LABELS, StaticRequirements, outcomes_to_frame, run_static_baselines
from isac.evaluation.plots import line_plot
from isac.evaluation.reporting import experiment_dir, print_config, print_frame, print_header
from isac.evaluation.scenario import build_system, common_parser, config_from_args, with_path_loss

EXPERIMENT = "exp09_snr_power_sweep"
METHODS = ("uniform", "water_filling", "max_rate_isac", "dinkelbach_ee_isac")


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--path-loss-min-db", type=float, default=100.0)
    parser.add_argument("--path-loss-max-db", type=float, default=150.0)
    parser.add_argument("--num-points", type=int, default=11)
    parser.add_argument("--num-channels", type=int, default=20)
    args = parser.parse_args()
    base_cfg = config_from_args(args)
    out = args.output_dir or experiment_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    print_config(base_cfg)

    rng = np.random.default_rng(base_cfg.simulation.seed)
    path_losses = np.linspace(args.path_loss_min_db, args.path_loss_max_db, args.num_points)
    frames = []
    for pl in path_losses:
        cfg = with_path_loss(base_cfg, float(pl))
        snr_avg = cfg.channel.path_gain * (cfg.ofdm.total_power_w / cfg.ofdm.num_subcarriers) / cfg.ofdm.noise_power_per_subcarrier_w
        for c in range(args.num_channels):
            gain, system = build_system(cfg, rng)
            req = StaticRequirements.from_system(system, cfg.optimization.rate_fraction_of_water_filling,
                                                 cfg.optimization.sensing_fraction_of_maximum)
            frame = outcomes_to_frame(run_static_baselines(system, req, methods=METHODS,
                                                           solver_preference=cfg.optimization.solver_preference))
            frame.insert(0, "path_loss_db", pl)
            frame.insert(1, "avg_snr_db", 10.0 * np.log10(snr_avg))
            frame.insert(2, "channel", c)
            frame["water_filling_capacity_se"] = req.water_filling_capacity_se
            frames.append(frame)
        print(f"  path loss {pl:6.1f} dB  (avg SNR {10*np.log10(snr_avg):6.1f} dB): done")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(out / "exp09_raw.csv", index=False)

    ok = raw[raw["status"] == "ok"]
    summary = ok.groupby(["path_loss_db", "avg_snr_db", "method"], sort=True).agg(
        rate_bps=("rate_bps", "mean"), rate_se=("rate_spectral_efficiency", "mean"), tx_power_w=("tx_power_w", "mean"),
        ee_bit_per_j=("energy_efficiency_bit_per_j", "mean"), delay_crb_s2=("delay_crb_s2", "mean"),
        range_rmse_m=("range_rmse_bound_m", "mean"), psl_db=("psl_db", "mean"), count=("rate_bps", "size"),
    ).reset_index()
    # Water-filling gain over uniform, computed per realisation and then averaged.
    pivot = ok.pivot_table(index=["path_loss_db", "avg_snr_db", "channel"], columns="method", values="rate_spectral_efficiency")
    gain_frame = pivot.reset_index()
    gain_frame["wf_gain_percent"] = 100.0 * (gain_frame["water_filling"] / gain_frame["uniform"] - 1.0)
    gain_frame["wf_gain_abs_se"] = gain_frame["water_filling"] - gain_frame["uniform"]
    gain_summary = gain_frame.groupby(["path_loss_db", "avg_snr_db"]).agg(
        wf_gain_percent_mean=("wf_gain_percent", "mean"), wf_gain_percent_std=("wf_gain_percent", "std"),
        wf_gain_abs_se_mean=("wf_gain_abs_se", "mean"),
    ).reset_index()
    summary.to_csv(out / "exp09_summary.csv", index=False)
    gain_summary.to_csv(out / "exp09_water_filling_gain.csv", index=False)

    status_counts = raw.groupby(["path_loss_db", "method"])["status"].value_counts().unstack(fill_value=0).reset_index()
    print_header("Status counts per path loss and method")
    print_frame(status_counts)
    print_header("Mean metrics vs average SNR")
    print_frame(summary)
    print_header("Water-filling gain over uniform vs average SNR")
    print_frame(gain_summary)

    x = gain_summary["avg_snr_db"].to_numpy()
    def series(col: str, scale: float = 1.0) -> dict[str, np.ndarray]:
        return {METHOD_LABELS[m]: summary[summary["method"] == m].sort_values("avg_snr_db")[col].to_numpy() * scale for m in METHODS
                if (summary["method"] == m).any()}

    figures = [
        line_plot(x, series("rate_bps", 1e-6), "average per-subcarrier SNR of uniform allocation [dB]", "mean achievable rate [Mbit/s]",
                  "Rate vs average SNR", out / "exp09_rate_vs_snr.png"),
        line_plot(x, series("ee_bit_per_j", 1e-6), "average per-subcarrier SNR of uniform allocation [dB]", "mean energy efficiency [Mbit/J]",
                  "Energy efficiency vs average SNR", out / "exp09_ee_vs_snr.png"),
        line_plot(x, series("delay_crb_s2"), "average per-subcarrier SNR of uniform allocation [dB]", "mean delay CRB [s$^2$]",
                  "Delay CRB vs average SNR (sensing SNR fixed; CRB changes only via the allocation)", out / "exp09_crb_vs_snr.png", logy=True),
        line_plot(x, series("tx_power_w"), "average per-subcarrier SNR of uniform allocation [dB]", "mean transmit power [W]",
                  "Transmit power vs average SNR", out / "exp09_tx_power_vs_snr.png"),
        line_plot(x, {"water-filling gain over uniform (mean over channels)": gain_summary["wf_gain_percent_mean"].to_numpy()},
                  "average per-subcarrier SNR of uniform allocation [dB]", "rate gain [%]",
                  "Water-filling gain over uniform allocation vs average SNR", out / "exp09_wf_gain_vs_snr.png"),
    ]
    print_header("Saved outputs")
    for p in [out / "exp09_raw.csv", out / "exp09_summary.csv", out / "exp09_water_filling_gain.csv", *figures]:
        print(f"  {p}")


if __name__ == "__main__":
    main()

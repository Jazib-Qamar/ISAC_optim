"""Experiment 2A - single-channel static OFDM-ISAC comparison.

For one deterministic Rayleigh realisation, compare

    uniform | water-filling | edge-weighted | sensing-optimal |
    min-power ISAC | max-rate ISAC | Dinkelbach EE-ISAC

under identical requirements.  Requirements are constructed relative to
baseline performance (not chosen arbitrarily):

    R_min   = rate_fraction    * C_WF     (C_WF: water-filling sum SE at P_total, with peak cap)
    Gamma_s = sensing_fraction * S_max    (S_max: greedy edge-fill maximum of sum_k w_k P_k)

Outputs (results/stage2/exp02_static_isac/):
    exp02_metrics.csv, exp02_power_allocations.csv, exp02_requirements.csv,
    exp02_dinkelbach_history.csv and figures listed at the end of the run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from isac.evaluation.baselines import ALL_METHODS, METHOD_LABELS, StaticRequirements, outcomes_to_frame, run_static_baselines
from isac.evaluation.plots import bar_metric, plot_channel_gain, plot_power_allocations
from isac.evaluation.reporting import experiment_dir, print_config, print_frame, print_header
from isac.evaluation.scenario import build_system, common_parser, config_from_args
from isac.optimization.dinkelbach import solve_dinkelbach_ee

EXPERIMENT = "exp02_static_isac"


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--rate-fraction", type=float, default=None, help="R_min / C_WF (default from config)")
    parser.add_argument("--sensing-fraction", type=float, default=None, help="Gamma_s / S_max (default from config)")
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = args.output_dir or experiment_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    print_config(cfg)

    rng = np.random.default_rng(cfg.simulation.seed)
    gain, system = build_system(cfg, rng)
    rate_fraction = cfg.optimization.rate_fraction_of_water_filling if args.rate_fraction is None else args.rate_fraction
    sensing_fraction = cfg.optimization.sensing_fraction_of_maximum if args.sensing_fraction is None else args.sensing_fraction
    req = StaticRequirements.from_system(system, rate_fraction, sensing_fraction)

    print_header("Threshold construction")
    print(req.describe())
    print()

    outcomes = run_static_baselines(
        system, req, methods=ALL_METHODS,
        solver_preference=cfg.optimization.solver_preference,
        dinkelbach_rel_tolerance=cfg.optimization.dinkelbach_rel_tolerance,
        dinkelbach_max_iterations=cfg.optimization.dinkelbach_max_iterations,
        abs_tol=cfg.optimization.feasibility_abs_tol, rel_tol=cfg.optimization.feasibility_rel_tol,
    )
    table = outcomes_to_frame(outcomes)

    print_header("Single-channel comparison (all metrics from the shared NumPy model)")
    show = table[[
        "label", "status", "solver_status", "rate_spectral_efficiency", "rate_bps", "tx_power_w", "system_power_w",
        "energy_efficiency_bit_per_j", "sensing_surrogate", "delay_crb_s2", "range_rmse_bound_m", "psl_db", "isl_db",
        "peak_power_w", "meets_requirements", "violated_constraints", "solve_time_s",
    ]].copy()
    show["sensing_surrogate_over_S_max"] = show["sensing_surrogate"] / req.max_sensing_surrogate
    print_frame(show.set_index("label").T.reset_index().rename(columns={"index": "metric"}))

    print_header("Constraint slacks (>= 0 satisfied; ~0 active)")
    slack_cols = ["label"] + [c for c in table.columns if c.startswith("slack_")]
    print_frame(table[slack_cols])

    # Dinkelbach history for this channel (same requirements as the table).
    dk = solve_dinkelbach_ee(system, req.min_sensing_surrogate, rel_tolerance=cfg.optimization.dinkelbach_rel_tolerance,
                             max_iterations=cfg.optimization.dinkelbach_max_iterations,
                             solver_preference=cfg.optimization.solver_preference)
    print_header("Dinkelbach convergence history")
    print_frame(dk.history_frame()[["iteration", "q_bit_per_j", "rate_bps", "tx_power_w", "system_power_w",
                                    "energy_efficiency_bit_per_j", "residual_bps", "residual_relative", "solver_status"]])

    # --------------------------------------------------------------- save CSV
    table.to_csv(out / "exp02_metrics.csv", index=False)
    pd.DataFrame(req.as_dict(), index=[0]).to_csv(out / "exp02_requirements.csv", index=False)
    dk.history_frame().to_csv(out / "exp02_dinkelbach_history.csv", index=False)
    alloc = {"subcarrier": np.arange(system.num_subcarriers), "frequency_hz": system.frequencies_hz, "channel_gain": gain}
    for o in outcomes:
        if o.power_allocation is not None:
            alloc[f"power_{o.method}_w"] = o.power_allocation
    pd.DataFrame(alloc).to_csv(out / "exp02_power_allocations.csv", index=False)

    # --------------------------------------------------------------- figures
    figures = [plot_channel_gain(gain, out / "exp02_channel_gain.png")]
    allocations = {METHOD_LABELS[o.method]: o.power_allocation for o in outcomes if o.power_allocation is not None}
    figures.append(plot_power_allocations(allocations, out / "exp02_power_allocations.png", system.peak_power_w,
                                          title="Power allocation per subcarrier (all methods)"))
    optimizer_labels = {METHOD_LABELS[m] for m in ("water_filling", "min_power_isac", "max_rate_isac",
                                                   "dinkelbach_ee_isac", "dinkelbach_ee_rmin_isac")}
    opt_alloc = {k: v for k, v in allocations.items() if k in optimizer_labels}
    figures.append(plot_power_allocations(opt_alloc, out / "exp02_power_allocations_optimizers.png", system.peak_power_w,
                                          title="Power allocation: water-filling vs ISAC optimisers"))
    ok = table[table["status"] == "ok"]
    figures.append(bar_metric(ok, "rate_bps", "achievable rate [Mbit/s]", "Communication rate", out / "exp02_rate.png", scale=1e-6))
    figures.append(bar_metric(ok, "tx_power_w", "transmit power [W]", "Transmit power", out / "exp02_tx_power.png"))
    figures.append(bar_metric(ok, "system_power_w", "system power [W]", "System power  P_sys = P_c + P_tx / eta_PA",
                              out / "exp02_system_power.png"))
    figures.append(bar_metric(ok, "energy_efficiency_bit_per_j", "energy efficiency [Mbit/J]", "Energy efficiency",
                              out / "exp02_energy_efficiency.png", scale=1e-6))
    figures.append(bar_metric(ok, "range_rmse_bound_m", "range RMSE bound sqrt(CRB_R) [m]", "Ranging accuracy bound",
                              out / "exp02_range_rmse.png"))
    figures.append(bar_metric(ok, "delay_crb_s2", "delay CRB [s^2]", "Delay CRB (log scale)", out / "exp02_delay_crb.png", log=True))
    figures.append(bar_metric(ok, "psl_db", "actual PSL [dB]", "Actual delay-domain peak sidelobe level", out / "exp02_psl.png"))

    print_header("Saved outputs")
    for p in [out / "exp02_metrics.csv", out / "exp02_power_allocations.csv", out / "exp02_requirements.csv",
              out / "exp02_dinkelbach_history.csv", *figures]:
        print(f"  {p}")


if __name__ == "__main__":
    main()

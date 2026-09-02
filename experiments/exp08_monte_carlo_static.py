"""Experiment 2G - Monte Carlo validation of the static baselines.

For ``N`` independent Rayleigh realisations (default 100, configurable), every
method is evaluated under requirements constructed per realisation with the
same rule (``R_min = f_R C_WF``, ``Gamma_s = f_S S_max``).  Because ``S_max`` does
not depend on the channel, ``Gamma_s`` is identical across realisations; ``R_min``
tracks the realisation's own water-filling capacity.

Reported per method: mean, std, median, 5th and 95th percentile of rate, tx
power, system power, EE, CRB / range RMSE, PSL, plus the infeasibility rate
(optimiser reports an empty feasible set), constraint-violation rate (returned
allocation fails the independent check), and solver-failure rate.  Nothing is
discarded: infeasible and failed cases are rows in the raw CSV.

Outputs (results/stage2/exp08_monte_carlo_static/): exp08_raw.csv, exp08_summary.csv,
exp08_rates.csv, figures.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from isac.evaluation.baselines import ALL_METHODS, METHOD_LABELS, StaticRequirements, outcomes_to_frame, run_static_baselines
from isac.evaluation.plots import boxplot_by_method
from isac.evaluation.reporting import experiment_dir, print_config, print_frame, print_header, summarize_by_method
from isac.evaluation.scenario import build_system, common_parser, config_from_args

EXPERIMENT = "exp08_monte_carlo_static"
METRICS = ["rate_bps", "rate_spectral_efficiency", "tx_power_w", "system_power_w", "energy_efficiency_bit_per_j",
           "delay_crb_s2", "range_rmse_bound_m", "psl_db", "isl_db", "solve_time_s"]


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--num-realizations", type=int, default=100)
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = args.output_dir or experiment_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    print_config(cfg)

    rng = np.random.default_rng(cfg.simulation.seed)
    frames = []
    start = time.perf_counter()
    for i in range(args.num_realizations):
        gain, system = build_system(cfg, rng)
        req = StaticRequirements.from_system(system, cfg.optimization.rate_fraction_of_water_filling,
                                             cfg.optimization.sensing_fraction_of_maximum)
        outcomes = run_static_baselines(
            system, req, methods=ALL_METHODS, solver_preference=cfg.optimization.solver_preference,
            dinkelbach_rel_tolerance=cfg.optimization.dinkelbach_rel_tolerance,
            dinkelbach_max_iterations=cfg.optimization.dinkelbach_max_iterations,
            abs_tol=cfg.optimization.feasibility_abs_tol, rel_tol=cfg.optimization.feasibility_rel_tol,
        )
        frame = outcomes_to_frame(outcomes)
        frame.insert(0, "realization", i)
        frame["min_rate_se"] = req.min_rate_se
        frame["gamma_s"] = req.min_sensing_surrogate
        frame["water_filling_capacity_se"] = req.water_filling_capacity_se
        frame["mean_channel_gain"] = gain.mean()
        frames.append(frame)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.num_realizations} realisations done ({time.perf_counter() - start:.1f} s)")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(out / "exp08_raw.csv", index=False)

    summary = summarize_by_method(raw, METRICS)
    summary["label"] = summary["method"].map(METHOD_LABELS)
    summary.to_csv(out / "exp08_summary.csv", index=False)

    rate_rows = []
    for method, sub in raw.groupby("method", sort=False):
        n = len(sub)
        ok = sub[sub["status"] == "ok"]
        rate_rows.append({
            "method": method, "label": METHOD_LABELS[method], "realizations": n,
            "infeasibility_rate": float((sub["status"] == "infeasible").mean()),
            "solver_failure_rate": float((sub["status"] == "solver_failure").mean()),
            "constraint_violation_rate": float((~ok["meets_requirements"].astype(bool)).sum() / n),
            "inaccurate_solve_rate": float(ok["solver_inaccurate"].fillna(False).astype(bool).mean()) if "solver_inaccurate" in ok else 0.0,
            "mean_dinkelbach_iterations": float(ok["dinkelbach_iterations"].mean()) if "dinkelbach_iterations" in ok and ok["dinkelbach_iterations"].notna().any() else np.nan,
        })
    rates = pd.DataFrame(rate_rows)
    rates.to_csv(out / "exp08_rates.csv", index=False)

    print_header(f"Monte Carlo summary over {args.num_realizations} realisations (status == ok rows)")
    for metric in ["rate_bps", "tx_power_w", "system_power_w", "energy_efficiency_bit_per_j", "range_rmse_bound_m", "delay_crb_s2", "psl_db"]:
        print(f"--- {metric}")
        print_frame(summary[summary["metric"] == metric][["label", "mean", "std", "median", "p05", "p95", "count"]])
    print_header("Infeasibility / violation / solver-failure rates")
    print("(heuristics ignore the requirements; their 'violation rate' is the fraction of realisations in which the")
    print(" fixed allocation happens NOT to meet R_min and Gamma_s.  Dinkelbach EE-ISAC without R_min is not required to")
    print(" meet R_min; its violation rate is reported for information only.)")
    print_frame(rates)

    ok = raw[raw["status"] == "ok"].copy()
    figures = [
        boxplot_by_method(ok, "rate_bps", "achievable rate [Mbit/s]", "Rate distribution", out / "exp08_rate_box.png", scale=1e-6),
        boxplot_by_method(ok, "tx_power_w", "transmit power [W]", "Transmit power distribution", out / "exp08_tx_power_box.png"),
        boxplot_by_method(ok, "energy_efficiency_bit_per_j", "energy efficiency [Mbit/J]", "Energy-efficiency distribution",
                          out / "exp08_ee_box.png", scale=1e-6),
        boxplot_by_method(ok, "range_rmse_bound_m", "range RMSE bound [m]", "Ranging-accuracy bound distribution", out / "exp08_range_rmse_box.png"),
        boxplot_by_method(ok, "psl_db", "actual PSL [dB]", "Actual PSL distribution", out / "exp08_psl_box.png"),
        boxplot_by_method(ok, "solve_time_s", "solve time [s]", "Runtime distribution (log scale)", out / "exp08_runtime_box.png", log=True),
    ]
    print_header("Saved outputs")
    for p in [out / "exp08_raw.csv", out / "exp08_summary.csv", out / "exp08_rates.csv", *figures]:
        print(f"  {p}")


if __name__ == "__main__":
    main()

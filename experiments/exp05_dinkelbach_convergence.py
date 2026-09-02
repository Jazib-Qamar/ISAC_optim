"""Experiment 2D - Dinkelbach convergence on representative scenarios.

Scenarios (same seeded channel):
    loose sensing (Gamma_s = 0.2 S_max), default (0.6 S_max), strict (0.95 S_max),
    default + R_min, and default with a 10x larger power budget.

For each scenario the per-iteration residual |F(q)|, parameter q and achieved EE
are recorded.  The residual must tend to zero (F(q*) = 0); q converges to the
achieved EE.

Outputs (results/stage2/exp05_dinkelbach_convergence/): exp05_histories.csv, exp05_summary.csv, figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from isac.evaluation.baselines import StaticRequirements
from isac.evaluation.plots import save_figure
from isac.evaluation.reporting import experiment_dir, print_config, print_frame, print_header
from isac.evaluation.scenario import build_system, common_parser, config_from_args
from isac.optimization.dinkelbach import solve_dinkelbach_ee

EXPERIMENT = "exp05_dinkelbach_convergence"


def main() -> None:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = args.output_dir or experiment_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    print_config(cfg)

    rng = np.random.default_rng(cfg.simulation.seed)
    _, system = build_system(cfg, rng)
    req = StaticRequirements.from_system(system, cfg.optimization.rate_fraction_of_water_filling,
                                         cfg.optimization.sensing_fraction_of_maximum)
    print(req.describe(), "\n")

    scenarios = {
        "loose sensing (0.2 S_max)": dict(min_sensing_surrogate=0.2 * req.max_sensing_surrogate),
        "default sensing (0.6 S_max)": dict(min_sensing_surrogate=req.min_sensing_surrogate),
        "strict sensing (0.95 S_max)": dict(min_sensing_surrogate=0.95 * req.max_sensing_surrogate),
        "default + R_min (0.9 C_WF)": dict(min_sensing_surrogate=req.min_sensing_surrogate, min_rate_se=req.min_rate_se),
        "default, P_max x10": dict(min_sensing_surrogate=req.min_sensing_surrogate, max_power_w=10.0 * system.total_power_w),
    }

    histories = []
    summary = []
    for name, kwargs in scenarios.items():
        dk = solve_dinkelbach_ee(system, rel_tolerance=1e-10, max_iterations=cfg.optimization.dinkelbach_max_iterations,
                                 solver_preference=cfg.optimization.solver_preference, **kwargs)
        hist = dk.history_frame()
        hist.insert(0, "scenario", name)
        histories.append(hist)
        summary.append({
            "scenario": name, "iterations": dk.num_iterations, "converged": dk.converged,
            "q_final_bit_per_j": dk.q_final_bit_per_j, "achieved_ee_bit_per_j": dk.result.energy_efficiency_bit_per_j,
            "final_residual_bps": dk.final_residual_bps, "final_residual_relative": dk.history[-1].residual_relative,
            "tx_power_w": dk.result.tx_power_w, "rate_bps": dk.result.rate_bps, "feasible": dk.result.feasible,
            "solver_status": dk.result.solver_status,
        })
        print_header(f"Scenario: {name}")
        print_frame(hist[["iteration", "q_bit_per_j", "rate_bps", "tx_power_w", "energy_efficiency_bit_per_j",
                          "residual_bps", "residual_relative", "solver_status"]])

    hist_all = pd.concat(histories, ignore_index=True)
    summary_frame = pd.DataFrame(summary)
    print_header("Summary")
    print_frame(summary_frame)
    print("Check F(q) -> 0: max final |F(q)| over scenarios = "
          f"{summary_frame['final_residual_bps'].abs().max():.3e} bit/s "
          f"(relative {summary_frame['final_residual_relative'].max():.3e}); q_final == achieved EE: "
          f"{bool(np.allclose(summary_frame['q_final_bit_per_j'], summary_frame['achieved_ee_bit_per_j'], rtol=1e-8))}\n")
    hist_all.to_csv(out / "exp05_histories.csv", index=False)
    summary_frame.to_csv(out / "exp05_summary.csv", index=False)

    import matplotlib.pyplot as plt

    figures = []
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name, h in hist_all.groupby("scenario", sort=False):
        ax.semilogy(h["iteration"], np.maximum(np.abs(h["residual_bps"]), 1e-12), marker="o", label=name)
    ax.set_xlabel("Dinkelbach iteration n (count)")
    ax.set_ylabel("residual $|F(q_n)|$ [bit/s]")
    ax.set_title("Dinkelbach residual convergence ($F(q^*) = 0$)")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="best")
    figures.append(save_figure(fig, out / "exp05_residual_vs_iteration.png"))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name, h in hist_all.groupby("scenario", sort=False):
        ax.plot(h["iteration"], h["q_bit_per_j"] / 1e6, marker="o", label=name)
    ax.set_xlabel("Dinkelbach iteration n (count)")
    ax.set_ylabel("parameter $q_n$ [Mbit/J]")
    ax.set_title("Dinkelbach parameter trajectory")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best")
    figures.append(save_figure(fig, out / "exp05_q_vs_iteration.png"))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name, h in hist_all.groupby("scenario", sort=False):
        ax.plot(h["iteration"], h["energy_efficiency_bit_per_j"] / 1e6, marker="o", label=name)
    ax.set_xlabel("Dinkelbach iteration n (count)")
    ax.set_ylabel("achieved EE of $P_n$ [Mbit/J]")
    ax.set_title("Achieved energy efficiency per iteration")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best")
    figures.append(save_figure(fig, out / "exp05_ee_vs_iteration.png"))

    print_header("Saved outputs")
    for p in [out / "exp05_histories.csv", out / "exp05_summary.csv", *figures]:
        print(f"  {p}")


if __name__ == "__main__":
    main()

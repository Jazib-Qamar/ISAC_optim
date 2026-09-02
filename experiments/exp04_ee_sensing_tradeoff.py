"""Experiment 2C - energy-efficiency frontier vs sensing requirement (Dinkelbach sweep).

For one deterministic channel, ``Gamma_s`` is swept from 0 to ``0.999 * S_max``
and the Dinkelbach EE optimiser is run for each value (sensing constraint
only, no ``R_min``).  This shows the energy-efficiency cost of increasingly
strict sensing requirements together with the number of Dinkelbach iterations.

Outputs (results/stage2/exp04_ee_sensing_tradeoff/): exp04_ee_frontier.csv and figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from isac.evaluation.baselines import StaticRequirements
from isac.evaluation.plots import line_plot
from isac.evaluation.reporting import experiment_dir, print_config, print_frame, print_header
from isac.evaluation.scenario import build_system, common_parser, config_from_args
from isac.optimization.dinkelbach import solve_dinkelbach_ee
from isac.optimization.exceptions import InfeasibleProblemError, SolverFailureError
from isac.optimization.max_rate import solve_max_rate

EXPERIMENT = "exp04_ee_sensing_tradeoff"


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--num-points", type=int, default=25)
    parser.add_argument("--max-fraction", type=float, default=0.999)
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = args.output_dir or experiment_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    print_config(cfg)

    rng = np.random.default_rng(cfg.simulation.seed)
    gain, system = build_system(cfg, rng)
    req = StaticRequirements.from_system(system, cfg.optimization.rate_fraction_of_water_filling,
                                         cfg.optimization.sensing_fraction_of_maximum)
    print_header("Sweep definition")
    print(req.describe())
    fractions = np.linspace(0.0, args.max_fraction, args.num_points)
    print(f"  Gamma_s / S_max sweep: {args.num_points} points in [0, {args.max_fraction}]\n")

    rows = []
    for frac in fractions:
        gamma = frac * req.max_sensing_surrogate
        row = {"sensing_fraction": frac, "gamma_s": gamma}
        try:
            dk = solve_dinkelbach_ee(system, gamma, rel_tolerance=cfg.optimization.dinkelbach_rel_tolerance,
                                     max_iterations=cfg.optimization.dinkelbach_max_iterations,
                                     solver_preference=cfg.optimization.solver_preference,
                                     abs_tol=cfg.optimization.feasibility_abs_tol, rel_tol=cfg.optimization.feasibility_rel_tol)
            res = dk.result
            row.update({"status": "ok", "solver_status": res.solver_status, "meets_constraints": res.feasible,
                        "dinkelbach_iterations": dk.num_iterations, "dinkelbach_converged": dk.converged,
                        "q_final_bit_per_j": dk.q_final_bit_per_j, "final_residual_bps": dk.final_residual_bps,
                        "total_solve_time_s": res.extra["total_solve_time_s"]})
            row.update(res.metrics.as_dict())
            # Reference: rate-maximising solution under the same sensing constraint.
            mr = solve_max_rate(system, gamma, solver_preference=cfg.optimization.solver_preference)
            row.update({"max_rate_ee_bit_per_j": mr.energy_efficiency_bit_per_j, "max_rate_rate_se": mr.rate_spectral_efficiency,
                        "max_rate_tx_power_w": mr.tx_power_w})
        except InfeasibleProblemError as exc:
            row.update({"status": "infeasible", "solver_status": exc.status})
        except SolverFailureError as exc:
            row.update({"status": "solver_failure", "solver_status": "failed", "error": str(exc)})
        rows.append(row)
    table = pd.DataFrame(rows)

    print_header("EE frontier (Dinkelbach, sensing constraint only)")
    print_frame(table[["sensing_fraction", "status", "dinkelbach_iterations", "dinkelbach_converged", "final_residual_bps",
                       "energy_efficiency_bit_per_j", "max_rate_ee_bit_per_j", "rate_spectral_efficiency", "tx_power_w",
                       "system_power_w", "delay_crb_s2", "range_rmse_bound_m", "psl_db", "meets_constraints"]])
    ok = table[table["status"] == "ok"]
    print(f"feasible points: {len(ok)}/{len(table)};  all converged: {bool(ok['dinkelbach_converged'].all())};  "
          f"max |F(q)| at convergence: {ok['final_residual_bps'].abs().max():.3e} bit/s;  "
          f"iterations: min {ok['dinkelbach_iterations'].min()}, max {ok['dinkelbach_iterations'].max()}")
    print(f"EE non-increasing along the sweep (feasible set shrinks): "
          f"{bool(np.all(np.diff(ok['energy_efficiency_bit_per_j'].to_numpy()) <= 1e-6 * ok['energy_efficiency_bit_per_j'].iloc[0]))}")
    print()
    table.to_csv(out / "exp04_ee_frontier.csv", index=False)

    x = ok["sensing_fraction"].to_numpy()
    figures = [
        line_plot(ok["delay_crb_s2"].to_numpy(), {"Dinkelbach EE-ISAC": ok["energy_efficiency_bit_per_j"].to_numpy() / 1e6,
                                                 "max-rate ISAC (same $\\Gamma_s$)": ok["max_rate_ee_bit_per_j"].to_numpy() / 1e6},
                  "delay CRB [s$^2$]", "energy efficiency [Mbit/J]", "Energy efficiency vs delay CRB", out / "exp04_ee_vs_crb.png", logx=True),
        line_plot(x, {"Dinkelbach EE-ISAC": ok["energy_efficiency_bit_per_j"].to_numpy() / 1e6,
                      "max-rate ISAC (same $\\Gamma_s$)": ok["max_rate_ee_bit_per_j"].to_numpy() / 1e6},
                  "$\\Gamma_s / S_{max}$ (dimensionless)", "energy efficiency [Mbit/J]", "Energy efficiency vs sensing requirement",
                  out / "exp04_ee_vs_gamma.png"),
        line_plot(x, {"Dinkelbach EE-ISAC": ok["tx_power_w"].to_numpy(), "max-rate ISAC (same $\\Gamma_s$)": ok["max_rate_tx_power_w"].to_numpy()},
                  "$\\Gamma_s / S_{max}$ (dimensionless)", "transmit power [W]", "Transmit power vs sensing requirement",
                  out / "exp04_tx_power_vs_gamma.png", hlines={"$P_{max}$": system.total_power_w}),
        line_plot(x, {"Dinkelbach EE-ISAC": ok["rate_spectral_efficiency"].to_numpy(), "max-rate ISAC (same $\\Gamma_s$)": ok["max_rate_rate_se"].to_numpy()},
                  "$\\Gamma_s / S_{max}$ (dimensionless)", "sum spectral efficiency [bit/s/Hz]", "Rate vs sensing requirement",
                  out / "exp04_rate_vs_gamma.png", hlines={"$C_{WF}$": req.water_filling_capacity_se}),
        line_plot(x, {"Dinkelbach iterations": ok["dinkelbach_iterations"].to_numpy()},
                  "$\\Gamma_s / S_{max}$ (dimensionless)", "iterations to $|F(q)| \\leq$ tol (count)", "Dinkelbach iterations vs sensing requirement",
                  out / "exp04_iterations_vs_gamma.png"),
        line_plot(x, {"actual PSL": ok["psl_db"].to_numpy(), "ISL": ok["isl_db"].to_numpy()},
                  "$\\Gamma_s / S_{max}$ (dimensionless)", "level [dB]", "Sidelobe levels of the EE-optimal allocation",
                  out / "exp04_psl_isl_vs_gamma.png"),
    ]
    print_header("Saved outputs")
    for p in [out / "exp04_ee_frontier.csv", *figures]:
        print(f"  {p}")


if __name__ == "__main__":
    main()

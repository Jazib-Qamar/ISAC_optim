"""Experiment 2B - rate vs sensing Pareto frontier (max-rate ISAC sweep over Gamma_s).

For one deterministic channel, ``Gamma_s`` is swept from 0 (pure water-filling)
to ``0.999 * S_max`` (near the largest attainable surrogate).  For each value the
max-rate ISAC problem is solved and all metrics are recomputed with the shared
NumPy model.  Infeasible / failed solves are recorded, never discarded.

Outputs (results/stage2/exp03_rate_sensing_pareto/): exp03_pareto.csv,
exp03_power_allocations.csv and the figures listed at the end of the run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from isac.evaluation.baselines import StaticRequirements
from isac.evaluation.plots import heatmap, line_plot
from isac.evaluation.reporting import experiment_dir, print_config, print_frame, print_header
from isac.evaluation.scenario import build_system, common_parser, config_from_args
from isac.optimization.exceptions import InfeasibleProblemError, SolverFailureError
from isac.optimization.max_rate import solve_max_rate

EXPERIMENT = "exp03_rate_sensing_pareto"


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--num-points", type=int, default=25, help="number of Gamma_s values")
    parser.add_argument("--max-fraction", type=float, default=0.999, help="largest Gamma_s / S_max")
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
    print(f"  Gamma_s / S_max sweep: {args.num_points} points in [0, {args.max_fraction}]")
    print()

    rows = []
    allocations = []
    for frac in fractions:
        gamma = frac * req.max_sensing_surrogate
        row = {"sensing_fraction": frac, "gamma_s": gamma}
        try:
            res = solve_max_rate(system, gamma, solver_preference=cfg.optimization.solver_preference,
                                 abs_tol=cfg.optimization.feasibility_abs_tol, rel_tol=cfg.optimization.feasibility_rel_tol)
            row.update({"status": "ok", "solver_status": res.solver_status, "solver_name": res.solver_name,
                        "solve_time_s": res.solve_time_s, "meets_constraints": res.feasible,
                        "max_violation": res.feasibility.max_violation})
            row.update(res.metrics.as_dict())
            allocations.append(res.power_allocation)
        except InfeasibleProblemError as exc:
            row.update({"status": "infeasible", "solver_status": exc.status, "solver_name": exc.solver_name})
            allocations.append(np.full(system.num_subcarriers, np.nan))
        except SolverFailureError as exc:
            row.update({"status": "solver_failure", "solver_status": "failed", "solver_name": "none", "error": str(exc)})
            allocations.append(np.full(system.num_subcarriers, np.nan))
        rows.append(row)
    table = pd.DataFrame(rows)
    table["rate_loss_vs_wf_percent"] = 100.0 * (1.0 - table["rate_spectral_efficiency"] / req.water_filling_capacity_se)

    print_header("Pareto sweep (max-rate ISAC)")
    print_frame(table[["sensing_fraction", "status", "solver_status", "rate_spectral_efficiency", "rate_loss_vs_wf_percent",
                       "tx_power_w", "energy_efficiency_bit_per_j", "delay_fim_per_s2", "delay_crb_s2",
                       "range_rmse_bound_m", "psl_db", "isl_db", "peak_power_w", "meets_constraints"]])
    ok = table[table["status"] == "ok"]
    print(f"feasible points: {len(ok)}/{len(table)};  infeasible: {(table['status'] == 'infeasible').sum()};  "
          f"solver failures: {(table['status'] == 'solver_failure').sum()}")
    rate_diffs = np.diff(ok["rate_spectral_efficiency"].to_numpy())
    print(f"rate non-increasing along the sweep (expected, feasible set shrinks): {bool(np.all(rate_diffs <= 1e-6 * ok['rate_spectral_efficiency'].iloc[0]))}")
    crb_diffs = np.diff(ok["delay_crb_s2"].to_numpy())
    print(f"CRB non-increasing along the sweep (up to solver tolerance 1e-6): "
          f"{bool(np.all(crb_diffs <= 1e-6 * ok['delay_crb_s2'].iloc[0]))} "
          "(with w_k = f_k^2 and the known-amplitude model, CRB = 1/(c S(P)) exactly, so this is expected;")
    print("   it is NOT guaranteed for the unknown-amplitude CRB or other weightings - see exp06)")
    active = ok[ok["sensing_surrogate"] <= ok["gamma_s"] * (1 + 1e-6)]
    print(f"sensing constraint active (S = Gamma_s) for Gamma_s / S_max >= {active['sensing_fraction'].min():.3f} "
          f"(water-filling already achieves S_WF / S_max = {ok['sensing_surrogate'].iloc[0] / req.max_sensing_surrogate:.3f})")
    print()

    table.to_csv(out / "exp03_pareto.csv", index=False)
    alloc_frame = pd.DataFrame(np.array(allocations).T, columns=[f"gamma_frac_{f:.4f}" for f in fractions])
    alloc_frame.insert(0, "subcarrier", np.arange(system.num_subcarriers))
    alloc_frame.to_csv(out / "exp03_power_allocations.csv", index=False)

    figures = []
    x_crb = ok["delay_crb_s2"].to_numpy()
    figures.append(line_plot(x_crb, {"max-rate ISAC frontier": ok["rate_spectral_efficiency"].to_numpy()},
                             "delay CRB [s$^2$]", "sum spectral efficiency [bit/s/Hz]",
                             "Rate vs delay CRB Pareto frontier (sweep over $\\Gamma_s$)", out / "exp03_rate_vs_crb.png",
                             logx=True, hlines={"water-filling capacity $C_{WF}$": req.water_filling_capacity_se}))
    figures.append(line_plot(ok["range_rmse_bound_m"].to_numpy(), {"max-rate ISAC frontier": ok["rate_spectral_efficiency"].to_numpy()},
                             "range RMSE bound [m]", "sum spectral efficiency [bit/s/Hz]",
                             "Rate vs range RMSE bound", out / "exp03_rate_vs_range_rmse.png",
                             hlines={"water-filling capacity $C_{WF}$": req.water_filling_capacity_se}))
    figures.append(line_plot(ok["sensing_surrogate"].to_numpy() / req.max_sensing_surrogate,
                             {"max-rate ISAC frontier": ok["rate_spectral_efficiency"].to_numpy()},
                             "achieved sensing surrogate $S(P)/S_{max}$ (dimensionless)", "sum spectral efficiency [bit/s/Hz]",
                             "Rate vs sensing-information surrogate", out / "exp03_rate_vs_surrogate.png",
                             hlines={"water-filling capacity $C_{WF}$": req.water_filling_capacity_se}))
    figures.append(line_plot(x_crb, {"max-rate ISAC frontier": ok["energy_efficiency_bit_per_j"].to_numpy() / 1e6},
                             "delay CRB [s$^2$]", "energy efficiency [Mbit/J]", "Energy efficiency vs delay CRB (max-rate solutions)",
                             out / "exp03_ee_vs_crb.png", logx=True))
    figures.append(line_plot(ok["sensing_fraction"].to_numpy(), {"actual PSL": ok["psl_db"].to_numpy(), "ISL": ok["isl_db"].to_numpy()},
                             "$\\Gamma_s / S_{max}$ (dimensionless)", "level [dB]", "Actual sidelobe levels along the frontier",
                             out / "exp03_psl_isl_vs_gamma.png"))
    figures.append(line_plot(ok["sensing_fraction"].to_numpy(), {"max$_k P_k$": ok["peak_power_w"].to_numpy() * 1e3},
                             "$\\Gamma_s / S_{max}$ (dimensionless)", "peak subcarrier power [mW]",
                             "Peak spectral power along the frontier (PSL proxy)", out / "exp03_peak_power_vs_gamma.png",
                             hlines={"$P_{peak}$ cap": system.peak_power_w * 1e3}))
    figures.append(heatmap(np.array(allocations).T * 1e3, fractions, "subcarrier index k", "$\\Gamma_s / S_{max}$ (dimensionless)",
                           "power $P_k$ [mW]", "Max-rate ISAC power allocation vs sensing requirement",
                           out / "exp03_power_heatmap.png"))

    print_header("Saved outputs")
    for p in [out / "exp03_pareto.csv", out / "exp03_power_allocations.csv", *figures]:
        print(f"  {p}")


if __name__ == "__main__":
    main()

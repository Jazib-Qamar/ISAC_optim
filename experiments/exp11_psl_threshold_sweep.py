"""Experiment 11 — sampled-PSL threshold sweep at a fixed unknown-amplitude FIM.

For one channel and a fixed ``Gamma_J = 0.45 J_max``, sweep ``PSL_max_db`` from
0 dB downward until the convex program is infeasible.  The feasible region is
reported; thresholds are not relaxed.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from isac.evaluation.plots import line_plot
from isac.evaluation.reporting import STAGE25_ROOT, experiment_dir, print_config, print_frame, print_header
from isac.evaluation.scenario import build_system, common_parser, config_from_args
from isac.evaluation.stage25 import allocation_row, failed_row, psl_grids
from isac.optimization.exceptions import InfeasibleProblemError, SolverFailureError
from isac.optimization.feasibility import max_unknown_amplitude_fim
from isac.optimization.max_rate import solve_max_rate
from isac.optimization.sensing_spec import UNKNOWN_AMPLITUDE_EXACT

EXPERIMENT = "exp11_psl_threshold_sweep"


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--fim-fraction", type=float, default=0.45)
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = args.output_dir or experiment_dir(EXPERIMENT, STAGE25_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    print_config(cfg)

    rng = np.random.default_rng(cfg.simulation.seed)
    _, system = build_system(cfg, rng)
    j_max, _ = max_unknown_amplitude_fim(system)
    gamma_j = args.fim_fraction * j_max
    delays, opt_grid, val_grid = psl_grids(
        system, cfg.ambiguity.optimization_oversampling_factor, cfg.ambiguity.validation_oversampling_factor
    )
    print_header("Fixed FIM requirement")
    print(f"  J_max = {j_max:.6e} 1/s^2")
    print(f"  Gamma_J = {args.fim_fraction:.2f} * J_max = {gamma_j:.6e} 1/s^2")
    print(f"  PSL SOC delays: {delays.size} (opt oversampling {cfg.ambiguity.optimization_oversampling_factor})")
    print()

    thresholds = np.arange(0.0, -25.5, -1.5)
    rows = []
    for psl_max in thresholds:
        start = time.perf_counter()
        try:
            res = solve_max_rate(
                system,
                sensing_model=UNKNOWN_AMPLITUDE_EXACT,
                min_unknown_fim=gamma_j,
                psl_max_db=float(psl_max),
                psl_delays_s=delays,
            )
            row = allocation_row(
                "exact_fim_psl", res.power_allocation, system,
                solve_time_s=res.solve_time_s, solver_status=res.solver_status,
                solver_inaccurate=res.solver_inaccurate,
                extra={"num_psl_soc_constraints": int(res.extra.get("num_psl_soc_constraints", delays.size))},
                psl_max_db=float(psl_max), opt_grid=opt_grid, val_grid=val_grid,
                opt_oversampling=cfg.ambiguity.optimization_oversampling_factor,
                val_oversampling=cfg.ambiguity.validation_oversampling_factor,
            )
        except InfeasibleProblemError as exc:
            row = failed_row("exact_fim_psl", "infeasible", str(exc), time.perf_counter() - start)
        except SolverFailureError as exc:
            row = failed_row("exact_fim_psl", "solver_failure", str(exc), time.perf_counter() - start)
        row["requested_psl_max_db"] = float(psl_max)
        row["gamma_j"] = gamma_j
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out / "exp11_psl_sweep.csv", index=False)
    print_header("PSL threshold sweep")
    cols = [c for c in (
        "requested_psl_max_db", "status", "rate_spectral_efficiency", "energy_efficiency_bit_per_j",
        "tx_power_w", "delay_crb_s2", "psl_db", "validation_grid_psl_db", "violation_margin_db",
        "num_psl_soc_constraints", "solve_time_s",
    ) if c in table.columns]
    print_frame(table[cols])

    ok = table[table["status"] == "ok"]
    infeas = table[table["status"] == "infeasible"]
    if not ok.empty:
        x = ok["requested_psl_max_db"].to_numpy()
        line_plot(x, {"rate": ok["rate_spectral_efficiency"].to_numpy()},
                  "requested sampled-grid PSL_max [dB]", "sum spectral efficiency [bit/s/Hz]",
                  "Rate vs sampled-PSL threshold (fixed unknown-amplitude FIM)",
                  out / "exp11_rate_vs_psl.png")
        line_plot(x, {"EE": ok["energy_efficiency_bit_per_j"].to_numpy() / 1e6},
                  "requested sampled-grid PSL_max [dB]", "energy efficiency [Mbit/J]",
                  "EE vs sampled-PSL threshold (fixed unknown-amplitude FIM)",
                  out / "exp11_ee_vs_psl.png")
        line_plot(x, {
            "optimisation-grid PSL": ok["optimization_grid_psl_db"].to_numpy() if "optimization_grid_psl_db" in ok else ok["psl_db"].to_numpy(),
            "validation-grid PSL": ok["validation_grid_psl_db"].to_numpy() if "validation_grid_psl_db" in ok else ok["psl_db"].to_numpy(),
        }, "requested sampled-grid PSL_max [dB]", "achieved PSL [dB]",
           "Achieved vs requested sampled-grid PSL (opt grid vs denser validation grid)",
           out / "exp11_achieved_psl.png")
        line_plot(x, {"solve time": ok["solve_time_s"].to_numpy()},
                  "requested sampled-grid PSL_max [dB]", "solver time [s]",
                  "Solve time vs PSL threshold", out / "exp11_solve_time.png")
    feasible_min = float(ok["requested_psl_max_db"].min()) if not ok.empty else float("nan")
    print(f"Feasible PSL_max region on this channel: [{feasible_min:.2f}, 0] dB"
          if not ok.empty else "No feasible PSL_max in the sweep.")
    if not infeas.empty:
        print(f"First infeasible PSL_max = {infeas['requested_psl_max_db'].max():.2f} dB")
    print(f"Saved outputs under {out}")


if __name__ == "__main__":
    main()

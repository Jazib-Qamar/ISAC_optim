"""Experiment 10 — CRB vs communication vs sampled-PSL tradeoff.

Sweep the unknown-amplitude FIM requirement and, at each level, solve

1. max-rate with the exact FIM constraint only
2. max-rate with exact FIM + a sampled-grid PSL cone family

The PSL threshold is the uniform-allocation validation-grid PSL (measured, not
assumed).  If that threshold makes a CRB level infeasible, the row is recorded
as infeasible — it is not relaxed.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from isac.communication.water_filling import uniform_power
from isac.evaluation.metrics import evaluate_allocation
from isac.evaluation.plots import allocation_heatmap, line_plot, plot_power_allocations, scatter_plot
from isac.evaluation.reporting import STAGE25_ROOT, experiment_dir, print_config, print_frame, print_header
from isac.evaluation.scenario import build_system, common_parser, config_from_args
from isac.evaluation.stage25 import allocation_row, failed_row, psl_grids
from isac.optimization.ambiguity_constraints import validation_delay_grid
from isac.optimization.exceptions import InfeasibleProblemError, SolverFailureError
from isac.optimization.feasibility import max_unknown_amplitude_fim
from isac.optimization.max_rate import solve_max_rate
from isac.optimization.sensing_spec import UNKNOWN_AMPLITUDE_EXACT
from isac.sensing.ambiguity import peak_sidelobe_level_db

EXPERIMENT = "exp10_crb_psl_tradeoff"


def _solve(system, method, gamma_j, psl_max_db, delays, val_oversampling):
    start = time.perf_counter()
    kwargs = dict(sensing_model=UNKNOWN_AMPLITUDE_EXACT, min_unknown_fim=gamma_j)
    if psl_max_db is not None:
        kwargs.update(psl_max_db=psl_max_db, psl_delays_s=delays)
    try:
        res = solve_max_rate(system, **kwargs)
        row = allocation_row(
            method, res.power_allocation, system,
            solve_time_s=res.solve_time_s, solver_status=res.solver_status,
            solver_inaccurate=res.solver_inaccurate,
            psl_max_db=psl_max_db,
            opt_grid=np.concatenate([[0.0], delays]) if (psl_max_db is not None and delays.size) else None,
            val_grid=None if psl_max_db is None else validation_delay_grid(system, val_oversampling),
            val_oversampling=val_oversampling,
        )
        return row, res.power_allocation
    except InfeasibleProblemError as exc:
        return failed_row(method, "infeasible", str(exc), time.perf_counter() - start), None
    except SolverFailureError as exc:
        return failed_row(method, "solver_failure", str(exc), time.perf_counter() - start), None


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--num-points", type=int, default=9)
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = args.output_dir or experiment_dir(EXPERIMENT, STAGE25_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    print_config(cfg)

    rng = np.random.default_rng(cfg.simulation.seed)
    _, system = build_system(cfg, rng)
    j_max, _ = max_unknown_amplitude_fim(system)
    delays, _, val_grid = psl_grids(
        system, cfg.ambiguity.optimization_oversampling_factor, cfg.ambiguity.validation_oversampling_factor
    )
    uniform = uniform_power(system.num_subcarriers, system.total_power_w, system.peak_power_w)
    psl_uniform = peak_sidelobe_level_db(uniform, system.frequencies_hz, val_grid, system.mainlobe_exclusion_s)
    print_header("Reference levels")
    print(f"  J_max unknown-amplitude FIM = {j_max:.6e} 1/s^2")
    print(f"  uniform-allocation validation-grid PSL = {psl_uniform:.3f} dB")
    print(f"  mainlobe exclusion = {system.mainlobe_exclusion_s:.6e} s")
    print()

    fractions = np.linspace(0.15, 0.85, args.num_points)
    rows: list[dict] = []
    alloc_fim: dict[float, np.ndarray] = {}
    alloc_psl: dict[float, np.ndarray] = {}
    val_os = cfg.ambiguity.validation_oversampling_factor
    for frac in fractions:
        gamma_j = float(frac) * j_max
        row, power = _solve(system, "exact_fim", gamma_j, None, delays, val_os)
        row["fim_fraction"] = float(frac)
        row["gamma_j"] = gamma_j
        row["psl_constrained"] = False
        rows.append(row)
        if power is not None:
            alloc_fim[float(frac)] = power

        row_p, power_p = _solve(system, "exact_fim_psl", gamma_j, psl_uniform, delays, val_os)
        row_p["fim_fraction"] = float(frac)
        row_p["gamma_j"] = gamma_j
        row_p["psl_constrained"] = True
        rows.append(row_p)
        if power_p is not None:
            alloc_psl[float(frac)] = power_p

    table = pd.DataFrame(rows)
    table.to_csv(out / "exp10_tradeoff.csv", index=False)
    print_header("CRB–PSL sweep (infeasible rows kept)")
    cols = [c for c in (
        "method", "status", "fim_fraction", "rate_spectral_efficiency", "tx_power_w",
        "energy_efficiency_bit_per_j", "delay_fim_unknown_amplitude_per_s2", "delay_crb_s2",
        "range_rmse_bound_m", "psl_db", "isl_db", "validation_grid_psl_db", "solve_time_s", "error",
    ) if c in table.columns]
    print_frame(table[cols])

    ok = table[table["status"] == "ok"]
    for method, sub in ok.groupby("method", sort=False):
        order = np.argsort(sub["fim_fraction"].to_numpy())
        frac = sub["fim_fraction"].to_numpy()[order]
        line_plot(frac, {"rate": sub["rate_spectral_efficiency"].to_numpy()[order]},
                  "FIM requirement as a fraction of J_max", "sum spectral efficiency [bit/s/Hz]",
                  f"Rate vs FIM requirement — {method}", out / f"exp10_rate_vs_frac_{method}.png")
        line_plot(frac, {"CRB": sub["delay_crb_s2"].to_numpy()[order]},
                  "FIM requirement as a fraction of J_max", "configured-model delay CRB [s^2]",
                  f"CRB vs FIM requirement — {method}", out / f"exp10_crb_vs_frac_{method}.png", logy=True)
        line_plot(frac, {"PSL": sub["psl_db"].to_numpy()[order]},
                  "FIM requirement as a fraction of J_max", "actual delay-domain PSL [dB]",
                  f"PSL vs FIM requirement — {method}", out / f"exp10_psl_vs_frac_{method}.png")
        line_plot(frac, {"EE": sub["energy_efficiency_bit_per_j"].to_numpy()[order] / 1e6},
                  "FIM requirement as a fraction of J_max", "energy efficiency [Mbit/J]",
                  f"EE vs FIM requirement — {method}", out / f"exp10_ee_vs_frac_{method}.png")
        scatter_plot(sub["delay_crb_s2"].to_numpy(), sub["rate_spectral_efficiency"].to_numpy(),
                     "delay CRB [s^2]", "sum spectral efficiency [bit/s/Hz]",
                     f"Rate vs CRB — {method}", out / f"exp10_rate_vs_crb_{method}.png",
                     label=method, logx=True)
        scatter_plot(sub["delay_crb_s2"].to_numpy(), sub["psl_db"].to_numpy(),
                     "delay CRB [s^2]", "actual PSL [dB]",
                     f"PSL vs CRB — {method}", out / f"exp10_psl_vs_crb_{method}.png",
                     label=method, logx=True)
        scatter_plot(sub["psl_db"].to_numpy(), sub["rate_spectral_efficiency"].to_numpy(),
                     "actual PSL [dB]", "sum spectral efficiency [bit/s/Hz]",
                     f"Rate vs PSL — {method}", out / f"exp10_rate_vs_psl_{method}.png", label=method)

    if alloc_fim:
        fracs = list(alloc_fim.keys())
        stack = np.column_stack([alloc_fim[f] for f in fracs])
        allocation_heatmap(stack, fracs, "FIM fraction of J_max",
                           "Power allocation vs CRB requirement (exact FIM, no PSL cone)",
                           out / "exp10_heatmap_fim_only.png")
        plot_power_allocations(
            {f"frac={fracs[0]:.2f}": alloc_fim[fracs[0]], f"frac={fracs[-1]:.2f}": alloc_fim[fracs[-1]], "uniform": uniform},
            out / "exp10_power_endpoints.png", peak_power_w=system.peak_power_w,
            title="Exact-FIM allocations at the weakest and strongest feasible CRB requests",
        )
    if alloc_psl:
        fracs = list(alloc_psl.keys())
        stack = np.column_stack([alloc_psl[f] for f in fracs])
        allocation_heatmap(stack, fracs, "FIM fraction of J_max",
                           "Power allocation vs CRB requirement (exact FIM + sampled-grid PSL)",
                           out / "exp10_heatmap_fim_psl.png")

    print(f"Saved outputs under {out}")
    uni = evaluate_allocation(uniform, system)
    print(f"PSL_max taken from uniform validation-grid PSL = {psl_uniform:.3f} dB; "
          f"uniform rate = {uni.rate_spectral_efficiency:.4f} bit/s/Hz")


if __name__ == "__main__":
    main()

"""Experiment 15 — Monte Carlo of linear vs exact-FIM vs exact-FIM+PSL oracles.

100 ordinary Rayleigh realisations and 100 controlled asymmetric realisations.
Infeasible and inaccurate solves are counted, not dropped.  Validation-grid PSL
violations are counted whenever a sampled-PSL solve returns a primal.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from isac.communication.water_filling import uniform_power
from isac.evaluation.plots import boxplot_by_method
from isac.evaluation.reporting import (
    STAGE25_ROOT,
    experiment_dir,
    print_config,
    print_frame,
    print_header,
    summarize_by_method,
)
from isac.evaluation.scenario import build_asymmetric_system, build_system, common_parser, config_from_args
from isac.evaluation.stage25 import four_formulation_rows
from isac.sensing.ambiguity import peak_sidelobe_level_db
from isac.optimization.ambiguity_constraints import validation_delay_grid

EXPERIMENT = "exp15_monte_carlo_stage25"
METRIC_COLS = [
    "rate_spectral_efficiency", "energy_efficiency_bit_per_j", "tx_power_w",
    "delay_fim_unknown_amplitude_per_s2", "delay_crb_s2", "range_rmse_bound_m",
    "psl_db", "spectral_centroid_hz", "solve_time_s",
]


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--num-realizations", type=int, default=100)
    parser.add_argument("--sensing-fraction", type=float, default=0.6)
    parser.add_argument("--fim-fraction", type=float, default=0.45)
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = args.output_dir or experiment_dir(EXPERIMENT, STAGE25_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    print_config(cfg)

    rows: list[dict] = []
    n = args.num_realizations
    t0 = time.perf_counter()
    for kind, builder in (
        ("ordinary", lambda rng: build_system(cfg, rng)[1]),
        ("asymmetric", lambda rng: build_asymmetric_system(cfg, rng, tilt=4.0)[1]),
    ):
        for i in range(n):
            rng = np.random.default_rng(cfg.simulation.seed + 1000 * (0 if kind == "ordinary" else 1) + i)
            system = builder(rng)
            val_grid = validation_delay_grid(system, cfg.ambiguity.validation_oversampling_factor)
            uniform = uniform_power(system.num_subcarriers, system.total_power_w, system.peak_power_w)
            psl_max = peak_sidelobe_level_db(uniform, system.frequencies_hz, val_grid, system.mainlobe_exclusion_s)
            chunk = four_formulation_rows(
                system,
                sensing_fraction=args.sensing_fraction,
                fim_fraction=args.fim_fraction,
                psl_max_db=psl_max,
                opt_oversampling=cfg.ambiguity.optimization_oversampling_factor,
            )
            for row in chunk:
                row["channel_kind"] = kind
                row["realization"] = i
                row["psl_max_db_request"] = psl_max
                if row.get("status") == "ok" and row.get("method") == "exact_fim_psl":
                    margin = row.get("violation_margin_db", np.nan)
                    row["validation_psl_violation"] = bool(np.isfinite(margin) and margin < -0.25)
                else:
                    row["validation_psl_violation"] = False
                rows.append(row)
            if (i + 1) % 20 == 0:
                print(f"  {kind}: {i + 1}/{n} realisations")

    table = pd.DataFrame(rows)
    table.to_csv(out / "exp15_raw.csv", index=False)
    elapsed = time.perf_counter() - t0

    print_header("Status counts")
    status = table.groupby(["channel_kind", "method", "status"]).size().reset_index(name="count")
    print_frame(status)

    print_header("Inaccurate-solve frequency")
    if "solver_inaccurate" in table.columns:
        acc = table.groupby(["channel_kind", "method"])["solver_inaccurate"].mean().reset_index()
        print_frame(acc)

    print_header("Validation-grid PSL violations (exact_fim_psl only, margin < -0.25 dB)")
    psl_v = table[table["method"] == "exact_fim_psl"].groupby("channel_kind")["validation_psl_violation"].mean()
    print(psl_v.to_string())
    print()

    print_header("Distribution summaries (successful solves)")
    for kind, sub in table.groupby("channel_kind", sort=False):
        summary = summarize_by_method(sub, METRIC_COLS)
        summary["channel_kind"] = kind
        summary.to_csv(out / f"exp15_summary_{kind}.csv", index=False)
        print(f"--- {kind} ---")
        print_frame(summary)

    ok = table[table["status"] == "ok"]
    for kind, sub in ok.groupby("channel_kind", sort=False):
        for col, ylabel, title, fname, scale, log in (
            ("rate_spectral_efficiency", "sum spectral efficiency [bit/s/Hz]", "Rate", "rate", 1.0, False),
            ("energy_efficiency_bit_per_j", "energy efficiency [Mbit/J]", "EE", "ee", 1e-6, False),
            ("delay_fim_unknown_amplitude_per_s2", "unknown-amplitude J_tau [1/s^2]", "Unknown FIM", "unknown_fim", 1.0, True),
            ("psl_db", "actual PSL [dB]", "PSL", "psl", 1.0, False),
            ("spectral_centroid_hz", "spectral centroid [Hz]", "Centroid", "centroid", 1.0, False),
            ("solve_time_s", "solver time [s]", "Runtime", "runtime", 1.0, True),
        ):
            if col not in sub.columns:
                continue
            boxplot_by_method(
                sub, col, ylabel, f"{title} ({kind} channels)", out / f"exp15_{fname}_{kind}.png",
                label_column="method", scale=scale, log=log,
            )

    mean_runtime = float(np.nanmean(ok["solve_time_s"])) if not ok.empty else float("nan")
    print_header("Runtime")
    print(f"  wall-clock for the whole Monte Carlo = {elapsed:.2f} s")
    print(f"  mean per-solve time (successful)     = {mean_runtime:.4f} s")
    print(f"Saved outputs under {out}")


if __name__ == "__main__":
    main()

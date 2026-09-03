"""Cutting-plane vs sampled-grid PSL Monte Carlo (dense-validation compliance)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.icc_common import config_from_icc_args, icc_parser, print_and_snapshot, write_csv, write_log
from isac.evaluation.icc_stats import describe
from isac.evaluation.icc_suite import build_physical_target, run_methods
from isac.evaluation.reporting import print_frame, print_header
from isac.evaluation.scenario import build_tdl_system

EXPERIMENT = "icc_cutting_plane_mc"
METHODS = (
    "conventional_s2_psl",
    "exact_efim_sampled_psl",
    "exact_efim_cutting_plane_psl",
)


def main() -> None:
    parser = icc_parser(__doc__)
    parser.add_argument("--fim-fraction", type=float, default=0.5)
    args = parser.parse_args()
    cfg = config_from_icc_args(args)
    n = args.num_realizations or (15 if args.quick else 200)
    print_and_snapshot(cfg, EXPERIMENT, {"n": n, "fim_fraction": args.fim_fraction, "methods": list(METHODS)})
    print_header(f"Cutting-plane PSL Monte Carlo n={n}")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i in range(n):
        rng = np.random.default_rng(cfg.simulation.seed + 70_000 + i)
        _, system = build_tdl_system(cfg, rng)
        target = build_physical_target(system, cfg, fim_fraction=args.fim_fraction)
        for row in run_methods(system, target, cfg, METHODS):
            row["realization"] = i
            row["channel_kind"] = "natural_tdl"
            rows.append(row)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i+1}/{n}  elapsed={time.perf_counter()-t0:.1f}s")

    raw = pd.DataFrame(rows)
    write_csv(raw, f"{EXPERIMENT}_raw.csv")
    stats_rows = []
    for method, sub in raw.groupby("method", sort=False):
        ok = sub[sub["status"] == "ok"]
        dense_viol = (~ok["dense_psl_satisfied"].astype(bool)).mean() if len(ok) else float("nan")
        stats_rows.append({
            "method": method,
            "n_ok": int(len(ok)),
            "n_infeasible": int((sub["status"] == "infeasible").sum()),
            "dense_violation_rate": float(dense_viol),
            "mean_psl_margin_db": describe(ok["psl_margin_db"]).get("mean") if "psl_margin_db" in ok else float("nan"),
            "mean_cutting_plane_iterations": describe(ok["cutting_plane_iterations"]).get("mean")
            if "cutting_plane_iterations" in ok.columns else float("nan"),
            "mean_solve_time_s": describe(ok["solve_time_s"]).get("mean"),
        })
    stats = pd.DataFrame(stats_rows)
    write_csv(stats, f"{EXPERIMENT}_summary.csv")
    print_frame(stats)
    elapsed = time.perf_counter() - t0
    write_log(f"{EXPERIMENT}.log", f"n={n} elapsed_s={elapsed:.3f}\n")
    print(f"elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()

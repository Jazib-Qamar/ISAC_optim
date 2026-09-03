"""ICC fair comparison: conventional S2 vs exact EFIM under the SAME physical target.

Runs the full method catalog on natural TDL realisations.  Conventional S2
optimises ``S(P)=sum f_k^2 P_k``; exact EFIM optimises ``G(P)``.  All returned
allocations are scored by the independent unknown-amplitude evaluator.
"""

from __future__ import annotations

import time

import pandas as pd

from experiments.icc_common import config_from_icc_args, icc_parser, print_and_snapshot, write_csv, write_log
from isac.evaluation.icc_stats import summarize_methods
from isac.evaluation.icc_suite import METHOD_ORDER, build_physical_target, run_methods
from isac.evaluation.reporting import print_frame, print_header
from isac.evaluation.scenario import build_tdl_system

EXPERIMENT = "icc_fair_comparison"
METRIC_COLS = [
    "rate_spectral_efficiency", "rate_mbps", "tx_power_w", "system_power_w",
    "ee_mbit_per_j", "delay_fim_known_amplitude_per_s2",
    "independently_evaluated_unknown_fim", "range_rmse_bound_m",
    "centroid_magnitude_hz", "dense_psl_db", "fim_mismatch_percent", "solve_time_s",
]


def main() -> None:
    parser = icc_parser(__doc__)
    parser.add_argument("--fim-fraction", type=float, default=0.5)
    args = parser.parse_args()
    cfg = config_from_icc_args(args)
    n = args.num_realizations or (12 if args.quick else 60)
    methods = METHOD_ORDER if not args.quick else (
        "uniform", "water_filling", "conventional_s2", "exact_efim",
        "exact_efim_sampled_psl", "exact_efim_cutting_plane_psl",
    )
    print_and_snapshot(cfg, EXPERIMENT, {"num_realizations": n, "fim_fraction": args.fim_fraction, "methods": list(methods)})
    print_header(f"{EXPERIMENT}: {n} natural TDL realisations, fair Gamma_J")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i in range(n):
        rng = np_rng(cfg.simulation.seed + i)
        _, system = build_tdl_system(cfg, rng)
        target = build_physical_target(cfg=cfg, system=system, fim_fraction=args.fim_fraction)
        chunk = run_methods(system, target, cfg, methods)
        for row in chunk:
            row["realization"] = i
            row["channel_kind"] = "natural_tdl"
            rows.append(row)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i + 1}/{n}  elapsed={time.perf_counter()-t0:.1f}s")

    raw = pd.DataFrame(rows)
    write_csv(raw, f"{EXPERIMENT}_raw.csv")
    summary = summarize_methods(raw, METRIC_COLS)
    write_csv(summary, f"{EXPERIMENT}_summary.csv")
    print_frame(summary)
    write_log(f"{EXPERIMENT}.log", f"n={n} methods={list(methods)} elapsed_s={time.perf_counter()-t0:.3f}\n")
    print(f"elapsed {time.perf_counter()-t0:.1f}s -> results/icc_paper/raw_data/{EXPERIMENT}_raw.csv")


def np_rng(seed: int):
    import numpy as np
    return np.random.default_rng(seed)


if __name__ == "__main__":
    main()

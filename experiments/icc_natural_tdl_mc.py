"""Natural (uncontrolled) exponential-PDP TDL Monte Carlo.

No sign-dependent frequency tilt.  Reports centroid magnitude, FIM mismatch,
false sensing-feasibility, and independent dense-grid PSL.  If natural channels
produce little mismatch, that is reported honestly.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.icc_common import config_from_icc_args, icc_parser, print_and_snapshot, write_csv, write_log
from isac.evaluation.icc_stats import pearson_spearman, describe, summarize_methods
from isac.evaluation.icc_suite import build_physical_target, run_methods
from isac.evaluation.reporting import print_frame, print_header
from isac.evaluation.scenario import build_tdl_system

EXPERIMENT = "icc_natural_tdl_mc"
METRIC_COLS = [
    "rate_spectral_efficiency", "rate_mbps", "tx_power_w", "ee_mbit_per_j",
    "independently_evaluated_unknown_fim", "delay_fim_known_amplitude_per_s2",
    "range_rmse_bound_m", "centroid_magnitude_hz", "dense_psl_db",
    "fim_mismatch_percent", "solve_time_s",
]


def main() -> None:
    parser = icc_parser(__doc__)
    parser.add_argument("--fim-fraction", type=float, default=0.5)
    args = parser.parse_args()
    cfg = config_from_icc_args(args)
    n = args.num_realizations or (25 if args.quick else 500)
    methods = (
        "uniform", "water_filling", "conventional_s2", "conventional_s2_psl",
        "exact_efim", "exact_efim_sampled_psl", "exact_efim_cutting_plane_psl",
    )
    if args.quick:
        methods = (
            "uniform", "water_filling", "conventional_s2", "exact_efim",
            "exact_efim_sampled_psl", "exact_efim_cutting_plane_psl",
        )
    print_and_snapshot(
        cfg, EXPERIMENT,
        {"num_realizations": n, "fim_fraction": args.fim_fraction, "channel": "exponential_tdl", "methods": list(methods)},
    )
    print_header(f"Natural TDL Monte Carlo  n={n}  (no constructed frequency tilt)")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i in range(n):
        rng = np.random.default_rng(cfg.simulation.seed + 10_000 + i)
        _, system = build_tdl_system(cfg, rng)
        target = build_physical_target(system, cfg, fim_fraction=args.fim_fraction)
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

    s2 = raw[(raw["method"] == "conventional_s2") & (raw["status"] == "ok")]
    mismatch_stats = describe(s2["fim_mismatch_percent"].tolist())
    mismatch_stats["frac_gt_1pct"] = float(np.mean(s2["fim_mismatch_percent"] > 1.0)) if len(s2) else float("nan")
    mismatch_stats["frac_gt_2pct"] = float(np.mean(s2["fim_mismatch_percent"] > 2.0)) if len(s2) else float("nan")
    mismatch_stats["frac_gt_5pct"] = float(np.mean(s2["fim_mismatch_percent"] > 5.0)) if len(s2) else float("nan")
    mismatch_stats["frac_gt_10pct"] = float(np.mean(s2["fim_mismatch_percent"] > 10.0)) if len(s2) else float("nan")
    corr = pearson_spearman(s2["centroid_magnitude_hz"], s2["fim_mismatch_percent"])
    write_csv(pd.DataFrame([{**mismatch_stats, **corr}]), f"{EXPERIMENT}_mismatch_stats.csv")

    print_header("Conventional S2 mismatch on natural TDL")
    print_frame(pd.DataFrame([mismatch_stats]))
    print_header("Centroid vs mismatch correlation")
    print_frame(pd.DataFrame([corr]))
    print_frame(summary)
    elapsed = time.perf_counter() - t0
    write_log(f"{EXPERIMENT}.log", f"n={n} elapsed_s={elapsed:.3f} mismatch_mean={mismatch_stats.get('mean')}\n")
    print(f"elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()

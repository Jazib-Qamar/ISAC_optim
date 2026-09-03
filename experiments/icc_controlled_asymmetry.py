"""Controlled asymmetry sweep (mechanism experiment — not a realistic channel).

Asymmetry strength ``a`` is the logistic spectral-tilt steepness.  ``a=0`` is
untilted i.i.d. Rayleigh.  Keep these results strictly separate from the
natural TDL Monte Carlo.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.icc_common import config_from_icc_args, icc_parser, print_and_snapshot, write_csv, write_log
from isac.evaluation.icc_stats import describe, mean_ci_95
from isac.evaluation.icc_suite import build_physical_target, run_methods
from isac.evaluation.reporting import print_frame, print_header
from isac.evaluation.scenario import build_controlled_tilt_system

EXPERIMENT = "icc_controlled_asymmetry"
DEFAULT_A = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
METHODS = ("uniform", "water_filling", "conventional_s2", "exact_efim")


def main() -> None:
    parser = icc_parser(__doc__)
    parser.add_argument("--fim-fraction", type=float, default=0.5)
    parser.add_argument("--per-level", type=int, default=None)
    args = parser.parse_args()
    cfg = config_from_icc_args(args)
    levels = DEFAULT_A
    n = args.per_level or (8 if args.quick else 40)
    print_and_snapshot(
        cfg, EXPERIMENT,
        {"channel": "controlled_logistic_tilt", "asymmetry_levels": list(levels),
         "per_level": n, "fim_fraction": args.fim_fraction, "methods": list(METHODS)},
    )
    print_header("Controlled asymmetry experiment (NOT a realistic channel model)")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for a in levels:
        for i in range(n):
            rng = np.random.default_rng(cfg.simulation.seed + 20_000 + int(10_000 * a) + i)
            _, system = build_controlled_tilt_system(cfg, rng, asymmetry=float(a))
            target = build_physical_target(system, cfg, fim_fraction=args.fim_fraction)
            chunk = run_methods(system, target, cfg, METHODS)
            for row in chunk:
                row["realization"] = i
                row["asymmetry"] = float(a)
                row["channel_kind"] = "controlled_tilt"
                rows.append(row)
        print(f"  a={a:.1f}  elapsed={time.perf_counter()-t0:.1f}s")

    raw = pd.DataFrame(rows)
    write_csv(raw, f"{EXPERIMENT}_raw.csv")

    agg_rows = []
    for a, sub_a in raw.groupby("asymmetry", sort=True):
        for method, sub in sub_a.groupby("method", sort=False):
            ok = sub[sub["status"] == "ok"]
            for metric in ("fim_mismatch_percent", "centroid_magnitude_hz",
                           "rate_spectral_efficiency", "ee_mbit_per_j", "dense_psl_db",
                           "independently_evaluated_unknown_fim"):
                if metric not in ok.columns:
                    continue
                stats = describe(ok[metric].tolist())
                mean, lo, hi = mean_ci_95(ok[metric].tolist())
                agg_rows.append({
                    "asymmetry": a, "method": method, "metric": metric,
                    **stats, "mean": mean, "ci95_low": lo, "ci95_high": hi,
                })
            phys = ok["physical_sensing_satisfied"].astype(bool) if "physical_sensing_satisfied" in ok else []
            if len(ok):
                rate, rlo, rhi = mean_ci_95(ok["physical_sensing_satisfied"].astype(float))
                agg_rows.append({
                    "asymmetry": a, "method": method, "metric": "physical_sensing_feasibility_rate",
                    "mean": rate, "ci95_low": rlo, "ci95_high": rhi, "count": len(ok),
                })
                if "false_sensing_feasibility" in ok:
                    fr, flo, fhi = mean_ci_95(ok["false_sensing_feasibility"].astype(float))
                    agg_rows.append({
                        "asymmetry": a, "method": method, "metric": "false_sensing_feasibility_rate",
                        "mean": fr, "ci95_low": flo, "ci95_high": fhi, "count": len(ok),
                    })
    agg = pd.DataFrame(agg_rows)
    write_csv(agg, f"{EXPERIMENT}_summary.csv")
    print_frame(agg[agg["metric"].isin(["fim_mismatch_percent", "centroid_magnitude_hz"])])
    elapsed = time.perf_counter() - t0
    write_log(f"{EXPERIMENT}.log", f"levels={list(levels)} per_level={n} elapsed_s={elapsed:.3f}\n")
    print(f"elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()

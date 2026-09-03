"""OFDM numerology robustness: K = 64, 128, 256 with fixed Δf = 15 kHz.

Occupied bandwidth scales as K Δf.  Both K and bandwidth are recorded; they are
not changed silently.  Conclusions are compared qualitatively across K.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.icc_common import config_from_icc_args, icc_parser, print_and_snapshot, write_csv, write_log
from isac.evaluation.icc_stats import summarize_methods
from isac.evaluation.icc_suite import build_physical_target, run_methods
from isac.evaluation.reporting import print_frame, print_header
from isac.evaluation.scenario import build_tdl_system, with_num_subcarriers

EXPERIMENT = "icc_numerology"
KS = (64, 128, 256)
METRIC_COLS = [
    "rate_spectral_efficiency", "independently_evaluated_unknown_fim",
    "fim_mismatch_percent", "centroid_magnitude_hz", "dense_psl_db", "ee_mbit_per_j",
]


def main() -> None:
    parser = icc_parser(__doc__)
    parser.add_argument("--fim-fraction", type=float, default=0.5)
    args = parser.parse_args()
    base = config_from_icc_args(args)
    n = args.num_realizations or (8 if args.quick else 40)
    ks = (64, 128) if args.quick else KS
    cheap = ("uniform", "water_filling", "conventional_s2", "exact_efim")
    psl_methods = ("conventional_s2_psl", "exact_efim_sampled_psl", "exact_efim_cutting_plane_psl")
    print_and_snapshot(
        base, EXPERIMENT,
        {"K": list(ks), "delta_f_hz": base.ofdm.subcarrier_spacing_hz,
         "bandwidth_note": "B = K * Delta_f (Delta_f held fixed)",
         "n_per_K": n, "fim_fraction": args.fim_fraction},
    )
    print_header("Numerology robustness (Δf fixed, B scales with K)")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for k in ks:
        cfg = with_num_subcarriers(base, int(k))
        methods = cheap + psl_methods if k == 64 else cheap
        if k == 128:
            methods = cheap + ("exact_efim_cutting_plane_psl",)
        n_k = n if k < 256 else max(8, n // 2)
        for i in range(n_k):
            rng = np.random.default_rng(cfg.simulation.seed + 50_000 + 1000 * k + i)
            _, system = build_tdl_system(cfg, rng)
            target = build_physical_target(system, cfg, fim_fraction=args.fim_fraction)
            chunk = run_methods(system, target, cfg, methods)
            for row in chunk:
                row["realization"] = i
                row["num_subcarriers"] = k
                row["subcarrier_spacing_hz"] = cfg.ofdm.subcarrier_spacing_hz
                row["bandwidth_hz"] = cfg.ofdm.bandwidth_hz
                row["num_symbols"] = cfg.sensing.num_symbols
                row["total_power_w"] = cfg.ofdm.total_power_w
                row["noise_bandwidth_hz"] = cfg.ofdm.subcarrier_spacing_hz
                row["channel_model"] = "exponential_tdl"
                rows.append(row)
        print(f"  K={k}  n={n_k}  elapsed={time.perf_counter()-t0:.1f}s")

    raw = pd.DataFrame(rows)
    write_csv(raw, f"{EXPERIMENT}_raw.csv")
    parts = []
    for k, sub in raw.groupby("num_subcarriers"):
        s = summarize_methods(sub, METRIC_COLS)
        s["num_subcarriers"] = k
        parts.append(s)
    summary = pd.concat(parts, ignore_index=True)
    write_csv(summary, f"{EXPERIMENT}_summary.csv")
    print_frame(summary)
    elapsed = time.perf_counter() - t0
    write_log(f"{EXPERIMENT}.log", f"K={list(ks)} elapsed_s={elapsed:.3f}\n")
    print(f"elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()

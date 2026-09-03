"""Exact physical ranging sweep: rate / EE / PSL vs common unknown-FIM target."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.icc_common import config_from_icc_args, icc_parser, print_and_snapshot, write_csv, write_log
from isac.evaluation.icc_suite import METHOD_ORDER, build_physical_target, run_methods
from isac.evaluation.reporting import print_header
from isac.evaluation.scenario import build_tdl_system

EXPERIMENT = "icc_frontier"
DEFAULT_FRACTIONS = (0.15, 0.25, 0.35, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)
METHODS = METHOD_ORDER


def main() -> None:
    parser = icc_parser(__doc__)
    parser.add_argument("--channels", type=int, default=None)
    args = parser.parse_args()
    cfg = config_from_icc_args(args)
    n_ch = args.channels or (2 if args.quick else 12)
    fractions = DEFAULT_FRACTIONS[:5] if args.quick else DEFAULT_FRACTIONS
    methods = METHODS if not args.quick else (
        "water_filling", "conventional_s2", "conventional_s2_psl",
        "exact_efim", "exact_efim_cutting_plane_psl", "exact_efim_cutting_plane_psl_ee",
    )
    print_and_snapshot(
        cfg, EXPERIMENT,
        {"n_channels": n_ch, "fim_fractions": list(fractions), "methods": list(methods), "channel": "natural_tdl"},
    )
    print_header(f"CRB/FIM–PSL–rate–EE frontier  channels={n_ch}")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i in range(n_ch):
        rng = np.random.default_rng(cfg.simulation.seed + 30_000 + i)
        _, system = build_tdl_system(cfg, rng)
        for frac in fractions:
            target = build_physical_target(system, cfg, fim_fraction=float(frac))
            chunk = run_methods(system, target, cfg, methods)
            for row in chunk:
                row["realization"] = i
                row["channel_kind"] = "natural_tdl"
                rows.append(row)
        print(f"  channel {i+1}/{n_ch}  elapsed={time.perf_counter()-t0:.1f}s")

    raw = pd.DataFrame(rows)
    write_csv(raw, f"{EXPERIMENT}_raw.csv")
    elapsed = time.perf_counter() - t0
    write_log(f"{EXPERIMENT}.log", f"n_ch={n_ch} fractions={list(fractions)} elapsed_s={elapsed:.3f}\n")
    print(f"elapsed {elapsed:.1f}s  rows={len(raw)}")


if __name__ == "__main__":
    main()

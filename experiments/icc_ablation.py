"""Ablation: what each component changes under identical physical conditions."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.icc_common import config_from_icc_args, icc_parser, print_and_snapshot, write_csv, write_log
from isac.evaluation.icc_suite import METHOD_ORDER, build_physical_target, run_methods
from isac.evaluation.reporting import print_frame, print_header
from isac.evaluation.scenario import build_controlled_tilt_system, build_tdl_system

EXPERIMENT = "icc_ablation"


def main() -> None:
    parser = icc_parser(__doc__)
    parser.add_argument("--fim-fraction", type=float, default=0.5)
    args = parser.parse_args()
    cfg = config_from_icc_args(args)
    n_tdl = 4 if args.quick else 20
    n_tilt = 2 if args.quick else 8
    print_and_snapshot(
        cfg, EXPERIMENT,
        {"n_tdl": n_tdl, "n_controlled_tilt": n_tilt, "tilt": 4.0, "fim_fraction": args.fim_fraction,
         "methods": list(METHOD_ORDER)},
    )
    print_header("Ablation (same target, same grids, all catalog methods)")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i in range(n_tdl):
        rng = np.random.default_rng(cfg.simulation.seed + 60_000 + i)
        _, system = build_tdl_system(cfg, rng)
        target = build_physical_target(system, cfg, fim_fraction=args.fim_fraction)
        for row in run_methods(system, target, cfg, METHOD_ORDER):
            row["realization"] = i
            row["channel_kind"] = "natural_tdl"
            rows.append(row)
    for i in range(n_tilt):
        rng = np.random.default_rng(cfg.simulation.seed + 61_000 + i)
        _, system = build_controlled_tilt_system(cfg, rng, asymmetry=4.0)
        target = build_physical_target(system, cfg, fim_fraction=args.fim_fraction)
        for row in run_methods(system, target, cfg, METHOD_ORDER):
            row["realization"] = i
            row["channel_kind"] = "controlled_tilt"
            row["asymmetry"] = 4.0
            rows.append(row)
        print(f"  progress elapsed={time.perf_counter()-t0:.1f}s")

    raw = pd.DataFrame(rows)
    write_csv(raw, f"{EXPERIMENT}_raw.csv")
    print_frame(raw.groupby(["channel_kind", "method", "status"]).size().reset_index(name="count"))
    elapsed = time.perf_counter() - t0
    write_log(f"{EXPERIMENT}.log", f"elapsed_s={elapsed:.3f}\n")
    print(f"elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()

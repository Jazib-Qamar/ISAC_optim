"""2-D FIM/CRB–PSL feasibility map for the proposed cutting-plane method.

Classifies each (Gamma_J, PSL_max) point as physically feasible, infeasible,
or sampled-grid feasible but dense-grid violating (the last uses a sampled-only
solve for diagnosis).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.icc_common import config_from_icc_args, icc_parser, print_and_snapshot, write_csv, write_log
from isac.evaluation.icc_suite import build_physical_target, run_method
from isac.evaluation.reporting import print_header
from isac.evaluation.scenario import build_tdl_system

EXPERIMENT = "icc_feasibility_map"
FIM_FRACS = (0.20, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
PSL_DBS = (0.0, -6.0, -10.0, -13.0, -15.0, -18.0, -21.0)


def classify(row: dict) -> str:
    if row.get("status") != "ok":
        return "infeasible"
    phys = bool(row.get("physical_sensing_satisfied"))
    dense = bool(row.get("dense_psl_satisfied"))
    opt_psl = row.get("optimization_grid_psl_db")
    req = row.get("requested_psl_max_db")
    sampled_ok = np.isfinite(opt_psl) and np.isfinite(req) and opt_psl <= req + 0.25
    if phys and dense:
        return "physically_feasible"
    if phys and sampled_ok and not dense:
        return "sampled_feasible_dense_violating"
    if not phys:
        return "physical_sensing_violated"
    return "dense_psl_violating"


def main() -> None:
    parser = icc_parser(__doc__)
    args = parser.parse_args()
    cfg = config_from_icc_args(args)
    n_ch = 1 if args.quick else 4
    fracs = FIM_FRACS[:4] if args.quick else FIM_FRACS
    psls = PSL_DBS[:4] if args.quick else PSL_DBS
    print_and_snapshot(cfg, EXPERIMENT, {"n_channels": n_ch, "fim_fractions": list(fracs), "psl_dbs": list(psls)})
    print_header("FIM–PSL feasibility map")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i in range(n_ch):
        rng = np.random.default_rng(cfg.simulation.seed + 40_000 + i)
        _, system = build_tdl_system(cfg, rng)
        for frac in fracs:
            for psl in psls:
                target = build_physical_target(system, cfg, fim_fraction=float(frac), psl_max_db=float(psl))
                sampled = run_method("exact_efim_sampled_psl", system, target, cfg)
                sampled["map_role"] = "sampled_only"
                sampled["realization"] = i
                sampled["classification"] = classify(sampled)
                rows.append(sampled)
                proposed = run_method("exact_efim_cutting_plane_psl", system, target, cfg)
                proposed["map_role"] = "cutting_plane"
                proposed["realization"] = i
                proposed["classification"] = classify(proposed)
                rows.append(proposed)
        print(f"  channel {i+1}/{n_ch}  elapsed={time.perf_counter()-t0:.1f}s")

    raw = pd.DataFrame(rows)
    write_csv(raw, f"{EXPERIMENT}_raw.csv")
    elapsed = time.perf_counter() - t0
    write_log(f"{EXPERIMENT}.log", f"n_ch={n_ch} elapsed_s={elapsed:.3f}\n")
    print(raw.groupby(["map_role", "classification"]).size())
    print(f"elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()

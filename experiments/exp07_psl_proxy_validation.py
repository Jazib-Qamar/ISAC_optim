"""Experiment 2F - is the per-tone peak power a useful proxy for the actual PSL?

Two populations are examined:

1. Random feasible allocations (six structural families, exp06 sampler).
2. Optimiser outputs: max-rate ISAC and Dinkelbach EE-ISAC solutions along a
   Gamma_s sweep over several channel realisations (the sub-population an
   optimiser actually visits).

For each allocation: max_k P_k (absolute and relative to the mean power), power
variance, actual PSL and ISL from the delay-domain ambiguity evaluator.  Pearson
and Spearman correlations are reported honestly; nothing is tuned to produce a
relationship.

Outputs (results/stage2/exp07_psl_proxy_validation/): exp07_random_samples.csv,
exp07_optimizer_samples.csv, exp07_correlations.csv, figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from isac.evaluation.baselines import StaticRequirements
from isac.evaluation.metrics import evaluate_allocation
from isac.evaluation.plots import scatter_plot
from isac.evaluation.reporting import experiment_dir, print_config, print_frame, print_header
from isac.evaluation.sampling import sample_feasible_allocations
from isac.evaluation.scenario import build_system, common_parser, config_from_args
from isac.optimization.dinkelbach import solve_dinkelbach_ee
from isac.optimization.exceptions import OptimizationError
from isac.optimization.max_rate import solve_max_rate

EXPERIMENT = "exp07_psl_proxy_validation"


def _corr_row(name: str, x: np.ndarray, y: np.ndarray, population: str) -> dict[str, float | str]:
    ok = np.isfinite(x) & np.isfinite(y)
    pearson = float(stats.pearsonr(x[ok], y[ok])[0]) if ok.sum() > 2 else np.nan
    spearman = float(stats.spearmanr(x[ok], y[ok])[0]) if ok.sum() > 2 else np.nan
    return {"population": population, "pair": name, "pearson": pearson, "spearman": spearman, "num_samples": int(ok.sum())}


def _row(power: np.ndarray, system, tag: str, **extra) -> dict:
    m = evaluate_allocation(power, system)
    mean_power = m.tx_power_w / system.num_subcarriers
    return {
        "population": tag, "tx_power_w": m.tx_power_w, "max_power_w": m.peak_power_w,
        "max_over_mean_power": m.peak_power_w / mean_power, "power_variance_w2": m.power_variance_w2,
        "power_cv": np.sqrt(m.power_variance_w2) / mean_power, "psl_db": m.psl_db, "isl_db": m.isl_db,
        "num_active": m.num_active_subcarriers, **extra,
    }


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--num-channels", type=int, default=10, help="channels for the optimiser population")
    parser.add_argument("--num-gamma", type=int, default=12)
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = args.output_dir or experiment_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    print_config(cfg)

    rng = np.random.default_rng(cfg.simulation.seed)
    _, system = build_system(cfg, rng)

    # ---- population 1: random feasible allocations
    powers, families = sample_feasible_allocations(system, args.num_samples, rng)
    random_rows = [_row(p, system, "random", family=f) for p, f in zip(powers, families)]
    random_frame = pd.DataFrame(random_rows)

    # ---- population 2: optimiser outputs along Gamma_s sweeps over several channels
    opt_rows = []
    failures = 0
    for c in range(args.num_channels):
        _, sys_c = build_system(cfg, rng)
        req = StaticRequirements.from_system(sys_c, cfg.optimization.rate_fraction_of_water_filling,
                                             cfg.optimization.sensing_fraction_of_maximum)
        for frac in np.linspace(0.0, 0.99, args.num_gamma):
            gamma = frac * req.max_sensing_surrogate
            try:
                mr = solve_max_rate(sys_c, gamma, solver_preference=cfg.optimization.solver_preference)
                opt_rows.append(_row(mr.power_allocation, sys_c, "optimizer", family="max_rate_isac", channel=c, sensing_fraction=frac))
                dk = solve_dinkelbach_ee(sys_c, gamma, solver_preference=cfg.optimization.solver_preference)
                opt_rows.append(_row(dk.result.power_allocation, sys_c, "optimizer", family="dinkelbach_ee_isac", channel=c, sensing_fraction=frac))
            except OptimizationError as exc:
                failures += 1
                print(f"  optimiser failure (channel {c}, Gamma_s/S_max={frac:.2f}): {exc}")
    opt_frame = pd.DataFrame(opt_rows)

    corr_rows = []
    for tag, frame in (("random", random_frame), ("optimizer", opt_frame)):
        for x_name in ("max_power_w", "max_over_mean_power", "power_variance_w2", "power_cv"):
            corr_rows.append(_corr_row(f"{x_name} vs psl_db", frame[x_name].to_numpy(), frame["psl_db"].to_numpy(), tag))
            corr_rows.append(_corr_row(f"{x_name} vs isl_db", frame[x_name].to_numpy(), frame["isl_db"].to_numpy(), tag))
        for fam, sub in frame.groupby("family", sort=False):
            corr_rows.append(_corr_row("max_power_w vs psl_db", sub["max_power_w"].to_numpy(), sub["psl_db"].to_numpy(), f"{tag}/{fam}"))
    corr = pd.DataFrame(corr_rows)

    print_header("PSL proxy validation: correlations")
    print(f"random samples: {len(random_frame)}; optimiser samples: {len(opt_frame)} (optimiser failures: {failures})")
    print_frame(corr)
    strong = corr[(corr["pair"] == "max_power_w vs psl_db") & corr["population"].isin(["random", "optimizer"])]
    for _, r in strong.iterrows():
        verdict = "strong" if abs(r["spearman"]) >= 0.8 else "moderate" if abs(r["spearman"]) >= 0.5 else "weak"
        print(f"  population={r['population']:9s}: Spearman(max P_k, PSL) = {r['spearman']:+.3f} -> {verdict} rank association")
    print("Note: a positive correlation means higher peak power tends to come with HIGHER (worse) PSL.\n")

    random_frame.to_csv(out / "exp07_random_samples.csv", index=False)
    opt_frame.to_csv(out / "exp07_optimizer_samples.csv", index=False)
    corr.to_csv(out / "exp07_correlations.csv", index=False)

    figures = []
    for tag, frame in (("random", random_frame), ("optimizer", opt_frame)):
        cats = frame["family"].tolist()
        figures.append(scatter_plot(frame["max_power_w"].to_numpy() * 1e3, frame["psl_db"].to_numpy(),
                                    "max$_k P_k$ [mW]", "actual PSL [dB]", f"Peak spectral power vs actual PSL ({tag} allocations)",
                                    out / f"exp07_max_power_vs_psl_{tag}.png", label=tag, categories=cats))
        figures.append(scatter_plot(frame["power_variance_w2"].to_numpy() * 1e6, frame["psl_db"].to_numpy(),
                                    "power variance across subcarriers [mW$^2$]", "actual PSL [dB]",
                                    f"Power variance vs actual PSL ({tag} allocations)", out / f"exp07_variance_vs_psl_{tag}.png",
                                    label=tag, categories=cats, logx=True))
        figures.append(scatter_plot(frame["max_over_mean_power"].to_numpy(), frame["isl_db"].to_numpy(),
                                    "max$_k P_k$ / mean power (dimensionless)", "ISL [dB]",
                                    f"Peak-to-mean power vs ISL ({tag} allocations)", out / f"exp07_peak_ratio_vs_isl_{tag}.png",
                                    label=tag, categories=cats))
    print_header("Saved outputs")
    for p in [out / "exp07_random_samples.csv", out / "exp07_optimizer_samples.csv", out / "exp07_correlations.csv", *figures]:
        print(f"  {p}")


if __name__ == "__main__":
    main()

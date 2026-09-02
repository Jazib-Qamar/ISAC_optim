"""Experiment 2E - validity of the linear sensing surrogate S(P) = sum_k f_k^2 P_k.

For many random feasible allocations (six structural families) the surrogate is
compared with

    J_known   - known-amplitude delay Fisher information,
    J_unknown - unknown-complex-amplitude delay Fisher information (Schur complement),
    CRB       - exact delay CRB of the configured model, and the range RMSE bound.

Expected mathematics (see isac/sensing/fim.py):
    J_known   = c * S(P),   c = N (2|beta|^2/sigma^2) (2 pi)^2      -> exact proportionality
    J_unknown = c * [S(P) - (sum f_k P_k)^2 / sum P_k] <= J_known    -> deviation for asymmetric spectra
    CRB       = 1 / J                                                -> hyperbolic, monotone in S only for J_known

Correlations (Pearson on linear values, Spearman on ranks) and the relative
deviation |c S - J_unknown| / J_known are quantified per family.

Outputs (results/stage2/exp06_sensing_surrogate_validation/): exp06_samples.csv, exp06_summary.csv, figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from isac.evaluation.metrics import evaluate_allocation
from isac.evaluation.plots import scatter_plot
from isac.evaluation.reporting import experiment_dir, print_config, print_frame, print_header
from isac.evaluation.sampling import sample_feasible_allocations
from isac.evaluation.scenario import build_system, common_parser, config_from_args
from isac.sensing.fim import surrogate_to_fisher_scale

EXPERIMENT = "exp06_sensing_surrogate_validation"


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    return float(stats.pearsonr(x[ok], y[ok])[0]), float(stats.spearmanr(x[ok], y[ok])[0])


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--num-samples", type=int, default=3000)
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = args.output_dir or experiment_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    print_config(cfg)

    rng = np.random.default_rng(cfg.simulation.seed)
    _, system = build_system(cfg, rng)
    powers, families = sample_feasible_allocations(system, args.num_samples, rng)
    scale = surrogate_to_fisher_scale(system.reflection_coefficient, system.noise_power_w, system.num_symbols)

    rows = []
    for power, family in zip(powers, families):
        m = evaluate_allocation(power, system)
        first_moment = float(np.dot(system.frequencies_hz, power))
        rows.append({
            "family": family,
            "tx_power_w": m.tx_power_w,
            "surrogate": m.sensing_surrogate,
            "scaled_surrogate": scale * m.sensing_surrogate,
            "fim_known": m.delay_fim_known_amplitude_per_s2,
            "fim_unknown": m.delay_fim_unknown_amplitude_per_s2,
            "crb_configured_s2": m.delay_crb_s2,
            "crb_known_s2": 1.0 / m.delay_fim_known_amplitude_per_s2,
            "crb_unknown_s2": 1.0 / m.delay_fim_unknown_amplitude_per_s2 if m.delay_fim_unknown_amplitude_per_s2 > 0 else np.inf,
            "range_rmse_m": m.range_rmse_bound_m,
            "spectral_centroid_hz": first_moment / m.tx_power_w,
        })
    table = pd.DataFrame(rows)
    table["rel_dev_known"] = np.abs(table["scaled_surrogate"] - table["fim_known"]) / table["fim_known"]
    table["rel_dev_unknown"] = (table["fim_known"] - table["fim_unknown"]) / table["fim_known"]  # >= 0 by theory
    table["unknown_over_known"] = table["fim_unknown"] / table["fim_known"]

    summary_rows = []
    groups = [("all", table)] + [(f, sub) for f, sub in table.groupby("family", sort=False)]
    for name, sub in groups:
        pk, sk = _corr(sub["surrogate"].to_numpy(), sub["fim_known"].to_numpy())
        pu, su = _corr(sub["surrogate"].to_numpy(), sub["fim_unknown"].to_numpy())
        pc, sc = _corr(sub["surrogate"].to_numpy(), sub["crb_configured_s2"].to_numpy())
        pcl, scl = _corr(np.log(sub["surrogate"].to_numpy()), np.log(sub["crb_configured_s2"].to_numpy()))
        summary_rows.append({
            "family": name, "num_samples": len(sub),
            "pearson_S_vs_J_known": pk, "spearman_S_vs_J_known": sk,
            "pearson_S_vs_J_unknown": pu, "spearman_S_vs_J_unknown": su,
            "pearson_S_vs_CRB": pc, "spearman_S_vs_CRB": sc,
            "pearson_logS_vs_logCRB": pcl,
            "max_rel_dev_known": sub["rel_dev_known"].max(),
            "mean_rel_dev_unknown": sub["rel_dev_unknown"].mean(),
            "max_rel_dev_unknown": sub["rel_dev_unknown"].max(),
            "frac_unknown_dev_gt_1pct": float((sub["rel_dev_unknown"] > 0.01).mean()),
            "frac_unknown_dev_gt_10pct": float((sub["rel_dev_unknown"] > 0.10).mean()),
        })
    summary = pd.DataFrame(summary_rows)

    print_header("Sensing surrogate validation")
    print(f"samples: {len(table)}   scale c = J_known / S = {scale:.6e} [1/(W s^2 Hz^2)]")
    print(f"max |c S - J_known| / J_known over all samples = {table['rel_dev_known'].max():.3e}  (exact proportionality expected)")
    print(f"J_unknown <= J_known for all samples: {bool((table['fim_unknown'] <= table['fim_known'] * (1 + 1e-12)).all())}")
    print_frame(summary)
    print("Interpretation: S(P) is an exact affine image of the *known-amplitude* Fisher information and a monotone")
    print("(hyperbolic) image of the known-amplitude CRB.  For the unknown-amplitude model, S(P) over-estimates the")
    print("information by (sum f_k P_k)^2 / sum P_k; the deviation is zero for spectra symmetric about the carrier and")
    print("becomes large for one-sided spectra.  The surrogate should therefore be described as 'FIM-equivalent for the")
    print("known-amplitude model' and 'an upper bound on the unknown-amplitude information', not as 'the CRB'.\n")

    table.to_csv(out / "exp06_samples.csv", index=False)
    summary.to_csv(out / "exp06_summary.csv", index=False)

    s = table["surrogate"].to_numpy()
    cats = table["family"].tolist()
    line_s = np.linspace(s.min(), s.max(), 200)
    figures = [
        scatter_plot(s, table["fim_known"].to_numpy(), "sensing surrogate $S(P)=\\sum_k f_k^2 P_k$ [W Hz$^2$]",
                     "known-amplitude $J_\\tau$ [1/s$^2$]", "Surrogate vs known-amplitude Fisher information",
                     out / "exp06_surrogate_vs_fim_known.png", label="random feasible allocations",
                     reference_line=(line_s, scale * line_s, "$J_\\tau = c\\,S(P)$ (theory)")),
        scatter_plot(s, table["fim_unknown"].to_numpy(), "sensing surrogate $S(P)$ [W Hz$^2$]",
                     "unknown-amplitude $J_\\tau$ [1/s$^2$]", "Surrogate vs unknown-amplitude Fisher information",
                     out / "exp06_surrogate_vs_fim_unknown.png", label="random feasible allocations",
                     color=np.abs(table["spectral_centroid_hz"].to_numpy()) / 1e3, color_label="|spectral centroid| [kHz]",
                     reference_line=(line_s, scale * line_s, "$c\\,S(P)$ (upper bound)")),
        scatter_plot(s, table["crb_configured_s2"].to_numpy(), "sensing surrogate $S(P)$ [W Hz$^2$]",
                     "delay CRB [s$^2$]", "Surrogate vs exact delay CRB (configured amplitude model)",
                     out / "exp06_surrogate_vs_crb.png", label="random feasible allocations", logx=True, logy=True,
                     reference_line=(line_s, 1.0 / (scale * line_s), "$1/(c\\,S(P))$ (known amplitude)")),
        scatter_plot(s, table["range_rmse_m"].to_numpy(), "sensing surrogate $S(P)$ [W Hz$^2$]",
                     "range RMSE bound [m]", "Surrogate vs range RMSE bound", out / "exp06_surrogate_vs_range_rmse.png",
                     label="random feasible allocations", logx=True, logy=True),
        scatter_plot(np.abs(table["spectral_centroid_hz"].to_numpy()) / 1e3, table["unknown_over_known"].to_numpy(),
                     "|power-weighted spectral centroid| [kHz]", "$J_{unknown} / J_{known}$ (dimensionless)",
                     "Information loss from unknown amplitude vs spectral asymmetry",
                     out / "exp06_unknown_ratio_vs_centroid.png", label="family", categories=cats),
    ]
    print_header("Saved outputs")
    for p in [out / "exp06_samples.csv", out / "exp06_summary.csv", *figures]:
        print(f"  {p}")


if __name__ == "__main__":
    main()

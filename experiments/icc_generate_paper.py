"""Generate ICC paper figures, tables, index and claims from stored raw CSVs.

Does not re-run optimisation.  Missing raw files are skipped with a log note.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.icc_common import ICC_CFG, ICC_FIG, ICC_RAW, ICC_SUM, ICC_TAB, icc_dirs
from isac.evaluation.icc_stats import describe, mean_ci_95, pearson_spearman
from isac.evaluation.icc_style import (
    METHOD_COLOR,
    apply_ieee_style,
    legend_outside,
    method_label,
    new_axes,
    plot_box_methods,
    plot_cdf_methods,
    plot_lines_ci,
    plot_scatter_groups,
    save_ieee,
    style_of,
)
from isac.evaluation.icc_suite import DISPLAY_NAME

NOTES: list[str] = []
FIGURES: list[dict] = []


def _load(name: str) -> pd.DataFrame | None:
    path = ICC_RAW / name
    if not path.exists():
        NOTES.append(f"missing raw file: {name}")
        return None
    return pd.read_csv(path)


def _ok(frame: pd.DataFrame, method: str | None = None) -> pd.DataFrame:
    sub = frame[frame["status"] == "ok"] if "status" in frame.columns else frame
    if method is not None:
        sub = sub[sub["method"] == method]
    return sub


def _pct(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{100.0 * x:.2f}%"


def _fmt(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}g}"


def fig1_natural_mismatch(natural: pd.DataFrame) -> None:
    groups = {}
    for method in ("conventional_s2", "exact_efim"):
        sub = _ok(natural, method)
        if sub.empty:
            continue
        groups[method] = (
            sub["centroid_magnitude_hz"].to_numpy(dtype=float) / 1e3,
            sub["fim_mismatch_percent"].to_numpy(dtype=float),
        )
    if not groups:
        return
    plot_scatter_groups(
        groups,
        r"$|\bar{f}_P|$ [kHz]",
        r"FIM mismatch $\varepsilon_J$ [%]",
        ICC_FIG / "fig1_natural_asymmetry_mismatch",
    )
    FIGURES.append({
        "id": "Fig. 1",
        "file": "fig1_natural_asymmetry_mismatch",
        "question": "Does naturally occurring communication-driven spectral asymmetry produce meaningful error in the conventional sensing metric?",
        "baselines": "Conventional S2, Exact EFIM",
        "x": "|f_bar_P| [kHz]",
        "y": "epsilon_J [%]",
        "sample": int(len(_ok(natural, "conventional_s2"))),
        "section": "Numerical Results — Natural frequency-selective channels",
    })


def fig2_controlled(ctrl: pd.DataFrame) -> None:
    methods = ("conventional_s2", "exact_efim")
    levels = np.sort(ctrl["asymmetry"].unique())
    for metric, ylabel, stem in (
        ("fim_mismatch_percent", r"FIM mismatch $\varepsilon_J$ [%]", "fig2a_controlled_mismatch"),
        ("centroid_magnitude_hz", r"$|\bar{f}_P|$ [kHz]", "fig2b_controlled_centroid"),
    ):
        series = {}
        for method in methods:
            means, los, his = [], [], []
            for a in levels:
                sub = _ok(ctrl[(ctrl["method"] == method) & (ctrl["asymmetry"] == a)])
                vals = (sub[metric] / (1e3 if metric == "centroid_magnitude_hz" else 1.0)).tolist()
                m, lo, hi = mean_ci_95(vals)
                means.append(m); los.append(lo); his.append(hi)
            series[method] = (np.array(means), np.array(los), np.array(his))
        plot_lines_ci(levels, series, r"Controlled asymmetry strength $a$", ylabel, ICC_FIG / stem)
    FIGURES.append({
        "id": "Fig. 2",
        "file": "fig2a_controlled_mismatch / fig2b_controlled_centroid",
        "question": "Does increasing spectral asymmetry systematically increase the nuisance-related sensing-model mismatch?",
        "baselines": "Conventional S2, Exact EFIM",
        "x": "asymmetry strength a",
        "y": "epsilon_J and |f_bar_P|",
        "sample": int(ctrl.groupby(["asymmetry", "realization"]).ngroups) if "realization" in ctrl else len(ctrl),
        "section": "Numerical Results — Controlled asymmetry mechanism",
    })


def fig3_tradeoff(front: pd.DataFrame) -> None:
    methods = [
        "water_filling", "conventional_s2", "conventional_s2_psl",
        "exact_efim", "exact_efim_cutting_plane_psl", "exact_efim_cutting_plane_psl_ee",
    ]
    fig, ax = new_axes(double=True)
    for method in methods:
        sub = _ok(front, method)
        if sub.empty:
            continue
        grouped = sub.groupby("fim_fraction", sort=True)
        xs, ys = [], []
        for frac, g in grouped:
            xs.append(g["max_range_rmse_m_target"].mean() if "max_range_rmse_m_target" in g else frac)
            ys.append(g["rate_mbps"].mean() if "rate_mbps" in g else g["rate_spectral_efficiency"].mean())
        st = style_of(method)
        ax.plot(xs, ys, **{k: st[k] for k in ("color", "linestyle", "marker", "label")})
    ax.set_xlabel(r"Physical ranging requirement (target RMSE) [m]")
    ax.set_ylabel("Rate [Mbit/s]")
    ax.grid(True, alpha=0.3)
    legend_outside(ax, ncol=2)
    save_ieee(fig, ICC_FIG / "fig3_rate_vs_ranging")
    FIGURES.append({
        "id": "Fig. 3",
        "file": "fig3_rate_vs_ranging",
        "question": "What communication rate is achieved under a common physical ranging requirement?",
        "baselines": ", ".join(method_label(m) for m in methods),
        "x": "target range RMSE [m]",
        "y": "rate [Mbit/s]",
        "sample": int(front["realization"].nunique()) if "realization" in front else 0,
        "section": "Numerical Results — Rate–ranging tradeoff",
    })


def fig4_feasibility(natural: pd.DataFrame) -> None:
    groups = {}
    for method in ("conventional_s2", "exact_efim"):
        sub = _ok(natural, method)
        if sub.empty:
            continue
        groups[method] = (
            sub["gamma_unknown_fim"].to_numpy(dtype=float),
            sub["independently_evaluated_unknown_fim"].to_numpy(dtype=float),
        )
    if not groups:
        return
    plot_scatter_groups(
        groups,
        r"$J_{\mathrm{required}}$ [1/s$^2$]",
        r"$J_{\mathrm{actual,unknown}}$ [1/s$^2$]",
        ICC_FIG / "fig4_actual_sensing_feasibility",
        identity=True,
        double=True,
    )
    FIGURES.append({
        "id": "Fig. 4",
        "file": "fig4_actual_sensing_feasibility",
        "question": "Does conventional S2 claim sensing feasibility while the independent unknown-reflectivity evaluator disagrees?",
        "baselines": "Conventional S2, Exact EFIM",
        "x": "J_required",
        "y": "J_actual,unknown",
        "sample": int(len(_ok(natural, "conventional_s2"))),
        "section": "Numerical Results — False sensing feasibility",
    })


def fig5_dense_psl(front: pd.DataFrame) -> None:
    methods = [
        "water_filling", "conventional_s2", "conventional_s2_psl",
        "exact_efim", "exact_efim_sampled_psl", "exact_efim_cutting_plane_psl",
    ]
    fig, ax = new_axes(double=True)
    for method in methods:
        sub = _ok(front, method)
        if sub.empty or "dense_psl_db" not in sub:
            continue
        xs, dense, opt = [], [], []
        for frac, g in sub.groupby("fim_fraction", sort=True):
            xs.append(frac)
            dense.append(g["dense_psl_db"].mean())
            opt.append(g["optimization_grid_psl_db"].mean() if "optimization_grid_psl_db" in g else np.nan)
        st = style_of(method)
        ax.plot(xs, dense, **{k: st[k] for k in ("color", "linestyle", "marker")}, label=st["label"] + " (dense)")
        if np.isfinite(opt).any():
            ax.plot(xs, opt, color=st["color"], ls=":", marker=None, alpha=0.7, label=st["label"] + " (opt. grid)")
    ax.set_xlabel(r"Physical FIM fraction $\Gamma_J / J_{\max}$")
    ax.set_ylabel("PSL [dB]")
    ax.grid(True, alpha=0.3)
    legend_outside(ax, ncol=2)
    save_ieee(fig, ICC_FIG / "fig5_dense_psl")
    FIGURES.append({
        "id": "Fig. 5",
        "file": "fig5_dense_psl",
        "question": "How does independent dense-grid PSL compare across methods under a common ranging requirement?",
        "baselines": ", ".join(method_label(m) for m in methods),
        "x": "FIM fraction",
        "y": "dense-validation PSL [dB]",
        "sample": int(front["realization"].nunique()) if "realization" in front else 0,
        "section": "Numerical Results — Ambiguity",
    })


def fig6_cutting_plane(cp: pd.DataFrame) -> None:
    apply_ieee_style()
    from matplotlib import pyplot as plt
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.16, 2.6))
    methods = ["conventional_s2_psl", "exact_efim_sampled_psl", "exact_efim_cutting_plane_psl"]
    rates, labels, colors = [], [], []
    for method in methods:
        sub = _ok(cp, method)
        if sub.empty:
            continue
        rates.append(1.0 - float(sub["dense_psl_satisfied"].astype(bool).mean()))
        labels.append(method_label(method))
        colors.append(METHOD_COLOR.get(method, "#333"))
        vals = sub["psl_margin_db"].to_numpy(dtype=float)
        vals = np.sort(vals[np.isfinite(vals)])
        if vals.size:
            y = np.arange(1, vals.size + 1) / vals.size
            st = style_of(method)
            ax1.step(vals, y, where="post", color=st["color"], ls=st["linestyle"], label=st["label"])
    ax0.bar(labels, rates, color=colors, alpha=0.75)
    ax0.set_ylabel("Dense-grid PSL violation rate")
    ax0.tick_params(axis="x", rotation=20)
    ax0.grid(True, axis="y", alpha=0.3)
    ax1.axvline(0.0, color="k", ls="--", lw=0.8)
    ax1.set_xlabel("PSL margin [dB] (positive = met)")
    ax1.set_ylabel("Empirical CDF")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=6, frameon=False)
    fig.tight_layout()
    save_ieee(fig, ICC_FIG / "fig6_cutting_plane_refinement")
    FIGURES.append({
        "id": "Fig. 6",
        "file": "fig6_cutting_plane_refinement",
        "question": "Does cutting-plane refinement solve the dense-grid PSL miss of sampled SOCs?",
        "baselines": "Conventional S2+PSL, Exact EFIM+sampled PSL, Exact EFIM+cutting-plane PSL",
        "x": "method / PSL margin",
        "y": "violation rate / CDF",
        "sample": int(cp["realization"].nunique()) if "realization" in cp else len(cp),
        "section": "Numerical Results — Cutting-plane PSL",
    })


def fig7_ee(front: pd.DataFrame) -> None:
    methods = [
        "water_filling", "conventional_s2_psl",
        "exact_efim_cutting_plane_psl", "exact_efim_cutting_plane_psl_ee",
    ]
    fig, ax = new_axes(double=True)
    for method in methods:
        sub = _ok(front, method)
        if sub.empty:
            continue
        xs, ee, rate = [], [], []
        for frac, g in sub.groupby("fim_fraction", sort=True):
            xs.append(g["max_range_rmse_m_target"].mean())
            ee.append(g["ee_mbit_per_j"].mean())
            rate.append(g["rate_mbps"].mean())
        st = style_of(method)
        ax.plot(xs, ee, **{k: st[k] for k in ("color", "linestyle", "marker", "label")})
    ax.set_xlabel(r"Physical ranging requirement (target RMSE) [m]")
    ax.set_ylabel("EE [Mbit/J]")
    ax.grid(True, alpha=0.3)
    legend_outside(ax, ncol=2)
    save_ieee(fig, ICC_FIG / "fig7_ee_vs_ranging")
    FIGURES.append({
        "id": "Fig. 7",
        "file": "fig7_ee_vs_ranging",
        "question": "Does EE optimisation provide a system-level gain once exact EFIM and dense PSL are enforced?",
        "baselines": ", ".join(method_label(m) for m in methods),
        "x": "target range RMSE [m]",
        "y": "EE [Mbit/J]",
        "sample": int(front["realization"].nunique()) if "realization" in front else 0,
        "section": "Numerical Results — Energy efficiency",
    })


def fig8_map(mp: pd.DataFrame) -> None:
    apply_ieee_style()
    from matplotlib import pyplot as plt
    sub = mp[mp["map_role"] == "cutting_plane"] if "map_role" in mp.columns else mp
    if sub.empty:
        return
    codes = {
        "physically_feasible": 2,
        "sampled_feasible_dense_violating": 1,
        "dense_psl_violating": 0,
        "physical_sensing_violated": -1,
        "infeasible": -2,
    }
    fracs = np.sort(sub["fim_fraction"].unique())
    psls = np.sort(sub["requested_psl_max_db"].unique())
    grid = np.full((len(psls), len(fracs)), np.nan)
    for i, psl in enumerate(psls):
        for j, frac in enumerate(fracs):
            cell = sub[(sub["fim_fraction"] == frac) & (sub["requested_psl_max_db"] == psl)]
            if cell.empty:
                continue
            # majority classification across channels
            mode = cell["classification"].mode()
            grid[i, j] = codes.get(str(mode.iloc[0]), np.nan) if len(mode) else np.nan
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2,
                   extent=(fracs.min() - 0.03, fracs.max() + 0.03, psls.min() - 0.5, psls.max() + 0.5))
    ax.set_xlabel(r"$\Gamma_J / J_{\max}$")
    ax.set_ylabel(r"Requested PSL [dB]")
    cbar = fig.colorbar(im, ax=ax, ticks=[-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels(["infeasible", "phys. viol.", "dense viol.", "sampled only", "feasible"])
    save_ieee(fig, ICC_FIG / "fig8_feasibility_map")
    FIGURES.append({
        "id": "Fig. 8",
        "file": "fig8_feasibility_map",
        "question": "Where is the joint FIM–PSL feasibility frontier for the proposed method?",
        "baselines": "Proposed cutting-plane (sampled-only overlay in raw data)",
        "x": "FIM fraction",
        "y": "requested PSL [dB]",
        "sample": int(sub["realization"].nunique()) if "realization" in sub else 0,
        "section": "Numerical Results — Feasibility frontier",
    })


def fig9_mc(natural: pd.DataFrame) -> None:
    methods = [
        "uniform", "water_filling", "conventional_s2", "conventional_s2_psl",
        "exact_efim", "exact_efim_sampled_psl", "exact_efim_cutting_plane_psl",
    ]
    plot_cdf_methods(natural, "rate_mbps", "Rate [Mbit/s]", ICC_FIG / "fig9a_cdf_rate", methods=methods, double=True)
    plot_box_methods(natural, "independently_evaluated_unknown_fim", r"$J_{\tau}^{\mathrm{eff}}$ [1/s$^2$]",
                     ICC_FIG / "fig9b_box_unknown_fim", methods=methods, double=True)
    plot_box_methods(natural, "dense_psl_db", "Dense-grid PSL [dB]", ICC_FIG / "fig9c_box_dense_psl",
                     methods=methods, double=True)
    plot_box_methods(natural, "ee_mbit_per_j", "EE [Mbit/J]", ICC_FIG / "fig9d_box_ee", methods=methods, double=True)
    FIGURES.append({
        "id": "Fig. 9",
        "file": "fig9a–d Monte Carlo distributions",
        "question": "How do methods compare statistically on natural TDL channels?",
        "baselines": ", ".join(method_label(m) for m in methods),
        "x": "method / metric value",
        "y": "CDF / distribution",
        "sample": int(natural["realization"].nunique()) if "realization" in natural else 0,
        "section": "Numerical Results — Monte Carlo comparison",
    })


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_\n"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _latex_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(frame.columns)
    align = "l" + "c" * (len(cols) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\hline",
        " & ".join(str(c).replace("_", r"\_") for c in cols) + r" \\",
        r"\hline",
    ]
    for _, row in frame.iterrows():
        lines.append(" & ".join(str(row[c]) for c in cols) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_table(name: str, frame: pd.DataFrame, caption: str, label: str) -> None:
    ICC_TAB.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ICC_TAB / f"{name}.csv", index=False)
    (ICC_TAB / f"{name}.md").write_text(f"## {caption}\n\n" + _md_table(frame))
    (ICC_TAB / f"{name}.tex").write_text(_latex_table(frame, caption, label))


def table1_parameters() -> None:
    cfg_path = ICC_CFG / "icc_natural_tdl_mc.json"
    if not cfg_path.exists():
        # any snapshot
        snaps = list(ICC_CFG.glob("*.json"))
        if not snaps:
            return
        cfg_path = snaps[0]
    cfg = json.loads(cfg_path.read_text())
    ofdm, ch, tdl, sen, en, amb, der = (
        cfg.get("ofdm", {}), cfg.get("channel", {}), cfg.get("tdl", {}),
        cfg.get("sensing", {}), cfg.get("energy", {}), cfg.get("ambiguity", {}),
        cfg.get("derived", {}),
    )
    rows = [
        ("Subcarriers K", ofdm.get("num_subcarriers")),
        ("Subcarrier spacing Δf", f"{ofdm.get('subcarrier_spacing_hz')} Hz"),
        ("Occupied bandwidth K Δf", f"{der.get('bandwidth_hz')} Hz"),
        ("Carrier frequency", f"{ofdm.get('carrier_frequency_hz')} Hz"),
        ("Transmit-power budget", f"{ofdm.get('total_power_w')} W"),
        ("Peak-power factor", ofdm.get("peak_power_factor")),
        ("Peak power P_peak", f"{der.get('peak_power_w')} W"),
        ("Noise PSD", f"{ofdm.get('noise_psd_dbm_per_hz')} dBm/Hz"),
        ("Noise figure", f"{ofdm.get('noise_figure_db')} dB"),
        ("Path loss", f"{ch.get('path_loss_db')} dB"),
        ("Sensing symbols N", sen.get("num_symbols")),
        ("Target range", f"{sen.get('target_range_m')} m"),
        ("Target reflectivity", f"{sen.get('reflection_gain_db')} dB"),
        ("Circuit power", f"{en.get('circuit_power_w')} W"),
        ("PA efficiency", en.get("pa_efficiency")),
        ("Channel (natural)", "exponential PDP TDL"),
        ("TDL taps (resolved)", der.get("tdl_num_taps")),
        ("RMS delay spread", f"{tdl.get('rms_delay_spread_s')} s"),
        ("Tap spacing", f"{tdl.get('tap_spacing_s')} s"),
        ("PSL opt. oversampling", amb.get("optimization_oversampling_factor")),
        ("PSL validation oversampling", amb.get("validation_oversampling_factor")),
        ("Cutting-plane max iterations", amb.get("cutting_plane_max_iterations")),
        ("PSL tolerance", f"{amb.get('psl_tolerance_db')} dB"),
    ]
    write_table("table1_simulation_parameters", pd.DataFrame(rows, columns=["Parameter", "Value"]),
                "Simulation parameters", "tab:params")


def table2_main(natural: pd.DataFrame | None, ablation: pd.DataFrame | None) -> None:
    src = natural if natural is not None else ablation
    if src is None:
        return
    # Prefer a single representative realisation from ablation TDL if available
    if ablation is not None and "channel_kind" in ablation.columns:
        one = ablation[ablation["channel_kind"] == "natural_tdl"]
        if not one.empty:
            rid = one["realization"].min()
            src = one[one["realization"] == rid]
    rows = []
    for method, sub in src.groupby("method", sort=False):
        ok = _ok(sub)
        if ok.empty:
            rows.append({"Method": method_label(method), "Status": sub["status"].iloc[0] if len(sub) else "n/a"})
            continue
        r = ok.iloc[0] if len(ok) == 1 else ok.mean(numeric_only=True)
        # mixed: use means for Monte Carlo
        def col(name, scale=1.0):
            if name in ok:
                return float(ok[name].mean()) * scale
            return float("nan")
        rows.append({
            "Method": method_label(method),
            "Rate [Mbit/s]": _fmt(col("rate_mbps"), 4),
            "EE [Mbit/J]": _fmt(col("ee_mbit_per_j"), 4),
            "Tx power [W]": _fmt(col("tx_power_w"), 4),
            "Unknown FIM": f"{col('independently_evaluated_unknown_fim'):.4e}" if np.isfinite(col("independently_evaluated_unknown_fim")) else "n/a",
            "Range RMSE [m]": _fmt(col("range_rmse_bound_m"), 4),
            "Dense PSL [dB]": _fmt(col("dense_psl_db"), 3),
            "|centroid| [kHz]": _fmt(col("centroid_magnitude_hz", 1e-3), 3),
            "Phys. sensing?": _pct(float(ok["physical_sensing_satisfied"].mean())) if "physical_sensing_satisfied" in ok else "n/a",
            "Dense PSL?": _pct(float(ok["dense_psl_satisfied"].mean())) if "dense_psl_satisfied" in ok else "n/a",
            "Runtime [s]": _fmt(col("solve_time_s"), 3),
        })
    write_table("table2_baseline_comparison", pd.DataFrame(rows), "Main baseline comparison", "tab:main")


def table3_mc(summary: pd.DataFrame | None, natural: pd.DataFrame | None) -> None:
    if natural is None:
        return
    rows = []
    for method, sub in natural.groupby("method", sort=False):
        ok = _ok(sub)
        for metric, label in (
            ("rate_mbps", "Rate [Mbit/s]"),
            ("ee_mbit_per_j", "EE [Mbit/J]"),
            ("independently_evaluated_unknown_fim", "Unknown FIM"),
            ("range_rmse_bound_m", "Range RMSE [m]"),
            ("dense_psl_db", "Dense PSL [dB]"),
            ("fim_mismatch_percent", "FIM mismatch [%]"),
        ):
            if metric not in ok.columns:
                continue
            d = describe(ok[metric])
            rows.append({
                "Method": method_label(method), "Metric": label, "N": d["count"],
                "Mean": _fmt(d["mean"], 4), "Median": _fmt(d["median"], 4),
                "Std": _fmt(d["std"], 4),
                "95% CI": f"[{_fmt(d['ci95_low'], 4)}, {_fmt(d['ci95_high'], 4)}]",
                "P5": _fmt(d["p05"], 4), "P95": _fmt(d["p95"], 4),
            })
        if len(ok):
            rows.append({
                "Method": method_label(method), "Metric": "Physical sensing feasibility",
                "N": len(ok), "Mean": _pct(float(ok["physical_sensing_satisfied"].mean())),
                "Median": "—", "Std": "—", "95% CI": "—", "P5": "—", "P95": "—",
            })
            rows.append({
                "Method": method_label(method), "Metric": "Dense PSL feasibility",
                "N": len(ok), "Mean": _pct(float(ok["dense_psl_satisfied"].mean())),
                "Median": "—", "Std": "—", "95% CI": "—", "P5": "—", "P95": "—",
            })
    write_table("table3_natural_mc_statistics", pd.DataFrame(rows),
                "Natural-channel Monte Carlo statistics", "tab:mc")


def table4_controlled(ctrl_sum: pd.DataFrame | None) -> None:
    if ctrl_sum is None:
        return
    keep = ctrl_sum[ctrl_sum["metric"].isin(
        ["fim_mismatch_percent", "centroid_magnitude_hz", "physical_sensing_feasibility_rate",
         "false_sensing_feasibility_rate", "rate_spectral_efficiency"]
    )].copy()
    keep["Method"] = keep["method"].map(method_label)
    out = keep.rename(columns={"asymmetry": "a", "metric": "Metric", "mean": "Mean",
                               "ci95_low": "CI low", "ci95_high": "CI high", "count": "N"})
    cols = [c for c in ["a", "Method", "Metric", "Mean", "CI low", "CI high", "N"] if c in out.columns]
    write_table("table4_controlled_asymmetry", out[cols], "Controlled asymmetry results", "tab:tilt")


def table5_ablation(ablation: pd.DataFrame | None) -> None:
    if ablation is None:
        return
    tdl = ablation[ablation["channel_kind"] == "natural_tdl"] if "channel_kind" in ablation.columns else ablation
    rows = []
    order = [
        "water_filling", "conventional_s2", "conventional_s2_psl",
        "exact_efim", "exact_efim_sampled_psl",
        "exact_efim_cutting_plane_psl", "exact_efim_cutting_plane_psl_ee",
    ]
    for method in order:
        ok = _ok(tdl, method)
        if ok.empty:
            rows.append({"Component": method_label(method), "Note": "no successful solves"})
            continue
        rows.append({
            "Component": method_label(method),
            "Rate [Mbit/s]": _fmt(ok["rate_mbps"].mean(), 4),
            "Unknown FIM": f"{ok['independently_evaluated_unknown_fim'].mean():.4e}",
            "Dense PSL [dB]": _fmt(ok["dense_psl_db"].mean(), 3),
            "EE [Mbit/J]": _fmt(ok["ee_mbit_per_j"].mean(), 4),
            "Phys. feas. [%]": _pct(float(ok["physical_sensing_satisfied"].mean())),
            "Dense PSL feas. [%]": _pct(float(ok["dense_psl_satisfied"].mean())),
            "Runtime [s]": _fmt(ok["solve_time_s"].mean(), 3),
        })
    write_table("table5_ablation", pd.DataFrame(rows), "Ablation study", "tab:ablation")


def _rate_diff(natural: pd.DataFrame) -> dict:
    s2 = _ok(natural, "conventional_s2")
    ef = _ok(natural, "exact_efim")
    if s2.empty or ef.empty:
        return {}
    merged = s2.merge(ef, on="realization", suffixes=("_s2", "_ef"))
    if merged.empty:
        return {}
    delta = 100.0 * (merged["rate_spectral_efficiency_s2"] - merged["rate_spectral_efficiency_ef"]) / merged["rate_spectral_efficiency_s2"]
    d = describe(delta)
    d["n_pairs"] = int(len(delta))
    return d


def write_claims(natural, ctrl, cp, front, numerology) -> dict:
    claims = []
    evidence = {}

    def add(claim, experiment, raw, n, numbers, kind, limitations):
        claims.append({
            "claim": claim, "experiment": experiment, "raw": raw, "n": n,
            "numbers": numbers, "kind": kind, "limitations": limitations,
        })

    if natural is not None:
        s2 = _ok(natural, "conventional_s2")
        ef = _ok(natural, "exact_efim")
        mm = describe(s2["fim_mismatch_percent"]) if not s2.empty else {}
        corr = pearson_spearman(s2["centroid_magnitude_hz"], s2["fim_mismatch_percent"]) if not s2.empty else {}
        false_s2 = float(s2["false_sensing_feasibility"].mean()) if not s2.empty else float("nan")
        false_ef = float(ef["false_sensing_feasibility"].mean()) if not ef.empty else float("nan")
        material_s2 = float(s2["material_false_sensing_feasibility_1pct"].mean()) if (not s2.empty and "material_false_sensing_feasibility_1pct" in s2.columns) else float("nan")
        phys_s2 = float(s2["physical_sensing_satisfied"].mean()) if not s2.empty else float("nan")
        phys_ef = float(ef["physical_sensing_satisfied"].mean()) if not ef.empty else float("nan")
        evidence.update({
            "natural_n": int(natural["realization"].nunique()) if "realization" in natural else 0,
            "natural_mismatch_mean": mm.get("mean"),
            "natural_mismatch_median": mm.get("median"),
            "natural_mismatch_p95": mm.get("p95"),
            "pearson_r": corr.get("pearson_r"),
            "spearman_rho": corr.get("spearman_rho"),
            "false_s2": false_s2,
            "false_ef": false_ef,
            "material_false_s2_1pct": material_s2,
            "phys_s2": phys_s2,
            "phys_ef": phys_ef,
        })
        add(
            f"Across {evidence['natural_n']} natural exponential-PDP TDL realisations, "
            f"conventional S2 allocations had mean unknown-vs-known FIM mismatch "
            f"{_fmt(mm.get('mean'), 3)}% (median {_fmt(mm.get('median'), 3)}%, "
            f"95% CI [{_fmt(mm.get('ci95_low'), 3)}, {_fmt(mm.get('ci95_high'), 3)}]).",
            "icc_natural_tdl_mc", "icc_natural_tdl_mc_raw.csv", evidence["natural_n"], mm,
            "Monte Carlo", "Mismatch is allocation-dependent; TDL RMS delay spread is a modelling choice.",
        )
        rd = _rate_diff(natural)
        evidence["rate_delta"] = rd
        if rd:
            add(
                f"Under the same physical unknown-FIM target, replacing conventional S2 with exact EFIM "
                f"changed rate by mean ΔR = {_fmt(rd.get('mean'), 3)}% "
                f"(median {_fmt(rd.get('median'), 3)}%, 95% CI [{_fmt(rd.get('ci95_low'), 3)}, {_fmt(rd.get('ci95_high'), 3)}]; "
                f"positive means conventional had higher rate).",
                "icc_natural_tdl_mc", "icc_natural_tdl_mc_raw.csv", rd.get("n_pairs"), rd,
                "Monte Carlo", "Compared only on realisations where both methods returned a primal.",
            )
        add(
            f"False sensing-feasibility rate (claims S-constraint met but independent J_unknown misses Γ_J within solver tolerance): "
            f"conventional S2 {_pct(false_s2)}; exact EFIM {_pct(false_ef)}. "
            f"Material (>1% relative shortfall) false-feasibility: conventional S2 {_pct(material_s2)}.",
            "icc_natural_tdl_mc", "icc_natural_tdl_mc_raw.csv", evidence["natural_n"],
            {"false_s2": false_s2, "false_ef": false_ef, "material_false_s2_1pct": material_s2},
            "Monte Carlo", "Depends on the chosen Γ_J / J_max fraction. Solver-tolerance violations can be much more frequent than 1% FIM shortfalls when S is an active constraint.",
        )

    if ctrl is not None:
        s2 = _ok(ctrl, "conventional_s2")
        if not s2.empty:
            by_a = s2.groupby("asymmetry")["fim_mismatch_percent"].mean()
            add(
                f"In the controlled-tilt mechanism experiment, mean conventional-S2 FIM mismatch increased from "
                f"{_fmt(float(by_a.iloc[0]), 3)}% at a={by_a.index[0]} to "
                f"{_fmt(float(by_a.iloc[-1]), 3)}% at a={by_a.index[-1]}.",
                "icc_controlled_asymmetry", "icc_controlled_asymmetry_raw.csv",
                int(s2["realization"].nunique()) if "realization" in s2 else len(s2),
                {"by_a": by_a.to_dict()},
                "Controlled mechanism (not a realistic channel)",
                "Logistic spectral tilt is synthetic; do not cite as a 3GPP channel result.",
            )
            evidence["controlled_mismatch_by_a"] = by_a.to_dict()

    if cp is not None:
        def viol(method):
            sub = _ok(cp, method)
            if sub.empty:
                return float("nan"), float("nan"), float("nan")
            return (
                1.0 - float(sub["dense_psl_satisfied"].mean()),
                float(sub["psl_margin_db"].mean()) if "psl_margin_db" in sub else float("nan"),
                float(sub["cutting_plane_iterations"].mean()) if "cutting_plane_iterations" in sub.columns else float("nan"),
            )
        v_s, m_s, _ = viol("exact_efim_sampled_psl")
        v_c, m_c, it_c = viol("exact_efim_cutting_plane_psl")
        evidence.update({"sampled_viol": v_s, "cp_viol": v_c, "cp_iters": it_c, "sampled_margin": m_s, "cp_margin": m_c})
        add(
            f"Cutting-plane refinement changed dense-grid PSL violation probability from "
            f"{_pct(v_s)} (sampled SOC) to {_pct(v_c)} (proposed), mean iterations {_fmt(it_c, 3)}.",
            "icc_cutting_plane_mc", "icc_cutting_plane_mc_raw.csv",
            int(cp["realization"].nunique()) if "realization" in cp else 0,
            {"sampled_violation": v_s, "cutting_plane_violation": v_c, "mean_iterations": it_c},
            "Monte Carlo", "Violation uses independent dense grid and the configured PSL tolerance.",
        )

    if front is not None:
        prop = _ok(front, "exact_efim_cutting_plane_psl")
        if not prop.empty:
            feas = prop.groupby("fim_fraction")["status"].apply(lambda s: float((s == "ok").mean()))
            # also dense
            if "dense_psl_satisfied" in prop:
                dens = prop.groupby("fim_fraction")["dense_psl_satisfied"].mean()
            else:
                dens = feas
            evidence["frontier_ok_by_frac"] = feas.to_dict()
            evidence["frontier_dense_by_frac"] = dens.to_dict()
            # boundary: largest frac with dense feasibility > 50%
            ok_fracs = [float(f) for f, v in dens.items() if v >= 0.5]
            boundary = max(ok_fracs) if ok_fracs else float("nan")
            evidence["fim_psl_boundary_frac"] = boundary
            add(
                f"On the joint FIM+uniform-PSL frontier, the proposed method remained dense-PSL feasible "
                f"for at least half of channels up to Γ_J / J_max ≈ {_fmt(boundary, 3)}.",
                "icc_frontier", "icc_frontier_raw.csv",
                int(front["realization"].nunique()) if "realization" in front else 0,
                {"boundary_frac": boundary},
                "Sweep (natural TDL channels)",
                "Boundary is empirical under the uniform-spectrum PSL request; not a continuous-delay certificate.",
            )
        ee_r = _ok(front, "exact_efim_cutting_plane_psl")
        ee_e = _ok(front, "exact_efim_cutting_plane_psl_ee")
        if not ee_r.empty and not ee_e.empty:
            # compare at common fractions
            merged = ee_r.groupby("fim_fraction")[["ee_mbit_per_j", "rate_mbps", "tx_power_w"]].mean().join(
                ee_e.groupby("fim_fraction")[["ee_mbit_per_j", "rate_mbps", "tx_power_w"]].mean(),
                lsuffix="_rate", rsuffix="_ee",
            )
            if not merged.empty:
                gain = 100.0 * (merged["ee_mbit_per_j_ee"] - merged["ee_mbit_per_j_rate"]) / merged["ee_mbit_per_j_rate"]
                rate_loss = 100.0 * (merged["rate_mbps_rate"] - merged["rate_mbps_ee"]) / merged["rate_mbps_rate"]
                evidence["ee_gain_pct_mean"] = float(gain.mean())
                evidence["ee_rate_loss_pct_mean"] = float(rate_loss.mean())
                add(
                    f"Enforcing the EE objective (Dinkelbach) under exact EFIM + cutting-plane PSL changed EE by "
                    f"mean {_fmt(float(gain.mean()), 3)}% relative to max-rate, with mean rate change "
                    f"{_fmt(float(rate_loss.mean()), 3)}% (positive rate change = max-rate had higher rate).",
                    "icc_frontier", "icc_frontier_raw.csv",
                    int(front["realization"].nunique()) if "realization" in front else 0,
                    {"ee_gain_pct": float(gain.mean()), "rate_loss_pct": float(rate_loss.mean())},
                    "Sweep",
                    "Dinkelbach is a standard fractional-programming solver, not a claimed novelty.",
                )

    if numerology is not None:
        s2 = _ok(numerology, "conventional_s2")
        byk = s2.groupby("num_subcarriers")["fim_mismatch_percent"].mean() if not s2.empty else pd.Series(dtype=float)
        evidence["mismatch_by_K"] = byk.to_dict()
        add(
            f"Mean conventional-S2 FIM mismatch by numerology (Δf fixed, B=KΔf): "
            + ", ".join(f"K={int(k)} → {_fmt(v, 3)}%" for k, v in byk.items()),
            "icc_numerology", "icc_numerology_raw.csv",
            int(numerology["realization"].nunique()) if "realization" in numerology else 0,
            {"by_K": byk.to_dict()},
            "Monte Carlo per K",
            "K and bandwidth are coupled; this is not a constant-bandwidth comparison.",
        )

    lines = ["# Paper-facing statistical claims", "",
             "Every claim below is generated from stored raw CSVs. Do not cite a claim if its raw file is missing.", ""]
    for i, c in enumerate(claims, 1):
        lines += [
            f"## Claim {i}",
            f"- **Claim:** {c['claim']}",
            f"- **Experiment:** `{c['experiment']}`",
            f"- **Raw data:** `results/icc_paper/raw_data/{c['raw']}`",
            f"- **Sample size:** {c['n']}",
            f"- **Evidence type:** {c['kind']}",
            f"- **Limitations:** {c['limitations']}",
            "",
        ]
    (ICC_SUM / "PAPER_CLAIMS.md").write_text("\n".join(lines))
    return evidence


def write_index(evidence: dict) -> None:
    lines = ["# ICC figure index", "",
             "Generated from stored raw data. Hypothesis verdicts use the numerical evidence only.", ""]
    natural_mm = evidence.get("natural_mismatch_mean")
    false_s2 = evidence.get("false_s2")
    for fig in FIGURES:
        verdict = "PARTIAL"
        if fig["id"] == "Fig. 1":
            if natural_mm is not None and np.isfinite(natural_mm):
                verdict = "SUPPORTS" if natural_mm >= 1.0 else "CONTRADICTS" if natural_mm < 0.2 else "PARTIALLY SUPPORTS"
        elif fig["id"] == "Fig. 4":
            if false_s2 is not None and np.isfinite(false_s2):
                verdict = "SUPPORTS" if false_s2 >= 0.05 else "CONTRADICTS" if false_s2 < 0.01 else "PARTIALLY SUPPORTS"
        elif fig["id"] == "Fig. 6":
            vs, vc = evidence.get("sampled_viol"), evidence.get("cp_viol")
            if vs is not None and vc is not None and np.isfinite(vs) and np.isfinite(vc):
                verdict = "SUPPORTS" if vc < vs - 0.05 else "PARTIALLY SUPPORTS" if vc <= vs else "CONTRADICTS"
        key = fig.get("key_finding", "")
        lines += [
            f"## {fig['id']} — `{fig['file']}`",
            f"- **Scientific question:** {fig['question']}",
            f"- **Baselines:** {fig['baselines']}",
            f"- **Axes:** x = {fig['x']}; y = {fig['y']}",
            f"- **Sample size:** {fig['sample']}",
            f"- **Hypothesis verdict:** {verdict}",
            f"- **Recommended section:** {fig['section']}",
            "",
        ]
    rec = [
        "Fig. 1 (natural mismatch)", "Fig. 2 (controlled mechanism)", "Fig. 4 (actual vs required J)",
        "Fig. 6 (cutting-plane PSL)", "Fig. 3 (rate–ranging)", "Table II (main comparison)", "Table III (MC stats)",
    ]
    lines += ["## Recommended 5–7 manuscript items (space-limited ICC)", ""]
    for item in rec:
        lines.append(f"- {item}")
    lines += ["", "Do not include Fig. 9 panels that merely restate Table III, or Fig. 8 if the map is dominated by a single colour.", ""]
    (ICC_SUM / "FIGURE_INDEX.md").write_text("\n".join(lines))
    (ICC_ROOT_INDEX := Path("results/icc_paper/FIGURE_INDEX.md")).write_text("\n".join(lines))


def write_final(evidence: dict, pytest_line: str, elapsed_note: str) -> None:
    q1 = "PARTIALLY"
    mm = evidence.get("natural_mismatch_mean")
    false_s2 = evidence.get("false_s2", float("nan"))
    if mm is not None and np.isfinite(mm) and mm >= 2.0 and np.isfinite(false_s2) and false_s2 >= 0.05:
        q1 = "YES"
    elif mm is not None and np.isfinite(mm) and mm < 0.5 and (not np.isfinite(false_s2) or false_s2 < 0.01):
        q1 = "NO"
    ctrl = evidence.get("controlled_mismatch_by_a", {})
    natural_vs_ctrl = "primarily under artificially controlled asymmetry" if (
        mm is not None and np.isfinite(mm) and mm < 1.0 and ctrl and max(ctrl.values()) > 3 * max(mm, 1e-9)
    ) else "present in both, typically larger under controlled tilt"
    strongest = (
        "When complex reflectivity is unknown, conventional uncentered S2 is an optimistic proxy for delay information "
        "on spectrally asymmetric allocations; exact EFIM removes that optimism at a measurable but scenario-dependent "
        "rate cost, and sampled PSL SOCs require cutting-plane / dense-grid validation before a PSL claim is made."
    )
    lines = f"""# Final ICC experimental report

This report is generated from `results/icc_paper/raw_data/` after the ICC experiment suite.
Exploratory Stage 1/2/2.5 outputs remain in `results/stage2/` and `results/stage2_5/`.

## 1. Pytest

{pytest_line}

## 2. Experiment commands

```
python -m pytest
python -m experiments.icc_fair_comparison
python -m experiments.icc_natural_tdl_mc
python -m experiments.icc_controlled_asymmetry
python -m experiments.icc_cutting_plane_mc
python -m experiments.icc_frontier
python -m experiments.icc_feasibility_map
python -m experiments.icc_numerology
python -m experiments.icc_ablation
python -m experiments.icc_generate_paper
```

`--quick` reduces sample sizes for smoke tests.

## 3. Configurations

JSON snapshots: `results/icc_paper/configs/`.  Default physics: K=64, Δf=15 kHz, P_total=1 W,
exponential-PDP TDL (τ_rms=300 ns, Δτ=50 ns) for natural experiments; logistic tilt for the
controlled mechanism experiment only.

## 4–18. Numerical evidence (auto)

- Natural-channel sample size: {evidence.get('natural_n')}
- Natural-channel mean FIM mismatch (conventional S2): {_fmt(mm, 4)} %
- Natural-channel median FIM mismatch: {_fmt(evidence.get('natural_mismatch_median'), 4)} %
- Pearson r(|f_bar|, ε_J): {_fmt(evidence.get('pearson_r'), 3)}
- Spearman ρ: {_fmt(evidence.get('spearman_rho'), 3)}
- Conventional false sensing-feasibility rate: {_pct(false_s2) if false_s2==false_s2 else 'n/a'}
- Exact-EFIM false sensing-feasibility rate: {_pct(evidence.get('false_ef')) if evidence.get('false_ef')==evidence.get('false_ef') else 'n/a'}
- Conventional vs exact rate ΔR mean: {_fmt((evidence.get('rate_delta') or {}).get('mean'), 4)} %
- Sampled-PSL dense violation rate: {_pct(evidence.get('sampled_viol')) if evidence.get('sampled_viol')==evidence.get('sampled_viol') else 'n/a'}
- Cutting-plane dense violation rate: {_pct(evidence.get('cp_viol')) if evidence.get('cp_viol')==evidence.get('cp_viol') else 'n/a'}
- Mean cutting-plane iterations: {_fmt(evidence.get('cp_iters'), 3)}
- Empirical FIM–PSL feasibility boundary (Γ_J/J_max, 50% channels): {_fmt(evidence.get('fim_psl_boundary_frac'), 3)}
- EE gain vs max-rate (mean %): {_fmt(evidence.get('ee_gain_pct_mean'), 3)}
- Associated rate reduction (mean %): {_fmt(evidence.get('ee_rate_loss_pct_mean'), 3)}
- Mismatch by K: {evidence.get('mismatch_by_K')}

{elapsed_note}

## Questions (evidence only)

### QUESTION 1
Does conventional uncentered S2 produce an optimistic ranging-feasibility assessment when complex reflectivity is unknown?

**{q1}**

Natural-channel mean mismatch {_fmt(mm, 3)}%; false-feasibility {_pct(false_s2) if false_s2==false_s2 else 'n/a'}.
A YES requires both a non-trivial mismatch and false-feasibility under the chosen Γ_J.

### QUESTION 2
Does this mismatch occur naturally in TDL channels, or primarily under controlled asymmetry?

**{natural_vs_ctrl}**

Natural mean ε_J = {_fmt(mm, 3)}%. Controlled mismatch-by-a = {evidence.get('controlled_mismatch_by_a')}.

### QUESTION 3
Association with spectral-centroid displacement: Pearson r = {_fmt(evidence.get('pearson_r'), 3)}, Spearman ρ = {_fmt(evidence.get('spearman_rho'), 3)}.

### QUESTION 4
Exact EFIM false-feasibility rate {_pct(evidence.get('false_ef')) if evidence.get('false_ef')==evidence.get('false_ef') else 'n/a'} vs conventional {_pct(false_s2) if false_s2==false_s2 else 'n/a'}.
Exact EFIM constrains the evaluated metric, so residual false-feasibility can only come from solver tolerance.

### QUESTION 5
Communication-rate cost of exact EFIM vs conventional S2 under the same physical target:
mean ΔR = {_fmt((evidence.get('rate_delta') or {}).get('mean'), 4)}%,
median {_fmt((evidence.get('rate_delta') or {}).get('median'), 4)}%,
CI [{_fmt((evidence.get('rate_delta') or {}).get('ci95_low'), 4)}, {_fmt((evidence.get('rate_delta') or {}).get('ci95_high'), 4)}],
worst (p95) {_fmt((evidence.get('rate_delta') or {}).get('p95'), 4)}%.

### QUESTION 6
Cutting-plane dense-grid PSL: sampled violation {_pct(evidence.get('sampled_viol')) if evidence.get('sampled_viol')==evidence.get('sampled_viol') else 'n/a'} → cutting-plane {_pct(evidence.get('cp_viol')) if evidence.get('cp_viol')==evidence.get('cp_viol') else 'n/a'}.

### QUESTION 7
Observed FIM/PSL frontier: proposed method dense-feasible on ≥50% of channels up to Γ_J/J_max ≈ {_fmt(evidence.get('fim_psl_boundary_frac'), 3)} with the uniform-spectrum PSL request.

### QUESTION 8
EE optimisation mean EE change {_fmt(evidence.get('ee_gain_pct_mean'), 3)}% with mean rate change {_fmt(evidence.get('ee_rate_loss_pct_mean'), 3)}% relative to max-rate under the same constraints.

### QUESTION 9
Numerology (Δf fixed): mismatch-by-K {evidence.get('mismatch_by_K')}. Qualitative consistency requires the same sign of mismatch and of ΔR at each K.

### QUESTION 10
Strongest scientifically defensible ICC claim:

> {strongest}

## Assumptions and limitations

- Single user, single point target, no clutter / self-interference / mobility.
- Unknown-amplitude delay EFIM is the Stage 1 Schur-complement model; it is not a new information-theoretic result.
- Sampled + cutting-plane PSL is an inner approximation, not a continuous-delay certificate.
- Conventional S2 is given the mapped threshold Γ_S = Γ_J / C_β so that J_known ≥ Γ_J is what it *claims*.
- Natural TDL uses a causal exponential PDP; that is frequency-selective but not sign-tilted by construction.
- Controlled tilt is synthetic.
- No published Yang/Iqbal (or other paper) baseline is plotted, because none was implemented.
- Dinkelbach, SOCs, water-filling and CRB optimisation are not claimed as novel.

## Missing raw files

{chr(10).join('- ' + n for n in NOTES) if NOTES else '- none'}
"""
    (ICC_SUM / "FINAL_ICC_RESULTS.md").write_text(lines)


def write_figure_readme() -> None:
    blocks = ["# ICC figures — reproducibility", "",
              "All figures are generated by `python -m experiments.icc_generate_paper` from CSV files in `raw_data/`.",
              "Optimisers are not re-run.", ""]
    meta = {
        "fig1_natural_asymmetry_mismatch": {
            "cmd": "python -m experiments.icc_natural_tdl_mc",
            "raw": "icc_natural_tdl_mc_raw.csv",
            "methods": "conventional_s2, exact_efim",
            "x": "|f_bar_P|", "y": "epsilon_J",
            "ci": "none (scatter)", "channel": "exponential PDP TDL",
            "target": "Γ_J = 0.5 J_max (unknown-amplitude), uniform-spectrum PSL request for PSL methods",
            "psl": "independent dense grid, mainlobe exclusion 1/B",
            "question": "Natural asymmetry vs conventional S2 mismatch",
        },
        "fig2a_controlled_mismatch": {
            "cmd": "python -m experiments.icc_controlled_asymmetry",
            "raw": "icc_controlled_asymmetry_raw.csv",
            "methods": "conventional_s2, exact_efim",
            "x": "asymmetry a", "y": "epsilon_J",
            "ci": "Student-t 95% interval across realisations at each a",
            "channel": "controlled logistic tilt (NOT a realistic model)",
            "target": "Γ_J = 0.5 J_max",
            "psl": "evaluated, not constrained in this sweep",
            "question": "Mechanism: tilt → mismatch",
        },
    }
    for name, m in meta.items():
        blocks += [f"## {name}",
                   f"- Experiment command: `{m['cmd']}`",
                   f"- Raw data: `results/icc_paper/raw_data/{m['raw']}`",
                   f"- Compared methods: {m['methods']}",
                   f"- x-axis: {m['x']}",
                   f"- y-axis: {m['y']}",
                   f"- CI method: {m['ci']}",
                   f"- Channel model: {m['channel']}",
                   f"- Physical sensing target: {m['target']}",
                   f"- PSL definition: {m['psl']}",
                   f"- Scientific question: {m['question']}", ""]
    (ICC_FIG / "README.md").write_text("\n".join(blocks))


def main() -> None:
    icc_dirs()
    apply_ieee_style()
    natural = _load("icc_natural_tdl_mc_raw.csv")
    natural_sum = _load("icc_natural_tdl_mc_summary.csv")
    ctrl = _load("icc_controlled_asymmetry_raw.csv")
    ctrl_sum = _load("icc_controlled_asymmetry_summary.csv")
    front = _load("icc_frontier_raw.csv")
    cp = _load("icc_cutting_plane_mc_raw.csv")
    mp = _load("icc_feasibility_map_raw.csv")
    numerology = _load("icc_numerology_raw.csv")
    ablation = _load("icc_ablation_raw.csv")
    fair = _load("icc_fair_comparison_raw.csv")
    if fair is not None and natural is None:
        natural = fair

    if natural is not None:
        fig1_natural_mismatch(natural)
        fig4_feasibility(natural)
        fig9_mc(natural)
    if ctrl is not None:
        fig2_controlled(ctrl)
    if front is not None:
        fig3_tradeoff(front)
        fig5_dense_psl(front)
        fig7_ee(front)
    if cp is not None:
        fig6_cutting_plane(cp)
    elif natural is not None:
        fig6_cutting_plane(natural)
    if mp is not None:
        fig8_map(mp)

    table1_parameters()
    table2_main(natural, ablation)
    table3_mc(natural_sum, natural)
    table4_controlled(ctrl_sum)
    table5_ablation(ablation)
    write_figure_readme()
    evidence = write_claims(natural, ctrl, cp if cp is not None else natural, front, numerology)
    write_index(evidence)
    pytest_line = Path(ICC_SUM / "pytest.txt").read_text() if (ICC_SUM / "pytest.txt").exists() else "see terminal / logs"
    write_final(evidence, pytest_line, "")
    print(f"wrote figures to {ICC_FIG}  notes={NOTES}")


if __name__ == "__main__":
    main()

"""IEEE-style plotting for ICC paper figures (PDF vector + PNG preview).

Figures have no plot titles (captions belong in the manuscript).  Method colours
and linestyles are consistent across figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from isac.evaluation.icc_suite import DISPLAY_NAME, METHOD_ORDER

IEEE_SINGLE_IN = 3.5
IEEE_DOUBLE_IN = 7.16
IEEE_HEIGHT_IN = 2.6
DPI_PNG = 300

METHOD_COLOR: dict[str, str] = {
    "uniform": "#7f7f7f",
    "water_filling": "#4c4c4c",
    "conventional_s2": "#d62728",
    "conventional_s2_psl": "#ff7f0e",
    "exact_efim": "#1f77b4",
    "exact_efim_sampled_psl": "#17becf",
    "exact_efim_cutting_plane_psl": "#2ca02c",
    "exact_efim_cutting_plane_psl_ee": "#9467bd",
}

METHOD_LS: dict[str, str] = {
    "uniform": ":",
    "water_filling": "--",
    "conventional_s2": "-.",
    "conventional_s2_psl": "-.",
    "exact_efim": "-",
    "exact_efim_sampled_psl": "-",
    "exact_efim_cutting_plane_psl": "-",
    "exact_efim_cutting_plane_psl_ee": "-",
}

METHOD_MARKER: dict[str, str] = {
    "uniform": "x",
    "water_filling": "+",
    "conventional_s2": "s",
    "conventional_s2_psl": "D",
    "exact_efim": "o",
    "exact_efim_sampled_psl": "^",
    "exact_efim_cutting_plane_psl": "v",
    "exact_efim_cutting_plane_psl_ee": "P",
}


def apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.1,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def method_label(method: str) -> str:
    return DISPLAY_NAME.get(method, method)


def save_ieee(fig: plt.Figure, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI_PNG)
    plt.close(fig)
    return pdf, png


def new_axes(*, double: bool = False, height: float | None = None) -> tuple[plt.Figure, plt.Axes]:
    apply_ieee_style()
    width = IEEE_DOUBLE_IN if double else IEEE_SINGLE_IN
    fig, ax = plt.subplots(figsize=(width, height or IEEE_HEIGHT_IN))
    return fig, ax


def style_of(method: str) -> dict[str, str]:
    return {
        "color": METHOD_COLOR.get(method, "#333333"),
        "linestyle": METHOD_LS.get(method, "-"),
        "marker": METHOD_MARKER.get(method, "o"),
        "label": method_label(method),
    }


def legend_outside(ax: plt.Axes, ncol: int = 2) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol, frameon=False, handlelength=2.2)


def plot_scatter_groups(
    groups: Mapping[str, tuple[np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    stem: Path,
    *,
    double: bool = False,
    identity: bool = False,
) -> tuple[Path, Path]:
    fig, ax = new_axes(double=double)
    for method, (x, y) in groups.items():
        st = style_of(method)
        ax.scatter(x, y, s=10, alpha=0.55, c=st["color"], marker=st["marker"], label=st["label"], linewidths=0)
    if identity:
        all_x = np.concatenate([np.asarray(v[0]) for v in groups.values()]) if groups else np.array([0.0, 1.0])
        lo, hi = float(np.nanmin(all_x)), float(np.nanmax(all_x))
        ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=0.8, label=r"$J=J_{\mathrm{req}}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    legend_outside(ax, ncol=2)
    return save_ieee(fig, stem)


def plot_lines_ci(
    x: np.ndarray,
    series: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    stem: Path,
    *,
    double: bool = False,
) -> tuple[Path, Path]:
    """``series[method] = (mean, ci_low, ci_high)`` aligned with ``x``."""
    fig, ax = new_axes(double=double)
    for method, (mean, lo, hi) in series.items():
        st = style_of(method)
        ax.plot(x, mean, ls=st["linestyle"], marker=st["marker"], color=st["color"], label=st["label"])
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.18, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    legend_outside(ax, ncol=2)
    return save_ieee(fig, stem)


def plot_box_methods(
    frame: pd.DataFrame,
    column: str,
    ylabel: str,
    stem: Path,
    methods: Sequence[str] | None = None,
    *,
    scale: float = 1.0,
    double: bool = False,
) -> tuple[Path, Path]:
    fig, ax = new_axes(double=double)
    order = list(methods or METHOD_ORDER)
    data, labels, colors = [], [], []
    for method in order:
        sub = frame[frame["method"] == method]
        if "status" in sub.columns:
            sub = sub[sub["status"] == "ok"]
        if column not in sub.columns or sub.empty:
            continue
        values = sub[column].to_numpy(dtype=float) * scale
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        data.append(values)
        labels.append(method_label(method))
        colors.append(METHOD_COLOR.get(method, "#333333"))
    if not data:
        ax.text(0.5, 0.5, "no finite samples", ha="center", va="center", transform=ax.transAxes)
        return save_ieee(fig, stem)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    return save_ieee(fig, stem)


def plot_cdf_methods(
    frame: pd.DataFrame,
    column: str,
    xlabel: str,
    stem: Path,
    methods: Sequence[str] | None = None,
    *,
    scale: float = 1.0,
    double: bool = False,
) -> tuple[Path, Path]:
    fig, ax = new_axes(double=double)
    order = list(methods or METHOD_ORDER)
    for method in order:
        sub = frame[frame["method"] == method]
        if "status" in sub.columns:
            sub = sub[sub["status"] == "ok"]
        if column not in sub.columns or sub.empty:
            continue
        values = np.sort(sub[column].to_numpy(dtype=float) * scale)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        y = np.arange(1, values.size + 1) / values.size
        st = style_of(method)
        ax.step(values, y, where="post", color=st["color"], ls=st["linestyle"], label=st["label"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    legend_outside(ax, ncol=2)
    return save_ieee(fig, stem)

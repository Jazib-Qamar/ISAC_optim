"""Matplotlib helpers shared by the experiments (title, axis labels with units, legend, grid).

All functions save to disk and close the figure; nothing is shown interactively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIG_SIZE = (9.0, 4.8)
DPI = 150


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_channel_gain(gain: np.ndarray, path: Path, title: str = "Channel power gain per subcarrier") -> Path:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(np.arange(gain.size), 10.0 * np.log10(gain), "o-", label=r"$|h_k|^2$")
    ax.set_xlabel("subcarrier index k")
    ax.set_ylabel("channel power gain [dB]")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best")
    return save_figure(fig, path)


def plot_power_allocations(
    allocations: Mapping[str, np.ndarray],
    path: Path,
    peak_power_w: float | None = None,
    title: str = "Power allocation per subcarrier",
) -> Path:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    styles = [("-", "."), ("--", "s"), ("-.", "^"), (":", "d"), ("-", "x"), ("--", "v"), ("-.", "o"), (":", "+")]
    for i, (label, power) in enumerate(allocations.items()):
        ls, marker = styles[i % len(styles)]
        ax.plot(np.arange(power.size), power * 1e3, ls=ls, marker=marker, ms=4, lw=1.2, label=label)
    if peak_power_w is not None:
        ax.axhline(peak_power_w * 1e3, color="k", ls=":", label="peak spectral power cap $P_{peak}$")
    ax.set_xlabel("subcarrier index k")
    ax.set_ylabel("power $P_k$ [mW]")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    return save_figure(fig, path)


def bar_metric(
    frame: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    path: Path,
    label_column: str = "label",
    scale: float = 1.0,
    log: bool = False,
    annotate: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    valid = frame[frame[column].notna()]
    values = valid[column].to_numpy(dtype=float) * scale
    labels = valid[label_column].tolist()
    bars = ax.bar(labels, values, color=plt.cm.tab10(np.arange(len(labels)) % 10))
    if annotate:
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.4g}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.4)
    ax.tick_params(axis="x", rotation=20)
    ax.legend([bars[0]], [column.replace("_", " ")], loc="best")
    return save_figure(fig, path)


def line_plot(
    x: np.ndarray,
    series: Mapping[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    logx: bool = False,
    logy: bool = False,
    markers: bool = True,
    hlines: Mapping[str, float] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for label, y in series.items():
        ax.plot(x, y, marker="o" if markers else None, ms=4, lw=1.4, label=label)
    for label, value in (hlines or {}).items():
        ax.axhline(value, ls="--", color="gray", label=label)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="best")
    return save_figure(fig, path)


def scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    label: str,
    color: np.ndarray | None = None,
    color_label: str | None = None,
    logx: bool = False,
    logy: bool = False,
    reference_line: tuple[np.ndarray, np.ndarray, str] | None = None,
    categories: Sequence[str] | None = None,
) -> Path:
    """Scatter plot; ``categories`` (one label per point) draws one legend entry per category."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    if categories is not None:
        cats = np.asarray(categories)
        for i, name in enumerate(dict.fromkeys(cats.tolist())):
            mask = cats == name
            ax.scatter(np.asarray(x)[mask], np.asarray(y)[mask], s=12, alpha=0.7, color=plt.cm.tab10(i % 10), label=f"{label}: {name}")
    else:
        sc = ax.scatter(x, y, s=12, alpha=0.7, c=color, cmap="viridis" if color is not None else None, label=label)
        if color is not None:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(color_label or "")
    if reference_line is not None:
        rx, ry, rlabel = reference_line
        ax.plot(rx, ry, "r--", lw=1.2, label=rlabel)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="best")
    return save_figure(fig, path)


def heatmap(
    matrix: np.ndarray,
    x_values: Sequence[float],
    y_label: str,
    x_label: str,
    color_label: str,
    title: str,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis",
                   extent=(min(x_values), max(x_values), -0.5, matrix.shape[0] - 0.5))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(color_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return save_figure(fig, path)


def boxplot_by_method(
    frame: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    path: Path,
    label_column: str = "label",
    scale: float = 1.0,
    log: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    groups = [(label, (sub[column].dropna().to_numpy(dtype=float) * scale)) for label, sub in frame.groupby(label_column, sort=False)]
    groups = [(label, values) for label, values in groups if values.size > 0]
    artists = ax.boxplot([v for _, v in groups], showfliers=True)
    ax.set_xticks(np.arange(1, len(groups) + 1))
    ax.set_xticklabels([label for label, _ in groups])
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.4)
    ax.tick_params(axis="x", rotation=20)
    ax.legend([artists["boxes"][0]], ["per-realisation distribution (box: IQR, whiskers: 1.5 IQR)"], loc="best")
    return save_figure(fig, path)

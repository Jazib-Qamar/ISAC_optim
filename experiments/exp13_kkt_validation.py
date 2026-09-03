"""Experiment 13 — KKT structure of the exact unknown-amplitude FIM.

Solves max-rate with the exact FIM constraint (no sampled-PSL SOCs) and
reports the power-weighted centroid, effective sensing weights
``(f_k - f_bar_P)^2`` versus ``f_k^2``, finite-difference gradient error, and
interior stationarity residuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from isac.evaluation.plots import line_plot, plot_power_allocations
from isac.evaluation.reporting import STAGE25_ROOT, experiment_dir, print_config, print_frame, print_header
from isac.evaluation.scenario import build_asymmetric_system, build_system, common_parser, config_from_args
from isac.optimization.feasibility import max_unknown_amplitude_fim
from isac.optimization.kkt_analysis import analyze_exact_fim_kkt
from isac.optimization.max_rate import solve_max_rate
from isac.optimization.sensing_spec import UNKNOWN_AMPLITUDE_EXACT

EXPERIMENT = "exp13_kkt_validation"


def _analyse(name, system, frac, rows, weights_store, allocations):
    j_max, _ = max_unknown_amplitude_fim(system)
    gamma_j = frac * j_max
    res = solve_max_rate(system, sensing_model=UNKNOWN_AMPLITUDE_EXACT, min_unknown_fim=gamma_j)
    report = analyze_exact_fim_kkt(res.power_allocation, system, min_unknown_fim=gamma_j, q_bit_per_j=0.0)
    row = report.as_dict()
    row.update({
        "channel": name,
        "fim_fraction": frac,
        "gamma_j": gamma_j,
        "j_unknown": res.metrics.delay_fim_unknown_amplitude_per_s2,
        "rate_spectral_efficiency": res.metrics.rate_spectral_efficiency,
        "tx_power_w": res.tx_power_w,
        "solver_status": res.solver_status,
    })
    rows.append(row)
    k = np.arange(system.num_subcarriers)
    weights_store[name] = (
        k, report.effective_weights_hz2, report.squared_frequency_weights_hz2, system.frequencies_hz,
    )
    allocations[name] = res.power_allocation
    return report


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--fim-fraction", type=float, default=0.5)
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = args.output_dir or experiment_dir(EXPERIMENT, STAGE25_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    print_config(cfg)

    rng = np.random.default_rng(cfg.simulation.seed)
    _, sys_sym = build_system(cfg, rng)
    _, sys_as = build_asymmetric_system(cfg, np.random.default_rng(cfg.simulation.seed + 17), tilt=4.0)

    rows: list[dict] = []
    weights: dict = {}
    allocs: dict = {}
    print_header("KKT reports")
    for name, system in (("symmetric", sys_sym), ("asymmetric", sys_as)):
        report = _analyse(name, system, args.fim_fraction, rows, weights, allocs)
        print(f"[{name}] centroid = {report.spectral_centroid_hz:.4e} Hz")
        print(f"         max |FD gradient error| = {report.gradient_max_abs_error:.4e} Hz^2")
        print(f"         max stationarity residual = {report.max_stationarity_residual:.4e}")
        print(f"         mean stationarity residual = {report.mean_stationarity_residual:.4e}")
        print(f"         FIM complementary slackness = {report.fim_complementary_slackness:.4e}")
        print(f"         power complementary slackness = {report.power_complementary_slackness:.4e}")
        print(f"         note: {report.note}")
        print()

    table = pd.DataFrame(rows)
    table.to_csv(out / "exp13_kkt_summary.csv", index=False)
    print_frame(table)

    for name, (k, w_eff, w_old, freqs) in weights.items():
        weight_frame = pd.DataFrame({
            "k": k, "f_hz": freqs, "w_eff_hz2": w_eff, "f_k_squared_hz2": w_old,
        })
        weight_frame.to_csv(out / f"exp13_weights_{name}.csv", index=False)
        line_plot(
            k,
            {r"$(f_k - \bar f_P)^2$": w_eff, r"$f_k^2$": w_old},
            "subcarrier index k",
            "sensing weight [Hz^2]",
            f"KKT effective sensing weight vs f_k^2 ({name} channel)",
            out / f"exp13_weights_{name}.png",
            markers=False,
        )
    plot_power_allocations(
        allocs, out / "exp13_allocations.png", peak_power_w=sys_sym.peak_power_w,
        title="Exact-FIM max-rate allocations used for the KKT check",
    )
    print(f"Saved outputs under {out}")


if __name__ == "__main__":
    main()

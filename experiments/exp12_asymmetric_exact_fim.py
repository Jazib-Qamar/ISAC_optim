"""Experiment 12 — asymmetric channel: linear surrogate vs exact unknown-amplitude FIM.

Constructs a reproducible one-sided spectral tilt so that strong communication
subcarriers sit on one side of the band.  Compares

* communication-only water filling
* max-rate with the old linear surrogate ``S(P) = sum f_k^2 P_k``
* max-rate with the exact unknown-amplitude FIM ``G(P) = S2 - S1^2/S0``

Relative requirements use the same fraction of each metric's own maximum
(``0.6 S_max`` vs ``0.6 J_max``).  That is an analogous operating point, not
an identical constraint.  Mainlobe exclusion 1/B and 2/B are both reported.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from isac.evaluation.plots import bar_metric, plot_channel_gain, plot_power_allocations
from isac.evaluation.reporting import STAGE25_ROOT, experiment_dir, print_config, print_frame, print_header
from isac.evaluation.scenario import build_asymmetric_system, build_system, common_parser, config_from_args
from isac.evaluation.stage25 import four_formulation_rows
from isac.sensing.ambiguity import default_mainlobe_exclusion_s

EXPERIMENT = "exp12_asymmetric_exact_fim"


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--tilt", type=float, default=4.0)
    parser.add_argument("--sensing-fraction", type=float, default=0.6)
    parser.add_argument("--fim-fraction", type=float, default=0.6)
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = args.output_dir or experiment_dir(EXPERIMENT, STAGE25_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    print_config(cfg)

    rng = np.random.default_rng(cfg.simulation.seed)
    gain_sym, sys_sym = build_system(cfg, rng)
    rng_a = np.random.default_rng(cfg.simulation.seed + 17)
    gain_as, sys_as = build_asymmetric_system(cfg, rng_a, tilt=args.tilt, strong_side="negative")

    table_bits = []
    for channel_name, gain, system in (
        ("symmetric_rayleigh", gain_sym, sys_sym),
        ("asymmetric_tilt", gain_as, sys_as),
    ):
        chunk = four_formulation_rows(
            system, sensing_fraction=args.sensing_fraction, fim_fraction=args.fim_fraction, psl_max_db=None,
        )
        for row in chunk:
            row["channel"] = channel_name
            row["exclusion"] = "1/B"
            table_bits.append(row)
        # 2/B exclusion: only changes PSL/ISL evaluation, not the FIM constraint.
        sys_2b = dataclasses.replace(
            system, mainlobe_exclusion_s=default_mainlobe_exclusion_s(system.bandwidth_hz, 2.0),
        )
        chunk2 = four_formulation_rows(
            sys_2b, sensing_fraction=args.sensing_fraction, fim_fraction=args.fim_fraction, psl_max_db=None,
        )
        for row in chunk2:
            row["channel"] = channel_name
            row["exclusion"] = "2/B"
            table_bits.append(row)

    table = pd.DataFrame(table_bits)
    table.to_csv(out / "exp12_comparison.csv", index=False)
    print_header("Symmetric vs asymmetric channel (1/B exclusion)")
    sub = table[(table["exclusion"] == "1/B")]
    cols = [c for c in (
        "channel", "method", "status", "rate_spectral_efficiency", "sensing_surrogate",
        "delay_fim_known_amplitude_per_s2", "delay_fim_unknown_amplitude_per_s2",
        "spectral_centroid_hz", "power_weighted_spectral_variance_hz2",
        "delay_crb_s2", "psl_db", "isl_db",
    ) if c in sub.columns]
    print_frame(sub[cols])

    print_header("Mainlobe exclusion sensitivity (PSL at 1/B vs 2/B)")
    psl_cols = [c for c in ("channel", "method", "exclusion", "psl_db", "isl_db", "mainlobe_exclusion_s") if c in table.columns]
    print_frame(table[psl_cols])

    plot_channel_gain(gain_sym, out / "exp12_channel_symmetric.png", "Symmetric Rayleigh |h_k|^2")
    plot_channel_gain(gain_as, out / "exp12_channel_asymmetric.png",
                      "Asymmetric (negative-frequency tilt) |h_k|^2")

    # Recover allocations for the asymmetric 1/B case by re-reading methods from a fresh solve.
    from isac.communication.water_filling import water_filling
    from isac.evaluation.stage25 import reference_levels
    from isac.optimization.max_rate import solve_max_rate
    from isac.optimization.sensing_spec import KNOWN_AMPLITUDE_LINEAR, UNKNOWN_AMPLITUDE_EXACT

    refs = reference_levels(sys_as)
    wf = water_filling(sys_as.channel_gain, sys_as.noise_power_w, sys_as.total_power_w, sys_as.peak_power_w)
    lin = solve_max_rate(sys_as, min_sensing_surrogate=args.sensing_fraction * refs["s_max"],
                         sensing_model=KNOWN_AMPLITUDE_LINEAR)
    ex = solve_max_rate(sys_as, sensing_model=UNKNOWN_AMPLITUDE_EXACT, min_unknown_fim=args.fim_fraction * refs["j_max"])
    plot_power_allocations(
        {"Water-filling": wf.power_w, "Linear S(P)": lin.power_allocation, "Exact unknown FIM": ex.power_allocation},
        out / "exp12_power_asymmetric.png", peak_power_w=sys_as.peak_power_w,
        title="Asymmetric channel: linear surrogate vs exact unknown-amplitude FIM",
    )

    asy = table[(table["channel"] == "asymmetric_tilt") & (table["exclusion"] == "1/B") & (table["status"] == "ok")]
    if not asy.empty:
        bar_metric(asy, "delay_fim_known_amplitude_per_s2", "known-amplitude J_tau [1/s^2]",
                   "Known-amplitude FIM (asymmetric channel)", out / "exp12_known_fim.png",
                   label_column="method")
        bar_metric(asy, "delay_fim_unknown_amplitude_per_s2", "unknown-amplitude J_tau [1/s^2]",
                   "Unknown-amplitude FIM (asymmetric channel)", out / "exp12_unknown_fim.png",
                   label_column="method")
        bar_metric(asy, "spectral_centroid_hz", "spectral centroid f_bar_P [Hz]",
                   "Power-weighted spectral centroid (asymmetric channel)", out / "exp12_centroid.png",
                   label_column="method")
        # Fix ylabel: scale 1e-3 would be kHz if we say so
        bar_metric(asy, "rate_spectral_efficiency", "sum spectral efficiency [bit/s/Hz]",
                   "Rate (asymmetric channel)", out / "exp12_rate.png", label_column="method")

    print(f"Saved outputs under {out}")


if __name__ == "__main__":
    main()

"""Experiment 14 — cutting-plane sampled-PSL generation.

Starts from a coarse optimisation delay grid, solves exact-FIM max-rate with
sampled-PSL SOCs, evaluates PSL on a denser validation grid, and adds the
worst-violating delay until the validation PSL meets the request (within
tolerance) or the iteration cap is hit.
"""

from __future__ import annotations

import numpy as np

from isac.communication.water_filling import uniform_power
from isac.evaluation.plots import line_plot, plot_power_allocations
from isac.evaluation.reporting import STAGE25_ROOT, experiment_dir, print_config, print_frame, print_header
from isac.evaluation.scenario import build_system, common_parser, config_from_args
from isac.optimization.ambiguity_constraints import validation_delay_grid
from isac.optimization.cutting_plane import solve_with_psl_cutting_plane
from isac.optimization.feasibility import max_unknown_amplitude_fim
from isac.optimization.max_rate import solve_max_rate
from isac.optimization.sensing_spec import UNKNOWN_AMPLITUDE_EXACT
from isac.sensing.ambiguity import peak_sidelobe_level_db

EXPERIMENT = "exp14_psl_cutting_plane"


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--fim-fraction", type=float, default=0.4)
    parser.add_argument("--max-iterations", type=int, default=12)
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = args.output_dir or experiment_dir(EXPERIMENT, STAGE25_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    print_config(cfg)

    rng = np.random.default_rng(cfg.simulation.seed)
    _, system = build_system(cfg, rng)
    j_max, _ = max_unknown_amplitude_fim(system)
    gamma_j = args.fim_fraction * j_max
    val_grid = validation_delay_grid(system, cfg.ambiguity.validation_oversampling_factor)
    uniform = uniform_power(system.num_subcarriers, system.total_power_w, system.peak_power_w)
    psl_uniform = peak_sidelobe_level_db(uniform, system.frequencies_hz, val_grid, system.mainlobe_exclusion_s)

    print_header("Cutting-plane setup")
    print(f"  Gamma_J = {args.fim_fraction:.2f} J_max = {gamma_j:.6e} 1/s^2")
    print(f"  PSL_max = uniform validation-grid PSL = {psl_uniform:.3f} dB")
    print(f"  optimisation oversampling = 2 (coarse, to expose dense-grid violations)")
    print(f"  validation oversampling   = {cfg.ambiguity.validation_oversampling_factor}")
    print()

    result = solve_with_psl_cutting_plane(
        solve_max_rate,
        system,
        psl_max_db=psl_uniform,
        optimization_oversampling=2,  # deliberately coarse so the dense grid can violate
        validation_oversampling=cfg.ambiguity.validation_oversampling_factor,
        max_iterations=args.max_iterations,
        psl_tolerance_db=0.05,
        solve_kwargs={"sensing_model": UNKNOWN_AMPLITUDE_EXACT, "min_unknown_fim": gamma_j},
    )
    hist = result.history_frame()
    hist.to_csv(out / "exp14_cutting_plane_history.csv", index=False)
    print_header("Cutting-plane history")
    print_frame(hist)
    print(f"converged = {result.converged}")
    extras = {k: result.result.extra.get(k) for k in (
        "requested_psl_max_db", "optimization_grid_psl_db", "validation_grid_psl_db",
        "worst_delay_s", "violation_margin_db", "cutting_plane_iterations",
    )}
    print("Final dual-grid PSL diagnostics:", extras)

    if not hist.empty:
        line_plot(
            hist["iteration"].to_numpy(),
            {"validation PSL": hist["worst_psl_db"].to_numpy()},
            "cutting-plane iteration",
            "validation-grid PSL [dB]",
            "Cutting-plane PSL (dense validation grid)",
            out / "exp14_psl_convergence.png",
            hlines={"PSL_max": psl_uniform},
        )
        line_plot(
            hist["iteration"].to_numpy(),
            {"number of PSL SOCs": hist["num_psl_constraints"].to_numpy()},
            "cutting-plane iteration",
            "number of sampled-PSL SOC constraints",
            "Cutting-plane constraint-set size",
            out / "exp14_constraint_count.png",
        )
    plot_power_allocations(
        {"cutting-plane solution": result.result.power_allocation, "uniform": uniform},
        out / "exp14_allocation.png", peak_power_w=system.peak_power_w,
        title="Exact-FIM + cutting-plane sampled PSL allocation",
    )
    print(f"Saved outputs under {out}")


if __name__ == "__main__":
    main()

# Final ICC experimental report

This report is generated from `results/icc_paper/raw_data/` after the ICC experiment suite.
Exploratory Stage 1/2/2.5 outputs remain in `results/stage2/` and `results/stage2_5/`.

## 1. Pytest

........................................................................ [ 57%]
.....................................................                    [100%]
=============================== warnings summary ===============================
tests/test_proposed_cutting_plane.py::test_cutting_plane_detects_sampled_miss_and_adds_dense_cuts
  /workspace/isac/optimization/solver.py:96: UserWarning: Solution may be inaccurate. Try another solver, adjusting the solver settings, or solve with verbose=True for more information.
    problem.solve(solver=solver_name, **options.get(solver_name, {}))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
125 passed, 1 warning in 8.93s


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

- Natural-channel sample size: 500
- Natural-channel methods: ['conventional_s2', 'conventional_s2_psl', 'exact_efim', 'exact_efim_cutting_plane_psl', 'exact_efim_sampled_psl', 'uniform', 'water_filling']
- Natural-channel EE method included?: False
- Natural-channel mean FIM mismatch (conventional S2): 0.7337 %
- Natural-channel median FIM mismatch: 0.0118 %
- Natural-channel p95 / max FIM mismatch: 3.317 % / 23.46 %
- Mismatch >1%/2%/5%/10%: 12.20% / 7.80% / 4.20% / 2.20% of realisations
- Pearson r(|f_bar|, ε_J): 0.92
- Spearman ρ: 1
- Conventional false sensing-feasibility rate (solver tolerance): 88.00%
- Conventional *material* (>1% FIM shortfall) false-feasibility: 11.20%
- Exact-EFIM false sensing-feasibility rate: 0.00%
- Conventional vs exact rate ΔR mean: 0.02805 %
- Sampled-PSL dense violation rate: 5.50%
- Conventional S2+PSL dense violation rate: 11.50%
- Cutting-plane dense violation rate: 0.00%
- Mean cutting-plane iterations: 1.05
- Mean cutting-plane solve time: 0.546 s
- Empirical FIM–PSL feasibility boundary (Γ_J/J_max, 50% channels): 0.5
- EE gain vs max-rate (mean %): 27.9
- Associated rate reduction (mean %): 16.6
- Mismatch by K: {64: 1.0890591732082533, 128: 0.7436881978103921, 256: 0.9274026199843484}
- Numerology methods by K: {64: ['conventional_s2', 'conventional_s2_psl', 'exact_efim', 'exact_efim_cutting_plane_psl', 'exact_efim_sampled_psl', 'uniform', 'water_filling'], 128: ['conventional_s2', 'exact_efim', 'exact_efim_cutting_plane_psl', 'uniform', 'water_filling'], 256: ['conventional_s2', 'exact_efim', 'uniform', 'water_filling']}
- Optimisation-result rows stored: 8060

- Suite wall-clock: 2049.8 s (34.2 min)

## Questions (evidence only)

### QUESTION 1
Does conventional uncentered S2 produce an optimistic ranging-feasibility assessment when complex reflectivity is unknown?

**PARTIALLY**

Natural-channel mean mismatch 0.734%; false-feasibility 88.00%.
A YES requires both a non-trivial mismatch and false-feasibility under the chosen Γ_J.

### QUESTION 2
Does this mismatch occur naturally in TDL channels, or primarily under controlled asymmetry?

**primarily under artificially controlled asymmetry**

Natural mean ε_J = 0.734%. Controlled mismatch-by-a = {0.0: 0.0435450264286804, 0.5: 0.0989920885105957, 1.0: 0.08938588196356494, 1.5: 0.13564783167942385, 2.0: 0.31879652234558664, 3.0: 1.2733423919275297, 4.0: 3.7934240389823826, 5.0: 9.419855753376265, 6.0: 20.15340921439894}.

### QUESTION 3
Association with spectral-centroid displacement: Pearson r = 0.92, Spearman ρ = 1.

### QUESTION 4
Exact EFIM false-feasibility rate 0.00% vs conventional 88.00%.
Exact EFIM constrains the evaluated metric, so residual false-feasibility can only come from solver tolerance.

### QUESTION 5
Communication-rate cost of exact EFIM vs conventional S2 under the same physical target:
mean ΔR = 0.02805%,
median 0.0003493%,
CI [0.0187, 0.03739],
worst (p95) 0.1042%.

### QUESTION 6
Cutting-plane dense-grid PSL: sampled violation 5.50% → cutting-plane 0.00%.

### QUESTION 7
Observed FIM/PSL frontier: proposed method dense-feasible on ≥50% of channels up to Γ_J/J_max ≈ 0.5 with the uniform-spectrum PSL request.

### QUESTION 8
EE optimisation mean EE change 27.9% with mean rate change 16.6% relative to max-rate under the same constraints.

### QUESTION 9
Numerology (Δf fixed): mismatch-by-K {64: 1.0890591732082533, 128: 0.7436881978103921, 256: 0.9274026199843484}. Qualitative consistency requires the same sign of mismatch and of ΔR at each K.

### QUESTION 10
Strongest scientifically defensible ICC claim:

> When complex reflectivity is unknown, conventional uncentered S2 is an optimistic proxy for delay information on spectrally asymmetric allocations; exact EFIM removes that optimism at a measurable but scenario-dependent rate cost, and sampled PSL SOCs require cutting-plane / dense-grid validation before a PSL claim is made.

## Assumptions and limitations

- Single user, single point target, no clutter / self-interference / mobility.
- Unknown-amplitude delay EFIM is the Stage 1 Schur-complement model; it is not a new information-theoretic result.
- Sampled + cutting-plane PSL is an inner approximation, not a continuous-delay certificate.
- Conventional S2 is given the mapped threshold Γ_S = Γ_J / C_β so that J_known ≥ Γ_J is what it *claims*.
- Natural TDL uses a causal exponential PDP; that is frequency-selective but not sign-tilted by construction.
- Controlled tilt is synthetic.
- No published Yang/Iqbal (or other paper) baseline is plotted, because none was implemented.
- Dinkelbach, SOCs, water-filling and CRB optimisation are not claimed as novel.
- Natural N=500 Monte Carlo omits `exact_efim_cutting_plane_psl_ee` (present in frontier and ablation).
- Numerology holds Δf fixed so B=KΔf; K=256 ran cheap methods only (no PSL SOCs).
- Power-vector NPZ dumps are not stored; CSVs are the archival format.

## Files under results/icc_paper/

- FIGURE_INDEX.md
- configs/icc_ablation.json
- configs/icc_controlled_asymmetry.json
- configs/icc_cutting_plane_mc.json
- configs/icc_fair_comparison.json
- configs/icc_feasibility_map.json
- configs/icc_frontier.json
- configs/icc_natural_tdl_mc.json
- configs/icc_numerology.json
- figures/README.md
- figures/fig1_natural_asymmetry_mismatch.pdf
- figures/fig1_natural_asymmetry_mismatch.png
- figures/fig2a_controlled_mismatch.pdf
- figures/fig2a_controlled_mismatch.png
- figures/fig2b_controlled_centroid.pdf
- figures/fig2b_controlled_centroid.png
- figures/fig3_rate_vs_ranging.pdf
- figures/fig3_rate_vs_ranging.png
- figures/fig4_actual_sensing_feasibility.pdf
- figures/fig4_actual_sensing_feasibility.png
- figures/fig5_dense_psl.pdf
- figures/fig5_dense_psl.png
- figures/fig6_cutting_plane_refinement.pdf
- figures/fig6_cutting_plane_refinement.png
- figures/fig7_ee_vs_ranging.pdf
- figures/fig7_ee_vs_ranging.png
- figures/fig8_feasibility_map.pdf
- figures/fig8_feasibility_map.png
- figures/fig9a_cdf_rate.pdf
- figures/fig9a_cdf_rate.png
- figures/fig9b_box_unknown_fim.pdf
- figures/fig9b_box_unknown_fim.png
- figures/fig9c_box_dense_psl.pdf
- figures/fig9c_box_dense_psl.png
- figures/fig9d_box_ee.pdf
- figures/fig9d_box_ee.png
- logs/experiments_icc_ablation.exit
- logs/experiments_icc_controlled_asymmetry.exit
- logs/experiments_icc_cutting_plane_mc.exit
- logs/experiments_icc_fair_comparison.exit
- logs/experiments_icc_feasibility_map.exit
- logs/experiments_icc_frontier.exit
- logs/experiments_icc_generate_paper.exit
- logs/experiments_icc_natural_tdl_mc.exit
- logs/experiments_icc_numerology.exit
- logs/icc_ablation.log
- logs/icc_controlled_asymmetry.log
- logs/icc_cutting_plane_mc.log
- logs/icc_fair_comparison.log
- logs/icc_feasibility_map.log
- logs/icc_frontier.log
- logs/icc_natural_tdl_mc.log
- logs/icc_numerology.log
- raw_data/icc_ablation_raw.csv
- raw_data/icc_controlled_asymmetry_raw.csv
- raw_data/icc_controlled_asymmetry_summary.csv
- raw_data/icc_cutting_plane_mc_raw.csv
- raw_data/icc_cutting_plane_mc_summary.csv
- raw_data/icc_fair_comparison_raw.csv
- raw_data/icc_fair_comparison_summary.csv
- raw_data/icc_feasibility_map_raw.csv
- raw_data/icc_frontier_raw.csv
- raw_data/icc_natural_tdl_mc_mismatch_stats.csv
- raw_data/icc_natural_tdl_mc_raw.csv
- raw_data/icc_natural_tdl_mc_summary.csv
- raw_data/icc_numerology_raw.csv
- raw_data/icc_numerology_summary.csv
- summary/FIGURE_INDEX.md
- summary/FINAL_ICC_RESULTS.md
- summary/PAPER_CLAIMS.md
- summary/pytest.txt
- summary/suite_runtime_s.txt
- tables/table1_simulation_parameters.csv
- tables/table1_simulation_parameters.md
- tables/table1_simulation_parameters.tex
- tables/table2_baseline_comparison.csv
- tables/table2_baseline_comparison.md
- tables/table2_baseline_comparison.tex
- tables/table3_natural_mc_statistics.csv
- tables/table3_natural_mc_statistics.md
- tables/table3_natural_mc_statistics.tex
- tables/table4_controlled_asymmetry.csv
- tables/table4_controlled_asymmetry.md
- tables/table4_controlled_asymmetry.tex
- tables/table5_ablation.csv
- tables/table5_ablation.md
- tables/table5_ablation.tex

## Missing raw files

- none

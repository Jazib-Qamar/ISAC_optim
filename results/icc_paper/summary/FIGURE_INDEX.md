# ICC figure index

Generated from stored raw data. Hypothesis verdicts use the numerical evidence only.

## Fig. 1 — `fig1_natural_asymmetry_mismatch`
- **Scientific question:** Does naturally occurring communication-driven spectral asymmetry produce meaningful error in the conventional sensing metric?
- **Baselines:** Conventional S2, Exact EFIM
- **Axes:** x = |f_bar_P| [kHz]; y = epsilon_J [%]
- **Sample size:** 500
- **Key numerical finding:** N=500 indoor TDL: conventional mean ε_J=0.734% (median 0.0118%, p95 3.32%, max 23.5%). Mismatch >1/2/5/10%: 12.20% / 7.80% / 4.20% / 2.20%. Pearson r(|f̄_P|, ε_J)=0.92, Spearman ρ=1.
- **Hypothesis verdict:** PARTIALLY SUPPORTS
- **Recommended section:** Numerical Results — Natural frequency-selective channels

## Fig. 4 — `fig4_actual_sensing_feasibility`
- **Scientific question:** Does conventional S2 claim sensing feasibility while the independent unknown-reflectivity evaluator disagrees?
- **Baselines:** Conventional S2, Exact EFIM
- **Axes:** x = J_required; y = J_actual,unknown
- **Sample size:** 500
- **Key numerical finding:** Conventional S2 false ranging-feasibility 88.00% at solver tolerance; material (>1% FIM shortfall) 11.20%. Exact EFIM false-feasibility 0.00%.
- **Hypothesis verdict:** SUPPORTS
- **Recommended section:** Numerical Results — False sensing feasibility

## Fig. 9 — `fig9a–d Monte Carlo distributions`
- **Scientific question:** How do methods compare statistically on natural TDL channels?
- **Baselines:** Uniform, Water-filling, Conventional $S_2$, Conventional $S_2$+PSL, Exact EFIM, Exact EFIM+sampled PSL, Exact EFIM+cutting-plane PSL
- **Axes:** x = method / metric value; y = CDF / distribution
- **Sample size:** 500
- **Key numerical finding:** N=500 TDL, 7 methods (no EE). Mean S2 ε_J=0.734%. The N=500 natural MC omits exact_efim_cutting_plane_psl_ee; EE is in the frontier and ablation.
- **Hypothesis verdict:** PARTIAL
- **Recommended section:** Numerical Results — Monte Carlo comparison

## Fig. 2 — `fig2a_controlled_mismatch / fig2b_controlled_centroid`
- **Scientific question:** Does increasing spectral asymmetry systematically increase the nuisance-related sensing-model mismatch?
- **Baselines:** Conventional S2, Exact EFIM
- **Axes:** x = asymmetry strength a; y = epsilon_J and |f_bar_P|
- **Sample size:** 360 (9 asymmetry levels × 40 realisations of synthetic logistic tilt)
- **Key numerical finding:** Synthetic logistic tilt (40 realisations × 9 values of a): mean conventional ε_J rises from 0.0435% at a=0.0 to 20.2% at a=6.0. Do not cite as a 3GPP/TDL result.
- **Hypothesis verdict:** SUPPORTS
- **Recommended section:** Numerical Results — Controlled asymmetry mechanism

## Fig. 3 — `fig3_rate_vs_ranging`
- **Scientific question:** What communication rate is achieved under a common physical ranging requirement?
- **Baselines:** Water-filling, Conventional $S_2$, Conventional $S_2$+PSL, Exact EFIM, Exact EFIM+cutting-plane PSL, Exact EFIM+cutting-plane PSL+EE
- **Axes:** x = target range RMSE [m]; y = rate [Mbit/s]
- **Sample size:** 12
- **Key numerical finding:** 12 TDL channels × FIM-fraction sweep. Proposed cutting-plane remains dense-feasible on ≥50% of channels up to Γ_J/J_max ≈ 0.5. PSL methods become infeasible at high Γ_J (see raw frontier CSV).
- **Hypothesis verdict:** PARTIAL
- **Recommended section:** Numerical Results — Rate–ranging tradeoff

## Fig. 5 — `fig5_dense_psl`
- **Scientific question:** How does independent dense-grid PSL compare across methods under a common ranging requirement?
- **Baselines:** Water-filling, Conventional $S_2$, Conventional $S_2$+PSL, Exact EFIM, Exact EFIM+sampled PSL, Exact EFIM+cutting-plane PSL
- **Axes:** x = FIM fraction; y = dense-validation PSL [dB]
- **Sample size:** 12
- **Key numerical finding:** Independent dense-grid PSL is the paper claim. Sampled-SOC dense violation 5.50%; conventional S2+PSL 11.50%; cutting-plane 0.00%.
- **Hypothesis verdict:** SUPPORTS
- **Recommended section:** Numerical Results — Ambiguity

## Fig. 7 — `fig7_ee_vs_ranging`
- **Scientific question:** Does EE optimisation provide a system-level gain once exact EFIM and dense PSL are enforced?
- **Baselines:** Water-filling, Conventional $S_2$+PSL, Exact EFIM+cutting-plane PSL, Exact EFIM+cutting-plane PSL+EE
- **Axes:** x = target range RMSE [m]; y = EE [Mbit/J]
- **Sample size:** 12
- **Key numerical finding:** Dinkelbach EE under exact EFIM+cutting-plane PSL: mean EE change 27.9% vs max-rate, mean rate change 16.6% (positive = max-rate had higher rate). Dinkelbach is the solver, not a claimed novelty.
- **Hypothesis verdict:** SUPPORTS
- **Recommended section:** Numerical Results — Energy efficiency

## Fig. 6 — `fig6_cutting_plane_refinement`
- **Scientific question:** Does cutting-plane refinement solve the dense-grid PSL miss of sampled SOCs?
- **Baselines:** Conventional S2+PSL, Exact EFIM+sampled PSL, Exact EFIM+cutting-plane PSL
- **Axes:** x = method / PSL margin; y = violation rate / CDF
- **Sample size:** 200
- **Key numerical finding:** N=200 TDL: sampled dense-PSL violation 5.50% → cutting-plane 0.00%, mean iterations 1.05, mean runtime 0.546 s.
- **Hypothesis verdict:** SUPPORTS
- **Recommended section:** Numerical Results — Cutting-plane PSL

## Fig. 8 — `fig8_feasibility_map`
- **Scientific question:** Where is the joint FIM–PSL feasibility frontier for the proposed method?
- **Baselines:** Proposed cutting-plane (sampled-only overlay in raw data)
- **Axes:** x = FIM fraction; y = requested PSL [dB]
- **Sample size:** 4
- **Key numerical finding:** Empirical joint (Γ_J, PSL) map on 4 TDL channels. Proposed method dense-feasible on ≥50% of frontier channels up to Γ_J/J_max ≈ 0.5 with the uniform-spectrum PSL request.
- **Hypothesis verdict:** PARTIAL
- **Recommended section:** Numerical Results — Feasibility frontier

## Recommended 5–7 manuscript items (space-limited ICC)

- Fig. 1 (natural mismatch)
- Fig. 2 (controlled mechanism)
- Fig. 4 (actual vs required J)
- Fig. 6 (cutting-plane PSL)
- Fig. 3 (rate–ranging)
- Table II (main comparison)
- Table III (MC stats)

Do not include Fig. 9 panels that merely restate Table III, or Fig. 8 if the map is dominated by a single colour.

## Scope notes (do not silently over-claim)

- Natural N=500 Monte Carlo omits `exact_efim_cutting_plane_psl_ee` (Dinkelbach+cutting-plane). EE is reported from the frontier and ablation.
- Numerology holds Δf=15 kHz fixed so occupied bandwidth B=KΔf scales with K. K=256 ran cheap methods only (no PSL SOCs).
- Archival format is CSV + JSON configs. Power-vector NPZ dumps are not stored.

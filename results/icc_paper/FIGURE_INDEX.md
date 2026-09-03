# ICC figure index

Generated from stored raw data. Hypothesis verdicts use the numerical evidence only.

## Fig. 1 — `fig1_natural_asymmetry_mismatch`
- **Scientific question:** Does naturally occurring communication-driven spectral asymmetry produce meaningful error in the conventional sensing metric?
- **Baselines:** Conventional S2, Exact EFIM
- **Axes:** x = |f_bar_P| [kHz]; y = epsilon_J [%]
- **Sample size:** 500
- **Hypothesis verdict:** PARTIALLY SUPPORTS
- **Recommended section:** Numerical Results — Natural frequency-selective channels

## Fig. 4 — `fig4_actual_sensing_feasibility`
- **Scientific question:** Does conventional S2 claim sensing feasibility while the independent unknown-reflectivity evaluator disagrees?
- **Baselines:** Conventional S2, Exact EFIM
- **Axes:** x = J_required; y = J_actual,unknown
- **Sample size:** 500
- **Hypothesis verdict:** SUPPORTS
- **Recommended section:** Numerical Results — False sensing feasibility

## Fig. 9 — `fig9a–d Monte Carlo distributions`
- **Scientific question:** How do methods compare statistically on natural TDL channels?
- **Baselines:** Uniform, Water-filling, Conventional $S_2$, Conventional $S_2$+PSL, Exact EFIM, Exact EFIM+sampled PSL, Exact EFIM+cutting-plane PSL
- **Axes:** x = method / metric value; y = CDF / distribution
- **Sample size:** 500
- **Hypothesis verdict:** PARTIAL
- **Recommended section:** Numerical Results — Monte Carlo comparison

## Fig. 2 — `fig2a_controlled_mismatch / fig2b_controlled_centroid`
- **Scientific question:** Does increasing spectral asymmetry systematically increase the nuisance-related sensing-model mismatch?
- **Baselines:** Conventional S2, Exact EFIM
- **Axes:** x = asymmetry strength a; y = epsilon_J and |f_bar_P|
- **Sample size:** 360
- **Hypothesis verdict:** PARTIAL
- **Recommended section:** Numerical Results — Controlled asymmetry mechanism

## Fig. 3 — `fig3_rate_vs_ranging`
- **Scientific question:** What communication rate is achieved under a common physical ranging requirement?
- **Baselines:** Water-filling, Conventional $S_2$, Conventional $S_2$+PSL, Exact EFIM, Exact EFIM+cutting-plane PSL, Exact EFIM+cutting-plane PSL+EE
- **Axes:** x = target range RMSE [m]; y = rate [Mbit/s]
- **Sample size:** 12
- **Hypothesis verdict:** PARTIAL
- **Recommended section:** Numerical Results — Rate–ranging tradeoff

## Fig. 5 — `fig5_dense_psl`
- **Scientific question:** How does independent dense-grid PSL compare across methods under a common ranging requirement?
- **Baselines:** Water-filling, Conventional $S_2$, Conventional $S_2$+PSL, Exact EFIM, Exact EFIM+sampled PSL, Exact EFIM+cutting-plane PSL
- **Axes:** x = FIM fraction; y = dense-validation PSL [dB]
- **Sample size:** 12
- **Hypothesis verdict:** PARTIAL
- **Recommended section:** Numerical Results — Ambiguity

## Fig. 7 — `fig7_ee_vs_ranging`
- **Scientific question:** Does EE optimisation provide a system-level gain once exact EFIM and dense PSL are enforced?
- **Baselines:** Water-filling, Conventional $S_2$+PSL, Exact EFIM+cutting-plane PSL, Exact EFIM+cutting-plane PSL+EE
- **Axes:** x = target range RMSE [m]; y = EE [Mbit/J]
- **Sample size:** 12
- **Hypothesis verdict:** PARTIAL
- **Recommended section:** Numerical Results — Energy efficiency

## Fig. 6 — `fig6_cutting_plane_refinement`
- **Scientific question:** Does cutting-plane refinement solve the dense-grid PSL miss of sampled SOCs?
- **Baselines:** Conventional S2+PSL, Exact EFIM+sampled PSL, Exact EFIM+cutting-plane PSL
- **Axes:** x = method / PSL margin; y = violation rate / CDF
- **Sample size:** 200
- **Hypothesis verdict:** SUPPORTS
- **Recommended section:** Numerical Results — Cutting-plane PSL

## Fig. 8 — `fig8_feasibility_map`
- **Scientific question:** Where is the joint FIM–PSL feasibility frontier for the proposed method?
- **Baselines:** Proposed cutting-plane (sampled-only overlay in raw data)
- **Axes:** x = FIM fraction; y = requested PSL [dB]
- **Sample size:** 4
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

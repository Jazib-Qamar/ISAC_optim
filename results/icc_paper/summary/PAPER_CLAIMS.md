# Paper-facing statistical claims

Every claim below is generated from stored raw CSVs. Do not cite a claim if its raw file is missing.

## Claim 1
- **Claim:** Across 500 natural exponential-PDP TDL realisations, conventional S2 allocations had mean unknown-vs-known FIM mismatch 0.734% (median 0.0118%, 95% CI [0.506, 0.961]).
- **Experiment:** `icc_natural_tdl_mc`
- **Raw data:** `results/icc_paper/raw_data/icc_natural_tdl_mc_raw.csv`
- **Sample size:** 500
- **Evidence type:** Monte Carlo
- **Limitations:** Mismatch is allocation-dependent; TDL RMS delay spread is a modelling choice.

## Claim 2
- **Claim:** Under the same physical unknown-FIM target, replacing conventional S2 with exact EFIM changed rate by mean ΔR = 0.028% (median 0.000349%, 95% CI [0.0187, 0.0374]; positive means conventional had higher rate).
- **Experiment:** `icc_natural_tdl_mc`
- **Raw data:** `results/icc_paper/raw_data/icc_natural_tdl_mc_raw.csv`
- **Sample size:** 500
- **Evidence type:** Monte Carlo
- **Limitations:** Compared only on realisations where both methods returned a primal.

## Claim 3
- **Claim:** False sensing-feasibility rate (claims S-constraint met but independent J_unknown misses Γ_J within solver tolerance): conventional S2 88.00%; exact EFIM 0.00%. Material (>1% relative shortfall) false-feasibility: conventional S2 11.20%.
- **Experiment:** `icc_natural_tdl_mc`
- **Raw data:** `results/icc_paper/raw_data/icc_natural_tdl_mc_raw.csv`
- **Sample size:** 500
- **Evidence type:** Monte Carlo
- **Limitations:** Depends on the chosen Γ_J / J_max fraction. Solver-tolerance violations can be much more frequent than 1% FIM shortfalls when S is an active constraint.

## Claim 4
- **Claim:** In the controlled-tilt mechanism experiment, mean conventional-S2 FIM mismatch increased from 0.0435% at a=0.0 to 20.2% at a=6.0.
- **Experiment:** `icc_controlled_asymmetry`
- **Raw data:** `results/icc_paper/raw_data/icc_controlled_asymmetry_raw.csv`
- **Sample size:** 40
- **Evidence type:** Controlled mechanism (not a realistic channel)
- **Limitations:** Logistic spectral tilt is synthetic; do not cite as a 3GPP channel result.

## Claim 5
- **Claim:** Cutting-plane refinement changed dense-grid PSL violation probability from 5.50% (sampled SOC) to 0.00% (proposed), mean iterations 1.05.
- **Experiment:** `icc_cutting_plane_mc`
- **Raw data:** `results/icc_paper/raw_data/icc_cutting_plane_mc_raw.csv`
- **Sample size:** 200
- **Evidence type:** Monte Carlo
- **Limitations:** Violation uses independent dense grid and the configured PSL tolerance.

## Claim 6
- **Claim:** On the joint FIM+uniform-PSL frontier, the proposed method remained dense-PSL feasible for at least half of channels up to Γ_J / J_max ≈ 0.5.
- **Experiment:** `icc_frontier`
- **Raw data:** `results/icc_paper/raw_data/icc_frontier_raw.csv`
- **Sample size:** 12
- **Evidence type:** Sweep (natural TDL channels)
- **Limitations:** Boundary is empirical under the uniform-spectrum PSL request; not a continuous-delay certificate.

## Claim 7
- **Claim:** Enforcing the EE objective (Dinkelbach) under exact EFIM + cutting-plane PSL changed EE by mean 27.9% relative to max-rate, with mean rate change 16.6% (positive rate change = max-rate had higher rate).
- **Experiment:** `icc_frontier`
- **Raw data:** `results/icc_paper/raw_data/icc_frontier_raw.csv`
- **Sample size:** 12
- **Evidence type:** Sweep
- **Limitations:** Dinkelbach is a standard fractional-programming solver, not a claimed novelty.

## Claim 8
- **Claim:** Mean conventional-S2 FIM mismatch by numerology (Δf fixed, B=KΔf): K=64 → 1.09%, K=128 → 0.744%, K=256 → 0.927%
- **Experiment:** `icc_numerology`
- **Raw data:** `results/icc_paper/raw_data/icc_numerology_raw.csv`
- **Sample size:** 40
- **Evidence type:** Monte Carlo per K
- **Limitations:** K and bandwidth are coupled; this is not a constant-bandwidth comparison.

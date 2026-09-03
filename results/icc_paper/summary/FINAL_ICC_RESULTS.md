# Final ICC experimental report

This report is generated from `results/icc_paper/raw_data/` after the ICC experiment suite.
Exploratory Stage 1/2/2.5 outputs remain in `results/stage2/` and `results/stage2_5/`.

## 1. Pytest

........................................................................ [ 58%]
...................................................                      [100%]
123 passed in 3.23s



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
- Natural-channel mean FIM mismatch (conventional S2): 0.7337 %
- Natural-channel median FIM mismatch: 0.0118 %
- Mismatch >1%/2%/5%/10%: 12.2% / 7.8% / 4.2% / 2.2% of realisations
- Pearson r(|f_bar|, ε_J): 0.920
- Spearman ρ: 1.000
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
- Mismatch by K: K=64 → 1.09%; K=128 → 0.744%; K=256 → 0.927%
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

## Missing raw files

- none

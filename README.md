# isac_ee — Energy-Efficient OFDM-ISAC with Optimizer-Guided, Safe, Structure-Aware RL

Research-grade Python simulation framework for a single OFDM transmitter that
serves one communication user and senses one target with the same waveform,
allocating per-subcarrier power `P_k` to trade off spectral efficiency,
delay/ranging accuracy, sidelobe performance and energy consumption.

The repository root is the `isac_ee` project. It is built incrementally:

* **Stage 1 (done)** — validated static foundation: system model, rate,
  water-filling, delay Fisher information / CRB, power model.
* **Stage 2 (done)** — static convex OFDM-ISAC optimisation oracle: min-power,
  max-rate and Dinkelbach energy-efficiency optimisers (CVXPY, CLARABEL with SCS
  fallback), independent feasibility verification, exact delay-domain
  ambiguity / PSL evaluation, baseline comparison, Pareto / EE frontiers,
  surrogate and PSL-proxy validation, Monte Carlo and SNR sweeps.
* **Stage 2.5 (done)** — exact unknown-complex-amplitude delay FIM constraint
  (convex quadratic-over-linear), sampled-grid PSL second-order cones,
  cutting-plane PSL generation, dense-grid PSL verification, and KKT analysis
  of the effective sensing weight ``(f_k - f_bar_P)^2``.
* **ICC experimental package (done)** — production proposed solver (exact EFIM
  + cutting-plane dense-grid PSL), fair conventional-``S_2`` vs exact-EFIM
  comparison under a common physical ranging target, exponential-PDP TDL
  channels, controlled-asymmetry mechanism sweep, and paper artifacts in
  ``results/icc_paper/``.  No RL.  The N=500 natural TDL Monte Carlo omits the
  Dinkelbach EE method (it is in the frontier and ablation).  Numerology holds
  ``Δf`` fixed so bandwidth scales with ``K``; ``K=256`` runs cheap methods only.

* Later stages — energy-harvesting MDP, Gymnasium environment, safe projection,
  SAC / PDS / structure-aware / optimizer-guided agents.

## Layout

```
configs/default.py                  frozen dataclass configuration (units documented)
isac/system.py                      ISACSystem: one channel realisation + all physical parameters
isac/communication/rate.py          SNR_k, log2(1+SNR_k), sum spectral efficiency, bit/s rate
isac/communication/water_filling.py classical water-filling (bisection on mu, optional peak cap)
isac/channels/rayleigh.py           i.i.d. frequency-selective Rayleigh CN(0, G)
isac/sensing/frequencies.py         centred grid f_k = (k-(K-1)/2) Delta_f, surrogate weights w_k
isac/sensing/fim.py                 delay Fisher information (known / unknown amplitude), surrogate S(P)
isac/sensing/crb.py                 CRB_tau = 1/J_tau, range CRB, summary dataclass
isac/sensing/ambiguity.py           delay-domain ambiguity profile, actual PSL / ISL
isac/energy/power_model.py          P_tx, P_sys = P_c + P_tx/eta_PA, energy per slot, EE [bit/J]
isac/optimization/solver.py         CVXPY solve helper: CLARABEL -> SCS fallback, strict status checks
isac/optimization/feasibility.py    independent constraint verification, slacks, S_max (greedy LP)
isac/optimization/common.py         scaled CVXPY model (x = P/P_peak), OptimizationResult
isac/optimization/{min_power,max_rate,dinkelbach}.py
                                        Stage 2 linear-S and Stage 2.5 exact-FIM / PSL
isac/optimization/fim_constraints.py    DCP unknown-amplitude FIM (quad_over_lin, optional RSOC)
isac/optimization/ambiguity_constraints.py  sampled-grid PSL SOC builder
isac/optimization/cutting_plane.py      optional PSL constraint generation
isac/optimization/kkt_analysis.py       G-gradient identity and stationarity residuals
isac/optimization/proposed.py           ICC proposed method: exact EFIM + cutting-plane PSL
isac/channels/tdl.py                    exponential-PDP TDL (natural frequency-selective channels)
isac/evaluation/physical.py             independent claimed-vs-actual sensing / dense-PSL evaluator
isac/evaluation/icc_suite.py            fair method catalog (S2 vs exact EFIM vs PSL vs EE)
isac/baselines/{base,internal}.py       internal baselines only (no published-paper labels)
experiments/icc_*.py                    ICC Monte Carlo, mechanism, frontier, numerology, paper generator
results/icc_paper/                      figures, tables, raw CSVs, config snapshots, FIGURE_INDEX.md
experiments/exp10 ... exp15             Stage 2.5 experiments (see below)
results/stage2_5/<experiment>/          CSV tables and PNG figures
isac/optimization/heuristics.py     edge-weighted and sensing-optimal heuristic allocations
isac/evaluation/metrics.py          single source of truth for all physical metrics of an allocation
isac/evaluation/baselines.py        common baseline evaluator + requirement construction
isac/evaluation/{plots,reporting,sampling,scenario}.py
experiments/exp01_basic_system.py   Stage 1 sanity experiment
experiments/exp02 ... exp09         Stage 2 experiments (see below)
tests/                              pytest suite (Stage 1 + Stage 2)
results/stage2/<experiment>/        CSV tables and PNG figures produced by the Stage 2 experiments
```

## Setup

Python >= 3.11 is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `isac` and `configs` importable from anywhere
```

If `python3 -m venv` fails with an `ensurepip` error (Debian/Ubuntu without
`python3-venv`), use `python3 -m venv --without-pip .venv` and bootstrap pip
with `get-pip.py`, or install `python3-venv`.

## Run

```bash
pytest                                                 # unit tests
python experiments/exp01_basic_system.py               # Stage 1 sanity check -> results/exp01/
python experiments/exp02_static_isac.py                # 2A single-channel baseline comparison
python experiments/exp03_rate_sensing_pareto.py        # 2B rate vs sensing Pareto frontier (max-rate sweep)
python experiments/exp04_ee_sensing_tradeoff.py        # 2C energy-efficiency frontier (Dinkelbach sweep)
python experiments/exp05_dinkelbach_convergence.py     # 2D Dinkelbach residual / q / EE per iteration
python experiments/exp06_sensing_surrogate_validation.py  # 2E S(P) vs FIM (known/unknown) vs CRB
python experiments/exp07_psl_proxy_validation.py       # 2F max(P_k) / variance vs actual PSL
python experiments/exp08_monte_carlo_static.py         # 2G 100-realisation Monte Carlo (--num-realizations)
python experiments/exp09_snr_power_sweep.py            # 2H path-loss (SNR) sweep
python experiments/exp10_crb_psl_tradeoff.py           # 2.5 CRB vs rate vs sampled PSL
python experiments/exp11_psl_threshold_sweep.py        # 2.5 PSL_max sweep at fixed FIM
python experiments/exp12_asymmetric_exact_fim.py       # 2.5 linear S(P) vs exact unknown FIM
python experiments/exp13_kkt_validation.py             # 2.5 KKT weights and residuals
python experiments/exp14_psl_cutting_plane.py          # 2.5 cutting-plane PSL
python experiments/exp15_monte_carlo_stage25.py        # 2.5 Monte Carlo (ordinary + asymmetric)
python -m experiments.icc_run_all                      # ICC paper suite -> results/icc_paper/
python -m experiments.icc_generate_paper               # regenerate figures/tables from stored CSVs
```

ICC paper-facing outputs live only in ``results/icc_paper/`` (figures, tables,
raw CSVs, config snapshots, ``FIGURE_INDEX.md``, ``summary/``).  Exploratory
Stage 1/2/2.5 figures stay under ``results/stage2/`` and ``results/stage2_5/``.
Use ``python -m experiments.icc_run_all --quick`` for a short smoke run.

Every ICC figure is regenerable from ``results/icc_paper/raw_data/`` without
re-solving.  The *proposed* ICC method is exact unknown-amplitude EFIM plus
cutting-plane dense-grid PSL; a sampled-grid primal that fails independent
dense validation is not labelled PSL-feasible.

Every experiment accepts `--seed` and `--output-dir`, prints the configuration,
saves raw CSV and figures, and records infeasible or failed solves instead of discarding them.

## Model summary

* Subcarriers: `f_k = (k - (K-1)/2) * Delta_f`, `K = 64`, `Delta_f = 15 kHz`.
* Channel: `h_k = sqrt(G) g_k`, `g_k ~ CN(0,1)` i.i.d., `G = 10^(-PL/10)`.
* Rate: `SNR_k = |h_k|^2 P_k / (N0 Delta_f)`, `R = sum_k log2(1 + SNR_k)`
  (bit/s/Hz summed over subcarriers; `Delta_f * R` gives bit/s).
* Water-filling: `P_k = clip(mu - 1/alpha_k, 0, P_peak)`, `alpha_k = |h_k|^2/(N0 Delta_f)`.
* Sensing: `y_k = beta x_k e^{-j 2 pi f_k tau} + n_k`, `|x_k|^2 = P_k`, `n_k ~ CN(0, sigma^2)`,
  `sigma^2 = N0 Delta_f` (real/imag parts `N(0, sigma^2/2)`, hence the factor `2/sigma^2`).
  Known amplitude: `J_tau = N (2|beta|^2/sigma^2) sum_k (2 pi f_k)^2 P_k`.
  Unknown complex amplitude: `J_tau = N (2|beta|^2/sigma^2)(2 pi)^2 [sum f_k^2 P_k - (sum f_k P_k)^2 / sum P_k]`.
  `CRB_tau = 1/J_tau` [s^2]; `CRB_R = (c/2)^2 CRB_tau` [m^2].
* Linear sensing surrogate: `S(P) = sum_k w_k P_k`, `w_k = f_k^2`. `S(P)` is
  **proportional to the known-amplitude FIM** and an **upper bound** on the
  unknown-amplitude FIM; it is never called "the CRB".
* Delay ambiguity: `A(tau) = sum_k P_k e^{j 2 pi f_k tau}`, `A_norm = |A|/A(0)`;
  PSL/ISL over `|tau| >= 1/B` (configurable). `P_k <= P_peak` is a **peak
  spectral power constraint / PSL proxy**, not a PSL constraint.
* Energy: `P_sys = P_circuit + sum_k P_k / eta_PA`, `EE = (Delta_f R) / P_sys` [bit/J].

## Static optimisation problems (all convex, solved directly in CVXPY)

```
A  min  sum_k P_k          s.t. R(P) >= R_min, S(P) >= Gamma_s, 0 <= P_k <= P_peak, sum P_k <= P_max
B  max  R(P)               s.t. S(P) >= Gamma_s, 0 <= P_k <= P_peak, sum P_k <= P_total [, R >= (1-gamma_c) C_WF]
C  max  R_bps(P)/P_sys(P)  s.t. S(P) >= Gamma_s, box, budget [, R >= R_min]   (Dinkelbach: q_{n+1} = R/P_sys, F(q*) = 0)
```

Requirements in the experiments are constructed relative to baseline
performance: `R_min = 0.9 C_WF` (water-filling capacity of the realisation) and
`Gamma_s = 0.6 S_max` (largest surrogate attainable under the peak/total power
limits). All reported metrics are recomputed with the NumPy model from the
returned allocation; every constraint is re-verified independently.

## Stage 2.5: exact unknown-amplitude FIM and sampled-grid PSL

The Stage 2 linear surrogate `S(P) = sum f_k^2 P_k` equals the known-amplitude
delay FIM up to `C_beta`, but **overestimates** ranging information when `beta`
is an unknown complex nuisance and the spectrum is asymmetric.  The exact
Schur-complement information is

```
G(P) = S2 - S1^2/S0 = sum_k P_k (f_k - f_bar_P)^2
J_tau^eff = C_beta * G(P),   C_beta = 8 pi^2 N |beta|^2 / sigma_n^2
```

`G` is concave (`S1^2/S0` is quadratic-over-linear), so `J_tau^eff >= Gamma_J`
is a convex constraint (`cp.quad_over_lin`, frequencies scaled by `max|f_k|`).

Direct ambiguity control is a **sampled-grid PSL** family of second-order cones
`||A(tau_m)|| <= rho sum P` on a moderate delay grid, verified afterwards on a
4x–8x denser independent grid.  `P_k <= P_peak` remains a peak spectral power
cap, not a PSL constraint.  An optional cutting-plane loop adds the worst
validation-grid violator until the request is met or the iteration cap is hit.

The KKT weight of `G` is `(f_k - f_bar_P)^2`.  This is coupled across
subcarriers through the centroid `f_bar_P`; it reduces to `f_k^2` when the
spectrum is symmetric.  That simple stationarity does **not** apply when
sampled-PSL SOCs are active.


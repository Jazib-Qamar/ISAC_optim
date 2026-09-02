# isac_ee — Energy-Efficient OFDM-ISAC with Optimizer-Guided, Safe, Structure-Aware RL

Research-grade Python simulation framework for a single OFDM transmitter that
serves one communication user and senses one target with the same waveform,
allocating per-subcarrier power `P_k` to trade off spectral efficiency,
delay/ranging accuracy, sidelobe performance and energy consumption.

The repository root is the `isac_ee` project. It is built incrementally; the
current stage is the **validated static foundation** (system model, rate,
water-filling, delay Fisher information / CRB, power model). Convex
optimisation (CVXPY), the Gymnasium environment and RL agents are added in
later stages once this foundation is verified.

## Layout (current stage)

```
configs/default.py                 frozen dataclass configuration (units documented)
isac/communication/rate.py         SNR_k, log2(1+SNR_k), sum spectral efficiency, bit/s rate
isac/communication/water_filling.py classical water-filling (bisection on mu, optional peak cap)
isac/channels/rayleigh.py          i.i.d. frequency-selective Rayleigh CN(0, G)
isac/sensing/frequencies.py        centred grid f_k = (k-(K-1)/2) Delta_f, surrogate weights w_k
isac/sensing/fim.py                delay Fisher information (known / unknown amplitude), surrogate S(P)
isac/sensing/crb.py                CRB_tau = 1/J_tau, range CRB, summary dataclass
isac/energy/power_model.py         P_tx, P_sys = P_c + P_tx/eta_PA, energy per slot, EE [bit/J]
experiments/exp01_basic_system.py  one channel: uniform vs water-filling, all metrics, plots, CSV
tests/                             pytest suite for every module above
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
pytest                                        # unit tests
python experiments/exp01_basic_system.py      # prints metrics, writes results/exp01/
python experiments/exp01_basic_system.py --seed 3 --show
```

`exp01` writes `exp01_metrics.csv`, `exp01_per_subcarrier.csv`,
`exp01_channel_gain.png` and `exp01_power_allocation.png` to `results/exp01/`.

## Model summary (stage 1)

* Subcarriers: `f_k = (k - (K-1)/2) * Delta_f`, `K = 64`, `Delta_f = 15 kHz`.
* Channel: `h_k = sqrt(G) g_k`, `g_k ~ CN(0,1)` i.i.d., `G = 10^(-PL/10)`.
* Rate: `SNR_k = |h_k|^2 P_k / (N0 Delta_f)`, `R = sum_k log2(1 + SNR_k)`
  (bit/s/Hz summed over subcarriers; `Delta_f * R` gives bit/s).
* Water-filling: `P_k = clip(mu - 1/alpha_k, 0, P_peak)`, `alpha_k = |h_k|^2/(N0 Delta_f)`,
  `mu` by bisection so that `sum P_k = P_total`.
* Sensing: `y_k = beta x_k e^{-j 2 pi f_k tau} + n_k`, `|x_k|^2 = P_k`, `n_k ~ CN(0, N0 Delta_f)`.
  Known amplitude: `J_tau = N (2|beta|^2/sigma^2) sum_k (2 pi f_k)^2 P_k`.
  Unknown complex amplitude (Schur complement):
  `J_tau = N (2|beta|^2/sigma^2)(2 pi)^2 [sum f_k^2 P_k - (sum f_k P_k)^2 / sum P_k]`.
  `CRB_tau = 1/J_tau` [s^2]; `CRB_R = (c/2)^2 CRB_tau` [m^2].
* Linear sensing surrogate: `S(P) = sum_k w_k P_k`, `w_k = f_k^2` (or normalised index).
  `S(P)` is proportional to the known-amplitude `J_tau`; it is **not** the CRB
  and is never named as such.
* Energy: `P_sys = P_circuit + sum_k P_k / eta_PA`, `EE = (Delta_f R) / P_sys` [bit/J].

## Roadmap

1. Static convex optimisation in CVXPY (min-power, max-rate, Dinkelbach EE), KKT water-filling.
2. Ambiguity function / actual PSL evaluator and validation experiments.
3. Time-slotted energy-harvesting model, Gymnasium environment, safe projection.
4. SAC baseline, post-decision state, structure-aware critic, optimizer-guided pretraining.
5. Ablations and sample-efficiency studies.

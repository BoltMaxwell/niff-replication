# niff-replication

A self-contained reimplementation of the numerical methods from **Hao & Bilionis,
*Neural Information Field Filter* (NIFF), Mechanical Systems and Signal Processing 226
(2025) 112253**, in [JAX](https://github.com/google/jax).

NIFF is a hierarchical Bayesian method for joint **state and parameter estimation** in
dynamical systems, built on Information Field Theory (IFT). The time-evolution state path is
represented by a finite basis; a physics-informed conditional prior couples the path to the
dynamics; and the intractable posterior over (path coefficients `w`, parameters `θ`, auxiliary
initial state `x0`) is approximated by either a variational method (**NSVI**) or a sampler
(**NPSGLD**).

This repo reproduces the paper's **Section 5.1** single-degree-of-freedom Duffing oscillator
end-to-end, and extends the sampler with a family of preconditioners.

## What's here

```
niff/
├── nsvi.py         Nested stochastic variational inference (optimizer)
├── npsgld.py       Nested (preconditioned) SGLD sampler
├── duffing_s51.py  Paper §5.1 Duffing problem: data (RK4), Fourier fields, energies
├── fields.py       Truncated Fourier basis
└── utils.py        Collocation grid
scripts/
├── run_duffing_s51.py   CLI: run a method → posterior .npz + summary.json
└── plot_duffing_s51.py  Overlay saved posteriors → figures
notes/lab.md        Chronological lab notebook (results, lessons, gotchas)
ROADMAP.md          Status of replicated phases + optional next examples
PROGRESS.md         Consolidated writeup + side-by-side paper comparison
figures/            Reproduced figures
```

## The §5.1 Duffing problem

```
x1'(t) = x2(t)
x2'(t) = -k1 x2 - k2 x1 - k3 x1^3 + gamma cos(omega t)
Y(t)   = x1(t) + sigma_y V(t)     (noisy, position-only)
```
Truth `k1=0.3, k2=-1, k3=1` (double-well), `gamma=0.37, omega=1.2`, IC `(1,0)`, `T=50`,
`sigma_y=0.075`, Fourier `K=40`. Two state-path parameterizations: **reparameterized** (IC
folded into `w`) and **relaxed** (free field + auxiliary `x0` coupled via a kernel — the NIFF
contribution).

## Methods

Both `run_nsvi` and `run_npsgld` share the same callback contract
(`log_likelihood_fn(w, θ)`, `energy_fn(w, θ, x0)`), so a problem written once runs under either.

- **NSVI** — nested variational inference. Fast; a diagonal/full-rank Gaussian guide, so
  posteriors are tight (and mildly overconfident).
- **NPSGLD** — nested SGLD with `preconditioner ∈`:

  | value | method | notes |
  |---|---|---|
  | `identity` | NSGLD (paper method 3) | un-preconditioned; step-size fragile |
  | `rmsprop` | NPSGLD (paper method 4) | diagonal RMSprop; **robust workhorse** |
  | `diag_fisher` | diagonal empirical Fisher | ≈ rmsprop empirically |
  | `dense_fisher` | dense empirical Fisher on the θ block | matrix preconditioner; finicky |

## Quick start

```bash
pip install -e .          # or: pip install -r requirements.txt

# NSVI, both variants (short local run; scale up --iterations for accuracy)
python scripts/run_duffing_s51.py --method nsvi --variants relaxed reparam \
    --iterations 30000 --num-obs 200 --n-colloc 120 --beta2 1e3 \
    --output-dir results/demo

# NPSGLD sampler
python scripts/run_duffing_s51.py --method npsgld --variants relaxed \
    --npsgld-iterations 200000 --npsgld-chains 4 --output-dir results/demo

# overlay whatever posteriors are in the directory
python scripts/plot_duffing_s51.py --results-dir results/demo
```

float64 is enabled automatically. On CPU, keep iteration counts modest; the paper-faithful runs
(2M iterations) are GPU jobs.

## Key results (see `notes/lab.md`, `PROGRESS.md` for detail)

- **Agreement with the paper.** State reconstruction matches Fig 2 (position tight, *unobserved*
  velocity recovered with a wider band); the reparameterized and relaxed variants agree; the
  relaxed prior reproduces the Fig 5 "slight differences" between `xhat(0;w)` and `x0`.
- **NSVI vs samplers.** NSVI posteriors are tight but slightly biased; the SGLD samplers are
  wider and **bracket the truth** — matching the paper's characterization. All recover
  `(k1,k2,k3)`.
- **Collocation matters.** A *fixed* collocation grid biases `θ`; *random per-iteration*
  subsampling (an unbiased MC estimate of the physics integral) removes it.
- **Preconditioner study.** On the strongly coupled `(k2,k3)` posterior (corr ≈ −0.97), the
  diagonal methods are the robust workhorse; the dense matrix preconditioner over-damps cold
  burn-in and over-disperses without careful tuning. (A lesson for higher-dimensional /
  more-coupled parameter posteriors.)

## Provenance

Ported from a research monorepo; the engines (`niff/nsvi.py`, `niff/npsgld.py`) are a
generalization of the vendored NIFF reference code. This standalone repo is a clean jumping-off
point for further work (see `ROADMAP.md` for the optional remaining paper examples). The paper itself is not
redistributed here — see the DOI: <https://doi.org/10.1016/j.ymssp.2024.112253>.

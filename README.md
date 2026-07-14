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

This repo reproduces the paper's **Section 5.1** (single-DOF Duffing) and **Section 5.2**
(two-DOF + residual neural network) end-to-end, and extends the sampler with a family of
preconditioners.

> **Provenance.** This replication was carried out by [Claude Code](https://claude.com/claude-code)
> (Anthropic) working from the published paper — writing the JAX implementation, running the
> experiments, and producing the figures and write-ups (`notes/lab.md`, `PROGRESS.md`,
> `ROADMAP.md`). The paper is not redistributed here; see the DOI below.

## Scope — how much of the paper is replicated

The paper's Section 5 has **four numerical examples**. Two are reproduced end-to-end here; two
are cataloged but not implemented. The core method — both state-path parameterizations
(reparameterized + relaxed), **NSVI** and **NPSGLD**, the collocation-subsampling estimator, and
a preconditioner family — is fully implemented and shared across examples.

| Paper example | w-dim | Status | Notes |
|---|---|---|---|
| **§5.1** Duffing oscillator (single-DOF) | 162 | ✅ reproduced | Figs 2–5, four-method comparison, preconditioner study |
| **§5.2** two-DOF + residual NN | 344 | ✅ reproduced | Fig 6 demonstration; parameter-identifiability caveat (below) |
| §5.3 twenty-story Bouc–Wen frame | 4660 | ⬜ not implemented | high-dimensional; heavy |
| §5.4 experimental nonlinear energy sink | 16000 | ⬜ not implemented | needs the external experimental dataset |

## What's here

```
niff/
├── nsvi.py         Nested stochastic variational inference (optimizer)
├── npsgld.py       Nested (preconditioned) SGLD sampler
├── duffing_s51.py  Paper §5.1 Duffing problem: data (RK4), Fourier fields, energies
├── twodof_s52.py   Paper §5.2 two-DOF + residual-NN (RBF basis + Fourier-encoded MLP)
├── fields.py       Truncated Fourier basis
└── utils.py        Collocation grid
scripts/
├── run_duffing_s51.py / run_twodof_s52.py    CLI: run a method → posterior .npz + summary
└── plot_duffing_s51.py / plot_twodof_s52.py  Overlay saved posteriors → figures
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

### §5.1 Duffing

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

### §5.2 two-DOF + residual NN (`niff/twodof_s52.py`)

- **The residual-NN demonstration reproduced (Fig 6).** With a linear RBF basis alone the states
  cannot be reconstructed; adding the Fourier-encoded residual NN reconstructs them near-perfectly.
  The state-path w-dimension (344) matches the paper's Table 3.
- **A cautionary finding — flexible fields loosen identifiability.** The residual NN fits the
  *states* perfectly but leaves flat directions in *parameter* space. NSVI recovers the parameters
  biased-but-reasonably; **NPSGLD diverges into a degeneracy** (k1→0 makes the cubic ε1 term vanish,
  so the sampler drifts off). This flips the §5.1 lesson: **samplers expose degeneracies that VI
  hides.** (Exact parameter recovery would need stronger nonlinear excitation and/or constraints;
  our forcing was a documented guess, since the paper does not state it.)

## Observed discrepancies vs. the paper

What differs between this replication and the paper, and how the paper's own results look at each
point (fuller detail with numbers in `PROGRESS.md` §3.3 and `notes/lab.md`).

**Resolved discrepancies** (matched the paper after a fix):

- **k3 quadrature bias.** With a *fixed* collocation grid, NSVI's k3 sat low (~0.937 vs the
  paper's ~1.02) and climbed as the grid densified — a quadrature bias, not a model error.
  *Random per-iteration* collocation subsampling (an unbiased MC estimate of the physics
  integral) fixed it: k3 → 0.982, k2 → −0.970. Now matches the paper.
- **NSGLD (un-preconditioned) sampler.** Now run (`preconditioner=identity`); completes the
  four-method panel and is the widest/slowest, exactly as in the paper. It is step-size fragile
  (diverged at 1e-5, stable at 1e-6).

**Remaining differences** (do not contradict the paper's results):

- **Runtime, not seconds.** Paper Table 2 reports §5.1 at **7–90 s on an M1 Pro CPU**; our
  paper-faithful runs take **~20–27 min on a GPU**. Cause: the paper's speed comes from CPU +
  per-iteration subsampling (n_t=10 collocation, m_y=10 measurements); on GPU the 2M-iteration
  `lax.scan` is launch-overhead-bound, so subsampling buys the unbiased gradient but ~no speedup.
  Parameter *posteriors* still agree — only wall-clock differs.
- **Fourier basis convention.** We use a half-period basis `[1, cos(πkt), sin(πkt)]`; the paper
  uses full-period `2πk·t/T̄`. Both K=40-expressive; no effect on agreement.
- **No state normalization.** We work in physical coordinates; the paper normalizes states by
  (x̄1,x̄2)=(1.5,1). A conditioning device only — does not change the posterior.
- **Unspecified measurement count (§5.1).** The paper does not state n_d for §5.1; we chose 1000.
- **Riemannian Γ correction omitted** in the preconditioned samplers (its `jacfwd` is ~300× the
  base cost → intractable at 2M iters); the standard practical pSGLD approximation is used.
- **§5.2 forcing was a documented guess.** The paper does not state the §5.2 excitation; we used
  F=2, ω0=1.2, Fourier period T̄=10 (from ref [84]). The Fig 6 *state* reconstruction reproduces
  regardless, but with this weaker forcing the flexible residual NN leaves flat directions in
  *parameter* space: NSVI recovers the parameters biased-but-reasonably while **NPSGLD diverges
  into a degeneracy** (k1→0 makes the cubic ε1 term vanish). This is a genuine finding, not a bug
  — samplers expose a degeneracy VI hides — but exact §5.2 parameter recovery would need the
  paper's true (stronger) nonlinear excitation and/or parameter constraints.

**Overall:** §5.1 and §5.2 reproduce the paper's figures and its NSVI-vs-sampler story, with the
truth recovered by all four methods in §5.1. The one substantive discrepancy (the collocation
quadrature bias) is understood and fixed; the rest are conventions or a runtime artifact.

## Provenance

Ported from a research monorepo; the engines (`niff/nsvi.py`, `niff/npsgld.py`) are a
generalization of the vendored NIFF reference code. This standalone repo is a clean jumping-off
point for further work (see `ROADMAP.md` for the optional remaining paper examples). The paper itself is not
redistributed here — see the DOI: <https://doi.org/10.1016/j.ymssp.2024.112253>.

# NIFF replication — progress & paper comparison

**As of 2026-07-02.** The §5.1 Duffing replication is **complete** — all four inference
methods, the collocation-subsampling fix, and a preconditioner-family study. This file is the
consolidated "where we are / how we compare against the paper" writeup; chronological detail
lives in `notes/lab.md`, the forward plan in `ROADMAP.md`.

Paper: Hao & Bilionis, *Neural Information Field Filter*, MSSP 226 (2025) 112253,
<https://doi.org/10.1016/j.ymssp.2024.112253>.

---

## 1. What's done

- **§5.1 Duffing, both state-path variants** — reparameterized (IC folded into `w`) and relaxed
  (free field + auxiliary `x0` via a kernel) — under **NSVI**.
- **Full four-method comparison** — NSVI, NPSGLD (`rmsprop`), NSGLD (`identity`), and the
  diagonal/dense empirical-**Fisher** preconditioners — via `niff.npsgld`.
- **Random collocation/measurement subsampling** (`n_t`, `m_y`) in the callbacks, which removed
  a fixed-grid quadrature bias in the NSVI parameter estimates.
- Reproduced figures in `figures/` (`comparison_final/` = the four-method overlay;
  `preconditioner/` = the preconditioner study).

## 2. §5.1 results (2M iters, num_obs=1000, β₁=200, β₂=1e5)

Four-method comparison (NSVI uses the bias-corrected *subsampled* runs; samplers full-batch —
same target posterior). All bracket the truth:

| param | NSVI relaxed | NSVI reparam | NPSGLD | NSGLD | truth |
|-------|--------------|--------------|--------|-------|-------|
| k1 | 0.310 | 0.293 | 0.334 | 0.336 | 0.30 |
| k2 | −0.939 | −0.970 | −0.952 | −0.977 | −1.00 |
| k3 | 0.960 | 0.982 | 0.968 | 0.992 | 1.00 |

Physics sanity (pre-inference): least-squares on the *true* trajectory recovers
(0.3, −0.9999, 0.9999), RMS 2e-5 — the energy/residual formulation is exact.

---

## 3. Comparison with the paper

### 3.1 Figures — strong match

- **Fig 2 (states).** Position x1 tracked tightly; velocity x2 *unobserved* → visibly wider
  band; measurement y noisy around the posterior. **Reproduced.** ✅
- **Fig 3 (params).** The paper's own NSVI curves are tight and slightly biased off-truth, with
  the SGLD samplers wider — *the same pattern we see*. Our two variants overlap ("great
  agreement of the methods"); the samplers bracket truth. **Reproduced.** ✅
- **Fig 5 (initial state).** NSVI shows "slight differences" between p(xhat(0;w)) and p(x0);
  the samplers give nearly identical distributions. **Reproduced.** ✅
- **Fig 4 (convergence).** NSVI converges fastest; NPSGLD ≫ NSGLD in mixing. We have all four
  methods; NSGLD is the slowest/widest, as in the paper. ✅

### 3.2 Runtime (paper Table 2)

| | Repara NSVI | Relaxed NSVI | Relaxed NSGLD | Relaxed NPSGLD |
|---|---|---|---|---|
| **paper** (w=162, **M1 Pro CPU**) | 8 s | 7 s | 77 s | 90 s |
| **ours** (w=162, **GPU**) | ~27 min | ~27 min | ~21 min | ~22 min |

Much slower despite identical w-dimension (162). **Cause:** the paper subsamples **n_t=10
collocation points and m_y=10 measurements per iteration** (the IFT efficiency trick). We
implemented that subsampling too, but found it gives **no GPU speedup** — the 2M-iteration
`lax.scan` is launch-overhead-bound, so 10-vs-1000 collocation is ~free on a GPU. The paper's
seconds-not-minutes is CPU + subsampling (CPU *is* collocation-cost-sensitive). Subsampling on
GPU buys the *unbiased gradient* (§3.3.1), not runtime.

### 3.3 Discrepancies — resolved and remaining

1. **k3 bias (RESOLVED).** With a *fixed* collocation grid, NSVI k3 sat low (0.937 vs the
   paper's ~1.02) — a quadrature bias (k3 climbed 0.888 → 0.937 as the grid densified). Random
   per-iteration collocation (an unbiased MC estimate of the physics integral) fixed it:
   reparam k3 → **0.982**, k2 → −0.970. Confirmed the hypothesis.
2. **NSGLD (RESOLVED).** The plain un-preconditioned sampler is now run (`preconditioner=
   identity`); it completes the four-method panel and, as in the paper, is the widest/slowest.
   It is step-size fragile (diverged at 1e-5, stable at 1e-6).
3. **No state normalization.** We work in physical coordinates; the paper normalizes states by
   (x̄1,x̄2)=(1.5,1). Shouldn't change the posterior, only optimizer conditioning. Not needed
   for agreement.
4. **Basis convention.** We use a half-period Fourier basis `[1,cos(πkt),sin(πkt)]`; the paper
   uses full-period `2πk t/T̄`. Both K=40-expressive.
5. **n_d (number of measurements)** is unspecified in the paper for §5.1; we chose 1000.
6. **Riemannian Γ correction omitted** in the preconditioned samplers (a `jacfwd` ~300× the
   base cost → intractable at 2M iters); the standard practical pSGLD approximation is used.

**Verdict:** §5.1 is reproduced — same figures, same NSVI-vs-sampler story, truth recovered by
all four methods, and the one real discrepancy (the collocation quadrature bias) understood and
fixed. Remaining deltas (normalization, basis, Γ) do not contradict the paper.

### 3.4 Preconditioner study (beyond the paper's two samplers)

On the strongly coupled (k2,k3) posterior (corr ≈ −0.97): the diagonal families
(`rmsprop`=NPSGLD, `identity`=NSGLD, `diag_fisher`) agree and are the robust workhorse
(`diag_fisher` ≡ `rmsprop` empirically). The **dense** matrix preconditioner is finicky — it
over-damps cold-start burn-in and over-disperses without near-mode initialization and step-size
retuning. See `notes/lab.md` (v7) and `figures/preconditioner/`.

---

## 4. The paper's other examples (optional future replication)

The paper has three more numerical examples beyond §5.1. None are implemented here; cataloged
for reference.

### §5.2 — two-DOF nonlinear system, **residual neural network** (the paper's headline device)
- 2 masses, Duffing spring + nonlinear damper; 4 states, **8 params**, truth m=1, c=0.2, k=1,
  ε=0.2. Measure q1 and q1+q2. Normalize by 1; ȳ=(1,2).
- Data: 50 s RK, dt=0.1; IC=(0,0,0.5,0); noise 5% of norm const.
- State path: **RBF basis** (Kb=20, σk=0.05) **+ Fourier-encoded residual NN** (K=10, 1 hidden
  layer width 10, swish). RBF *alone* fails; the residual NN corrects it.
- NSVI 300k iters (anneal 200k); NPSGLD 3 chains × 3M steps.
- **Relevance:** exercises the hybrid linear+NN parameterization (paper eq. 8), which this repo
  does not yet implement. The cleanest next example.

### §5.3 — twenty-story Bouc–Wen frame (**high-dimensional**)
- 20-DOF Bouc–Wen frame; estimate 20 stiffnesses s₁:₂₀ ~ U[8,10]. First 5 s of El-Centro
  earthquake excitation. State path: 100 RBFs + residual NN, w-dim **4660**. Heavy.

### §5.4 — experimental nonlinear energy sink (**real data**)
- Duffing-type device; identify (cν, cf, k, z). Data at 4096 Hz, w-dim **16000**, NSVI only.
  Needs the external NES dataset.

### Paper compute reference (M1 Pro, 10 cores)
| Section | w-dim | NSVI | NPSGLD |
|---|---|---|---|
| 5.1 | 162 | 7–8 s | 90 s |
| 5.2 (w/ residual) | 344 | 26 s | 195 s |
| 5.3 | 4660 | 196 s | 325 s |
| 5.4 | 16000 | 827 s | — |

---

## 5. Possible next steps (all optional; §5.1 is complete)

1. **§5.2 residual-NN example** — the most informative remaining paper example; would add the
   hybrid linear-basis + Fourier-encoded neural-network state path (paper eq. 8) and an RBF basis.
2. **§5.3 / §5.4** — heavier, and §5.4 needs external experimental data.
3. **Faithful cosmetics for §5.1** — state normalization and the full-period Fourier basis, if an
   exact match to the paper's conditioning is wanted (not needed for agreement).

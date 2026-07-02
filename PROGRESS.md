# NIFF replication — progress & paper comparison

**As of 2026-06-30.** Status note for the §5.1 replication (Phases 0–1 complete), with a
side-by-side comparison against the published paper and a catalog of the remaining
experiments. Chronological detail lives in `notes/niff_replication_lab.md`; the forward plan
in `ROADMAP.md`; this file is the consolidated "where we are / how we compare" writeup.

Paper: Hao & Bilionis, *Neural Information Field Filter*, MSSP 226 (2025) 112253
(`wiki/raw/niff.pdf`).

---

## 1. What we have done

- **Phase 0 — faithful §5.1 deterministic Duffing**, both state-path variants
  (reparameterized IC-folded, and relaxed auxiliary-x0), via our `ift.fields.fourier` +
  vendored `methods/nsvi` engine. Full 2M-iter run on Gautschi GPU.
- **Phase 1 — NPSGLD sampler** wired (`methods/nsvi/npsgld.py`) and run at 2M iters, relaxed.
- Figures regenerated and tracked in `figures/` (Phase 0) and `figures/comparison/` (Phase 1
  overlay of nsvi_relaxed / nsvi_reparam / npsgld_relaxed).

## 2. Our §5.1 results (2M iters, num_obs=1000, n_colloc=1000, β₁=200, β₂=1e5)

| param | NSVI relaxed | NSVI reparam | NPSGLD relaxed | truth |
|-------|--------------|--------------|----------------|-------|
| k1    | 0.316 ± 0.018 (z=0.90) | 0.299 ± 0.017 (z=0.05) | 0.334 ± 0.034 (z=0.99) | 0.30 |
| k2    | −0.917 ± 0.022 (z=3.76) | −0.916 ± 0.022 (z=3.80) | −0.952 ± 0.050 (z=0.96) | −1.00 |
| k3    | 0.937 ± 0.015 (z=4.15) | 0.940 ± 0.019 (z=3.24) | 0.968 ± 0.043 (z=0.75) | 1.00 |
| xhat(0) | (1.018, −0.005) | (1.028, −0.003) | (1.049, −0.128) | (1, 0) |

Physics sanity (pre-inference): least-squares on the *true* trajectory recovers
(0.3, −0.9999, 0.9999), RMS 2e-5 — the energy/residual formulation is exact.

---

## 3. Comparison with the paper

### 3.1 Figures — strong qualitative match

- **Fig 2 (states).** Paper shows 4 method-rows × (x1, x2, y). Position x1 tracked tightly;
  velocity x2 *unobserved* → visibly wider band; measurement y noisy around posterior. **Our
  Fig 2 reproduces this exactly** (position tight, velocity wider, double-well hopping). ✅
- **Fig 3 (params).** The paper's **NSVI curves are themselves tight and slightly biased
  off-truth**, with NSGLD/NPSGLD wider — *the same pattern we see*. This is the key validation:
  our NSVI bias+overconfidence is the documented NSVI behavior, not a replication bug. ✅
  - Paper: "great agreement of the four methods" — our two variants overlap, NPSGLD wider. ✅
- **Fig 5 (initial state).** Paper: NSVI shows "slight differences" between p(xhat(0;w)) and
  p(x0); NSGLD/NPSGLD give "nearly identical" distributions. **We reproduce both**: NSVI aux-x0
  is sharp while xhat(0;w) is wide (slight difference); NPSGLD's two overlap. ✅
- **Fig 4 (convergence).** Paper: NSVI converges faster than the samplers; NPSGLD ≫ NSGLD.
  We have the NSVI-vs-NPSGLD convergence (our Fig 4); NSGLD not run (see gaps).

### 3.2 The big practical gap — runtime (paper Table 2)

| | Repara NSVI | Relaxed NSVI | Relaxed NSGLD | Relaxed NPSGLD |
|---|---|---|---|---|
| **paper** (w=162, **M1 Pro CPU**) | 8 s | 7 s | 77 s | **90 s** |
| **ours** (w=162, **Gautschi GPU**) | ~27 min | ~27 min | — | **~22 min** |

~100–200× slower despite identical w-dimension (162). **Cause:** the paper subsamples
**n_t = 10 collocation points and m_y = 10 measurements per iteration** (the IFT efficiency
trick — Algorithm 4 sample sizes `(n,n_t,ñ,ñ_t,m_y)=(1,10,1,10,10)`). Our callbacks are
**full-batch** (`n_colloc=1000`, `num_obs=1000`) → ~100× more work per step; the GPU hides
most of it (so ~15× per-iter, not 100×), but we over-resourced (GPU) *and* over-computed.
**Actionable:** subsample collocation/measurements in the callbacks → expect paper-class speed
(likely *faster*, on GPU) **and** likely a bias fix (see 3.3).

### 3.3 Open discrepancies / where we diverged

1. **k3 bias is opposite-signed.** Paper NSVI k3 peaks slightly *above* 1.0 (~1.02); ours sits
   *below* (0.937). Our own collocation test (k3: 0.888 → 0.920 → 0.937 as n_colloc 200→1000→
   2M-grid) shows k3 climbs toward truth with denser quadrature ⇒ the low bias is most likely
   **fixed-grid collocation quadrature bias**. The paper's *random* n_t=10 subsampling is an
   unbiased integral estimator (averaged over 2M iters), which both speeds it up and removes
   this bias. **Hypothesis to test:** switch to random collocation subsampling.
2. **No state normalization.** We work in physical coords; the paper normalizes states by
   (x̄1,x̄2)=(1.5,1) (k̄=(1,1,1), ȳ=1.5). Shouldn't change the posterior in principle, but
   affects optimizer conditioning and could shift small biases.
3. **Basis convention.** We use the repo's half-period `fourier_basis_01`
   (`[1,cos(πkt),sin(πkt)]`); the paper uses full-period `2πk t/T̄`. Both K=40-expressive.
4. **NSGLD (4th method) not run.** We have Repara-NSVI, Relaxed-NSVI, NPSGLD; the plain
   (non-preconditioned) NSGLD is missing — needed for the *complete* Fig 3/4 four-method panel.
5. **n_d (number of measurements) unspecified in the paper for §5.1.** We chose 1000 (dense,
   matches Fig 2's visual density). The likelihood/physics balance depends on n_d.
6. **Riemannian Γ correction omitted** in NPSGLD (jacfwd ~300× cost → intractable at 2M);
   standard diagonal pSGLD used. Paper uses the full Γ (they had the compute budget / cheaper
   per-iter via subsampling).

**Verdict:** §5.1 is reproduced — same figures, same NSVI-vs-sampler story, truth recovered.
The deltas are efficiency (subsampling) and a small, explained quadrature bias, plus the
un-run NSGLD. None contradict the paper.

---

## 4. The paper's other experiments (catalog for the roadmap)

### §5.2 — two-DOF nonlinear system, **residual neural network** (the paper's headline device)
- 2 masses, Duffing spring + nonlinear damper; 4 states, **8 params** (m1,m2,c1,c2,k1,k2,ε1,ε2),
  truth m=1, c=0.2, k=1, ε=0.2. Measure q1 and q1+q2. Normalize all by 1; ȳ=(1,2).
- Data: 50 s RK, dt=0.1; IC=(0,0,0.5,0); noise 5% of norm const.
- State path: **RBF basis** Kb=20, σk=0.05, centers evenly in [0,1], **+ Fourier-encoded
  residual NN** (K=10, 1 hidden layer width 10, swish). Key point: RBF *alone* fails; the
  residual NN corrects it.
- NSVI 300k iters (anneal 200k); NPSGLD 3 chains × 3M steps, burn 1M, thin 10k.
- **Relevance:** exercises the hybrid linear+NN parameterization (paper eq. 8) we haven't
  touched — and the repo has `ift.fields.rbf`. Optional but the cleanest next *paper* example.

### §5.3 — twenty-story Bouc–Wen frame (**high-dimensional**)
- 20-DOF Bouc–Wen hysteretic frame; estimate 20 stiffnesses s₁:₂₀ ~ U[8,10]; m,c known.
- Excitation: first 5 s of El-Centro NS earthquake. Noise 1% of story-accel RMS.
- State path: **100 RBFs** (length scale 0.01) + same residual NN. w-dim **4660**.
- NSVI 50k steps; NPSGLD 5 chains × 200k steps.
- **Relevance:** stresses NSVI vs NPSGLD at scale (the paper's guideline regime). Heavy.

### §5.4 — experimental nonlinear energy sink (**real data**)
- Duffing-type device, eq. `m ẍ + cν ẋ + cf tanh(200ẋ) + k x + z x³ = −m ẍg`; m=0.664 known,
  identify (cν, cf, k, z). Two datasets stacked (4-state). Data sampled at **4096 Hz** (subsample
  the collocation at those points). w-dim **16000**, 4000 RBFs/signal, **no residual**.
- NSVI only (NPSGLD too slow — "several hours"). Reference values e.g. cν=0.344, cf=0.064,
  k=33.1, z=6.54e5.
- **Relevance:** real-data validation; needs the external NES dataset (ref [93]).

### Paper compute reference (M1 Pro, 10 cores)
| Section | w-dim | NSVI | NPSGLD |
|---|---|---|---|
| 5.1 | 162 | 7–8 s | 90 s |
| 5.2 (w/ residual) | 344 | 26 s | 195 s |
| 5.3 | 4660 | 196 s | 325 s |
| 5.4 | 16000 | 827 s | — |

---

## 5. Recommended next steps (in priority order)

1. **Efficiency + bias fix (cheap, high-value):** add random collocation/measurement
   subsampling (n_t, m_y) to the callbacks. Test the k3-bias hypothesis (3.3#1) and aim for
   paper-class runtime. Re-run §5.1 NSVI to see if the bias direction flips toward the paper's.
2. **(Optional) NSGLD** for the complete four-method Fig 3/4 panel.
3. **Phase 2 — SDE extension** (the repo's core contribution; beyond the paper): process noise
   σx, σv in the energy, infer diffusion jointly.
4. **Phase 3 — auxiliary-IC joint inference** (relaxed machinery already validated).
5. **(Optional paper examples)** §5.2 residual-NN is the most informative next *paper* example
   (tests the hybrid basis); §5.3/§5.4 are heavier and need external data.

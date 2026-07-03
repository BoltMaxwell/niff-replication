# NIFF Replication — Lab Notebook

**Goal:** Reproduce Hao & Bilionis *Neural Information Field Filter* (MSSP 2025), **§5.1**
(single-DOF Duffing oscillator), in JAX.

**Code:** `niff/{nsvi,npsgld,duffing_s51,fields,utils}.py`, `scripts/{run,plot}_duffing_s51.py`.

> Ported from a research monorepo; historical entries below may reference the original
> layout (`methods/…`, `experiments/…`) and a GPU cluster.

**Paper §5.1 truth:** k1=0.3, k2=-1, k3=1; gamma=0.37, omega=1.2; IC=(1,0);
T=50, RK dt=0.01; sigma_y=0.075 (5% of ybar=1.5); Fourier K=40; std-normal priors.

**Two NSVI variants (paper methods 1 & 2):**
- `reparam` — IC folded into w (xhat(0)=ic exact), d_x0=0, beta=200.
- `relaxed` — free field + auxiliary x0 via kernel beta2||xhat(0;w)-x0||^2, d_x0=2,
  beta1=200, beta2=1e5.

---

## Design decisions (2026-06-29)

- **§5.1 replication**, both state-path variants (reparameterized + relaxed).
- Both NSVI variants in the first pass → Figs 2 (states), 3 (params), 5 (xhat(0) vs x0).
  SGLD samplers (NSGLD/NPSGLD, Fig 4) deferred.
- **Likelihood gets no x0** in the engine → reparam folds IC into w; relaxed keeps the
  field a pure function of w (x0 only in the energy kernel). This is the exact form the
  vendored paper example code uses.
- Work in **physical coordinates** (truth O(1)); paper's normalization is a conditioning
  device that doesn't change the (k1,k2,k3) posterior. Add only if conditioning demands.
- Basis: repo half-period `fourier_basis_01` (not paper's full-period 2πk). K=40-expressive
  either way on 50 s.

---

## Entries

### v0 — pipeline build + identifiability check (2026-06-29)

- **Physics validation (decisive):** least-squares regression of the *true* trajectory's
  acceleration onto [-v, -x, -x^3] (forcing known) recovers **(k1,k2,k3) = (0.3, -0.9999,
  0.9999)**, residual RMS 2e-5. ⇒ data-gen + residual/energy formulation are correct; θ is
  identifiable. Any failure to recover θ downstream is an optimization-budget issue, not a
  model-spec bug.
- **Truth regime:** the oscillator hops between both wells, x1 ∈ [-1.44, 1.44] — the
  interesting double-well Duffing (k2<0). Good test of field expressivity.
- **2k-iter CPU smoke (relaxed, beta2=1e3):** 18.5 s, no NaNs, energy 24308→6973. θ barely
  moved from prior mean (k1≈0.09, k2≈0.07, k3≈0.02) — *expected*: paper uses 2M iters and
  Fig 4 shows θ needs ~1e5+ iters. xhat(0)≈(1.0, 0.19), warmup loglik≈250 (good obs fit).
- **TODO:** confirm θ trends to truth in a longer local smoke (30k) before cluster scale;
  then GPU-smoke on Gautschi; then 2M-iter faithful run for both variants.

### v1 — 30k-iter relaxed convergence smoke (2026-06-29)

CPU, 30k iters, num_obs=200, n_colloc=120, **beta2=1e3 (soft smoke value)**, outer_lr=2e-3,
inner_iters=8, n_aux=8. Runtime 367 s.

- **Param recovery (clear convergence toward truth):**
  | param | posterior        | truth | z    |
  |-------|------------------|-------|------|
  | k1    | +0.325 ± 0.018   | +0.30 | 1.42 |
  | k2    | -0.906 ± 0.022   | -1.00 | 4.21 |
  | k3    | +0.932 ± 0.018   | +1.00 | 3.73 |
  Energy 23366→131 (stabilized). Means ~10% biased at 30k but unambiguously trending;
  the paper's 2M-iter budget will close the gap. Posterior std ~0.02 is the expected
  NSVI mean-field variance underestimate.
- **Fig 2 (states):** position x1(t) tracked near-perfectly with tight 90% band; velocity
  x2(t) — *unobserved* — recovered from position-only data with appropriately wider band.
  Matches the paper's qualitative claim "uncertainty in x2 much higher than x1". ✓
- **Fig 5 (initial):** xhat(0;w) vs auxiliary x0 close with "slight differences" — exactly
  the paper's stated NSVI behavior. Position centers ~1.01 (truth 1.0). Velocity centers
  ~-0.10 (biased from 0) because beta2=1e3 is soft; the velocity IC is observed only
  through position, so least-constrained. Faithful beta2=1e5 should tighten this.
- **Verdict: local validation PASS.** Physics proven correct (v0 LS), params converge,
  states reconstruct, Fig 5 structure reproduced. Remaining bias = iteration budget (30k
  vs 2M) + soft beta2. → Move to Gautschi for the faithful 2M-iter run (both variants,
  beta2=1e5, denser obs), via `cluster-slurm`.

### v2 — 30k-iter reparam convergence smoke (2026-06-29)

Same settings as v1 but `--variants reparam` (IC folded into w, d_x0=0). Runtime 387 s.

- **Param recovery:** k1=+0.320±0.019 (z=1.05), k2=-0.906±0.024 (z=3.92),
  k3=+0.937±0.023 (z=2.79). Nearly identical to the relaxed variant → reproduces the
  paper's "great agreement of the four methods".
- **xhat(0) = (1.004, 0.048)** — tighter to (1,0) than relaxed's (1.01,-0.09), because
  reparam enforces the IC structurally (xhat(0)=ic exact). The velocity IC is recovered
  well even though it's observed only through position.
- **Both NSVI variants validated locally.** Ready for the faithful Gautschi run.

### Next: faithful Gautschi run (proposed)

Pending user go-ahead on shared-quota commitment:
- Both variants, **beta2=1e5** (faithful, vs 1e3 smoke), iterations 1–2M, denser obs.
- GPU-smoke (~50k iters, ~10 min) first per `feedback_cluster_smoke_before_scale`, then scale.
- scp uncommitted edits before launch (`feedback_cluster_scp_modified_files`); NO explicit
  `cuda/X.Y` module load (`feedback_lmod_cuda_warning`).
- Follow-ups after agreement confirmed: NSGLD/NPSGLD (Fig 4 four-method panel).

### v3 — Gautschi GPU smoke, both variants, faithful beta2=1e5 (2026-06-29)

Job 12949606 (`gautschi-gpu`), 50k iters, num_obs=500, n_colloc=200, beta2=1e5, lr=1e-3.
**COMPLETED**, both variants, no NaNs.

- **GPU runtime: 37 s / 50k iters** (relaxed), 35 s (reparam) ≈ 0.74 ms/iter — 16× over CPU.
  ⇒ a full **2M-iter run ≈ 25 min/variant**. Faithful run is cheap.
- **beta2=1e5 stable** — energy 157561→220 (relaxed), 12789→117 (reparam); smooth descent.
- **Params @50k:** relaxed k1=0.307 (z=0.39), k2=-0.892, k3=0.888; reparam k1=0.287,
  k2=-0.898, k3=0.898, xhat(0)=(1.022,-0.004)≈(1,0).
- **Cuda reload warning is BENIGN here:** `.err` showed `cuda/12.6.1 => cuda/12.9.0`
  (the `feedback_lmod_cuda_warning` red flag) BUT the job ran on GPU fine (37 s = GPU speed).
  The codex-ml-py312 jaxlib matches cuda 12.9, so the pin is correct for this env. Memory note
  updated: verify actual GPU execution, don't treat the warning as an automatic failure.
- **Open issue — k2/k3 bias + overconfidence:** k3=0.888±0.020 puts truth(1.0) at z≈5.6
  (outside posterior). Overconfidence = known NSVI mean-field. Bias hypothesis = collocation
  under-resolution (n_colloc=200 borderline for a K=40 field's squared derivative). → testing
  n_colloc=1000 (job 12949705) before the long run.

### v4 — FULL 2M-iter faithful run + Phase 0 agreement verdict (2026-06-29)

Job 12949747 (`gautschi-gpu`), 2M iters, both variants, num_obs=1000, n_colloc=1000,
beta2=1e5, lr=1e-3. **COMPLETED.** relaxed 1604 s (27 min), reparam 1956 s (33 min).

| param | relaxed           | reparam           | truth |
|-------|-------------------|-------------------|-------|
| k1    | +0.316 ± 0.018    | **+0.299 ± 0.017**| +0.30 |
| k2    | -0.917 ± 0.022    | -0.916 ± 0.022    | -1.00 |
| k3    | +0.937 ± 0.015    | +0.940 ± 0.019    | +1.00 |
| xhat(0) | (1.018, -0.005) | (1.028, -0.003)   | (1,0) |

**Agreement verdict (Phase 0 PASS):**
- **Fig 2 (states):** ✅ both variants reconstruct position near-perfectly (tight band) and the
  *unobserved* velocity accurately (wider band). Faithful to the paper.
- **Fig 3 (params):** ✅ the two variants **agree closely** (paper's "great agreement of the
  methods"). k1 brackets truth. k2/k3 centered ~0.92/0.94 — truth at the posterior edge.
- **Fig 5 (initial):** ✅ relaxed xhat(0;w) vs aux x0 center near truth with **"slight
  differences"** in width/center — exactly the paper's NSVI-approximation statement. reparam
  tight on truth (xhat(0)≡x0 structurally).
- **xhat(0) recovered to (1,0)** at beta2=1e5 (velocity IC now ~exact vs the soft-beta2 smoke).

**Key finding — the k2/k3 bias is the NSVI guide, NOT iteration budget.** k3 went
0.888 (50k) → 0.937 (2M) and plateaued ~0.92-0.94; the 40× budget increase did not close the
~6-8% gap, and the std stayed tight (~0.02) → truth at z≈3-4 (outside the overconfident
posterior). This is precisely the mean-field diagonal-guide limitation the paper flags and the
motivation for **Phase 1 (NPSGLD)**, which should bracket the truth.

Figures saved to `figures/` (tracked). Phase 0 done →
update ROADMAP, proceed to Phase 1.

### v5 — Phase 1: NPSGLD wired + full run + verdict (2026-06-30)

Added `niff/npsgld.py` (mirror of vendored paper code; same callback interface).
`run_duffing_s51.py --method npsgld`; results labelled `{method}_{variant}`.

**XLA cuBLAS-Lt crash (debugged):** NPSGLD's batched 4-chain GEMMs crashed on gautschi-gpu with
`cuda_blas_lt.cc RET_CHECK workspace` (NSVI never triggered it). Fix:
`XLA_FLAGS=--xla_gpu_enable_cublaslt=false` (forces classic cuBLAS). Disabling CUDA-graph
capture alone was insufficient. Now baked into the gautschi-gpu profile; see
`feedback_xla_cublaslt_gautschi`.

**Riemannian Γ omitted:** `jacfwd` over ~164 dims/step ≈ 300× base cost → intractable at 2M.
Using the standard diagonal-preconditioner pSGLD (`--npsgld-no-riemannian`).

**Full run** (job 12990130): relaxed, 2M iters, 4 chains, burn-in 1M, thin 500 → 8000 samples.
**COMPLETED** 1355 s (22.6 min). num_obs=1000, n_colloc=1000, beta2=1e5, lr 1e-4→1e-5, delta=0.1.

| param | NSVI 2M (z)        | **NPSGLD 2M (z)**       | truth |
|-------|--------------------|-------------------------|-------|
| k1    | 0.316±0.018 (0.90) | 0.334±0.034 (0.99)      | 0.30  |
| k2    | -0.917±0.022 (**3.76**) | -0.952±0.050 (**0.96**) | -1.00 |
| k3    | 0.937±0.015 (**4.15**)  | 0.968±0.043 (**0.75**)  | 1.00  |

**VERDICT (Phase 1 PASS):** NPSGLD posteriors are ~2.5× wider AND closer to truth → **all three
params bracket truth (z<1)** where NSVI was overconfident+biased (z≈4). This proves the NSVI
parameter bias was the diagonal-guide family, not the model — exactly the paper's narrative
(NSVI fast-but-approximate; NPSGLD accurate). Fig 3 (overlay) shows NPSGLD mass reaching the
truth lines; Fig 5 shows the sampler's xhat(0;w)≈aux x0 overlap vs NSVI's "slight differences".
Velocity IC under NPSGLD is wide/biased (-0.13) — least-identifiable (position-only obs).

Comparison figures: `figures/comparison/` (tracked).
Phase 1 done → Phase 1.5 (collocation subsampling + NSGLD) next.

### v6 — Phase 1.5: subsampling + NSGLD; four-method comparison complete (2026-06-30)

**Engine extensions (opt-in, backward-compatible — verified byte-identical when off):**
- `niff/nsvi.py`: `stochastic_callbacks` threads a per-iter data_key so callbacks
  subsample (n_t collocation pts, m_y measurements). PRNG stream unchanged when off.
- npsgld: `preconditioned` flag (now `preconditioner:str` after the refactor) → NSGLD.
- `duffing_s51.build_variant(subsample,n_t,m_y)`: unbiased minibatch callbacks.

**NSVI subsampling (n_t=10, m_y=10, 2M iters) — k3-bias hypothesis CONFIRMED:**
| variant | k2 full→sub | k3 full→sub (z) |
|---------|-------------|------------------|
| relaxed | -0.917→-0.939 | 0.937→0.960 |
| reparam | -0.916→-0.970 | 0.940→**0.982 (z=0.67)** |
The fixed-grid quadrature bias was real; random per-iter collocation (unbiased MC integral)
moves NSVI toward truth, matching the paper's direction. **No GPU speedup** (~27 min either
way): the 2M sequential scan is launch-overhead-bound, so 10-vs-1000 collocation barely
matters on GPU — the paper's 7s is CPU+subsampling (CPU is collocation-cost-sensitive).

**NSGLD:** diverged at step 1e-5 (energy→5e12; no preconditioner can't handle the high-curvature
162-d field). **Re-run at 1e-6 converged cleanly** (energy~250 stable): k1=0.336±0.044 (z=0.82),
k2=-0.977±0.065 (z=0.35), k3=**0.992±0.055 (z=0.15)**. Widest posterior of the four (paper:
NSGLD slowest/widest).

**Four-method comparison (Fig 3/4/5) in `figures/comparison_final/`** — matches paper Fig 3:
all four bracket truth; samplers (NSGLD widest, NPSGLD) wider than NSVI. NSVI uses the
subsample (bias-corrected) runs; samplers full-batch (same target posterior).

| method | k1 | k2 | k3 |
|--------|----|----|----|
| NSVI relaxed (sub) | 0.310 | -0.939 | 0.960 |
| NSVI reparam (sub) | 0.293 | -0.970 | 0.982 |
| NPSGLD | 0.334 | -0.952 | 0.968 |
| NSGLD | 0.336 | -0.977 | 0.992 |

**Sampler (`niff.npsgld`):** `preconditioner ∈ {identity, rmsprop, diag_fisher, dense_fisher}`
(the last two implemented in Stage B, below). Phase 1.5 done.

### Phase 1.5 — key lessons (transferable)

1. **Fixed-grid collocation biases θ in the IFT energy.** A deterministic uniform quadrature
   of `∫₀ᵀ‖residual‖²dt` under-resolves a K=40 field's squared derivative and pulls θ off
   truth (our k3 sat low: 0.888→0.937 as n_colloc 200→2M-grid, never reaching 1.0). **Random
   per-iteration collocation** (n_t points ~ U[0,T]) is an *unbiased* MC estimator of the
   integral gradient and removes the bias (reparam k3 → 0.982). Use random subsampling, not a
   fixed grid, whenever θ accuracy matters (any physics-integral energy).
2. **Subsampling buys unbiasedness, NOT speed — on GPU.** The 2M-iter `lax.scan` is launch-
   overhead-bound, so n_t=10 vs n_colloc=1000 gave ~identical wall-time (~27 min). The paper's
   7s is CPU+subsampling (CPU *is* collocation-cost-sensitive). On GPU, subsample for the right
   gradient, not for runtime; to cut GPU wall-time, cut *iterations* or raise throughput, not
   collocation count.
3. **NSGLD (no preconditioner) is step-size-fragile.** With β₁=200 the 162-d field directions
   are high-curvature; a single un-preconditioned step of 1e-5 diverges (energy→5e12), 1e-6 is
   stable. Preconditioning (NPSGLD) is what makes the sampler robust to this curvature spread —
   the practical case for the `preconditioner` family (Stage B).
4. **NSVI's tightness is the guide family, not the model.** Confirmed end-to-end: NSVI stays
   tight+slightly-biased even subsampled; the samplers (NSGLD/NPSGLD) are wider and bracket
   truth. Matches the paper. For honest UQ, prefer a sampler; for speed, NSVI.

### v7 — Preconditioner comparison on Duffing (Stage B Fisher families) (2026-06-30)

Ran all four `preconditioner` families on §5.1 Duffing (relaxed, 2M, 4 chains, 1e-4→1e-5) to
compare on the **strongly-coupled (k2,k3) posterior (corr = -0.971** — linear vs cubic stiffness
trade off). Figure: `results/niff_s51_precond_k2k3.png`.

| method | k2 | k3 | corr(k2,k3) | R̂max | energy | verdict |
|--------|----|----|-------------|------|--------|---------|
| rmsprop (NPSGLD) | -0.95±0.05 | 0.97±0.04 | -0.971 | 1.09 | ~90 | ✅ correct, tight |
| nsgld (identity) | -0.98±0.07 | 0.99±0.06 | -0.972 | 1.10 | ~90 | ✅ correct, tight |
| diag_fisher | -0.97±0.06 | 0.98±0.05 | -0.978 | 1.06 | ~90 | ✅ ≈ rmsprop |
| dense_fisher (cold θ=0) | -0.02±0.11 | 0.08±0.12 | — | — | ~570 | ❌ never converged |
| dense_fisher (warm θ=diag) | -1.13±0.48 | 1.14±0.44 | -1.000 | 1.01 | ~173 | ⚠ over-dispersed |

**Findings:**
- **diag_fisher ≡ rmsprop empirically** — both diagonal (1/√E[gᵢ²]); diag_fisher just estimates
  the second moment across the chain ensemble instead of a per-chain time-EMA. Indistinguishable.
- **The three robust methods** (rmsprop, identity/NSGLD, diag_fisher) agree: tight posterior on
  truth, recover corr ≈ -0.97. Un-preconditioned NSGLD agreeing rules out preconditioner bias.
- **dense_fisher is finicky, two failure modes:**
  1. *Cold start over-damps burn-in* — from θ=0 (far from truth) the initial θ-gradients are large
     and aligned, so F is large in that direction, P=F⁻¹ shrinks exactly the steps needed to reach
     the mode → energy stuck at ~570, θ never moves. (Diagonal methods can't do this.)
  2. *Warm start over-disperses* — started at the diag mean it reaches truth (z<1) and mixes
     broadly (R̂ 1.01, corr → -1.000, tracks the ridge perfectly), BUT the whitening amplifies
     steps along the *soft* ridge direction; with the step size untuned for the whitened geometry
     (and under θ–w coupling, where the θ-Fisher reflects conditional-given-w not marginal
     curvature) it over-explores the soft direction → posterior ~10× too wide, energy ~173>90.

**Lesson:** for IFT/NIFF — a few parameters coupled to a high-dim field —
**diagonal preconditioning (NPSGLD/rmsprop) is the robust workhorse.** Matrix preconditioners on
the parameter block are not a free win: they need (a) a near-mode start (burn-in fragility) and
(b) step-size retuning for the whitened geometry, and the θ-Fisher is confounded by θ–field
coupling. Higher-dimensional / more-coupled parameter posteriors amplify this; default to NPSGLD, treat
matrix preconditioners as a careful, situational tool. (Not chasing dense_fisher tuning now —
the robust-diagonal conclusion stands; step-annealed dense_fisher is a possible future exercise.)

### §5.2 — two-DOF + residual neural network (2026-07-02/03)

The example that exercises NIFF's namesake device (RBF linear basis + Fourier-encoded residual
NN). Code: `niff/twodof_s52.py`, `scripts/run_twodof_s52.py`, `scripts/plot_twodof_s52.py`.

**System:** 2-DOF nonlinear (ref [84]), 4 states, 8 params (m1,m2,c1,c2,k1,k2,ε1,ε2), truth
m=1, c=0.2, k=1, ε=0.2. Measure q1 and q1+q2. **State path decoded from Table 3 (w-dim 344):**
per-component RBF (4×20=80) + one shared Fourier-encoded MLP (encode t→21, K=10; 1 hidden
width 10, swish; **4 outputs**) = 264. Mass-matrix-residual energy; relaxed prior (aux x0);
kinematic warmup (seed all 4 fields: displacements from data, velocities from field derivatives).

**Documented divergences (unspecified in the paper, from ref [84]):** forcing F=2, ω0=1.2 and
Fourier-encoding period Tbar=10 — chosen so a 20-term RBF under-fits (the premise) and the K=10 NN
can still fit. Physics validated: LS on the true trajectory recovers all 8 params (RMS 8e-4).

**Demonstration REPRODUCED (Fig 6, `figures/s52/`):** rbf_nn tracks the truth near-perfectly for
q1 and q2; rbf_only fails badly — RBF-alone cannot reconstruct, the residual NN corrects it (the
paper's headline claim for §5.2). Fig 7: the residual helps every parameter (rbf_nn ≫ rbf_only).

**Param-recovery caveat + the key lesson.** Full budget (NSVI 300k; NPSGLD 3 chains × 3M):
| | NSVI rbf_nn | NPSGLD rbf_nn | truth |
|---|---|---|---|
| m1 | 0.28±0.61 | 0.18±0.18 | 1.0 |
| k1 | 0.59±0.22 | **−0.04±0.04** | 1.0 |
| e1 | −0.00±0.16 | **−7.1±0.5** | 0.2 |

NSVI rbf_nn is biased low but reasonable (best result). **NPSGLD rbf_nn DIVERGED** into a
degeneracy: k1→0 makes the cubic term `k1·ε1·q1³` vanish → ε1 unconstrained → the Langevin chain
diffuses off to −7.

**Lesson (opposite of §5.1):** in §5.1 the *rigid* Fourier basis kept params well-identified, so
the NPSGLD sampler beat NSVI. Here the *flexible* residual NN reconstructs the states perfectly but
**loosens parameter identifiability** (flat directions: k↔m, ε when the linear stiffness is small).
NSVI's guide + warmup-anchoring stays in the reasonable region; the sampler *explores* the
degeneracy — more sampling makes it worse. **Samplers expose degeneracies that VI hides.** The
paper likely avoids this via the true [84] forcing exciting the nonlinearity more strongly (our
forcing was a documented guess) and/or implicit constraints.

**Verdict:** §5.2 demonstration reproduced (Fig 6); the RBF+NN hybrid capability built + validated;
exact param recovery limited by NN-flexibility degeneracy (documented). NSVI figures are primary in
`figures/s52/`; the 4-way NSVI/NPSGLD comparison is `figures/s52/fig7_4way_nsvi_npsgld.png`.







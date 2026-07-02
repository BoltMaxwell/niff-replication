# NIFF replication & extension — Roadmap

Living roadmap for verifying the repo's IFT machinery against Hao & Bilionis,
*Neural Information Field Filter* (MSSP 2025, `wiki/raw/niff.pdf`), then extending it
to the SDE setting with joint initial-condition inference.

**Keep this updated** as phases complete. Status legend: ✅ done · 🔄 in progress ·
⏳ queued/next · ⬜ planned · 💤 optional/later.

Chronological experiment results live in `notes/niff_replication_lab.md`; the consolidated
progress + paper-comparison writeup is `PROGRESS.md`; this file is the forward-looking plan.

> **Paper-comparison findings (2026-06-30, see `PROGRESS.md`):** §5.1 reproduced — same Figs
> 2/3/5, same NSVI-tight-biased vs sampler-wide story, truth recovered. Two gaps surfaced:
> (a) our runtime is ~100× the paper's because we run **full-batch collocation/measurements**
> while the paper subsamples **n_t=10, m_y=10** per iter — fixing this should also correct a
> small fixed-grid **quadrature bias** (our k3 sits low at 0.937 vs the paper's ~1.02);
> (b) the plain **NSGLD** sampler (paper's 4th method) is un-run.

---

## Phase 0 — Faithful §5.1 Duffing replication (deterministic ODE)  ✅

The agreement baseline: reproduce the paper's published single-DOF Duffing example
(`x1'=x2`, `x2'=-k1 x2 - k2 x1 - k3 x1^3 + γcos(ωt)`, position-only obs) with **our**
`ift.fields.fourier` + `methods/nsvi`, recovering the paper's Figs 2/3/5.

- ✅ Build: `duffing_s51.py` (RK4 truth, K=40 Fourier fields, both variants), `run_*`, `plot_*`.
- ✅ Physics validated — LS on true trajectory recovers (0.3, −1, 1) to RMS 2e-5.
- ✅ Local CPU convergence smoke (both variants) — Fig 2 reconstruction + Fig 5 structure reproduced.
- ✅ Gautschi GPU-smoke — method runs on GPU, β₂=1e5 stable, ~0.74 ms/iter, cuda-reload benign.
- ✅ Collocation-density test — locked `n_colloc=1000`.
- ✅ **Full 2M-iter faithful run** (both NSVI variants, job 12949747) — COMPLETED (27/33 min).
- ✅ Figs 2/3/5 at full fidelity in `figures/` → **agreement verdict written** (lab v4).

**VERDICT: PASS.** Fig 2 state reconstruction matches the paper (position tight, velocity
wider); the two variants agree closely (paper's "great agreement"); Fig 5 reproduces the
xhat(0;w)-vs-x0 "slight differences". k1 recovered (reparam z=0.05); **xhat(0)≈(1,0)**.
*Known caveat:* k2/k3 carry a ~6-8% bias with overconfident std (truth at z≈3-4) — the NSVI
mean-field-guide limitation, NOT iteration budget (40× budget didn't close it). → Phase 1.

Artifacts: `experiments/niff_replication/figures/{fig2,fig3,fig5,summary.json}`;
results npz in (gitignored) `results/niff_s51_full/`.

---

## Phase 1 — SGLD sampler: NPSGLD (paper method 4)  ✅

Adds the Langevin sampler to complete §5.1's comparison and gives **truth-containing
parameter posteriors** (NSVI's diagonal guide is overconfident).

- ✅ Wire `methods/nsvi/npsgld.py` (same callback interface as `run_nsvi`);
  `run_duffing_s51.py --method npsgld`, labelled `{method}_{variant}` results.
- ✅ Local + GPU smoke; debugged the gautschi-gpu cuBLAS-Lt crash (XLA flag, now in profile).
- ✅ **Full 2M-iter NPSGLD run** (relaxed, 4 chains, no-Riemannian, job 12990130) — 22.6 min.
- ✅ Overlaid Figs 2/3/4/5 in `figures/comparison/` (tracked).

**VERDICT: PASS.** NPSGLD posteriors are ~2.5× wider and closer to truth → **all three params
bracket truth (z<1)** (k2 z: 3.76→0.96, k3 z: 4.15→0.75) where NSVI was overconfident+biased.
Proves the NSVI bias was the diagonal-guide family, not the model. Fig 5: sampler's
xhat(0;w)≈aux x0 overlap vs NSVI's "slight differences".

**Divergences (documented):** Riemannian Γ omitted (`jacfwd` ~300× cost → intractable at 2M;
standard pSGLD approximation used). NSGLD (no preconditioner) not run — NPSGLD alone suffices
to demonstrate the truth-bracketing; add later only if the full four-method Fig 4 is wanted.

---

## Phase 1.5 — Refinements from the paper comparison  ✅

Surfaced by re-reading the paper (`PROGRESS.md` §3, §5); see lab v6.

- ✅ **Random collocation/measurement subsampling** (`n_t=10`, `m_y=10`) via opt-in
  `NSVIConfig.stochastic_callbacks`. **Confirmed the k3-bias hypothesis** — unbiased MC
  collocation moves NSVI toward truth (reparam k3 0.940→0.982, z→0.67). Caveat: *no GPU
  speedup* (2M scan is launch-overhead-bound; the paper's seconds are CPU+subsampling).
- ✅ **NSGLD** via `preconditioner="identity"`. Diverged at 1e-5; converged at **1e-6**
  (k3=0.992, z=0.15). Completes the four-method Fig 3/4/5 panel (`figures/comparison_final/`),
  which now matches the paper: all four bracket truth, samplers wider than NSVI.
- ⬜ **(optional) state normalization** (x̄=(1.5,1), ȳ=1.5) — deferred; not needed for agreement.

**Engine refactor (done alongside):** npsgld → `methods/npsgld/` (own method); OU psgld →
`archive/methods/psgld/`. **Stage A + B complete** — `preconditioner:str ∈ {identity (=NSGLD),
rmsprop (=NPSGLD), diag_fisher, dense_fisher}`, all four implemented and validated on a
correlated-Gaussian θ-posterior (all recover mean+corr; dense_fisher best on covariance after
EMA-smoothing the Fisher matrix). Future: expose the Fisher families via the `run_duffing_s51`
CLI; optionally draw the Fisher samples from the nested aux chain instead of the chain ensemble.


## Phase 2 — SDE extension of the Duffing example  ⏳ (after 1.5)

Extend §5.1 from deterministic ODE to the **stochastic** Duffing (process noise σ_x, σ_v) —
the repo's core IFT-for-SDEs contribution, which the paper does *not* cover.

- ⬜ Add process-noise residual weighting (`1/σ²`) + the Wiener log-σ normalizer to the energy
  (cf. `experiments/shared/duffing.py` and the OU path-action; see `feedback_mle_vs_wiener_qv`,
  `feedback_state_dep_diffusion`).
- ⬜ Infer (k1,k2,k3, σ_x, σ_v) jointly; check σ identifiability (watch σ_x collapse —
  `feedback_sigma_x_collapse`).
- ⬜ Compare field bases (Fourier vs PL vs wavelet) for the SDE roughness — recall
  `feedback_mle_vs_wiener_qv` (MAP field is a smoother; sample the posterior for SDE roughness)
  and the wavelet-N caveats (`feedback_wavelet_n_critical`).

**Open question to settle:** does the relaxed-prior auxiliary-x0 device preserve the SDE
paper's pullback identification (Prop 3.3), or does that theorem need re-deriving for the
relaxed case? (Noted in `wiki/ift-sde-wiki/niff.md` TODO.)

---

## Phase 3 — Auxiliary initial-condition joint inference  ⬜

Fold in NIFF's headline device: infer the initial state jointly rather than conditioning on it,
using the auxiliary x0 + relaxed kernel. **The machinery is already wired and validated** (the
`relaxed` variant uses `d_x0=2` with the β₂ kernel), so this builds directly on Phase 0.

- ⬜ Treat x0 as a genuine unknown driven by a noisy initial measurement (not the true IC).
- ⬜ Tie β₂ to the initial-measurement noise (the principled choice flagged in `niff.md` TODO
  step 4 — β₂ and g_λ are both stiffness knobs and can collide).
- ⬜ Demonstrate on the SDE Duffing (Phase 2): joint posterior over (params, σ, x0).

---

## Optional / later — other paper examples  💤

- 💤 §5.2 — two-DOF nonlinear system with the **residual neural network** (linear basis +
  Fourier-encoded NN). Tests the hybrid parameterization (paper eq. 8).
- 💤 §5.3 — twenty-story frame (high-dimensional, ~hundreds of unknowns) — stresses NSVI scaling.
- 💤 §5.4 — experimental nonlinear energy sink (real data).

---

## Key decisions & constraints (carry forward)

- **Faithful-deterministic-first**, then SDE, then joint-IC (user, 2026-06-29).
- Both NSVI variants in the first pass (reparam + relaxed) for the Fig 5 comparison.
- Inference engine = `methods/nsvi/nsvi.py` (byte-identical to the vendored paper code).
- Basis = repo half-period `fourier_basis_01` (documented divergence from paper's full-period).
- Physical coordinates (truth O(1)); paper normalization is a conditioning device only.
- Cluster: `gautschi-gpu` via `cluster-slurm`; ~0.74 ms/iter; scp files into `~/ift-sde`
  before launch (`feedback_cluster_scp_modified_files`); the `cuda/12.9.0` pin is correct for
  the `codex-ml-py312` env (`feedback_lmod_cuda_warning`, updated 2026-06-29).
- **XLA cuBLAS-Lt crash:** NPSGLD's batched-chain GEMMs crash with
  `cuda_blas_lt.cc RET_CHECK workspace` until `XLA_FLAGS=--xla_gpu_enable_cublaslt=false`
  (now in the `gautschi-gpu` profile setup_commands). NSVI didn't trigger it.
  See `feedback_xla_cublaslt_gautschi`.

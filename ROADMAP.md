# NIFF replication — roadmap / status

Status of reproducing Hao & Bilionis, *Neural Information Field Filter*
(<https://doi.org/10.1016/j.ymssp.2024.112253>). The **§5.1 Duffing example is complete**;
this file records the phases done and the optional remaining paper examples.

Detailed results: `notes/lab.md` (chronological) and `PROGRESS.md` (paper comparison).
Legend: ✅ done · ⬜ optional / not started.

---

## Completed — §5.1 Duffing oscillator

### Phase 0 — NSVI, both state-path variants  ✅
Build (`niff/duffing_s51.py`: RK4 truth, K=40 Fourier fields, reparameterized + relaxed
variants; `scripts/run_*`, `scripts/plot_*`). Physics validated (least-squares on the true
trajectory recovers (0.3, −1, 1) to RMS 2e-5). Full 2M-iteration runs reproduce **Figs 2/3/5**.
*Verdict: PASS.* NSVI posteriors are tight and slightly biased on k2/k3 — the documented NSVI
mean-field-guide behavior, matching the paper's own NSVI.

### Phase 1 — NPSGLD sampler  ✅
`niff.npsgld.run_npsgld` (same callback interface as `run_nsvi`), `--method npsgld`. The
sampler posteriors are ~2.5× wider and **bracket the truth** (k2/k3 z: ~4 → <1), confirming the
NSVI tightness is the guide family, not the model. Fig 5: the sampler's xhat(0;w) and auxiliary
x0 overlap, vs NSVI's "slight differences" — as in the paper.

### Phase 1.5 — collocation subsampling + NSGLD  ✅
- **Random per-iteration collocation/measurement subsampling** (`--subsample`, `n_t`/`m_y`) via
  `NSVIConfig.stochastic_callbacks`. Removed a fixed-grid quadrature bias — NSVI reparam k3
  0.940 → **0.982**. (No GPU speedup: the long scan is launch-overhead-bound.)
- **NSGLD** via `--method nsgld` (`preconditioner="identity"`). Completes the four-method
  panel (`figures/comparison_final/`): all four bracket truth, samplers wider than NSVI.

### Stage B — preconditioner family  ✅
`niff.npsgld` exposes `preconditioner ∈ {identity (=NSGLD), rmsprop (=NPSGLD), diag_fisher,
dense_fisher}`, validated on a correlated-Gaussian θ-posterior. Study on the coupled §5.1
(k2,k3) posterior (`figures/preconditioner/`, lab v7): diagonal families are the robust
workhorse (`diag_fisher` ≡ `rmsprop`); the dense matrix preconditioner is finicky (over-damps
cold burn-in, over-disperses without near-mode start + step retuning).

---

## Optional — remaining paper examples  ⬜

- ⬜ **§5.2** — two-DOF nonlinear system with a **residual neural network** (linear basis +
  Fourier-encoded NN, paper eq. 8) on an RBF basis. The cleanest next example; the only one
  that exercises the hybrid parameterization this repo doesn't yet have.
- ⬜ **§5.3** — twenty-story Bouc–Wen frame (high-dimensional, w-dim 4660). Heavy.
- ⬜ **§5.4** — experimental nonlinear energy sink (real data; needs the external dataset).
- ⬜ **Faithful cosmetics for §5.1** — state normalization (x̄=(1.5,1)) and the full-period
  Fourier basis, if an exact match to the paper's conditioning is wanted (not needed for
  agreement).

---

## Notes / conventions

- **Basis:** half-period Fourier `[1, cos(πkt), sin(πkt)]` (the paper uses full-period
  `2πk t/T̄`; both K=40-expressive).
- **Coordinates:** physical (truth is O(1)); the paper's state normalization is a
  conditioning device only and does not change the posterior.
- **Riemannian Γ** correction is omitted in the preconditioned samplers (a `jacfwd` ~300× the
  base cost); the standard practical pSGLD approximation is used.
- **GPU gotchas (some jaxlib/CUDA builds):** batched-chain GEMMs may hit a cuBLAS-Lt workspace
  crash — run with `XLA_FLAGS=--xla_gpu_enable_cublaslt=false`; and cuSOLVER factorizations
  (`jnp.linalg.inv`/`cholesky`) can be unavailable, so `dense_fisher` uses a manual Cholesky +
  `solve_triangular` (cuBLAS) instead. Both pass silently on CPU.

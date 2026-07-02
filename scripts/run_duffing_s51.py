#!/usr/bin/env python3
"""Run NIFF §5.1 Duffing replication with NSVI or NPSGLD.

Each run is labelled ``{method}_{variant}`` (e.g. ``nsvi_relaxed``,
``npsgld_relaxed``, ``nsvi_reparam``) and writes ``{label}_posterior.npz`` +
``{label}_history.json`` into ``--output-dir``, so multiple methods/variants can
be dropped into one directory and overlaid by ``plot_duffing_s51.py``.

Examples
--------
NSVI, both variants (Phase 0 faithful)::

    ... --method nsvi --variants relaxed reparam --iterations 2000000 \\
        --num-obs 1000 --n-colloc 1000 --beta2 1e5

NPSGLD, relaxed (Phase 1 — paper methods 3/4 use the relaxed prior)::

    ... --method npsgld --variants relaxed --npsgld-iterations 2000000 \\
        --npsgld-chains 4 --npsgld-burn-in 1000000 --npsgld-thinning 1000 \\
        --num-obs 1000 --n-colloc 1000 --beta2 1e5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

jax.config.update("jax_enable_x64", True)

from niff.nsvi import NSVIConfig, draw_nsvi_samples, run_nsvi
from niff.npsgld import NPSGLDConfig, run_npsgld
from niff.duffing_s51 import (
    DuffingS51Config,
    build_variant,
    simulate_truth,
)

PARAM_NAMES = ("k1", "k2", "k3")


def _posterior_summary(method, variant_name, theta_samples, xhat0, truths, runtime, extra):
    summary = {"method": method, "variant": variant_name, "runtime_s": runtime, "parameters": {}}
    for i, name in enumerate(PARAM_NAMES):
        col = theta_samples[:, i]
        summary["parameters"][name] = {
            "true": float(truths[name]),
            "post_mean": float(np.mean(col)),
            "post_std": float(np.std(col)),
            "z": float(abs(np.mean(col) - truths[name]) / max(np.std(col), 1e-9)),
        }
    summary["xhat0_pos_mean"] = float(np.mean(xhat0[:, 0]))
    summary["xhat0_vel_mean"] = float(np.mean(xhat0[:, 1]))
    summary.update(extra)
    return summary


def _xhat0_from_w(variant, w_samples):
    cols = jax.vmap(variant.x_at_0_w)(jnp.asarray(w_samples))
    return np.column_stack([np.asarray(cols[0]), np.asarray(cols[1])])


def run_nsvi_variant(cfg, data, variant_name, args):
    variant = build_variant(cfg, data, variant=variant_name, beta1=args.beta1, beta2=args.beta2,
                            warmup=not args.no_warmup, subsample=args.subsample, n_t=args.n_t, m_y=args.m_y)
    nsvi_cfg = NSVIConfig(
        iterations=args.iterations, inner_iterations=args.inner_iterations,
        outer_lr=args.outer_lr, inner_lr=args.inner_lr, n_outer_samples=1,
        n_aux_samples=args.n_aux_samples, theta_guide=args.theta_guide,
        init_mean=variant.init_mean, init_std=0.1, inner_init_std=0.5,
        theta_prior_std=1.0, x0_prior_std=1.0, partition_weight=1.0,
        partition_anneal_steps=max(1, args.iterations // 10), log_every=args.log_every, quiet=False,
        stochastic_callbacks=args.subsample,
    )
    sub = f" subsample(n_t={args.n_t},m_y={args.m_y})" if args.subsample else ""
    print(f"\n=== nsvi_{variant_name}  d_w={variant.d_w} d_theta={variant.d_theta} "
          f"d_x0={variant.d_x0}  beta1={args.beta1} beta2={args.beta2}{sub} ===", flush=True)
    t0 = time.time()
    result = run_nsvi(jr.PRNGKey(cfg.seed + 1000), d_w=variant.d_w, d_theta=variant.d_theta,
                      d_x0=variant.d_x0, log_likelihood_fn=variant.log_likelihood_fn,
                      energy_fn=variant.energy_fn, config=nsvi_cfg)
    runtime = time.time() - t0
    samples = draw_nsvi_samples(jr.PRNGKey(cfg.seed + 2000), result.params, n_samples=args.posterior_draws,
                                d_w=variant.d_w, d_theta=variant.d_theta, d_x0=variant.d_x0, config=nsvi_cfg)
    theta_s = np.asarray(samples["theta_samples"]); w_s = np.asarray(samples["w_samples"])
    x0_s = np.asarray(samples["x0_samples"]); xhat0 = _xhat0_from_w(variant, w_s)
    truths = {"k1": cfg.k1, "k2": cfg.k2, "k3": cfg.k3}
    summary = _posterior_summary("nsvi", variant_name, theta_s, xhat0, truths, runtime,
                                 {"elbo_final": float(result.history["elbo"][-1])})
    return {"summary": summary, "theta_samples": theta_s, "w_samples": w_s, "x0_samples": x0_s,
            "xhat0_samples": xhat0, "history": result.history, "coeff_len": variant.coeff_len}


def run_npsgld_variant(cfg, data, variant_name, args, *, method_label="npsgld", preconditioner="rmsprop"):
    # Samplers run full-batch (the npsgld engine has no stochastic-callback hook).
    variant = build_variant(cfg, data, variant=variant_name, beta1=args.beta1, beta2=args.beta2,
                            warmup=not args.no_warmup, subsample=False)
    # optional theta warm-start (e.g. diagonal-burn-in mean) -> near-mode start for dense_fisher
    init_mean = list(variant.init_mean)
    if args.npsgld_theta_init is not None:
        init_mean[variant.d_w:variant.d_w + variant.d_theta] = list(args.npsgld_theta_init)
    init_mean = tuple(init_mean)
    npsgld_cfg = NPSGLDConfig(
        iterations=args.npsgld_iterations, chains=args.npsgld_chains, burn_in=args.npsgld_burn_in,
        thinning=args.npsgld_thinning, step_size=args.npsgld_step_size,
        step_size_final=args.npsgld_step_size_final, aux_step_size=args.npsgld_step_size,
        aux_step_size_final=args.npsgld_step_size_final, aux_iterations=args.npsgld_aux_iterations,
        alpha_initial=0.99, alpha_final=1.0, alpha_anneal_steps=max(1, args.npsgld_iterations // 2),
        aux_alpha_initial=0.99, aux_alpha_final=1.0, aux_alpha_anneal_steps=max(1, args.npsgld_iterations // 2),
        delta=args.npsgld_delta, theta_prior_std=1.0, x0_prior_std=1.0,
        init_mean=init_mean, init_std=0.1,
        include_riemannian_correction=(preconditioner == "rmsprop") and not args.npsgld_no_riemannian,
        preconditioner=preconditioner,
        trace_every=args.log_every, quiet=False,
    )
    print(f"\n=== {method_label}_{variant_name}  d_w={variant.d_w} d_theta={variant.d_theta} "
          f"d_x0={variant.d_x0}  chains={args.npsgld_chains} preconditioner={preconditioner} ===",
          flush=True)
    t0 = time.time()
    result = run_npsgld(jr.PRNGKey(cfg.seed + 3000), d_w=variant.d_w, d_theta=variant.d_theta,
                        d_x0=variant.d_x0, log_likelihood_fn=variant.log_likelihood_fn,
                        energy_fn=variant.energy_fn, config=npsgld_cfg)
    runtime = time.time() - t0
    theta_s = np.asarray(result.samples["theta_samples"]); w_s = np.asarray(result.samples["w_samples"])
    x0_s = np.asarray(result.samples["x0_samples"]); xhat0 = _xhat0_from_w(variant, w_s)
    truths = {"k1": cfg.k1, "k2": cfg.k2, "k3": cfg.k3}
    summary = _posterior_summary(method_label, variant_name, theta_s, xhat0, truths, runtime,
                                 {"n_samples": int(theta_s.shape[0]), "chains": args.npsgld_chains})
    return {"summary": summary, "theta_samples": theta_s, "w_samples": w_s, "x0_samples": x0_s,
            "xhat0_samples": xhat0, "history": result.history, "coeff_len": variant.coeff_len}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", default="nsvi",
                   choices=["nsvi", "npsgld", "nsgld", "diag_fisher", "dense_fisher"])
    p.add_argument("--variants", nargs="+", default=["relaxed", "reparam"], choices=["relaxed", "reparam"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-obs", type=int, default=500)
    p.add_argument("--n-colloc", type=int, default=200)
    p.add_argument("--fourier-order", type=int, default=40)
    p.add_argument("--beta1", type=float, default=200.0)
    p.add_argument("--beta2", type=float, default=1.0e5)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--posterior-draws", type=int, default=4000)
    # subsampling (nsvi only): IFT n_t collocation points + m_y measurements per iter
    p.add_argument("--subsample", action="store_true",
                   help="per-iteration random collocation/measurement subsampling (nsvi only)")
    p.add_argument("--n-t", type=int, default=10, help="collocation points per iter when --subsample")
    p.add_argument("--m-y", type=int, default=10, help="measurement minibatch per iter when --subsample")
    # NSVI
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--inner-iterations", type=int, default=10)
    p.add_argument("--outer-lr", type=float, default=1.0e-3)
    p.add_argument("--inner-lr", type=float, default=3.0e-3)
    p.add_argument("--n-aux-samples", type=int, default=10)
    p.add_argument("--theta-guide", default="full_rank", choices=["diag", "full_rank"])
    # NPSGLD
    p.add_argument("--npsgld-iterations", type=int, default=200000)
    p.add_argument("--npsgld-chains", type=int, default=4)
    p.add_argument("--npsgld-burn-in", type=int, default=100000)
    p.add_argument("--npsgld-thinning", type=int, default=100)
    p.add_argument("--npsgld-step-size", type=float, default=1.0e-4)
    p.add_argument("--npsgld-step-size-final", type=float, default=1.0e-5)
    p.add_argument("--npsgld-aux-iterations", type=int, default=10)
    p.add_argument("--npsgld-delta", type=float, default=0.1)
    p.add_argument("--npsgld-theta-init", type=float, nargs=3, default=None,
                   help="warm-start theta at (k1 k2 k3); e.g. a diagonal-burn-in mean for dense_fisher")
    p.add_argument("--npsgld-no-riemannian", action="store_true",
                   help="omit the expensive Riemannian Gamma correction (standard pSGLD approximation)")
    p.add_argument("--output-dir", type=Path, default=Path("results/niff_s51"))
    args = p.parse_args()

    if args.subsample and args.method != "nsvi":
        p.error("--subsample is only supported for --method nsvi "
                "(the npsgld/nsgld engine has no stochastic-callback hook).")

    cfg = DuffingS51Config(gamma=0.37, omega=1.2, num_obs=args.num_obs, n_colloc=args.n_colloc,
                           fourier_order=args.fourier_order, seed=args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = simulate_truth(cfg)
    np.savez(out_dir / "data.npz", times=np.asarray(data["times"]), path_x=np.asarray(data["path_x"]),
             path_v=np.asarray(data["path_v"]), obs_times=np.asarray(data["obs_times"]),
             obs_x=np.asarray(data["obs_x"]))

    # sampler methods -> (label, engine preconditioner)
    _sampler = {
        "nsgld": ("nsgld", "identity"),
        "npsgld": ("npsgld", "rmsprop"),
        "diag_fisher": ("diagfisher", "diag_fisher"),
        "dense_fisher": ("densefisher", "dense_fisher"),
    }

    def runner(cfg_, data_, variant_, args_):
        if args_.method == "nsvi":
            return run_nsvi_variant(cfg_, data_, variant_, args_)
        label, precond = _sampler[args_.method]
        return run_npsgld_variant(cfg_, data_, variant_, args_, method_label=label, preconditioner=precond)

    summary_path = out_dir / "summary.json"
    all_runs = json.loads(summary_path.read_text())["runs"] if summary_path.exists() else {}

    for variant_name in args.variants:
        res = runner(cfg, data, variant_name, args)
        label = f"{args.method}_{variant_name}"
        np.savez(out_dir / f"{label}_posterior.npz", theta_samples=res["theta_samples"],
                 w_samples=res["w_samples"], x0_samples=res["x0_samples"],
                 xhat0_samples=res["xhat0_samples"], coeff_len=res["coeff_len"])
        with (out_dir / f"{label}_history.json").open("w") as f:
            json.dump(res["history"], f)
        all_runs[label] = res["summary"]

        s = res["summary"]
        print(f"\n--- {label} posterior (runtime {s['runtime_s']:.1f}s) ---")
        for name in PARAM_NAMES:
            pp = s["parameters"][name]
            print(f"  {name}: {pp['post_mean']:+.4f} ± {pp['post_std']:.4f}  "
                  f"(true {pp['true']:+.2f}, z={pp['z']:.2f})")
        print(f"  xhat(0) mean = ({s['xhat0_pos_mean']:.3f}, {s['xhat0_vel_mean']:.3f})")

    with summary_path.open("w") as f:
        json.dump({"config": cfg.__dict__, "runs": all_runs}, f, indent=2)
    print(f"\n✓ wrote {summary_path}")


if __name__ == "__main__":
    main()

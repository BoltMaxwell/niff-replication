#!/usr/bin/env python3
"""Run NIFF §5.2 two-DOF + residual-NN replication (NSVI or NPSGLD).

Variants: ``rbf_only`` (linear RBF basis) vs ``rbf_nn`` (RBF + residual NN).
The paper's demonstration: rbf_only under-fits → biased parameters; rbf_nn
reconstructs the states → recovers the 8 parameters.

Each run writes ``{method}_{variant}_posterior.npz`` + summary.json into
``--output-dir``.

Example (short local smoke)::

    python experiments/niff_replication/run_twodof_s52.py --method nsvi \\
        --variants rbf_only rbf_nn --iterations 20000 --num-obs 300 \\
        --n-colloc 200 --beta2 1e3 --output-dir results/s52_smoke
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
from niff.twodof_s52 import (
    TwoDOFConfig, build_variant, simulate_truth, PARAM_NAMES, TRUTH,
)


def _summ(method, variant, theta_s, runtime, extra):
    truths = dict(zip(PARAM_NAMES, TRUTH))
    s = {"method": method, "variant": variant, "runtime_s": runtime, "parameters": {}}
    for i, name in enumerate(PARAM_NAMES):
        col = theta_s[:, i]
        s["parameters"][name] = {"true": truths[name], "post_mean": float(np.mean(col)),
                                 "post_std": float(np.std(col)),
                                 "z": float(abs(np.mean(col) - truths[name]) / max(np.std(col), 1e-9))}
    s.update(extra)
    return s


def run_one(cfg, data, variant_name, args):
    residual = variant_name == "rbf_nn"
    warmup_iters = args.warmup_iters_nn if residual else args.warmup_iters_rbf
    variant = build_variant(cfg, data, residual=residual, beta1=args.beta1, beta2=args.beta2,
                            warmup=not args.no_warmup, warmup_iters=warmup_iters)
    print(f"\n=== {args.method}_{variant_name}  d_w={variant.d_w} d_theta={variant.d_theta} "
          f"d_x0={variant.d_x0}  beta1={args.beta1} beta2={args.beta2} ===", flush=True)
    t0 = time.time()
    if args.method == "nsvi":
        nsvi_cfg = NSVIConfig(
            iterations=args.iterations, inner_iterations=args.inner_iterations,
            outer_lr=args.outer_lr, inner_lr=args.inner_lr, n_outer_samples=1,
            n_aux_samples=args.n_aux_samples, theta_guide=args.theta_guide,
            init_mean=variant.init_mean, init_std=0.1, inner_init_std=0.5,
            theta_prior_std=1.0, x0_prior_std=1.0, partition_weight=1.0,
            partition_anneal_steps=max(1, 2 * args.iterations // 3), log_every=args.log_every, quiet=False,
        )
        result = run_nsvi(jr.PRNGKey(cfg.seed + 1000), d_w=variant.d_w, d_theta=variant.d_theta,
                          d_x0=variant.d_x0, log_likelihood_fn=variant.log_likelihood_fn,
                          energy_fn=variant.energy_fn, config=nsvi_cfg)
        samples = draw_nsvi_samples(jr.PRNGKey(cfg.seed + 2000), result.params, n_samples=args.posterior_draws,
                                    d_w=variant.d_w, d_theta=variant.d_theta, d_x0=variant.d_x0, config=nsvi_cfg)
        theta_s = np.asarray(samples["theta_samples"]); w_s = np.asarray(samples["w_samples"])
        x0_s = np.asarray(samples["x0_samples"]); history = result.history
        extra = {"elbo_final": float(result.history["elbo"][-1])}
    else:
        npsgld_cfg = NPSGLDConfig(
            iterations=args.npsgld_iterations, chains=args.npsgld_chains, burn_in=args.npsgld_burn_in,
            thinning=args.npsgld_thinning, step_size=args.npsgld_step_size,
            step_size_final=args.npsgld_step_size_final, aux_step_size=args.npsgld_step_size,
            aux_step_size_final=args.npsgld_step_size_final, aux_iterations=args.npsgld_aux_iterations,
            alpha_initial=0.99, alpha_final=1.0, alpha_anneal_steps=max(1, args.npsgld_iterations // 2),
            aux_alpha_initial=0.99, aux_alpha_final=1.0, aux_alpha_anneal_steps=max(1, args.npsgld_iterations // 2),
            delta=args.npsgld_delta, theta_prior_std=1.0, x0_prior_std=1.0,
            init_mean=variant.init_mean, init_std=0.1,
            include_riemannian_correction=False, preconditioner="rmsprop",
            trace_every=args.log_every, quiet=False,
        )
        result = run_npsgld(jr.PRNGKey(cfg.seed + 3000), d_w=variant.d_w, d_theta=variant.d_theta,
                            d_x0=variant.d_x0, log_likelihood_fn=variant.log_likelihood_fn,
                            energy_fn=variant.energy_fn, config=npsgld_cfg)
        theta_s = np.asarray(result.samples["theta_samples"]); w_s = np.asarray(result.samples["w_samples"])
        x0_s = np.asarray(result.samples["x0_samples"]); history = result.history
        extra = {"n_samples": int(theta_s.shape[0]), "chains": args.npsgld_chains}
    runtime = time.time() - t0
    return {"summary": _summ(args.method, variant_name, theta_s, runtime, extra),
            "theta_samples": theta_s, "w_samples": w_s, "x0_samples": x0_s, "history": history}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", default="nsvi", choices=["nsvi", "npsgld"])
    p.add_argument("--variants", nargs="+", default=["rbf_only", "rbf_nn"], choices=["rbf_only", "rbf_nn"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-obs", type=int, default=500)
    p.add_argument("--n-colloc", type=int, default=400)
    p.add_argument("--beta1", type=float, default=200.0)
    p.add_argument("--beta2", type=float, default=1.0e5)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--warmup-iters-rbf", type=int, default=8000)
    p.add_argument("--warmup-iters-nn", type=int, default=40000)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--posterior-draws", type=int, default=4000)
    # NSVI
    p.add_argument("--iterations", type=int, default=300000)
    p.add_argument("--inner-iterations", type=int, default=10)
    p.add_argument("--outer-lr", type=float, default=1.0e-3)
    p.add_argument("--inner-lr", type=float, default=3.0e-3)
    p.add_argument("--n-aux-samples", type=int, default=10)
    p.add_argument("--theta-guide", default="full_rank", choices=["diag", "full_rank"])
    # NPSGLD
    p.add_argument("--npsgld-iterations", type=int, default=3000000)
    p.add_argument("--npsgld-chains", type=int, default=3)
    p.add_argument("--npsgld-burn-in", type=int, default=1000000)
    p.add_argument("--npsgld-thinning", type=int, default=1000)
    p.add_argument("--npsgld-step-size", type=float, default=1.0e-4)
    p.add_argument("--npsgld-step-size-final", type=float, default=1.0e-5)
    p.add_argument("--npsgld-aux-iterations", type=int, default=10)
    p.add_argument("--npsgld-delta", type=float, default=0.1)
    p.add_argument("--output-dir", type=Path, default=Path("results/niff_s52"))
    args = p.parse_args()

    cfg = TwoDOFConfig(num_obs=args.num_obs, n_colloc=args.n_colloc, seed=args.seed)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    data = simulate_truth(cfg)
    np.savez(out / "data.npz", times=np.asarray(data["times"]), path=np.asarray(data["path"]),
             obs_times=np.asarray(data["obs_times"]), y1=np.asarray(data["y1"]), y2=np.asarray(data["y2"]))

    summ_path = out / "summary.json"
    runs = json.loads(summ_path.read_text())["runs"] if summ_path.exists() else {}
    for variant_name in args.variants:
        res = run_one(cfg, data, variant_name, args)
        label = f"{args.method}_{variant_name}"
        np.savez(out / f"{label}_posterior.npz", theta_samples=res["theta_samples"],
                 w_samples=res["w_samples"], x0_samples=res["x0_samples"])
        with (out / f"{label}_history.json").open("w") as f:
            json.dump(res["history"], f)
        runs[label] = res["summary"]
        s = res["summary"]
        print(f"\n--- {label} (runtime {s['runtime_s']:.1f}s) ---")
        for name in PARAM_NAMES:
            pp = s["parameters"][name]
            print(f"  {name}: {pp['post_mean']:+.3f} ± {pp['post_std']:.3f}  (true {pp['true']}, z={pp['z']:.2f})")

    with summ_path.open("w") as f:
        json.dump({"config": cfg.__dict__, "runs": runs}, f, indent=2)
    print(f"\n✓ wrote {summ_path}")


if __name__ == "__main__":
    main()

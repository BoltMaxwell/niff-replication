#!/usr/bin/env python3
"""Recreate NIFF §5.1 figures from saved posteriors (NSVI and/or NPSGLD).

Discovers every ``{method}_{variant}_posterior.npz`` in ``--results-dir`` and
overlays them, so a directory holding nsvi_relaxed / nsvi_reparam /
npsgld_relaxed produces a multi-method comparison.  Outputs:

    fig2_states.png   — posterior state reconstruction x1(t), x2(t) (90% bands)
    fig3_params.png   — model-parameter posteriors (k1, k2, k3), all runs overlaid
    fig4_convergence.png — parameter running-mean vs iteration (from histories)
    fig5_initial.png  — xhat(0;w) vs auxiliary x0 posteriors
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

jax.config.update("jax_enable_x64", True)

from niff.duffing_s51 import make_relaxed_field, make_reparam_field

PARAM_NAMES = ("k1", "k2", "k3")
TRUTH = {"k1": 0.3, "k2": -1.0, "k3": 1.0}
TRUTH0 = [1.0, 0.0]
# stable colour per label
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


def discover_runs(results_dir: Path) -> list[str]:
    labels = sorted(p.name[: -len("_posterior.npz")] for p in results_dir.glob("*_posterior.npz"))
    return labels


def variant_of(label: str) -> str:
    return "reparam" if label.endswith("reparam") else "relaxed"


def field_fns(variant: str, final_time: float, order: int):
    return make_relaxed_field(final_time, order) if variant == "relaxed" else make_reparam_field(final_time, order)


def eval_state_bands(label, post, cfg, t_grid, n_use=600):
    order = int(cfg["fourier_order"]); final_time = float(cfg["final_time"])
    L = int(post["coeff_len"]); variant = variant_of(label)
    x_of_t, _, _ = field_fns(variant, final_time, order)
    w = post["w_samples"]
    if w.shape[0] > n_use:
        w = w[:: max(1, w.shape[0] // n_use)]
    c_x, c_v = jnp.asarray(w[:, :L]), jnp.asarray(w[:, L:])
    tg = jnp.asarray(t_grid)
    X = np.asarray(jax.vmap(lambda c: jax.vmap(lambda t: x_of_t(t, c))(tg))(c_x))
    V = np.asarray(jax.vmap(lambda c: jax.vmap(lambda t: x_of_t(t, c))(tg))(c_v))
    q = lambda A, p: np.quantile(A, p, axis=0)
    return q(X, 0.5), q(X, 0.05), q(X, 0.95), q(V, 0.5), q(V, 0.05), q(V, 0.95)


def fig2_states(rd, data, cfg, labels, colors):
    t = data["times"]; t_grid = np.linspace(0.0, float(cfg["final_time"]), 1000)
    n = len(labels)
    fig, axes = plt.subplots(2, n, figsize=(6.5 * n, 7), squeeze=False)
    for j, label in enumerate(labels):
        post = dict(np.load(rd / f"{label}_posterior.npz"))
        xm, xlo, xhi, vm, vlo, vhi = eval_state_bands(label, post, cfg, t_grid)
        c = colors[label]
        ax = axes[0][j]
        ax.fill_between(t_grid, xlo, xhi, color=c, alpha=0.3, label="90% band")
        ax.plot(t_grid, xm, color=c, lw=1.4, label="post. median")
        ax.plot(t, data["path_x"], "k-", lw=1.0, alpha=0.7, label="truth")
        ax.plot(data["obs_times"], data["obs_x"], ".", ms=2, color="0.4", alpha=0.4, label="obs")
        ax.set_title(f"{label}: position x1(t)"); ax.set_xlabel("t"); ax.legend(fontsize=8, frameon=False)
        ax = axes[1][j]
        ax.fill_between(t_grid, vlo, vhi, color=c, alpha=0.3)
        ax.plot(t_grid, vm, color=c, lw=1.4)
        ax.plot(t, data["path_v"], "k-", lw=1.0, alpha=0.7)
        ax.set_title(f"{label}: velocity x2(t) (unobserved)"); ax.set_xlabel("t")
    fig.suptitle("NIFF §5.1 — posterior state reconstruction (cf. paper Fig 2)", fontsize=13)
    fig.tight_layout(); fig.savefig(rd / "fig2_states.png", dpi=160); plt.close(fig)


def fig3_params(rd, labels, colors):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, name in enumerate(PARAM_NAMES):
        ax = axes[i]
        for label in labels:
            col = dict(np.load(rd / f"{label}_posterior.npz"))["theta_samples"][:, i]
            ax.hist(col, bins=45, density=True, alpha=0.4, color=colors[label], label=label)
        ax.axvline(TRUTH[name], color="k", ls="--", lw=1.2, label="truth")
        ax.set_title(name); ax.set_xlabel(name)
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("NIFF §5.1 — model-parameter posteriors (cf. paper Fig 3)", fontsize=13)
    fig.tight_layout(); fig.savefig(rd / "fig3_params.png", dpi=160); plt.close(fig)


def fig4_convergence(rd, labels, colors):
    """Parameter running-mean (from history['mean']) vs logged-iteration index."""
    have_hist = [l for l in labels if (rd / f"{l}_history.json").exists()]
    if not have_hist:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for label in have_hist:
        hist = json.loads((rd / f"{label}_history.json").read_text())
        means = np.asarray(hist.get("mean", []))
        if means.ndim != 2 or means.shape[0] < 2:
            continue
        L = int(dict(np.load(rd / f"{label}_posterior.npz"))["coeff_len"]); d_w = 2 * L
        theta_means = means[:, d_w:d_w + 3]  # (n_log, 3)
        xs = np.arange(theta_means.shape[0])
        for i in range(3):
            axes[i].plot(xs, theta_means[:, i], color=colors[label], lw=1.3, label=label)
    for i, name in enumerate(PARAM_NAMES):
        axes[i].axhline(TRUTH[name], color="k", ls="--", lw=1.0)
        axes[i].set_title(name); axes[i].set_xlabel("logged step")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("NIFF §5.1 — parameter convergence (cf. paper Fig 4)", fontsize=13)
    fig.tight_layout(); fig.savefig(rd / "fig4_convergence.png", dpi=160); plt.close(fig)


def fig5_initial(rd, labels, colors):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    titles = ["position xhat1(0)", "velocity xhat2(0)"]
    for k, ax in enumerate(axes):
        for label in labels:
            post = dict(np.load(rd / f"{label}_posterior.npz"))
            ax.hist(post["xhat0_samples"][:, k], bins=45, density=True, alpha=0.35,
                    color=colors[label], label=f"{label}: xhat(0;w)")
            x0s = post["x0_samples"]
            if x0s.shape[1] == 2:  # relaxed variants carry an auxiliary x0
                ax.hist(x0s[:, k], bins=45, density=True, histtype="step", lw=1.6,
                        color=colors[label], label=f"{label}: aux x0")
        ax.axvline(TRUTH0[k], color="k", ls="--", lw=1.0, label="truth")
        ax.set_title(titles[k]); ax.legend(fontsize=7, frameon=False)
    fig.suptitle("NIFF §5.1 — state-path initial value vs auxiliary x0 (cf. paper Fig 5)", fontsize=13)
    fig.tight_layout(); fig.savefig(rd / "fig5_initial.png", dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True)
    args = p.parse_args()
    rd = args.results_dir
    cfg = json.loads((rd / "summary.json").read_text())["config"]
    labels = discover_runs(rd)
    if not labels:
        raise SystemExit(f"no *_posterior.npz found in {rd}")
    colors = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}
    data = {k: np.asarray(v) for k, v in np.load(rd / "data.npz").items()}
    print(f"runs: {labels}")

    fig2_states(rd, data, cfg, labels, colors)
    fig3_params(rd, labels, colors)
    fig4_convergence(rd, labels, colors)
    fig5_initial(rd, labels, colors)
    print(f"✓ wrote fig2/fig3/fig4/fig5 to {rd}")


if __name__ == "__main__":
    main()

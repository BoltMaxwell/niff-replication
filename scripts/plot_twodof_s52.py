#!/usr/bin/env python3
"""Recreate NIFF §5.2 figures from saved posteriors.

Reads ``{method}_{rbf_only,rbf_nn}_posterior.npz`` + ``data.npz`` + summary.json
from ``--results-dir`` and produces:
    fig7_params.png  — 8 model-parameter posteriors, with vs without the residual NN
    fig6_states.png  — q1(t), q2(t) reconstruction (90% bands), with vs without residual
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

from niff.twodof_s52 import TwoDOFConfig, make_state_path, PARAM_NAMES, TRUTH

COLORS = {"rbf_only": "#E45756", "rbf_nn": "#54A24B"}  # red = w/o residual, green = w/ residual


def _labels(rd):
    return sorted(p.name[: -len("_posterior.npz")] for p in rd.glob("*_posterior.npz"))


def variant_of(label):
    return "rbf_nn" if label.endswith("rbf_nn") else "rbf_only"


def fig7_params(rd, labels):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for i, (name, ax) in enumerate(zip(PARAM_NAMES, axes.ravel())):
        for label in labels:
            col = dict(np.load(rd / f"{label}_posterior.npz"))["theta_samples"][:, i]
            ax.hist(col, bins=40, density=True, alpha=0.5, color=COLORS[variant_of(label)], label=label)
        ax.axvline(TRUTH[i], color="k", ls="--", lw=1.2)
        ax.set_title(name)
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("NIFF §5.2 — model-parameter posteriors, with vs without residual NN (cf. Fig 7)", fontsize=13)
    fig.tight_layout(); fig.savefig(rd / "fig7_params.png", dpi=150); plt.close(fig)


def fig6_states(rd, cfg, data, labels, n_use=400):
    T = float(cfg["final_time"])
    t_grid = np.linspace(0.0, T, 800)
    tg = jnp.asarray(t_grid)
    truth = data["path"]; ts = data["times"]
    # component index -> (row, label)
    comps = [(0, "q1 = x1"), (2, "q2 = x3")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    cfg_obj = TwoDOFConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()})
    for label in labels:
        variant = variant_of(label)
        state_path, _, _, _ = make_state_path(cfg_obj, residual=(variant == "rbf_nn"))
        w = dict(np.load(rd / f"{label}_posterior.npz"))["w_samples"]
        if w.shape[0] > n_use:
            w = w[:: max(1, w.shape[0] // n_use)]
        X = np.asarray(jax.vmap(lambda wi: jax.vmap(lambda t: state_path(t, wi))(tg))(jnp.asarray(w)))  # (S,T,4)
        for ax, (ci, _) in zip(axes, comps):
            lo, mid, hi = np.quantile(X[:, :, ci], [0.05, 0.5, 0.95], axis=0)
            ax.fill_between(t_grid, lo, hi, color=COLORS[variant], alpha=0.25)
            ax.plot(t_grid, mid, color=COLORS[variant], lw=1.3, label=label)
    for ax, (ci, name) in zip(axes, comps):
        ax.plot(ts, truth[:, ci], "k-", lw=1.0, alpha=0.8, label="truth")
        ax.set_title(f"state {name}"); ax.set_xlabel("t"); ax.legend(fontsize=8, frameon=False)
    fig.suptitle("NIFF §5.2 — state reconstruction, with vs without residual NN (cf. Fig 6)", fontsize=13)
    fig.tight_layout(); fig.savefig(rd / "fig6_states.png", dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True)
    args = p.parse_args()
    rd = args.results_dir
    cfg = json.loads((rd / "summary.json").read_text())["config"]
    labels = _labels(rd)
    if not labels:
        raise SystemExit(f"no *_posterior.npz in {rd}")
    data = {k: np.asarray(v) for k, v in np.load(rd / "data.npz").items()}
    print(f"runs: {labels}")
    fig7_params(rd, labels)
    fig6_states(rd, cfg, data, labels)
    print(f"✓ wrote fig6_states.png, fig7_params.png to {rd}")


if __name__ == "__main__":
    main()

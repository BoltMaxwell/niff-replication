#!/usr/bin/env python3
"""NIFF paper Section 5.2 — two-DOF nonlinear system with a residual neural network.

Hao & Bilionis, *Neural Information Field Filter* (MSSP 2025), §5.2.  This is the
example that exercises NIFF's namesake device: the state path is a **linear RBF
basis plus a Fourier-encoded residual neural network** (paper eq. 8).  The RBF
basis alone cannot represent the true state path; the residual NN corrects it.

System (2-DOF, from ref [84]; q1 = y1, q2 = y2 - y1):

    m1 q1'' + c1 q1' - c2 q2' + k1 q1 - k2 q2 + k1 e1 q1^3 - c2 e2 q2'^3 = F sin(w0 t)
    m2 q1'' + m2 q2'' + c2 q2' + k2 q2 + c2 e2 q2'^3                     = 0

State x = (q1, q1', q2, q2'); measurements y1 = q1, y2 = q1 + q2.
Truth: m1=m2=1, c1=c2=0.2, k1=k2=1, e1=e2=0.2 (8 params).  IC = (0,0,0.5,0).

**Divergence from the paper (documented):** F and w0 are not stated in the paper
(they come from ref [84]); we use F=2.0, w0=1.2, chosen so the response is rich
enough that a 20-term RBF basis under-fits it (the premise of the example).  The
Fourier-encoding period Tbar=10 is likewise a chosen value.

State-path parameterization (per Table 3 of the paper, w-dim = 344):
  * per-component RBF: 4 x Kb=20 = 80 coefficients;
  * a single shared Fourier-encoded MLP (encode t -> 2K+1=21 dims, K=10;
    one hidden layer width 10, swish; **4 outputs** — one residual per state
    component) = 264 parameters.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import numpy as np
import optax

from niff.utils import collocation_grid

jax.config.update("jax_enable_x64", True)

Array = jax.Array
LOG_2PI = math.log(2.0 * math.pi)
PARAM_NAMES = ("m1", "m2", "c1", "c2", "k1", "k2", "e1", "e2")
TRUTH = (1.0, 1.0, 0.2, 0.2, 1.0, 1.0, 0.2, 0.2)


@dataclass(frozen=True)
class TwoDOFConfig:
    # known forcing (unspecified in the paper; see module docstring)
    F: float = 2.0
    omega0: float = 1.2
    # initial condition
    x0: tuple[float, float, float, float] = (0.0, 0.0, 0.5, 0.0)
    # data
    final_time: float = 50.0
    rk_dt: float = 0.1
    sigma_y: tuple[float, float] = (0.05, 0.10)   # 5% of ybar = (1, 2)
    num_obs: int = 500
    # RBF linear basis
    n_rbf: int = 20
    rbf_sigma: float = 0.05
    # Fourier-encoded residual NN
    n_fourier: int = 10
    fourier_period: float = 10.0
    nn_hidden: int = 10
    # collocation
    n_colloc: int = 400
    seed: int = 0


# ======================================================================
# True dynamics (state-space) + RK4 data generation
# ======================================================================
def _rhs(x: Array, t: Array, cfg: TwoDOFConfig, p) -> Array:
    m1, m2, c1, c2, k1, k2, e1, e2 = p
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    b1 = cfg.F * jnp.sin(cfg.omega0 * t) - c1 * x2 + c2 * x4 - k1 * x1 + k2 * x3 - k1 * e1 * x1**3 + c2 * e2 * x4**3
    b2 = -(c2 * x4 + k2 * x3 + c2 * e2 * x4**3)
    q1ddot = b1 / m1
    q2ddot = b2 / m2 - q1ddot
    return jnp.stack([x2, q1ddot, x4, q2ddot])


def simulate_truth(cfg: TwoDOFConfig) -> dict[str, Array]:
    n_steps = int(round(cfg.final_time / cfg.rk_dt))
    ts = jnp.arange(n_steps + 1, dtype=jnp.float64) * cfg.rk_dt
    h = cfg.rk_dt
    p = jnp.asarray(TRUTH, dtype=jnp.float64)

    def step(x, t):
        k1_ = _rhs(x, t, cfg, p)
        k2_ = _rhs(x + 0.5 * h * k1_, t + 0.5 * h, cfg, p)
        k3_ = _rhs(x + 0.5 * h * k2_, t + 0.5 * h, cfg, p)
        k4_ = _rhs(x + h * k3_, t + h, cfg, p)
        nxt = x + (h / 6.0) * (k1_ + 2 * k2_ + 2 * k3_ + k4_)
        return nxt, nxt

    x0 = jnp.asarray(cfg.x0, dtype=jnp.float64)
    _, tail = lax.scan(step, x0, ts[:-1])
    path = jnp.concatenate([x0[None, :], tail], axis=0)  # (n+1, 4)
    if not np.isfinite(np.asarray(path)).all():
        raise ValueError("2-DOF RK4 produced non-finite states; check forcing/params.")

    obs_idx = jnp.linspace(0, n_steps, cfg.num_obs, dtype=jnp.int32)
    obs_times = ts[obs_idx]
    q1 = path[obs_idx, 0]
    q1q2 = path[obs_idx, 0] + path[obs_idx, 2]  # y2 = q1 + q2
    key = jr.PRNGKey(cfg.seed)
    ky1, ky2 = jr.split(key)
    y1 = q1 + cfg.sigma_y[0] * jr.normal(ky1, q1.shape, dtype=jnp.float64)
    y2 = q1q2 + cfg.sigma_y[1] * jr.normal(ky2, q1q2.shape, dtype=jnp.float64)
    return {
        "times": ts, "path": path, "obs_times": obs_times,
        "y1": y1, "y2": y2, "y1_clean": q1, "y2_clean": q1q2,
    }


# ======================================================================
# State path: per-component RBF + shared Fourier-encoded residual MLP
# ======================================================================
def make_state_path(cfg: TwoDOFConfig, residual: bool):
    """Return (state_path, dstate_dt, n_rbf_total, n_nn, split) where
    ``state_path(t, w) -> (4,)`` and ``w = [rbf (4*n_rbf), nn (n_nn if residual)]``.
    """
    T = cfg.final_time
    Kb, sig = cfg.n_rbf, cfg.rbf_sigma
    Kf, Tbar, hid = cfg.n_fourier, cfg.fourier_period, cfg.nn_hidden
    centers = jnp.linspace(0.0, 1.0, Kb)
    d_in = 2 * Kf + 1
    n_rbf_total = 4 * Kb
    n_nn = hid * d_in + hid + 4 * hid + 4  # W1,b1,W2,b2

    def rbf(t01: Array) -> Array:
        return jnp.exp(-(t01 - centers) ** 2 / (2.0 * sig**2))  # (Kb,)

    def fourier_encode(t01: Array) -> Array:
        ks = jnp.arange(1, Kf + 1)
        a = 2.0 * jnp.pi * ks * t01 / Tbar
        return jnp.concatenate([jnp.ones((1,)), jnp.sin(a), jnp.cos(a)])  # (2Kf+1,)

    def nn(t01: Array, p: Array) -> Array:
        i = 0
        W1 = p[i:i + hid * d_in].reshape(hid, d_in); i += hid * d_in
        b1 = p[i:i + hid]; i += hid
        W2 = p[i:i + 4 * hid].reshape(4, hid); i += 4 * hid
        b2 = p[i:i + 4]
        h = jax.nn.swish(W1 @ fourier_encode(t01) + b1)
        return W2 @ h + b2  # (4,)

    def split(w: Array) -> tuple[Array, Array]:
        w_rbf = w[:n_rbf_total].reshape(4, Kb)
        w_nn = w[n_rbf_total:]
        return w_rbf, w_nn

    def state_path(t: Array, w: Array) -> Array:
        t01 = t / T
        w_rbf, w_nn = split(w)
        lin = w_rbf @ rbf(t01)  # (4,)
        return lin + nn(t01, w_nn) if residual else lin

    dstate_dt = jax.jacfwd(lambda t, w: state_path(t, w), argnums=0)  # (4,) d/dt

    d_w = n_rbf_total + (n_nn if residual else 0)
    return state_path, dstate_dt, d_w, split


# ======================================================================
# Variant builder: returns NSVI/NPSGLD callbacks + dims + warmup init
# ======================================================================
@dataclass
class Variant:
    name: str
    d_w: int
    d_theta: int
    d_x0: int
    log_likelihood_fn: callable
    energy_fn: callable
    x_at_0_w: callable
    state_path: callable
    init_mean: tuple
    residual: bool


def _warmup_w(state_path, dstate_dt, d_w, cfg, data, *, iters=6000, lr=5e-3):
    """Seed the state-path w so all four components start kinematically consistent:
    fit the displacement fields to the measurements (x1->y1, x1+x3->y2) and the
    velocity fields to the derivatives of the displacement fields (x2->dx1/dt,
    x4->dx3/dt). Without the velocity seeding, x2/x4 start unconstrained and the
    inference must align them from scratch."""
    obs_times, y1, y2 = data["obs_times"], data["y1"], data["y2"]
    colloc = cfg.final_time * collocation_grid(min(cfg.n_colloc, 300))

    def loss(w):
        preds = jax.vmap(lambda t: state_path(t, w))(obs_times)
        r_disp = jnp.mean((preds[:, 0] - y1) ** 2) + jnp.mean((preds[:, 0] + preds[:, 2] - y2) ** 2)
        xc = jax.vmap(lambda t: state_path(t, w))(colloc)
        xdc = jax.vmap(lambda t: dstate_dt(t, w))(colloc)
        r_vel = jnp.mean((xc[:, 1] - xdc[:, 0]) ** 2) + jnp.mean((xc[:, 3] - xdc[:, 2]) ** 2)
        return r_disp + 0.3 * r_vel

    w = 0.01 * jr.normal(jr.PRNGKey(cfg.seed + 5), (d_w,), dtype=jnp.float64)
    sched = optax.cosine_decay_schedule(lr, iters)
    opt = optax.adam(sched)
    state = opt.init(w)

    @jax.jit
    def step(w, state):
        g = jax.grad(loss)(w)
        upd, state = opt.update(g, state, w)
        return optax.apply_updates(w, upd), state

    for _ in range(iters):
        w, state = step(w, state)
    return w


def build_variant(cfg: TwoDOFConfig, data: dict, *, residual: bool,
                  beta1: float = 200.0, beta2: float = 1.0e5, warmup: bool = True,
                  warmup_iters: int = 6000) -> Variant:
    state_path, dstate_dt, d_w, split = make_state_path(cfg, residual)
    d_theta = 8
    d_x0 = 4
    obs_times, y1, y2 = data["obs_times"], data["y1"], data["y2"]
    sy1, sy2 = cfg.sigma_y
    colloc_t = cfg.final_time * collocation_grid(cfg.n_colloc)
    T = cfg.final_time

    def log_likelihood_fn(w: Array, theta: Array) -> Array:
        preds = jax.vmap(lambda t: state_path(t, w))(obs_times)  # (N,4)
        r1 = (preds[:, 0] - y1) / sy1
        r2 = (preds[:, 0] + preds[:, 2] - y2) / sy2
        ll1 = jnp.sum(-0.5 * (r1**2 + LOG_2PI + 2.0 * math.log(sy1)))
        ll2 = jnp.sum(-0.5 * (r2**2 + LOG_2PI + 2.0 * math.log(sy2)))
        return ll1 + ll2

    def physics_energy(w: Array, theta: Array) -> Array:
        m1, m2, c1, c2, k1, k2, e1, e2 = [theta[i] for i in range(8)]
        x = jax.vmap(lambda t: state_path(t, w))(colloc_t)   # (M,4)
        xd = jax.vmap(lambda t: dstate_dt(t, w))(colloc_t)   # (M,4) d/dt
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        # mass-matrix residual form (no division by inferred masses)
        r1 = xd[:, 0] - x2
        r2 = m1 * xd[:, 1] + c1 * x2 - c2 * x4 + k1 * x1 - k2 * x3 + k1 * e1 * x1**3 \
            - c2 * e2 * x4**3 - cfg.F * jnp.sin(cfg.omega0 * colloc_t)
        r3 = xd[:, 2] - x4
        r4 = m2 * xd[:, 1] + m2 * xd[:, 3] + c2 * x4 + k2 * x3 + c2 * e2 * x4**3
        return T * jnp.mean(r1**2 + r2**2 + r3**2 + r4**2)

    def x_at_0_w(w: Array) -> Array:
        return state_path(jnp.asarray(0.0, dtype=jnp.float64), w)  # (4,)

    def energy_fn(w: Array, theta: Array, x0: Array) -> Array:
        kernel = jnp.sum((x_at_0_w(w) - x0) ** 2)
        return beta1 * physics_energy(w, theta) + beta2 * kernel

    if warmup:
        w_init = _warmup_w(state_path, dstate_dt, d_w, cfg, data, iters=warmup_iters)
    else:
        w_init = jnp.zeros((d_w,), dtype=jnp.float64)
    theta_init = jnp.zeros((d_theta,), dtype=jnp.float64)
    x0_init = jnp.asarray(cfg.x0, dtype=jnp.float64)
    init_mean = tuple(np.asarray(jnp.concatenate([w_init, theta_init, x0_init])).tolist())

    return Variant(
        name="rbf_nn" if residual else "rbf_only",
        d_w=d_w, d_theta=d_theta, d_x0=d_x0,
        log_likelihood_fn=log_likelihood_fn, energy_fn=energy_fn,
        x_at_0_w=x_at_0_w, state_path=state_path, init_mean=init_mean, residual=residual,
    )

#!/usr/bin/env python3
"""NIFF paper Section 5.1 — single-DOF Duffing oscillator (deterministic ODE).

Faithful replication target for Hao & Bilionis, *Neural Information Field Filter*
(MSSP 2025), `wiki/raw/niff.pdf` §5.1.  See `README.md` in this directory.

Dynamics (physical coordinates):

    x1'(t) = x2(t)
    x2'(t) = -k1 x2 - k2 x1 - k3 x1^3 + gamma cos(omega t)
    Y(t)   = x1(t) + sigma_y V(t)                       (position-only obs)

Truth: k1=0.3, k2=-1, k3=1, gamma=0.37, omega=1.2, IC=(1,0), sigma_y=0.075.

The state path is a truncated Fourier series with K=40 modes per component, built
from `niff.fields.fourier_basis_01`.  Inference uses the NSVI engine
(`niff.nsvi.run_nsvi`) or the NPSGLD sampler (`niff.npsgld.run_npsgld`).  Two variants:

  * "reparam"  — IC folded into w (x_hat(0)=ic exact);  d_x0=0;  beta = 200.
  * "relaxed"  — free field + auxiliary x0 coupled via kernel; d_x0=2;
                 beta1 = 200, beta2 = 1e5.
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

from niff.fields import fourier_basis_01  # half-period Fourier basis
from niff.utils import collocation_grid

jax.config.update("jax_enable_x64", True)

Array = jax.Array
LOG_2PI = math.log(2.0 * math.pi)


# ======================================================================
# Problem configuration (paper §5.1 defaults)
# ======================================================================
@dataclass(frozen=True)
class DuffingS51Config:
    # true model parameters (the inference targets are k1, k2, k3)
    k1: float = 0.3
    k2: float = -1.0
    k3: float = 1.0
    # known forcing
    gamma: float = 0.37
    omega: float = 1.2
    # initial condition
    x0_pos: float = 1.0
    x0_vel: float = 0.0
    # data generation
    final_time: float = 50.0
    rk_dt: float = 0.01
    sigma_y: float = 0.075          # 5% of ybar = 1.5
    num_obs: int = 500              # downsample of the 5001-pt grid for the likelihood
    # field
    fourier_order: int = 40         # K
    # collocation for the physics-energy integral
    n_colloc: int = 200
    seed: int = 0


# ======================================================================
# Forward simulation (RK4 fixed step, dt = 0.01)
# ======================================================================
def duffing_rhs(state: Array, t: Array, cfg: DuffingS51Config) -> Array:
    x, v = state[0], state[1]
    dx = v
    dv = -cfg.k1 * v - cfg.k2 * x - cfg.k3 * x**3 + cfg.gamma * jnp.cos(cfg.omega * t)
    return jnp.stack([dx, dv])


def simulate_truth(cfg: DuffingS51Config) -> dict[str, Array]:
    """Integrate the true Duffing ODE with fixed-step RK4; sample noisy obs."""
    n_steps = int(round(cfg.final_time / cfg.rk_dt))
    ts = jnp.arange(n_steps + 1, dtype=jnp.float64) * cfg.rk_dt
    h = cfg.rk_dt

    def step(state, t):
        k1_ = duffing_rhs(state, t, cfg)
        k2_ = duffing_rhs(state + 0.5 * h * k1_, t + 0.5 * h, cfg)
        k3_ = duffing_rhs(state + 0.5 * h * k2_, t + 0.5 * h, cfg)
        k4_ = duffing_rhs(state + h * k3_, t + h, cfg)
        nxt = state + (h / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)
        return nxt, nxt

    x0 = jnp.array([cfg.x0_pos, cfg.x0_vel], dtype=jnp.float64)
    _, tail = lax.scan(step, x0, ts[:-1])
    path = jnp.concatenate([x0[None, :], tail], axis=0)  # (n_steps+1, 2)

    if not np.isfinite(np.asarray(path)).all():
        raise ValueError("Duffing RK4 produced non-finite states; check params/dt.")

    obs_idx = jnp.linspace(0, n_steps, cfg.num_obs, dtype=jnp.int32)
    obs_times = ts[obs_idx]
    obs_clean = path[obs_idx, 0]
    key_obs = jr.PRNGKey(cfg.seed)
    obs_x = obs_clean + cfg.sigma_y * jr.normal(key_obs, obs_clean.shape, dtype=jnp.float64)

    return {
        "times": ts,
        "path_x": path[:, 0],
        "path_v": path[:, 1],
        "obs_times": obs_times,
        "obs_x": obs_x,
        "obs_clean": obs_clean,
    }


# ======================================================================
# Field parameterizations (per scalar component)
# ======================================================================
def make_relaxed_field(final_time: float, order: int):
    """Free Fourier field:  xhat(t) = basis(t/T) @ c.   coeff length L = 1+2K."""
    def x_of_t(t: Array, c: Array) -> Array:
        return fourier_basis_01(t / final_time, order) @ c

    dx_dt = jax.grad(lambda t, c: x_of_t(t, c), argnums=0)

    def x_at_0(c: Array) -> Array:
        return fourier_basis_01(jnp.asarray(0.0, dtype=jnp.float64), order) @ c

    return x_of_t, dx_dt, x_at_0


def make_reparam_field(final_time: float, order: int):
    """IC-folded field:  xhat(t) = c[0] + (t/T)*(basis(t/T) @ c[1:]).

    xhat(0) = c[0] exactly (Lagaris-style).  coeff length L = 2+2K.
    """
    def x_of_t(t: Array, c: Array) -> Array:
        t01 = t / final_time
        return c[0] + t01 * (fourier_basis_01(t01, order) @ c[1:])

    dx_dt = jax.grad(lambda t, c: x_of_t(t, c), argnums=0)

    def x_at_0(c: Array) -> Array:
        return c[0]

    return x_of_t, dx_dt, x_at_0


# ======================================================================
# Variant builder: returns NSVI callbacks + dims + init
# ======================================================================
@dataclass
class Variant:
    name: str
    d_w: int
    d_theta: int
    d_x0: int
    coeff_len: int           # per-component coefficient length
    log_likelihood_fn: callable
    energy_fn: callable
    x_at_0_w: callable       # (w) -> (xhat0_pos, xhat0_vel) for Fig 5
    init_mean: tuple
    beta1: float
    beta2: float


def _warmup_field_coeffs(x_of_t, coeff_len, obs_times, obs_x, *, iters=1500, lr=2e-2, seed=0):
    """Least-squares fit of a single field's coefficients to position obs."""
    def loss(c):
        preds = jax.vmap(lambda t: x_of_t(t, c))(obs_times)
        return jnp.mean((preds - obs_x) ** 2)

    opt = optax.adam(lr)
    c = jnp.zeros((coeff_len,), dtype=jnp.float64)
    state = opt.init(c)

    @jax.jit
    def step(c, state):
        g = jax.grad(loss)(c)
        upd, state = opt.update(g, state, c)
        return optax.apply_updates(c, upd), state

    for _ in range(iters):
        c, state = step(c, state)
    return c


def _warmup_velocity_coeffs(v_of_t, coeff_len, grid_t, target_v, *, iters=1500, lr=2e-2):
    """Fit a velocity field's coefficients to a target velocity on a grid."""
    def loss(c):
        preds = jax.vmap(lambda t: v_of_t(t, c))(grid_t)
        return jnp.mean((preds - target_v) ** 2)

    opt = optax.adam(lr)
    c = jnp.zeros((coeff_len,), dtype=jnp.float64)
    state = opt.init(c)

    @jax.jit
    def step(c, state):
        g = jax.grad(loss)(c)
        upd, state = opt.update(g, state, c)
        return optax.apply_updates(c, upd), state

    for _ in range(iters):
        c, state = step(c, state)
    return c


def build_variant(
    cfg: DuffingS51Config,
    data: dict[str, Array],
    *,
    variant: str,
    beta1: float = 200.0,
    beta2: float = 1.0e5,
    warmup: bool = True,
    subsample: bool = False,
    n_t: int = 10,
    m_y: int = 10,
) -> Variant:
    """Construct NSVI/NPSGLD callbacks for the 'reparam' or 'relaxed' variant.

    With ``subsample=False`` the callbacks are full-batch and use the original
    ``(w, theta)`` / ``(w, theta, x0)`` signatures.  With ``subsample=True`` the
    callbacks take an extra per-iteration PRNG ``data_key`` and form unbiased
    minibatch estimates: the physics integral from ``n_t`` random collocation
    times in [0,T], and the likelihood from an ``m_y``-sized measurement minibatch
    (scaled by num_obs/m_y).  This matches the paper's IFT subsampling and pairs
    with ``NSVIConfig(stochastic_callbacks=True)``.
    """
    order = cfg.fourier_order
    final_time = cfg.final_time
    obs_times = data["obs_times"]
    obs_x = data["obs_x"]
    n_obs = int(obs_times.shape[0])
    colloc_t = final_time * collocation_grid(cfg.n_colloc)

    if variant == "relaxed":
        x_of_t, dx_dt, x_at_0 = make_relaxed_field(final_time, order)
        v_of_t, dv_dt, v_at_0 = make_relaxed_field(final_time, order)
        L = 1 + 2 * order
        d_x0 = 2
    elif variant == "reparam":
        x_of_t, dx_dt, x_at_0 = make_reparam_field(final_time, order)
        v_of_t, dv_dt, v_at_0 = make_reparam_field(final_time, order)
        L = 2 + 2 * order
        d_x0 = 0
    else:
        raise ValueError(f"unknown variant {variant!r}")

    d_w = 2 * L
    d_theta = 3  # (k1, k2, k3)
    log_sigma_y = math.log(cfg.sigma_y)

    def split_w(w: Array) -> tuple[Array, Array]:
        return w[:L], w[L:]

    # ---- likelihood: per-point Gaussian log-density on position ----
    def _gauss_logpdf(preds: Array, y: Array) -> Array:
        resid = (preds - y) / cfg.sigma_y
        return -0.5 * (resid**2 + LOG_2PI + 2.0 * log_sigma_y)

    def loglik_full(w: Array, theta: Array) -> Array:
        c_x, _ = split_w(w)
        preds = jax.vmap(lambda t: x_of_t(t, c_x))(obs_times)
        return jnp.sum(_gauss_logpdf(preds, obs_x))

    def loglik_minibatch(w: Array, theta: Array, data_key: Array) -> Array:
        _, obs_key = jax.random.split(data_key)
        idx = jax.random.randint(obs_key, (m_y,), 0, n_obs)
        c_x, _ = split_w(w)
        preds = jax.vmap(lambda t: x_of_t(t, c_x))(obs_times[idx])
        # unbiased estimate of the full sum: (N / m_y) * sum_minibatch
        return (n_obs / m_y) * jnp.sum(_gauss_logpdf(preds, obs_x[idx]))

    # ---- physics residual^2 per collocation point ----
    def residual_sq_at(w: Array, theta: Array, pts: Array) -> Array:
        c_x, c_v = split_w(w)
        k1, k2, k3 = theta[0], theta[1], theta[2]
        x_vals = jax.vmap(lambda t: x_of_t(t, c_x))(pts)
        v_vals = jax.vmap(lambda t: v_of_t(t, c_v))(pts)
        dx_vals = jax.vmap(lambda t: dx_dt(t, c_x))(pts)
        dv_vals = jax.vmap(lambda t: dv_dt(t, c_v))(pts)
        r_x = dx_vals - v_vals
        drift_v = -k1 * v_vals - k2 * x_vals - k3 * x_vals**3 + cfg.gamma * jnp.cos(cfg.omega * pts)
        r_v = dv_vals - drift_v
        return r_x**2 + r_v**2

    # H = integral_0^T ||residual||^2 dt ~ T * mean over collocation points.
    # Fixed grid (deterministic) or n_t uniform random points (unbiased MC).
    def physics_energy_grid(w: Array, theta: Array) -> Array:
        return final_time * jnp.mean(residual_sq_at(w, theta, colloc_t))

    def physics_energy_random(w: Array, theta: Array, data_key: Array) -> Array:
        colloc_key, _ = jax.random.split(data_key)
        pts = jax.random.uniform(colloc_key, (n_t,), minval=0.0, maxval=final_time, dtype=jnp.float64)
        return final_time * jnp.mean(residual_sq_at(w, theta, pts))

    def _kernel_or_ic(w: Array, x0: Array) -> Array:
        c_x, c_v = split_w(w)
        if variant == "relaxed":  # soft IC coupling to the auxiliary x0
            return beta2 * ((x_at_0(c_x) - x0[0]) ** 2 + (v_at_0(c_v) - x0[1]) ** 2)
        return 0.5 * (x_at_0(c_x) ** 2 + v_at_0(c_v) ** 2)  # reparam: N(0,1) on folded IC

    if subsample:
        def log_likelihood_fn(w: Array, theta: Array, data_key: Array) -> Array:
            return loglik_minibatch(w, theta, data_key)

        def energy_fn(w: Array, theta: Array, x0: Array, data_key: Array) -> Array:
            return beta1 * physics_energy_random(w, theta, data_key) + _kernel_or_ic(w, x0)
    else:
        def log_likelihood_fn(w: Array, theta: Array) -> Array:
            return loglik_full(w, theta)

        def energy_fn(w: Array, theta: Array, x0: Array) -> Array:
            return beta1 * physics_energy_grid(w, theta) + _kernel_or_ic(w, x0)

    def x_at_0_w(w: Array) -> tuple[Array, Array]:
        c_x, c_v = split_w(w)
        return x_at_0(c_x), v_at_0(c_v)

    # ---- warmup init for w; theta at prior mean (0); x0 at observed IC ----
    if warmup:
        c_x0 = _warmup_field_coeffs(x_of_t, L, obs_times, obs_x, seed=cfg.seed)
        # velocity target: derivative of the warmed position field on the colloc grid
        target_v = jax.vmap(lambda t: dx_dt(t, c_x0))(colloc_t)
        c_v0 = _warmup_velocity_coeffs(v_of_t, L, colloc_t, target_v)
        w_init = jnp.concatenate([c_x0, c_v0])
    else:
        w_init = jnp.zeros((d_w,), dtype=jnp.float64)

    theta_init = jnp.zeros((d_theta,), dtype=jnp.float64)
    if d_x0 == 2:
        x0_init = jnp.array([cfg.x0_pos, cfg.x0_vel], dtype=jnp.float64)
    else:
        x0_init = jnp.zeros((0,), dtype=jnp.float64)
    init_mean = tuple(np.asarray(jnp.concatenate([w_init, theta_init, x0_init])).tolist())

    return Variant(
        name=variant,
        d_w=d_w,
        d_theta=d_theta,
        d_x0=d_x0,
        coeff_len=L,
        log_likelihood_fn=log_likelihood_fn,
        energy_fn=energy_fn,
        x_at_0_w=x_at_0_w,
        init_mean=init_mean,
        beta1=beta1,
        beta2=beta2,
    )

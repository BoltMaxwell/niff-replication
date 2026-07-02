"""Nested stochastic variational inference for the relaxed NIFF posterior.

NSVI approximates the relaxed NIFF posterior with a variational guide and uses
a persistent inner variational guide to estimate the partition-gradient term.
The method requires two differentiable callbacks:

``log_likelihood_fn(w, theta)``
    Full-data log likelihood for path weights ``w`` and parameters ``theta``.

``energy_fn(w, theta, x0)``
    Positive NIFF relaxed-prior energy ``beta_1 H_1 + beta_2 H_2``.

The target posterior is

    p(w, theta, x0 | y) proportional to
        p(y | w, theta) exp(-energy(w, theta, x0))
        p(theta) p(x0) / Z(theta, x0),

where ``Z(theta, x0) = int exp(-energy(w, theta, x0)) dw``.  As in the paper,
the outer guide factorizes as
``q_w(w) q_theta(theta) q_x0(x0)``; the low-dimensional ``theta`` block can be
diagonal or full-rank.  The implementation follows Algorithm SVI_POSTERIOR's
single outer reparameterized sample per iteration, which keeps one persistent
conditional inner guide.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax

Array = jax.Array
LogLikelihoodFn = Callable[[Array, Array], Array]
EnergyFn = Callable[[Array, Array, Array], Array]


@dataclass(frozen=True)
class NSVIConfig:
    """Configuration for :func:`run_nsvi`.

    ``theta_guide`` can be ``"diag"`` or ``"full_rank"``.  The path guide
    ``q_w`` and auxiliary-initial-state guide ``q_x0`` remain diagonal, matching
    Algorithm SVI_POSTERIOR in the paper source and the reproduction code.
    """

    iterations: int = 4_000
    inner_iterations: int = 8
    outer_lr: float = 2.0e-3
    inner_lr: float = 3.0e-3
    n_outer_samples: int = 1
    n_aux_samples: int = 8
    theta_guide: str = "diag"
    init_mean: tuple[float, ...] | None = None
    init_std: float = 0.75
    inner_init_std: float = 0.75
    theta_prior_std: float = 1.0
    x0_prior_std: float = 1.0
    partition_weight: float = 1.0
    partition_anneal_steps: int = 0
    min_std: float = 1.0e-5
    max_log_std: float = 3.0
    grad_clip: float | None = 1.0e3
    log_every: int = 250
    quiet: bool = False
    # Opt-in: when True, the callbacks are invoked as
    # ``log_likelihood_fn(w, theta, data_key)`` and
    # ``energy_fn(w, theta, x0, data_key)`` with a fresh per-iteration PRNG key,
    # so the callback can subsample collocation points / measurements (the IFT
    # n_t / m_y minibatching).  Default False keeps the original 2-/3-arg
    # callback contract and byte-identical behavior for every existing caller.
    stochastic_callbacks: bool = False


@dataclass
class NSVIResult:
    """Result returned by :func:`run_nsvi`."""

    params: dict[str, Array]
    inner_params: dict[str, Array]
    history: dict[str, list[float | list[float]]]
    config: NSVIConfig


def _as_std_vector(std: float | Array, dim: int, dtype: jnp.dtype) -> Array:
    arr = jnp.asarray(std, dtype=dtype)
    if arr.ndim == 0:
        return jnp.full((dim,), arr, dtype=dtype)
    arr = arr.reshape((-1,))
    if arr.shape[0] != dim:
        raise ValueError(f"prior std has length {arr.shape[0]}, expected {dim}")
    return arr


def _inverse_softplus(x: Array) -> Array:
    x = jnp.asarray(x)
    return jnp.log(jnp.expm1(jnp.maximum(x, jnp.asarray(1.0e-12, dtype=x.dtype))))


def _split_z(z: Array, d_w: int, d_theta: int) -> tuple[Array, Array, Array]:
    w = z[..., :d_w]
    theta = z[..., d_w : d_w + d_theta]
    x0 = z[..., d_w + d_theta :]
    return w, theta, x0


def _diag_logpdf(x: Array, mean: Array, log_std: Array) -> Array:
    standardized = (x - mean) / jnp.exp(log_std)
    return jnp.sum(-0.5 * jnp.log(2.0 * jnp.pi) - log_std - 0.5 * standardized**2)


def _normal_logpdf(x: Array, std: Array) -> Array:
    standardized = x / std
    return jnp.sum(-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(std) - 0.5 * standardized**2)


def _scale_tril(raw_tril: Array, min_std: float) -> Array:
    lower = jnp.tril(raw_tril, k=-1)
    raw_diag = jnp.diag(raw_tril)
    diag = jax.nn.softplus(raw_diag) + min_std
    return lower + jnp.diag(diag)


def _full_rank_logpdf(x: Array, mean: Array, raw_tril: Array, min_std: float) -> Array:
    scale_tril = _scale_tril(raw_tril, min_std)
    diff = x - mean
    whitened = jsp.linalg.solve_triangular(scale_tril, diff, lower=True)
    dim = x.shape[0]
    log_det = jnp.sum(jnp.log(jnp.diag(scale_tril)))
    return -0.5 * dim * jnp.log(2.0 * jnp.pi) - log_det - 0.5 * jnp.sum(whitened**2)


def _init_outer_params(config: NSVIConfig, d_w: int, d_theta: int, d_x0: int, dtype: jnp.dtype) -> dict[str, Array]:
    d_total = d_w + d_theta + d_x0
    if config.init_mean is None:
        init_mean = jnp.zeros((d_total,), dtype=dtype)
    else:
        init_mean = jnp.asarray(config.init_mean, dtype=dtype).reshape((-1,))
        if init_mean.shape[0] != d_total:
            raise ValueError(f"init_mean has length {init_mean.shape[0]}, expected {d_total}")

    init_log_std = jnp.log(jnp.asarray(config.init_std, dtype=dtype))
    theta_guide = config.theta_guide.lower()
    w_mean, theta_mean, x0_mean = _split_z(init_mean, d_w, d_theta)
    params = {
        "w_mean": w_mean,
        "w_log_std": jnp.full((d_w,), init_log_std, dtype=dtype),
        "theta_mean": theta_mean,
        "x0_mean": x0_mean,
        "x0_log_std": jnp.full((d_x0,), init_log_std, dtype=dtype),
    }
    if theta_guide == "diag":
        params["theta_log_std"] = jnp.full((d_theta,), init_log_std, dtype=dtype)
        return params
    if theta_guide == "full_rank":
        raw_diag = _inverse_softplus(jnp.full((d_theta,), config.init_std - config.min_std, dtype=dtype))
        params["theta_tril_raw"] = jnp.diag(raw_diag)
        return params
    raise ValueError(f"Unsupported theta_guide={config.theta_guide!r}")


def _clip_params(params: dict[str, Array], config: NSVIConfig) -> dict[str, Array]:
    out = dict(params)
    for key in ("w_log_std", "theta_log_std", "x0_log_std"):
        if key in out:
            out[key] = jnp.clip(out[key], jnp.log(config.min_std), config.max_log_std)
    return out


def _sample_outer(params: Mapping[str, Array], eps: Array, config: NSVIConfig, d_w: int, d_theta: int) -> tuple[Array, Array, Array, Array]:
    w_eps, theta_eps, x0_eps = _split_z(eps, d_w, d_theta)
    w = params["w_mean"] + jnp.exp(params["w_log_std"]) * w_eps
    x0 = params["x0_mean"] + jnp.exp(params["x0_log_std"]) * x0_eps
    log_q_w = _diag_logpdf(w, params["w_mean"], params["w_log_std"])
    log_q_x0 = _diag_logpdf(x0, params["x0_mean"], params["x0_log_std"])
    if "theta_tril_raw" in params:
        scale_tril = _scale_tril(params["theta_tril_raw"], config.min_std)
        theta = params["theta_mean"] + scale_tril @ theta_eps
        log_q_theta = _full_rank_logpdf(theta, params["theta_mean"], params["theta_tril_raw"], config.min_std)
    else:
        theta = params["theta_mean"] + jnp.exp(params["theta_log_std"]) * theta_eps
        log_q_theta = _diag_logpdf(theta, params["theta_mean"], params["theta_log_std"])
    log_q = log_q_w + log_q_theta + log_q_x0
    return w, theta, x0, log_q


def _partition_weight(step: int, config: NSVIConfig) -> float:
    if config.partition_anneal_steps <= 0:
        return float(config.partition_weight)
    ratio = min(1.0, float(step + 1) / float(config.partition_anneal_steps))
    return float(config.partition_weight * ratio)


def _make_optimizer(lr: float, grad_clip: float | None) -> optax.GradientTransformation:
    if grad_clip is None:
        return optax.adam(lr)
    return optax.chain(optax.clip_by_global_norm(float(grad_clip)), optax.adam(lr))


def _log_indices(iterations: int, log_every: int) -> np.ndarray:
    return np.asarray(
        [step for step in range(iterations) if (step % log_every) == 0 or step == iterations - 1],
        dtype=np.int32,
    )


def run_nsvi(
    rng_key: Array,
    *,
    d_w: int,
    d_theta: int,
    d_x0: int,
    log_likelihood_fn: LogLikelihoodFn,
    energy_fn: EnergyFn,
    config: NSVIConfig | None = None,
) -> NSVIResult:
    """Run nested SVI with a single compiled device loop."""

    if config is None:
        config = NSVIConfig()
    if config.n_outer_samples < 1 or config.n_aux_samples < 1:
        raise ValueError("n_outer_samples and n_aux_samples must be positive")
    if config.n_outer_samples != 1:
        raise ValueError(
            "Paper-faithful NSVI uses one outer reparameterized sample per iteration; "
            "set n_outer_samples=1."
        )
    if config.log_every < 1:
        raise ValueError("log_every must be positive")

    dtype = jnp.asarray(0.0).dtype
    d_total = d_w + d_theta + d_x0
    theta_prior_std = _as_std_vector(config.theta_prior_std, d_theta, dtype)
    x0_prior_std = _as_std_vector(config.x0_prior_std, d_x0, dtype)

    outer_params = _init_outer_params(config, d_w, d_theta, d_x0, dtype)
    inner_params = {
        "w_mean": outer_params["w_mean"],
        "w_log_std": jnp.full((d_w,), jnp.log(jnp.asarray(config.inner_init_std, dtype=dtype)), dtype=dtype),
    }

    outer_optimizer = _make_optimizer(config.outer_lr, config.grad_clip)
    inner_optimizer = _make_optimizer(config.inner_lr, config.grad_clip)
    outer_opt_state = outer_optimizer.init(outer_params)
    inner_opt_state = inner_optimizer.init(inner_params)

    part_weights_np = np.asarray(
        [_partition_weight(step, config) for step in range(config.iterations)],
        dtype=np.asarray(0.0, dtype=np.float64 if jax.config.jax_enable_x64 else np.float32).dtype,
    )
    log_steps_np = _log_indices(config.iterations, config.log_every)
    log_mask_np = np.zeros((config.iterations,), dtype=np.bool_)
    log_mask_np[log_steps_np] = True

    part_weights = jnp.asarray(part_weights_np, dtype=dtype)
    log_steps = jnp.asarray(log_steps_np, dtype=jnp.int32)
    log_mask = jnp.asarray(log_mask_np)
    n_logs = int(log_steps_np.size)

    def sample_inner(params: Mapping[str, Array], eps: Array) -> Array:
        return params["w_mean"][None, :] + jnp.exp(params["w_log_std"])[None, :] * eps

    # Callback adapters: pass the per-iteration data key only when the caller
    # opted into stochastic (subsampling) callbacks. Default path is the
    # original 2-/3-arg contract, so existing callers are unaffected.
    if config.stochastic_callbacks:
        def _loglik(w: Array, theta: Array, data_key: Array) -> Array:
            return log_likelihood_fn(w, theta, data_key)

        def _energy(w: Array, theta: Array, x0: Array, data_key: Array) -> Array:
            return energy_fn(w, theta, x0, data_key)
    else:
        def _loglik(w: Array, theta: Array, data_key: Array) -> Array:
            return log_likelihood_fn(w, theta)

        def _energy(w: Array, theta: Array, x0: Array, data_key: Array) -> Array:
            return energy_fn(w, theta, x0)

    def inner_loss(params: Mapping[str, Array], theta: Array, x0: Array, eps_aux: Array, data_key: Array) -> Array:
        w_aux = sample_inner(params, eps_aux)

        def one(wi: Array) -> Array:
            return _energy(wi, theta, x0, data_key) + _diag_logpdf(wi, params["w_mean"], params["w_log_std"])

        return jnp.mean(jax.vmap(one)(w_aux))

    inner_value_and_grad = jax.value_and_grad(inner_loss)

    def outer_loss(
        params: Mapping[str, Array],
        eps_outer: Array,
        aux_w_samples: Array,
        part_weight: Array,
        data_key: Array,
    ) -> tuple[Array, dict[str, Array]]:
        def one_sample(eps: Array) -> tuple[Array, Array, Array, Array]:
            w, theta, x0, log_q = _sample_outer(params, eps, config, d_w, d_theta)
            log_lik = _loglik(w, theta, data_key)
            energy = _energy(w, theta, x0, data_key)
            log_prior = _normal_logpdf(theta, theta_prior_std) + _normal_logpdf(x0, x0_prior_std)

            def correction_one(w_aux: Array) -> Array:
                return _energy(jax.lax.stop_gradient(w_aux), theta, x0, data_key)

            correction = jnp.mean(jax.vmap(correction_one)(aux_w_samples))
            elbo = log_lik - energy + log_prior - log_q + part_weight * correction
            return -elbo, log_lik, energy, correction

        losses, log_liks, energies, corrections = jax.vmap(one_sample)(eps_outer)
        loss = jnp.mean(losses)
        aux = {
            "elbo": -loss,
            "log_likelihood": jnp.mean(log_liks),
            "energy": jnp.mean(energies),
            "partition_correction": jnp.mean(corrections),
        }
        return loss, aux

    outer_value_and_grad = jax.value_and_grad(outer_loss, has_aux=True)

    def run_inner_updates(
        key: Array,
        params: dict[str, Array],
        opt_state: optax.OptState,
        theta_probe: Array,
        x0_probe: Array,
        data_key: Array,
    ) -> tuple[Array, dict[str, Array], optax.OptState, Array]:
        def body(carry, _unused):
            key_local, params_local, opt_state_local = carry
            key_local, eps_key = jax.random.split(key_local)
            eps_aux = jax.random.normal(eps_key, shape=(config.n_aux_samples, d_w))
            loss_value, grads = inner_value_and_grad(params_local, theta_probe, x0_probe, eps_aux, data_key)
            updates, opt_state_new = inner_optimizer.update(grads, opt_state_local, params_local)
            params_new = optax.apply_updates(params_local, updates)
            params_new = {
                "w_mean": params_new["w_mean"],
                "w_log_std": jnp.clip(params_new["w_log_std"], jnp.log(config.min_std), config.max_log_std),
            }
            return (key_local, params_new, opt_state_new), loss_value

        (key_out, params_out, opt_state_out), losses = jax.lax.scan(
            body,
            (key, params, opt_state),
            xs=None,
            length=config.inner_iterations,
        )
        return key_out, params_out, opt_state_out, losses[-1]

    def one_step(
        key: Array,
        outer_params_in: dict[str, Array],
        outer_opt_state_in: optax.OptState,
        inner_params_in: dict[str, Array],
        inner_opt_state_in: optax.OptState,
        part_weight: Array,
    ) -> tuple[Array, dict[str, Array], optax.OptState, dict[str, Array], optax.OptState, Array, dict[str, Array], Array]:
        # Preserve the exact PRNG stream of the original 4-way split when the
        # subsampling hook is off, so existing callers are byte-identical.
        if config.stochastic_callbacks:
            key, outer_eps_key, inner_key, aux_key, data_key = jax.random.split(key, 5)
        else:
            key, outer_eps_key, inner_key, aux_key = jax.random.split(key, 4)
            data_key = aux_key  # unused by the non-stochastic callback adapters
        eps_outer = jax.random.normal(outer_eps_key, shape=(config.n_outer_samples, d_total))
        probe_w, probe_theta, probe_x0, _ = _sample_outer(outer_params_in, eps_outer[0], config, d_w, d_theta)
        del probe_w

        inner_key, inner_params_out, inner_opt_state_out, inner_loss_value = run_inner_updates(
            inner_key,
            inner_params_in,
            inner_opt_state_in,
            probe_theta,
            probe_x0,
            data_key,
        )

        eps_aux = jax.random.normal(aux_key, shape=(config.n_aux_samples, d_w))
        aux_w_samples = sample_inner(inner_params_out, eps_aux)
        (loss_value, aux_terms), grads = outer_value_and_grad(
            outer_params_in,
            eps_outer,
            aux_w_samples,
            part_weight,
            data_key,
        )
        updates, outer_opt_state_out = outer_optimizer.update(grads, outer_opt_state_in, outer_params_in)
        outer_params_out = optax.apply_updates(outer_params_in, updates)
        outer_params_out = _clip_params(outer_params_out, config)
        return (
            key,
            outer_params_out,
            outer_opt_state_out,
            inner_params_out,
            inner_opt_state_out,
            loss_value,
            aux_terms,
            inner_loss_value,
        )

    def empty_history() -> dict[str, Array]:
        return {
            "objective": jnp.empty((n_logs,), dtype=dtype),
            "elbo": jnp.empty((n_logs,), dtype=dtype),
            "log_likelihood": jnp.empty((n_logs,), dtype=dtype),
            "energy": jnp.empty((n_logs,), dtype=dtype),
            "partition_correction": jnp.empty((n_logs,), dtype=dtype),
            "partition_weight": jnp.empty((n_logs,), dtype=dtype),
            "inner_objective": jnp.empty((n_logs,), dtype=dtype),
            "mean": jnp.empty((n_logs, d_total), dtype=dtype),
            "std": jnp.empty((n_logs, d_total), dtype=dtype),
        }

    def run_all_steps(
        key: Array,
        outer_params_init: dict[str, Array],
        outer_opt_state_init: optax.OptState,
        inner_params_init: dict[str, Array],
        inner_opt_state_init: optax.OptState,
    ):
        hist0 = empty_history()
        carry0 = (
            key,
            outer_params_init,
            outer_opt_state_init,
            inner_params_init,
            inner_opt_state_init,
            jnp.asarray(0, dtype=jnp.int32),
            hist0,
        )

        def body(carry, xs):
            (
                key_local,
                outer_params_local,
                outer_opt_state_local,
                inner_params_local,
                inner_opt_state_local,
                hist_i,
                hist,
            ) = carry
            step_i, part_weight, should_log = xs
            (
                key_next,
                outer_params_next,
                outer_opt_state_next,
                inner_params_next,
                inner_opt_state_next,
                loss_value,
                aux_terms,
                inner_loss_value,
            ) = one_step(
                key_local,
                outer_params_local,
                outer_opt_state_local,
                inner_params_local,
                inner_opt_state_local,
                part_weight,
            )

            def write_history(args):
                hist_in, write_i = args
                mean, cov = nsvi_moments(outer_params_next, d_w=d_w, d_theta=d_theta, d_x0=d_x0, config=config)
                std = jnp.sqrt(jnp.diag(cov))
                hist_out = {
                    "objective": hist_in["objective"].at[write_i].set(loss_value),
                    "elbo": hist_in["elbo"].at[write_i].set(aux_terms["elbo"]),
                    "log_likelihood": hist_in["log_likelihood"].at[write_i].set(aux_terms["log_likelihood"]),
                    "energy": hist_in["energy"].at[write_i].set(aux_terms["energy"]),
                    "partition_correction": hist_in["partition_correction"].at[write_i].set(aux_terms["partition_correction"]),
                    "partition_weight": hist_in["partition_weight"].at[write_i].set(part_weight),
                    "inner_objective": hist_in["inner_objective"].at[write_i].set(inner_loss_value),
                    "mean": hist_in["mean"].at[write_i].set(mean),
                    "std": hist_in["std"].at[write_i].set(std),
                }
                return hist_out, write_i + jnp.asarray(1, dtype=jnp.int32)

            hist_next, hist_i_next = jax.lax.cond(
                should_log,
                write_history,
                lambda args: args,
                (hist, hist_i),
            )
            del step_i
            return (
                key_next,
                outer_params_next,
                outer_opt_state_next,
                inner_params_next,
                inner_opt_state_next,
                hist_i_next,
                hist_next,
            ), None

        carry_out, _ = jax.lax.scan(
            body,
            carry0,
            (jnp.arange(config.iterations, dtype=jnp.int32), part_weights, log_mask),
        )
        key_out, outer_params_out, _outer_opt_state_out, inner_params_out, _inner_opt_state_out, _hist_i, hist = carry_out
        return key_out, outer_params_out, inner_params_out, hist

    run_all_steps_jit = jax.jit(run_all_steps)
    _key, outer_params, inner_params, hist_arrays = run_all_steps_jit(
        rng_key,
        outer_params,
        outer_opt_state,
        inner_params,
        inner_opt_state,
    )
    jax.block_until_ready((outer_params, inner_params, hist_arrays["objective"]))

    hist_np = {name: np.asarray(value) for name, value in hist_arrays.items()}
    history: dict[str, list[float | list[float]]] = {
        "objective": hist_np["objective"].astype(float).tolist(),
        "elbo": hist_np["elbo"].astype(float).tolist(),
        "log_likelihood": hist_np["log_likelihood"].astype(float).tolist(),
        "energy": hist_np["energy"].astype(float).tolist(),
        "partition_correction": hist_np["partition_correction"].astype(float).tolist(),
        "partition_weight": hist_np["partition_weight"].astype(float).tolist(),
        "inner_objective": hist_np["inner_objective"].astype(float).tolist(),
        "mean": hist_np["mean"].astype(float).tolist(),
        "std": hist_np["std"].astype(float).tolist(),
    }

    if not config.quiet:
        for i, step in enumerate(log_steps_np):
            print(
                "[NSVI] "
                f"step={int(step) + 1}/{config.iterations} "
                f"obj={history['objective'][i]:.6f} "
                f"energy={history['energy'][i]:.6f} "
                f"pc={history['partition_correction'][i]:.6f}",
                flush=True,
            )

    return NSVIResult(params=dict(outer_params), inner_params=dict(inner_params), history=history, config=replace(config))


def nsvi_moments(
    params: Mapping[str, Array],
    *,
    d_w: int,
    d_theta: int,
    d_x0: int,
    config: NSVIConfig,
) -> tuple[Array, Array]:
    """Return mean and covariance for ``z = [w, theta, x0]``."""

    mean = jnp.concatenate([params["w_mean"], params["theta_mean"], params["x0_mean"]])
    d_total = d_w + d_theta + d_x0
    if mean.shape[0] != d_total:
        raise ValueError("parameter dimensions do not match supplied dimensions")
    cov = jnp.zeros((d_total, d_total), dtype=mean.dtype)
    w_var = jnp.exp(2.0 * params["w_log_std"])
    x0_var = jnp.exp(2.0 * params["x0_log_std"])
    cov = cov.at[:d_w, :d_w].set(jnp.diag(w_var))
    if "theta_tril_raw" in params:
        theta_scale = _scale_tril(params["theta_tril_raw"], config.min_std)
        theta_cov = theta_scale @ theta_scale.T
    else:
        theta_cov = jnp.diag(jnp.exp(2.0 * params["theta_log_std"]))
    cov = cov.at[d_w : d_w + d_theta, d_w : d_w + d_theta].set(theta_cov)
    cov = cov.at[d_w + d_theta :, d_w + d_theta :].set(jnp.diag(x0_var))
    return mean, cov


def draw_nsvi_samples(
    rng_key: Array,
    params: Mapping[str, Array],
    *,
    n_samples: int,
    d_w: int,
    d_theta: int,
    d_x0: int,
    config: NSVIConfig,
) -> dict[str, np.ndarray]:
    """Draw samples from the fitted NSVI outer guide."""

    d_total = d_w + d_theta + d_x0
    eps = jax.random.normal(rng_key, shape=(n_samples, d_total))
    eps_w, eps_theta, eps_x0 = _split_z(eps, d_w, d_theta)
    w = params["w_mean"][None, :] + jnp.exp(params["w_log_std"])[None, :] * eps_w
    x0 = params["x0_mean"][None, :] + jnp.exp(params["x0_log_std"])[None, :] * eps_x0
    if "theta_tril_raw" in params:
        scale_tril = _scale_tril(params["theta_tril_raw"], config.min_std)
        theta = params["theta_mean"][None, :] + eps_theta @ scale_tril.T
    else:
        theta = params["theta_mean"][None, :] + jnp.exp(params["theta_log_std"])[None, :] * eps_theta
    z = jnp.concatenate([w, theta, x0], axis=1)
    z_arr = np.asarray(z, dtype=np.float64)
    return {
        "z_samples": z_arr,
        "w_samples": z_arr[:, :d_w],
        "theta_samples": z_arr[:, d_w : d_w + d_theta],
        "x0_samples": z_arr[:, d_w + d_theta :],
    }

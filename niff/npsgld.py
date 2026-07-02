"""Nested preconditioned SGLD for the relaxed NIFF posterior.

NPSGLD samples the relaxed NIFF posterior with a preconditioned Langevin chain
and uses a persistent auxiliary PSGLD chain to estimate the partition-gradient
term.  The method has three moving parts:

1. A persistent auxiliary PSGLD chain approximates
   ``p(w_tilde | theta, x0) proportional to exp(-energy(w_tilde, theta, x0))``.
2. The main chain updates ``(w, theta, x0)`` using the likelihood gradient, the
   unnormalized NIFF prior gradient, Gaussian priors on ``theta`` and ``x0``,
   and ``+ grad energy(w_tilde, theta, x0)`` as the partition-gradient
   correction.
3. Both chains use RMSprop-style diagonal preconditioning and, by default, the
   Riemannian ``Gamma`` correction from the paper's PSGLD/NPSGLD equations.

Set ``include_riemannian_correction=False`` in :class:`NPSGLDConfig` to use the
common practical pSGLD approximation that omits ``Gamma``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

Array = jax.Array
LogLikelihoodFn = Callable[[Array, Array], Array]
EnergyFn = Callable[[Array, Array, Array], Array]


def _chol_lower(mat: Array) -> Array:
    """Lower-triangular Cholesky factor L (L L^T = mat) for a small SPD matrix.

    Manual Cholesky-Banachiewicz, unrolled over the static dimension: pure
    arithmetic, so it avoids the cuSOLVER ``potrf``/``getrf`` factorizations
    (whose FFI handlers are missing on some GPU/jaxlib builds). Pairs with
    ``solve_triangular`` (a cuBLAS ``trsm``, which is available).
    """
    d = mat.shape[0]
    L = jnp.zeros_like(mat)
    for j in range(d):
        s = mat[j, j] - jnp.sum(L[j, :j] ** 2)
        Ljj = jnp.sqrt(jnp.maximum(s, 1.0e-30))
        L = L.at[j, j].set(Ljj)
        for i in range(j + 1, d):
            off = mat[i, j] - jnp.sum(L[i, :j] * L[j, :j])
            L = L.at[i, j].set(off / Ljj)
    return L


@dataclass(frozen=True)
class NPSGLDConfig:
    iterations: int = 20_000
    chains: int = 4
    burn_in: int = 5_000
    thinning: int = 10
    step_size: float = 2.0e-4
    step_size_final: float | None = None
    aux_step_size: float = 2.0e-4
    aux_step_size_final: float | None = None
    aux_iterations: int = 5
    alpha_initial: float = 0.95
    alpha_final: float = 1.0
    alpha_anneal_steps: int = 5_000
    aux_alpha_initial: float = 0.95
    aux_alpha_final: float = 1.0
    aux_alpha_anneal_steps: int = 5_000
    delta: float = 0.1
    aux_delta: float | None = None
    theta_prior_std: float = 1.0
    x0_prior_std: float = 1.0
    init_mean: tuple[float, ...] | None = None
    init_std: float = 0.5
    grad_clip: float = 1.0e3
    state_clip: float = 1.0e6
    include_riemannian_correction: bool = True
    # Preconditioner family for both the main and auxiliary chains:
    #   "rmsprop"      RMSprop-style diagonal preconditioner -> NPSGLD (paper method 4).
    #   "identity"     no preconditioner (and no Gamma)      -> plain nested SGLD / NSGLD
    #                  (paper method 3).
    #   "diag_fisher"  diagonal empirical Fisher (cross-chain second moment of the gradients).
    #   "dense_fisher" dense empirical Fisher on the low-dim theta block (corrects parameter
    #                  coupling); diagonal Fisher on the w / x0 blocks. Benefits from
    #                  chains >= d_theta. (design notes: archive/methods/psgld/preconditioned_sgld/)
    preconditioner: str = "rmsprop"
    trace_every: int = 1_000
    quiet: bool = False


@dataclass
class NPSGLDResult:
    samples: dict[str, np.ndarray]
    history: dict[str, list[float | list[float]]]
    final_state: dict[str, np.ndarray]
    config: NPSGLDConfig


def _as_std_vector(std: float | Array, dim: int, dtype: jnp.dtype) -> Array:
    arr = jnp.asarray(std, dtype=dtype)
    if arr.ndim == 0:
        return jnp.full((dim,), arr, dtype=dtype)
    arr = arr.reshape((-1,))
    if arr.shape[0] != dim:
        raise ValueError(f"prior std has length {arr.shape[0]}, expected {dim}")
    return arr


def _normal_logpdf(x: Array, std: Array) -> Array:
    standardized = x / std
    return jnp.sum(-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(std) - 0.5 * standardized**2)


def _split_z(z: Array, d_w: int, d_theta: int) -> tuple[Array, Array, Array]:
    return z[..., :d_w], z[..., d_w : d_w + d_theta], z[..., d_w + d_theta :]


def _schedule(step: int, start: float, final: float | None, total_steps: int) -> float:
    if final is None:
        return float(start)
    if total_steps <= 1:
        return float(final)
    ratio = min(1.0, max(0.0, float(step) / float(total_steps - 1)))
    if start > 0.0 and final > 0.0:
        return float(start * np.exp(np.log(final / start) * ratio))
    return float(start + ratio * (final - start))


def _alpha(step: int, start: float, final: float, anneal_steps: int) -> float:
    if anneal_steps <= 0:
        return float(final)
    ratio = min(1.0, max(0.0, float(step) / float(anneal_steps)))
    return float(start + ratio * (final - start))


def _sanitize_grad(grad: Array, clip: float) -> Array:
    return jnp.clip(jnp.nan_to_num(grad, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)


def _sanitize_state(candidate: Array, old: Array, clip: float) -> Array:
    return jnp.clip(jnp.where(jnp.isfinite(candidate), candidate, old), -clip, clip)


def _trace_indices(iterations: int, trace_every: int) -> np.ndarray:
    return np.asarray(
        [step for step in range(iterations) if (step % trace_every) == 0 or step == iterations - 1],
        dtype=np.int32,
    )


def _sample_indices(iterations: int, burn_in: int, thinning: int) -> np.ndarray:
    return np.asarray(
        [step for step in range(iterations) if step >= burn_in and ((step - burn_in) % thinning) == 0],
        dtype=np.int32,
    )


def run_npsgld(
    rng_key: Array,
    *,
    d_w: int,
    d_theta: int,
    d_x0: int,
    log_likelihood_fn: LogLikelihoodFn,
    energy_fn: EnergyFn,
    config: NPSGLDConfig | None = None,
) -> NPSGLDResult:
    """Run nested preconditioned SGLD with one compiled device loop."""

    if config is None:
        config = NPSGLDConfig()
    if config.burn_in >= config.iterations:
        raise ValueError("burn_in must be smaller than iterations")
    if config.thinning < 1 or config.chains < 1 or config.aux_iterations < 1:
        raise ValueError("chains, thinning, and aux_iterations must be positive")
    if config.trace_every < 1:
        raise ValueError("trace_every must be positive")
    precond_kind = config.preconditioner.lower()
    if precond_kind not in ("identity", "rmsprop", "diag_fisher", "dense_fisher"):
        raise ValueError(f"unknown preconditioner {config.preconditioner!r}")
    # The auxiliary (field-w) chain uses a diagonal RMSprop preconditioner under every
    # non-identity family; only the OUTER (z) update differs between families.
    use_precond = precond_kind != "identity"
    # The Riemannian Gamma term is only defined for the rmsprop diagonal preconditioner.
    use_rmsprop_gamma = precond_kind == "rmsprop"

    dtype = jnp.asarray(0.0).dtype
    d_total = d_w + d_theta + d_x0
    theta_prior_std = _as_std_vector(config.theta_prior_std, d_theta, dtype)
    x0_prior_std = _as_std_vector(config.x0_prior_std, d_x0, dtype)
    aux_delta = config.delta if config.aux_delta is None else config.aux_delta

    if config.init_mean is None:
        init_mean = jnp.zeros((d_total,), dtype=dtype)
    else:
        init_mean = jnp.asarray(config.init_mean, dtype=dtype).reshape((-1,))
        if init_mean.shape[0] != d_total:
            raise ValueError(f"init_mean has length {init_mean.shape[0]}, expected {d_total}")

    key, init_key = jax.random.split(rng_key)
    init_noise = jax.random.normal(init_key, shape=(config.chains, d_total))
    z0 = init_mean[None, :] + config.init_std * init_noise
    w_state, theta_state, x0_state = _split_z(z0, d_w, d_theta)
    w_aux_state = w_state
    outer_v_state = jnp.zeros((config.chains, d_total), dtype=dtype)
    aux_v_state = jnp.zeros((config.chains, d_w), dtype=dtype)
    # EMA empirical-Fisher matrix for the theta block (dense_fisher only; identity otherwise).
    f_theta_state = jnp.eye(d_theta, dtype=dtype)

    schedule_dtype = np.asarray(0.0, dtype=np.float64 if jax.config.jax_enable_x64 else np.float32).dtype
    lr_schedule = jnp.asarray(
        [_schedule(step, config.step_size, config.step_size_final, config.iterations) for step in range(config.iterations)],
        dtype=dtype,
    )
    lr_aux_schedule = jnp.asarray(
        [_schedule(step, config.aux_step_size, config.aux_step_size_final, config.iterations) for step in range(config.iterations)],
        dtype=dtype,
    )
    alpha_schedule = jnp.asarray(
        [_alpha(step, config.alpha_initial, config.alpha_final, config.alpha_anneal_steps) for step in range(config.iterations)],
        dtype=dtype,
    )
    alpha_aux_schedule = jnp.asarray(
        [_alpha(step, config.aux_alpha_initial, config.aux_alpha_final, config.aux_alpha_anneal_steps) for step in range(config.iterations)],
        dtype=dtype,
    )
    del schedule_dtype

    sample_steps_np = _sample_indices(config.iterations, config.burn_in, config.thinning)
    trace_steps_np = _trace_indices(config.iterations, config.trace_every)
    sample_mask_np = np.zeros((config.iterations,), dtype=np.bool_)
    trace_mask_np = np.zeros((config.iterations,), dtype=np.bool_)
    sample_mask_np[sample_steps_np] = True
    trace_mask_np[trace_steps_np] = True

    sample_mask = jnp.asarray(sample_mask_np)
    trace_mask = jnp.asarray(trace_mask_np)
    n_sample_steps = int(sample_steps_np.size)
    n_trace_steps = int(trace_steps_np.size)

    def log_conditional_path_args(cur_w: Array, theta: Array, x0: Array) -> Array:
        return -energy_fn(cur_w, theta, x0)

    grad_conditional_path_raw = jax.grad(log_conditional_path_args, argnums=0)

    def grad_conditional_path(cur_w: Array, theta: Array, x0: Array) -> Array:
        return _sanitize_grad(grad_conditional_path_raw(cur_w, theta, x0), config.grad_clip)

    def aux_precond_for_gamma_args(
        cur_w: Array,
        theta: Array,
        x0: Array,
        v_prev: Array,
        alpha_aux: Array,
    ) -> Array:
        grad_cur = grad_conditional_path(cur_w, theta, x0)
        v_cur = alpha_aux * v_prev + (1.0 - alpha_aux) * grad_cur**2
        return 1.0 / (aux_delta + jnp.sqrt(v_cur + 1.0e-18))

    aux_precond_jacfwd = jax.jacfwd(aux_precond_for_gamma_args, argnums=0)

    def log_posterior_surrogate_args(cur_w: Array, cur_theta: Array, cur_x0: Array, w_aux: Array) -> Array:
        log_lik = log_likelihood_fn(cur_w, cur_theta)
        energy = energy_fn(cur_w, cur_theta, cur_x0)
        correction = energy_fn(jax.lax.stop_gradient(w_aux), cur_theta, cur_x0)
        log_prior = _normal_logpdf(cur_theta, theta_prior_std) + _normal_logpdf(cur_x0, x0_prior_std)
        return log_lik - energy + log_prior + correction

    posterior_value_and_grad = jax.value_and_grad(log_posterior_surrogate_args, argnums=(0, 1, 2))

    def log_posterior_surrogate_z_args(cur_z: Array, w_aux_one: Array) -> Array:
        cur_w, cur_theta, cur_x0 = _split_z(cur_z, d_w, d_theta)
        return log_posterior_surrogate_args(cur_w, cur_theta, cur_x0, w_aux_one)

    grad_log_posterior_z = jax.grad(log_posterior_surrogate_z_args, argnums=0)

    def outer_precond_for_gamma_args(
        cur_z: Array,
        w_aux_one: Array,
        v_prev: Array,
        alpha: Array,
    ) -> Array:
        grad_cur = _sanitize_grad(grad_log_posterior_z(cur_z, w_aux_one), config.grad_clip)
        v_cur = alpha * v_prev + (1.0 - alpha) * grad_cur**2
        return 1.0 / (config.delta + jnp.sqrt(v_cur + 1.0e-18))

    outer_precond_jacfwd = jax.jacfwd(outer_precond_for_gamma_args, argnums=0)

    def run_aux_chain(
        key_in: Array,
        w_aux: Array,
        v_aux: Array,
        theta: Array,
        x0: Array,
        lr_aux: Array,
        alpha_aux: Array,
    ) -> tuple[Array, Array, Array]:
        def body(carry, _unused):
            key_local, w_local, v_local = carry
            key_local, noise_key = jax.random.split(key_local)

            grad_w = grad_conditional_path(w_local, theta, x0)
            v_new = alpha_aux * v_local + (1.0 - alpha_aux) * grad_w**2
            if use_precond:
                precond = 1.0 / (aux_delta + jnp.sqrt(v_new + 1.0e-18))
            else:
                precond = jnp.ones_like(w_local)
            if use_rmsprop_gamma and config.include_riemannian_correction:
                gamma = jnp.diag(aux_precond_jacfwd(w_local, theta, x0, v_local, alpha_aux))
            else:
                gamma = jnp.zeros_like(w_local)
            noise = jax.random.normal(noise_key, shape=w_local.shape)
            proposal = w_local + lr_aux * (precond * grad_w + gamma) + jnp.sqrt(2.0 * lr_aux * precond) * noise
            w_new = _sanitize_state(proposal, w_local, config.state_clip)
            return (key_local, w_new, v_new), None

        (key_out, w_out, v_out), _ = jax.lax.scan(
            body,
            (key_in, w_aux, v_aux),
            xs=None,
            length=config.aux_iterations,
        )
        return key_out, w_out, v_out

    def posterior_grad_and_terms(w: Array, theta: Array, x0: Array, w_aux: Array) -> tuple[Array, dict[str, Array]]:
        value, grads = posterior_value_and_grad(w, theta, x0, w_aux)
        grad_vec = jnp.concatenate(grads)
        terms = {
            "log_posterior_surrogate": value,
            "log_likelihood": log_likelihood_fn(w, theta),
            "energy": energy_fn(w, theta, x0),
            "partition_correction": energy_fn(jax.lax.stop_gradient(w_aux), theta, x0),
            "log_prior_theta_x0": _normal_logpdf(theta, theta_prior_std) + _normal_logpdf(x0, x0_prior_std),
        }
        return _sanitize_grad(grad_vec, config.grad_clip), terms

    aux_many = jax.vmap(run_aux_chain, in_axes=(0, 0, 0, 0, 0, None, None))
    posterior_many = jax.vmap(posterior_grad_and_terms, in_axes=(0, 0, 0, 0))

    def one_step(
        key_in: Array,
        w: Array,
        theta: Array,
        x0: Array,
        outer_v: Array,
        w_aux: Array,
        aux_v: Array,
        f_theta: Array,
        lr: Array,
        lr_aux: Array,
        alpha: Array,
        alpha_aux: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, dict[str, Array]]:
        split = jax.random.split(key_in, config.chains * 2 + 1)
        key_out = split[0]
        chain_keys = split[1:].reshape((config.chains, 2, 2))
        aux_keys = chain_keys[:, 0, :]
        noise_keys = chain_keys[:, 1, :]

        _, w_aux_new, aux_v_new = aux_many(aux_keys, w_aux, aux_v, theta, x0, lr_aux, alpha_aux)
        grad_vec, terms = posterior_many(w, theta, x0, w_aux_new)

        outer_v_new = alpha * outer_v + (1.0 - alpha) * grad_vec**2  # per-chain EMA (rmsprop + trace)
        eps_p = 1.0e-18
        if precond_kind == "rmsprop":
            precond = 1.0 / (config.delta + jnp.sqrt(outer_v_new + eps_p))  # (chains, d_total), per-chain
        elif precond_kind in ("diag_fisher", "dense_fisher"):
            # diagonal empirical Fisher = cross-chain mean of the per-chain EMA second moments
            # (EMA-smoothed so the preconditioner varies slowly -> the omitted-Gamma
            # approximation stays good and does not inflate the sampled variance).
            v_fisher = jnp.mean(outer_v_new, axis=0)  # (d_total,)
            precond = jnp.broadcast_to(1.0 / (config.delta + jnp.sqrt(v_fisher + eps_p)), grad_vec.shape)
        else:  # identity
            precond = jnp.ones_like(grad_vec)
        z_old = jnp.concatenate([w, theta, x0], axis=1)
        if use_rmsprop_gamma and config.include_riemannian_correction:
            def gamma_one(z_vec: Array, w_aux_one: Array, v_prev: Array) -> Array:
                return jnp.diag(outer_precond_jacfwd(z_vec, w_aux_one, v_prev, alpha))

            gamma = jax.vmap(gamma_one)(z_old, w_aux_new, outer_v)
        else:
            gamma = jnp.zeros_like(grad_vec)
        noise = jax.vmap(lambda kk: jax.random.normal(kk, shape=(d_total,)))(noise_keys)
        step_vec = lr * (precond * grad_vec + gamma) + jnp.sqrt(2.0 * lr * precond) * noise

        f_theta_new = f_theta
        if precond_kind == "dense_fisher" and d_theta > 0:
            # Dense empirical-Fisher preconditioner on the low-dimensional theta block.
            #   F_inst = (1/chains) sum_c g_c g_c^T  (cross-chain empirical Fisher),
            #   F = alpha F_prev + (1-alpha) F_inst  (EMA-smoothed, so P varies slowly and the
            #       omitted-Gamma approximation does not inflate the sampled variance),
            #   P = (F + delta I)^{-1}.  Langevin: drift = P g_c, noise ~ N(0, P).
            # Realized cuSOLVER-free: chol F = L L^T (manual), then triangular solves (cuBLAS):
            #   drift_c = F^{-1} g_c = L^{-T} L^{-1} g_c ;  noise_c = L^{-T} eps_c  (cov = F^{-1}).
            g_theta = grad_vec[:, d_w : d_w + d_theta]  # (chains, d_theta)
            f_inst = (g_theta.T @ g_theta) / jnp.asarray(config.chains, dtype=g_theta.dtype)
            f_theta_new = alpha * f_theta + (1.0 - alpha) * f_inst
            fisher = f_theta_new + config.delta * jnp.eye(d_theta, dtype=g_theta.dtype)
            l_f = _chol_lower(fisher)  # F = l_f l_f^T
            gt = g_theta.T  # (d_theta, chains)
            drift_theta = solve_triangular(
                l_f.T, solve_triangular(l_f, gt, lower=True), lower=False
            ).T  # (chains, d_theta) = F^{-1} g_c
            eps_theta = noise[:, d_w : d_w + d_theta].T  # (d_theta, chains) ~ N(0, I)
            noise_theta = solve_triangular(l_f.T, eps_theta, lower=False).T  # cov = F^{-1}
            step_theta = lr * drift_theta + jnp.sqrt(2.0 * lr) * noise_theta
            step_vec = step_vec.at[:, d_w : d_w + d_theta].set(step_theta)

        z_prop = z_old + step_vec
        z_new = _sanitize_state(z_prop, z_old, config.state_clip)
        w_new, theta_new, x0_new = _split_z(z_new, d_w, d_theta)
        return key_out, w_new, theta_new, x0_new, outer_v_new, w_aux_new, aux_v_new, f_theta_new, terms

    def empty_history() -> dict[str, Array]:
        return {
            "log_likelihood": jnp.empty((n_trace_steps,), dtype=dtype),
            "energy": jnp.empty((n_trace_steps,), dtype=dtype),
            "partition_correction": jnp.empty((n_trace_steps,), dtype=dtype),
            "log_prior_theta_x0": jnp.empty((n_trace_steps,), dtype=dtype),
            "lr": jnp.empty((n_trace_steps,), dtype=dtype),
            "alpha": jnp.empty((n_trace_steps,), dtype=dtype),
            "sample_count": jnp.empty((n_trace_steps,), dtype=jnp.int32),
            "mean": jnp.empty((n_trace_steps, d_total), dtype=dtype),
        }

    def run_all_steps(
        key_in: Array,
        w_in: Array,
        theta_in: Array,
        x0_in: Array,
        outer_v_in: Array,
        w_aux_in: Array,
        aux_v_in: Array,
        f_theta_in: Array,
    ):
        sample_buffer0 = jnp.empty((n_sample_steps, config.chains, d_total), dtype=dtype)
        hist0 = empty_history()
        carry0 = (
            key_in,
            w_in,
            theta_in,
            x0_in,
            outer_v_in,
            w_aux_in,
            aux_v_in,
            f_theta_in,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            sample_buffer0,
            hist0,
        )

        def body(carry, xs):
            (
                key_local,
                w_local,
                theta_local,
                x0_local,
                outer_v_local,
                w_aux_local,
                aux_v_local,
                f_theta_local,
                sample_i,
                hist_i,
                sample_buffer,
                hist,
            ) = carry
            lr, lr_aux, alpha, alpha_aux, should_sample, should_trace = xs
            (
                key_next,
                w_next,
                theta_next,
                x0_next,
                outer_v_next,
                w_aux_next,
                aux_v_next,
                f_theta_next,
                terms,
            ) = one_step(
                key_local,
                w_local,
                theta_local,
                x0_local,
                outer_v_local,
                w_aux_local,
                aux_v_local,
                f_theta_local,
                lr,
                lr_aux,
                alpha,
                alpha_aux,
            )

            z_next = jnp.concatenate([w_next, theta_next, x0_next], axis=1)

            def write_sample(args):
                buffer_in, write_i = args
                return buffer_in.at[write_i].set(z_next), write_i + jnp.asarray(1, dtype=jnp.int32)

            sample_buffer_next, sample_i_next = jax.lax.cond(
                should_sample,
                write_sample,
                lambda args: args,
                (sample_buffer, sample_i),
            )

            def write_history(args):
                hist_in, write_i = args
                hist_out = {
                    "log_likelihood": hist_in["log_likelihood"].at[write_i].set(jnp.mean(terms["log_likelihood"])),
                    "energy": hist_in["energy"].at[write_i].set(jnp.mean(terms["energy"])),
                    "partition_correction": hist_in["partition_correction"].at[write_i].set(jnp.mean(terms["partition_correction"])),
                    "log_prior_theta_x0": hist_in["log_prior_theta_x0"].at[write_i].set(jnp.mean(terms["log_prior_theta_x0"])),
                    "lr": hist_in["lr"].at[write_i].set(lr),
                    "alpha": hist_in["alpha"].at[write_i].set(alpha),
                    "sample_count": hist_in["sample_count"].at[write_i].set(sample_i_next * config.chains),
                    "mean": hist_in["mean"].at[write_i].set(jnp.mean(z_next, axis=0)),
                }
                return hist_out, write_i + jnp.asarray(1, dtype=jnp.int32)

            hist_next, hist_i_next = jax.lax.cond(
                should_trace,
                write_history,
                lambda args: args,
                (hist, hist_i),
            )
            return (
                key_next,
                w_next,
                theta_next,
                x0_next,
                outer_v_next,
                w_aux_next,
                aux_v_next,
                f_theta_next,
                sample_i_next,
                hist_i_next,
                sample_buffer_next,
                hist_next,
            ), None

        carry_out, _ = jax.lax.scan(
            body,
            carry0,
            (lr_schedule, lr_aux_schedule, alpha_schedule, alpha_aux_schedule, sample_mask, trace_mask),
        )
        (
            key_out,
            w_out,
            theta_out,
            x0_out,
            outer_v_out,
            w_aux_out,
            aux_v_out,
            _f_theta_out,
            _sample_i,
            _hist_i,
            sample_buffer,
            hist,
        ) = carry_out
        del key_out, outer_v_out, aux_v_out, _f_theta_out
        return w_out, theta_out, x0_out, w_aux_out, sample_buffer, hist

    run_all_steps_jit = jax.jit(run_all_steps)
    w_state, theta_state, x0_state, w_aux_state, sample_buffer, hist_arrays = run_all_steps_jit(
        key,
        w_state,
        theta_state,
        x0_state,
        outer_v_state,
        w_aux_state,
        aux_v_state,
        f_theta_state,
    )
    jax.block_until_ready((w_state, theta_state, x0_state, w_aux_state, sample_buffer, hist_arrays["energy"]))

    z_samples = np.asarray(sample_buffer.reshape((n_sample_steps * config.chains, d_total)), dtype=np.float64)
    chain_id = np.tile(np.arange(config.chains, dtype=np.int32), n_sample_steps)
    samples = {
        "z_samples": z_samples,
        "w_samples": z_samples[:, :d_w],
        "theta_samples": z_samples[:, d_w : d_w + d_theta],
        "x0_samples": z_samples[:, d_w + d_theta :],
        "chain_id": chain_id,
    }

    hist_np = {name: np.asarray(value) for name, value in hist_arrays.items()}
    history: dict[str, list[float | list[float]]] = {
        "log_likelihood": hist_np["log_likelihood"].astype(float).tolist(),
        "energy": hist_np["energy"].astype(float).tolist(),
        "partition_correction": hist_np["partition_correction"].astype(float).tolist(),
        "log_prior_theta_x0": hist_np["log_prior_theta_x0"].astype(float).tolist(),
        "lr": hist_np["lr"].astype(float).tolist(),
        "alpha": hist_np["alpha"].astype(float).tolist(),
        "sample_count": hist_np["sample_count"].astype(int).tolist(),
        "mean": hist_np["mean"].astype(float).tolist(),
    }

    if not config.quiet:
        for i, step in enumerate(trace_steps_np):
            print(
                "[NPSGLD] "
                f"step={int(step) + 1}/{config.iterations} "
                f"energy={history['energy'][i]:.6f} "
                f"pc={history['partition_correction'][i]:.6f} "
                f"samples={history['sample_count'][i]}",
                flush=True,
            )

    final_state = {
        "w": np.asarray(w_state, dtype=np.float64),
        "theta": np.asarray(theta_state, dtype=np.float64),
        "x0": np.asarray(x0_state, dtype=np.float64),
        "w_aux": np.asarray(w_aux_state, dtype=np.float64),
    }
    return NPSGLDResult(samples=samples, history=history, final_state=final_state, config=replace(config))

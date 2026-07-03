"""Field parameterizations for NIFF state paths.

Two shared bases, ported from the ift-sde repo:

* the truncated half-period Fourier basis on ``[0, 1]``
  (``ift/fields/fourier.py``; see ``notes/lab.md`` for the half- vs full-period
  discussion), used by §5.1 and the §5.2 residual-NN encoder; and
* the Gaussian RBF basis on the physical grid ``[0, T]``
  (``ift/fields/rbf.py``), used for the §5.2 linear state path.

Both are the single source of truth for their basis — no per-example copies.
"""

import jax.numpy as jnp


def fourier_basis_01(t01, m):
    """Half-period Fourier basis on ``[0, 1]``.

    ``[1, cos(pi t), ..., cos(m pi t), sin(pi t), ..., sin(m pi t)]``.

    Args:
        t01: time in ``[0, 1]``.
        m: number of sine/cosine pairs.

    Returns:
        basis vector of shape ``(1 + 2m,)``.
    """
    ks = jnp.arange(1, m + 1)
    c = jnp.cos(jnp.pi * ks * t01)
    s = jnp.sin(jnp.pi * ks * t01)
    return jnp.concatenate([jnp.ones((1,)), c, s], axis=0)


def rbf_centers(final_time, num_centers):
    """Evenly spaced RBF centers on ``[0, final_time]``, endpoints included."""
    return jnp.linspace(0.0, final_time, num_centers)


def _rbf_sigma(final_time, num_centers, spread_ratio):
    return spread_ratio * final_time / (num_centers - 1)


def rbf_basis(t, final_time, num_centers, spread_ratio):
    """Gaussian RBF basis at scalar time ``t`` on ``[0, final_time]``.

    ``phi_k(t) = exp(-(t - c_k)^2 / (2 sigma^2))`` with centers at
    ``linspace(0, final_time, num_centers)`` and width
    ``sigma = spread_ratio * center-spacing`` (unit max at ``t = c_k``).

    Args:
        t: scalar time in ``[0, final_time]``.
        final_time: T.
        num_centers: N (>= 2 for endpoints-included spacing).
        spread_ratio: sigma as a multiple of the center spacing ``T/(N-1)``.

    Returns:
        ``phi(t)`` of shape ``(num_centers,)``.
    """
    centers = rbf_centers(final_time, num_centers)
    sigma = _rbf_sigma(final_time, num_centers, spread_ratio)
    return jnp.exp(-0.5 * ((t - centers) / sigma) ** 2)

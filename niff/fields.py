"""Field parameterizations for NIFF state paths.

The truncated Fourier basis on ``[0, 1]``. Ported from the ift-sde repo
(``ift/fields/fourier.py``); this is the half-period basis used throughout the
replication (see ``notes/lab.md`` for the half- vs full-period discussion).
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

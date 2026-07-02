"""Small shared utilities."""

import jax.numpy as jnp


def collocation_grid(num_points):
    """Uniform midpoint collocation grid on ``[0, 1]``: ``(i + 0.5) / N``."""
    if num_points <= 1:
        return jnp.asarray([0.5], dtype=jnp.float64)
    return (jnp.arange(num_points, dtype=jnp.float64) + 0.5) / float(num_points)

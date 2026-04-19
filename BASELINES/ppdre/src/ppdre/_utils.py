import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.jit
def _truncate_to_min_positive(x: ArrayLike):
    x = jnp.asarray(x)
    min_positive_value = jnp.min(jnp.where(x > 0, x, jnp.inf))
    return jnp.where(x <= 0, min_positive_value, x)

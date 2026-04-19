import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


@jax.jit
def _gaussian_basis_function(x: ArrayLike, args: ArrayLike) -> Array:
    mu = args
    sigma = 1
    return jnp.exp(-0.5 * jnp.subtract(x, mu) ** 2 / sigma**2)


@jax.jit
def _expand_gaussian_basis_function(x: ArrayLike, args: ArrayLike) -> Array:
    return jax.vmap(_gaussian_basis_function, in_axes=[None, 0], out_axes=1)(x, args)

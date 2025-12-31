import jax
from flax import nnx
from jax import numpy as jnp


def load_param(target: nnx.Param[jax.Array], value):
    value = jnp.asarray(value, device=target.device)
    assert value.shape == target.shape
    assert value.dtype == target.dtype
    target[...] = value

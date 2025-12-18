from flax import nnx
from jax import numpy as jnp
import jax


class ValueNetwork(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, *, rngs: nnx.Rngs):
        super().__init__()

        self.up = nnx.Linear(
            in_features,
            hidden_features,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )
        self.down = nnx.Linear(
            hidden_features, 1, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=rngs
        )

    def __call__(self, x):
        x = self.up(x)
        x = jax.nn.silu(x)
        x = self.down(x)
        return x

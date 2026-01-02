from flax import nnx
from flax.nnx import variablelib
from jax import numpy as jnp
import jax

import typing as tp

A = tp.TypeVar('A')

class ValueParam(variablelib.Param[A]):
    pass

class ValueNetwork(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, *, rngs: nnx.Rngs):
        super().__init__()

        initializer = nnx.initializers.he_normal()
        # nnx.LoRAParam()
        self.up = ValueParam(initializer(rngs.param(), (in_features, hidden_features), jnp.bfloat16))
        self.up_bias = ValueParam(jnp.zeros(hidden_features, dtype=jnp.bfloat16))
        self.down = ValueParam(initializer(rngs.param(), (hidden_features, 1), jnp.bfloat16))
        self.down_bias = ValueParam(jnp.zeros(1, dtype=jnp.bfloat16))

        # self.up = nnx.Linear(
        #     in_features,
        #     hidden_features,
        #     dtype=jnp.bfloat16,
        #     param_dtype=jnp.bfloat16,
        #     rngs=rngs,
        # )
        # self.down = nnx.Linear(
        #     hidden_features, 1, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=rngs
        # )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x @ self.up
        x = x + self.up_bias
        x = jax.nn.silu(x)
        x = x @ self.down
        x = x + self.down_bias
        x = x * 0.01
        
        return x.squeeze(-1).astype(jnp.float32)

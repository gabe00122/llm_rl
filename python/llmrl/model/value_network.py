from llmrl.config import HlGaussConfig

import typing as tp

import jax
from flax import nnx
from flax.nnx import variablelib
from jax import numpy as jnp

A = tp.TypeVar("A")

from typing import Any
from einops import rearrange
from flax import nnx
import jax.numpy as jnp
from jax.scipy.stats import norm
import optax


def calculate_supports(config: HlGaussConfig):
    support = jnp.linspace(
        config.min, config.max, config.n_logits + 1, dtype=jnp.float32
    )
    centers = (support[:-1] + support[1:]) / 2
    support = support[None, :]

    return support, centers


class HlGaussValue(nnx.Module):
    def __init__(
        self, in_features: int, hl_gauss_config: HlGaussConfig, *, rngs: nnx.Rngs
    ) -> None:
        self._min = hl_gauss_config.min
        self._max = hl_gauss_config.max
        self._sigma = hl_gauss_config.sigma

        self.dense = nnx.Linear(in_features, hl_gauss_config.n_logits, param_dtype=jnp.bfloat16, rngs=rngs)
        self._supports, self._centers = calculate_supports(hl_gauss_config)

    def __call__(self, x) -> Any:
        return self.dense(x).astype(jnp.float32)

    def get_value(self, logits):
        probs = nnx.softmax(logits, axis=-1)
        return (probs * self._centers).sum(-1)

    def get_loss(self, logits, target_values):
        b, t = target_values.shape

        logits = rearrange(logits, "b t l -> (b t) l")
        target_values = rearrange(target_values, "b t -> (b t)")

        targets = jnp.clip(target_values, self._min, self._max)

        cdf_evals = norm.cdf(self._supports, loc=targets[:, None], scale=self._sigma)

        z = cdf_evals[:, -1] - cdf_evals[:, 0]

        bin_probs = cdf_evals[:, 1:] - cdf_evals[:, :-1]

        target_probs = bin_probs / z[:, None]

        loss = optax.softmax_cross_entropy(logits, target_probs, axis=-1)
        return loss.reshape(b, t)


class MseValue(nnx.Module):
    def __init__(self, in_features: int, *, rngs: nnx.Rngs) -> None:
        self.dense = nnx.Linear(in_features, 1, rngs=rngs)

    def __call__(self, x) -> Any:
        x = self.dense(x)
        return x.squeeze(axis=-1)

    def get_value(self, value):
        return value

    def get_loss(self, values, target_values):
        return 0.5 * jnp.square(values - target_values)


class ValueParam(variablelib.Param[A]):
    pass


class ValueNetwork(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, *, rngs: nnx.Rngs):
        super().__init__()

        initializer = nnx.initializers.he_normal()
        self.up = ValueParam(
            initializer(rngs.param(), (in_features, hidden_features), jnp.bfloat16)
        )
        self.up_bias = ValueParam(jnp.zeros(hidden_features, dtype=jnp.bfloat16))
        self.output = HlGaussValue(hidden_features, HlGaussConfig(
            min=0.0,
            max=1.0,
            n_logits=51,
            sigma=0.01
        ), rngs=rngs)
        # self.output = MseValue(hidden_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x = jax.lax.stop_gradient(x)
        x = x @ self.up[:]
        x = x + self.up_bias[:]
        x = jax.nn.silu(x)
        x = self.output(x)

        return x

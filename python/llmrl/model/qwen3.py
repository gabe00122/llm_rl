from typing import Any

import jax
from flax import nnx
from jax import numpy as jnp
from llmrl.config import LLMConfig, LoraConfig
from llmrl.model.attention import KVCache
from llmrl.model.layer import Qwen3Layer
from llmrl.model.util import load_param
from llmrl.model.value_network import ValueNetwork


class Qwen3(nnx.Module):
    def __init__(
        self,
        config: LLMConfig,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self._head_dim = config.head_dim
        self._rope_theta = config.rope_theta

        self.embeddings = nnx.Embed(
            config.vocab_size,
            config.embed,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )

        layers = []
        for _ in range(config.num_layers):
            layers.append(
                Qwen3Layer(
                    config=config,
                    rngs=rngs,
                )
            )
        self.layers = nnx.List(layers)

        self.final_norm = nnx.RMSNorm(
            config.embed,
            epsilon=config.norm_eps,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )

        self.value_net = ValueNetwork(config.embed, 4096 * 2, rngs=rngs)

    def initialize_lora(self, lora_config: LoraConfig, *, rngs: nnx.Rngs):
        for layer in self.layers:
            layer.initialize_lora(lora_config, rngs=rngs)

    def load_params(self, params: dict[str, Any]):
        embed_params = jnp.asarray(
            params["model"]["embed_tokens"]["weight"],
            device=self.embeddings.embedding.device,
        )
        assert embed_params.shape == self.embeddings.embedding.shape

        self.embeddings.embedding[...] = embed_params

        for i, layer in enumerate(self.layers):
            layer_params = params["model"]["layers"][f"{i}"]
            layer.load_params(layer_params)

        load_param(self.final_norm.scale, params["model"]["norm"]["weight"])

    def __call__(
        self,
        tokens: jax.Array,
        positions: jax.Array,
        carry: tuple[KVCache, ...] | None = None,
    ) -> tuple[jax.Array, jax.Array, tuple[KVCache, ...] | None]:
        x = self.embeddings(tokens)

        half_layers = len(self.layers) // 2

        if carry is not None:
            out_carry = []
            for i, (layer, layer_carry_in) in enumerate(zip(self.layers, carry)):
                if i == half_layers:
                    value_in = layer.attn_pre_norm(x)

                x, layer_carry_out = layer(x, positions, layer_carry_in)
                out_carry.append(layer_carry_out)
            carry = tuple(out_carry)
        else:
            for i, layer in enumerate(self.layers):
                if i == half_layers:
                    value_in = layer.attn_pre_norm(x)

                x, _ = jax.checkpoint(layer)(x, positions)

        x = self.final_norm(x)
        logits = x @ self.embeddings.embedding.T

        logits = logits.astype(jnp.float32)

        value = self.value_net(value_in)

        return logits, value, carry

    def initialize_carry(self, batch_size: int, seq_length: int):
        return tuple(
            layer.initialize_carry(batch_size, seq_length) for layer in self.layers
        )

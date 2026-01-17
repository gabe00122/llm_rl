from llmrl.model.value_network import ValueParam
from transformers import Qwen3Config
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

        value_layer_config = config.model_copy(
            update={
                "embed": 256,
                "mlp_ffw_size": 256 * 2,
                "head_dim": 32,
                "q_heads": 8,
                "kv_heads": 8,
            }
        )

        self.embeddings = nnx.Embed(
            config.vocab_size,
            config.embed,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )

        layers = []
        value_layers = []
        for _ in range(config.num_layers):
            layers.append(
                Qwen3Layer(
                    config=config,
                    rngs=rngs,
                )
            )
            value_layer = Qwen3Layer(
                config=value_layer_config,
                rngs=rngs,
            )
            value_layer.wrap_param_type(ValueParam)
            value_layers.append(value_layer)

        self.layers = nnx.List(layers)
        self.value_layers = nnx.List(value_layers)

        self.final_norm = nnx.RMSNorm(
            config.embed,
            epsilon=config.norm_eps,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )

        self.final_norm_value = nnx.RMSNorm(
            value_layer_config.embed,
            epsilon=config.norm_eps,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )
        self.final_norm_value.scale = ValueParam(self.final_norm_value.scale[:])

        self.value_net = ValueNetwork(value_layer_config.embed, 512, rngs=rngs)

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
        value_x = x[:, :, :256]

        if carry is not None:
            out_carry = []
            for i, (layer, value_layer, (layer_carry_in, value_carry_in)) in enumerate(zip(self.layers, self.value_layers, carry)):
                value_x, value_carry_out = value_layer(value_x, positions, value_carry_in)

                x, layer_carry_out = layer(x, positions, layer_carry_in)
                out_carry.append((layer_carry_out, value_carry_out))

            carry = tuple(out_carry)
        else:
            for i, (layer, value_layer) in enumerate(zip(self.layers, self.value_layers)):
                value_x, _ = value_layer(value_x, positions)

                x, _ = jax.checkpoint(layer)(x, positions)

        x = self.final_norm(x)
        value_x = self.final_norm_value(value_x)
        logits = x @ self.embeddings.embedding.T

        logits = logits.astype(jnp.float32)

        value = self.value_net(value_x)

        return logits, value, carry

    def initialize_carry(self, batch_size: int, seq_length: int):
        return tuple(
            (layer.initialize_carry(batch_size, seq_length), value_layer.initialize_carry(batch_size, seq_length)) for layer, value_layer in zip(self.layers, self.value_layers)
        )

from llmrl.model.value_network import ValueParam
from typing import Any

import jax
from flax import nnx
from jax import numpy as jnp
from llmrl.config import LLMConfig, LoraConfig, ValueConfig
from llmrl.model.attention import KVCache
from llmrl.model.layer import Qwen3Layer
from llmrl.model.util import load_param
from llmrl.model.value_network import ValueBackbone
from llmrl.model.util import wrap_param


class ValueLayer(nnx.Module):
    def __init__(self, config: LLMConfig, in_latent: int, *, rngs: nnx.Rngs):
        encode_rank = 64

        self.layer = Qwen3Layer(config=config, rngs=rngs)
        self.in_norm = nnx.RMSNorm(in_latent, rngs=rngs)
        self.in_proj = nnx.Linear(in_latent, encode_rank, rngs=rngs)
        self.in_up_proj = nnx.Linear(encode_rank, config.embed, rngs=rngs)

    def initialize_carry(self, batch_size: int, seq_length: int):
        return self.layer.initialize_carry(batch_size, seq_length)

    def __call__(self, latent: jax.Array, value_latent: jax.Array, positions: jax.Array, carry: KVCache | None = None):
        in_latents = self.in_up_proj(nnx.silu(self.in_proj(self.in_norm(jax.lax.stop_gradient(latent)))))
        return self.layer(value_latent + in_latents, positions, carry)

class Qwen3(nnx.Module):
    def __init__(
        self,
        config: LLMConfig,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self._embed = config.embed
        self._head_dim = config.head_dim
        self._rope_theta = config.rope_theta

        self.embeddings = nnx.Embed(
            config.vocab_size,
            config.embed,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )

        self.layers = nnx.List([Qwen3Layer(
            config=config,
            rngs=rngs,
        ) for _ in range(config.num_layers)])

        self.final_norm = nnx.RMSNorm(
            config.embed,
            epsilon=config.norm_eps,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )

        self.value_net = None

    def initalize_value_net(self, value_config: ValueConfig, *, rngs: nnx.Rngs):
        self.value_net = ValueBackbone(value_config, self._embed, rngs=rngs)
        wrap_param(self.value_net, ValueParam)

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
        carry: Any = None,
    ) -> tuple[jax.Array, jax.Array, Any]:
        x = self.embeddings(tokens)

        if carry is not None:
            base_carry, value_carry = carry

            out_carry = []
            latents = [x]
            for layer, carry_in in zip(self.layers, base_carry):
                x, layer_carry_out = layer(x, positions, carry_in)
                out_carry.append(layer_carry_out)
                latents.append(x)

            base_carry = tuple(out_carry)

            if self.value_net is not None:
                value_repr, value_carry = self.value_net(latents, positions, value_carry)
            carry = base_carry, value_carry
        else:
            latents = [x]
            for layer in self.layers:
                x, _ = jax.checkpoint(layer)(x, positions)
                latents.append(x)

            if self.value_net is not None:
                value_repr, _ = self.value_net(latents, positions)

        x = self.final_norm(x)
        logits = x @ self.embeddings.embedding.T

        logits = logits.astype(jnp.float32)

        return logits, value_repr, carry

    def initialize_carry(self, batch_size: int, seq_length: int):
        base_carry = tuple(layer.initialize_carry(batch_size, seq_length) for layer in self.layers)
        value_carry = self.value_net.initialize_carry(batch_size, seq_length) if self.value_net is not None else None
        return base_carry, value_carry

    def get_value(self, repr: jax.Array) -> jax.Array:
        return self.value_net.get_value(repr)

    def get_value_loss(self, repr: jax.Array, target_values: jax.Array) -> jax.Array:
        return self.value_net.get_loss(repr, target_values)

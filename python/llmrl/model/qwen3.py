from llmrl.model.value_network import ValueParam
from typing import Any

import jax
from flax import nnx
from jax import numpy as jnp
from llmrl.config import LLMConfig, LoraConfig
from llmrl.model.attention import KVCache
from llmrl.model.layer import Qwen3Layer
from llmrl.model.util import load_param
from llmrl.model.value_network import ValueNetwork


def wrap_param(node: nnx.Module, param):
    for path, value in nnx.iter_graph(node):
        if isinstance(value, nnx.Param):
            *path, key = path

            target = node
            for p in path:
                if isinstance(p, int):
                    target = target[p]
                else:
                    target = getattr(target, p)

            setattr(target, key, param(value[...]))

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
        self._head_dim = config.head_dim
        self._rope_theta = config.rope_theta

        value_layer_config = config.model_copy(
            update={
                "embed": 256,
                "mlp_ffw_size": 256*2,
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
            value_layer = ValueLayer(
                config=value_layer_config,
                in_latent=config.embed,
                rngs=rngs,
            )
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
        self.value_net = ValueNetwork(value_layer_config.embed, 512, rngs=rngs)

        wrap_param(self.value_layers, ValueParam)
        wrap_param(self.final_norm_value, ValueParam)
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
        carry: tuple[KVCache, ...] | None = None,
    ) -> tuple[jax.Array, jax.Array, tuple[KVCache, ...] | None]:
        x = self.embeddings(tokens)
        value_x = x[:, :, :256]

        if carry is not None:
            out_carry = []
            for i, (layer, value_layer, (layer_carry_in, value_carry_in)) in enumerate(zip(self.layers, self.value_layers, carry)):
                x, layer_carry_out = layer(x, positions, layer_carry_in)
                value_x, value_carry_out = value_layer(x, value_x, positions, value_carry_in)

                out_carry.append((layer_carry_out, value_carry_out))

            carry = tuple(out_carry)
        else:
            for i, (layer, value_layer) in enumerate(zip(self.layers, self.value_layers)):
                x, _ = jax.checkpoint(layer)(x, positions)
                value_x, _ = jax.checkpoint(value_layer)(x, value_x, positions)

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

    def get_value(self, repr: jax.Array) -> jax.Array:
        return self.value_net.output.get_value(repr)

    def get_value_loss(self, repr: jax.Array, target_values: jax.Array) -> jax.Array:
        return self.value_net.output.get_loss(repr, target_values)

import math
from typing import Any, NamedTuple

import jax
from flax import nnx
from jax import numpy as jnp

from llmrl.config import Config, LoraConfig
from llmrl.rope import apply_rope


def _load_param(target: nnx.Param[jax.Array], value):
    value = jnp.asarray(value, device=target.device)
    assert value.shape == target.shape
    assert value.dtype == target.dtype
    target[...] = value


class LoRAGeneral(nnx.Module):
    def __init__(
        self,
        in_features: int | tuple[int, ...],
        rank: int,
        out_features: int | tuple[int, ...],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        def prod_features(feat):
            return feat if isinstance(feat, int) else math.prod(feat)

        prod_in = prod_features(in_features)
        prod_out = prod_features(out_features)

        self._out_shape = (
            (out_features,) if isinstance(out_features, int) else out_features
        )

        self.lora = nnx.LoRA(
            prod_in,
            rank,
            prod_out,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        batch = x.shape[0]
        seq_length = x.shape[1]

        x = x.reshape(batch, seq_length, -1)
        x = self.lora(x)
        x = x.reshape(batch, seq_length, *self._out_shape)

        return x


class MlpLayer(nnx.Module):
    def __init__(self, config: Config, *, rngs: nnx.Rngs):
        super().__init__()
        self._embed_dim = config.embed
        self._ffw_dim = config.mlp_ffw_size

        self.up_gate = nnx.Linear(
            config.embed,
            config.mlp_ffw_size,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            config.embed,
            config.mlp_ffw_size,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            config.mlp_ffw_size,
            config.embed,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self._use_lora = False

    def initialize_lora(self, lora_config: LoraConfig, *, rngs: nnx.Rngs):
        if not lora_config.mlp_lora:
            self._use_lora = False
            return

        self._use_lora = True
        self.up_gate_lora = nnx.LoRA(
            self._embed_dim,
            lora_config.rank,
            self._ffw_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )
        self.up_proj_lora = nnx.LoRA(
            self._embed_dim,
            lora_config.rank,
            self._ffw_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )
        self.down_proj_lora = nnx.LoRA(
            self._ffw_dim,
            lora_config.rank,
            self._embed_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )

    def load_params(self, params):
        # pass in the mlp dict
        _load_param(self.up_gate.kernel, params["gate_proj"]["weight"].T)
        _load_param(self.up_proj.kernel, params["up_proj"]["weight"].T)
        _load_param(self.down_proj.kernel, params["down_proj"]["weight"].T)

    def __call__(self, inputs):
        up = self.up_proj(inputs)
        gate_in = self.up_gate(inputs)

        if self._use_lora:
            up = up + self.up_proj_lora(inputs)
            gate_in = gate_in + self.up_gate_lora(inputs)

        down_in = up * jax.nn.silu(gate_in)
        out = self.down_proj(down_in)

        if self._use_lora:
            out = out + self.down_proj_lora(down_in)

        return out


class KVCache(NamedTuple):
    key: jax.Array
    value: jax.Array
    # length: jax.Array


class AttentionLayer(nnx.Module):
    def __init__(self, config: Config, *, rngs: nnx.Rngs) -> None:
        super().__init__()

        self._num_kv_heads = config.kv_heads
        self._q_heads = config.q_heads
        self._embed_dim = config.embed
        self._head_dim = config.head_dim
        self._rope_theta = config.rope_theta

        self.key_proj = nnx.LinearGeneral(
            in_features=config.embed,
            out_features=(config.kv_heads, config.head_dim),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self.value_proj = nnx.LinearGeneral(
            in_features=config.embed,
            out_features=(config.kv_heads, config.head_dim),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self.query_proj = nnx.LinearGeneral(
            in_features=config.embed,
            out_features=(config.q_heads, config.head_dim),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self.out = nnx.LinearGeneral(
            in_features=(config.q_heads, config.head_dim),
            out_features=config.embed,
            axis=(-2, -1),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self.query_norm = nnx.RMSNorm(
            config.head_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            epsilon=config.norm_eps,
            rngs=rngs,
        )
        self.key_norm = nnx.RMSNorm(
            config.head_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            epsilon=config.norm_eps,
            rngs=rngs,
        )

        self._use_lora = False

    def initialize_lora(self, lora_config: LoraConfig, *, rngs: nnx.Rngs):
        if not lora_config.attn_lora:
            self._use_lora = False
            return

        self._use_lora = True
        self.key_proj_lora = LoRAGeneral(
            self._embed_dim,
            lora_config.rank,
            (self._num_kv_heads, self._head_dim),
            rngs=rngs,
        )
        self.value_proj_lora = LoRAGeneral(
            self._embed_dim,
            lora_config.rank,
            (self._num_kv_heads, self._head_dim),
            rngs=rngs,
        )
        self.query_proj_lora = LoRAGeneral(
            self._embed_dim,
            lora_config.rank,
            (self._q_heads, self._head_dim),
            rngs=rngs,
        )
        self.out_lora = LoRAGeneral(
            (self._q_heads, self._head_dim),
            lora_config.rank,
            self._embed_dim,
            rngs=rngs,
        )

    def initialize_carry(self, batch_size: int, seq_length: int):
        shape = (batch_size, seq_length, self._num_kv_heads, self._head_dim)
        key = jnp.zeros(shape, dtype=jnp.bfloat16)
        value = jnp.zeros(shape, dtype=jnp.bfloat16)

        return KVCache(key, value)

    def _update_carry(
        self,
        carry: KVCache,
        positions: jax.Array,
        key_update: jax.Array,
        value_update: jax.Array,
    ) -> KVCache:
        scatter_indices = positions

        dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(1, 2),
            inserted_window_dims=(1,),
            scatter_dims_to_operand_dims=(1,),
            operand_batching_dims=(0,),
            scatter_indices_batching_dims=(0,),
        )

        new_key = jax.lax.scatter(
            carry.key,
            scatter_indices,
            key_update.squeeze(1),
            dnums,
            unique_indices=True,
        )
        new_value = jax.lax.scatter(
            carry.value,
            scatter_indices,
            value_update.squeeze(1),
            dnums,
            unique_indices=True,
        )

        return KVCache(new_key, new_value)

    def __call__(
        self, inputs: jax.Array, positions: jax.Array, carry: KVCache | None = None
    ) -> tuple[jax.Array, KVCache | None]:
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)
        query = self.query_proj(inputs)

        if self._use_lora:
            key = key + self.key_proj_lora(inputs)
            value = value + self.value_proj_lora(inputs)
            query = query + self.query_proj_lora(inputs)

        key = self.key_norm(key)
        query = self.query_norm(query)

        key = apply_rope(key, positions, self._head_dim, self._rope_theta)
        query = apply_rope(query, positions, self._head_dim, self._rope_theta)

        if carry is not None:
            carry = self._update_carry(carry, positions, key, value)

            x = jax.nn.dot_product_attention(
                query,
                carry.key,
                carry.value,
                key_value_seq_lengths=positions.squeeze(-1) + 1,
                implementation="cudnn",
            )
        else:
            x = jax.nn.dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                implementation="cudnn",
            )

        out = self.out(x)
        if self._use_lora:
            out = out + self.out_lora(x)

        return out, carry

    def load_params(self, params):
        k_proj = params["k_proj"]["weight"].T.reshape(self.key_proj.kernel.shape)
        q_proj = params["q_proj"]["weight"].T.reshape(self.query_proj.kernel.shape)
        v_proj = params["v_proj"]["weight"].T.reshape(self.value_proj.kernel.shape)
        o_proj = params["o_proj"]["weight"].T.reshape(self.out.kernel.shape)

        _load_param(self.key_proj.kernel, k_proj)
        _load_param(self.query_proj.kernel, q_proj)
        _load_param(self.value_proj.kernel, v_proj)
        _load_param(self.out.kernel, o_proj)

        _load_param(self.query_norm.scale, params["q_norm"]["weight"])
        _load_param(self.key_norm.scale, params["k_norm"]["weight"])


class Qwen3Layer(nnx.Module):
    def __init__(self, config: Config, *, rngs: nnx.Rngs):
        super().__init__()
        self.attn = AttentionLayer(config, rngs=rngs)
        self.mlp = MlpLayer(config, rngs=rngs)

        self.attn_pre_norm = nnx.RMSNorm(
            config.embed,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            epsilon=config.norm_eps,
            rngs=rngs,
        )
        self.attn_post_norm = nnx.RMSNorm(
            config.embed,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            epsilon=config.norm_eps,
            rngs=rngs,
        )

    def initialize_lora(self, lora_config: LoraConfig, *, rngs: nnx.Rngs):
        self.attn.initialize_lora(lora_config, rngs=rngs)
        self.mlp.initialize_lora(lora_config, rngs=rngs)

    def __call__(
        self, inputs: jax.Array, positions: jax.Array, carry: KVCache | None = None
    ) -> tuple[jax.Array, KVCache | None]:
        attn_in = self.attn_pre_norm(inputs)
        attn_out, carry = self.attn(attn_in, positions, carry)
        x = inputs + attn_out

        ff_in = self.attn_post_norm(x)
        ff_out = self.mlp(ff_in)
        x = x + ff_out

        return x, carry

    def initialize_carry(self, batch_size: int, seq_length: int):
        return self.attn.initialize_carry(batch_size, seq_length)

    def load_params(self, params):
        _load_param(self.attn_pre_norm.scale, params["input_layernorm"]["weight"])
        _load_param(
            self.attn_post_norm.scale, params["post_attention_layernorm"]["weight"]
        )
        self.attn.load_params(params["self_attn"])
        self.mlp.load_params(params["mlp"])


class ValueNetwork(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, *, rngs: nnx.Rngs):
        super().__init__()

        self.up = nnx.Linear(in_features, hidden_features, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=rngs)
        self.down = nnx.Linear(hidden_features, 1, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=rngs)        

    def __call__(self, x):
        x = self.up(x)
        x = jax.nn.silu(x)
        x = self.down(x)
        return x


class Qwen3(nnx.Module):
    def __init__(
        self,
        config: Config,
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

        self.value_net = ValueNetwork(config.embed, 512, rngs=rngs)

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

        _load_param(self.final_norm.scale, params["model"]["norm"]["weight"])

    def __call__(
        self,
        tokens: jax.Array,
        positions: jax.Array,
        carry: tuple[KVCache, ...] | None = None,
    ) -> tuple[jax.Array, tuple[KVCache, ...] | None]:
        x = self.embeddings(tokens)

        if carry is not None:
            out_carry = []
            for layer, layer_carry_in in zip(self.layers, carry):
                x, layer_carry_out = layer(x, positions, layer_carry_in)
                out_carry.append(layer_carry_out)
            carry = tuple(out_carry)
        else:
            for layer in self.layers:
                x, _ = layer(x, positions)

        x = self.final_norm(x)
        logits = x @ self.embeddings.embedding.T

        logits = logits.astype(jnp.float32)

        value = self.value_net(x)

        return logits, value, carry

    def initialize_carry(self, batch_size: int, seq_length: int):
        return tuple(
            layer.initialize_carry(batch_size, seq_length) for layer in self.layers
        )

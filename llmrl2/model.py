from typing import Any
import jax
from jax import numpy as jnp
from flax import nnx
from llmrl2.config import Config

import tensorflow_probability.substrates.jax.distributions as tfd

from llmrl2.rope import apply_rotary_embedding, generate_pos_embeddings


def _load_param(target: nnx.Param[jax.Array], value):
    value = jnp.asarray(value, device=target.value.device)
    assert value.shape == target.value.shape
    assert value.dtype == target.value.dtype
    target.value = value


class MlpLayer(nnx.Module):
    def __init__(self, config: Config, *, rngs: nnx.Rngs):
        super().__init__()

        self.up_gate = nnx.Linear(
            config.embed,
            config.mlp_ffw_size,
            dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            config.embed,
            config.mlp_ffw_size,
            dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            config.mlp_ffw_size,
            config.embed,
            dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self.down_proj.kernel.value.device

    def load_params(self, params):
        # pass in the mlp dict
        _load_param(self.up_gate.kernel, params["gate_proj"]["weight"].T)
        _load_param(self.up_proj.kernel, params["up_proj"]["weight"].T)
        _load_param(self.down_proj.kernel, params["down_proj"]["weight"].T)

    def __call__(self, inputs):
        up = self.up_proj(inputs)
        gate = jax.nn.silu(self.up_gate(inputs))
        out = self.down_proj(up * gate)
        return out
    

class AttentionLayer(nnx.Module):
    def __init__(self, config: Config, *, rngs: nnx.Rngs) -> None:
        super().__init__()

        self.key_proj = nnx.LinearGeneral(
            in_features=config.embed,
            out_features=(config.kv_heads, config.head_dim),
            dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self.value_proj = nnx.LinearGeneral(
            in_features=config.embed,
            out_features=(config.kv_heads, config.head_dim),
            dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self.query_proj = nnx.LinearGeneral(
            in_features=config.embed,
            out_features=(config.q_heads, config.head_dim),
            dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self.out = nnx.LinearGeneral(
            in_features=(config.q_heads, config.head_dim),
            out_features=config.embed,
            axis=(-2, -1),
            dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
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

    def __call__(self, inputs, sin, cos) -> Any:
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)
        query = self.query_proj(inputs)

        key = self.key_norm(key)
        query = self.query_norm(query)

        key = apply_rotary_embedding(key, sin, cos)
        query = apply_rotary_embedding(query, sin, cos)

        x = jax.nn.dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            implementation="cudnn",
        )

        out = self.out(x)
        return out

    def load_params(self, params):
        k_proj = params["k_proj"]["weight"].T.reshape(self.key_proj.kernel.value.shape)
        q_proj = params["q_proj"]["weight"].T.reshape(self.query_proj.kernel.value.shape)
        v_proj = params["v_proj"]["weight"].T.reshape(self.value_proj.kernel.value.shape)
        o_proj = params["o_proj"]["weight"].T.reshape(self.out.kernel.value.shape)

        _load_param(self.key_proj.kernel, k_proj)
        _load_param(self.query_proj.kernel, q_proj)
        _load_param(self.value_proj.kernel, v_proj)
        _load_param(self.out.kernel, o_proj)

        _load_param(self.query_norm.scale, params["q_norm"]["weight"])
        _load_param(self.key_norm.scale, params["k_norm"]["weight"])


class Qwen3Layer(nnx.Module):
    def __init__(
        self,
        config: Config,
        *,
        rngs: nnx.Rngs
    ):
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

    def __call__(self, x, sin, cos) -> Any:
        attn_in = self.attn_pre_norm(x)
        attn_out = self.attn(attn_in, sin, cos)
        x = x + attn_out

        ff_in = self.attn_post_norm(x)
        ff_out = self.mlp(ff_in)
        x = x + ff_out

        return x, None

    def load_params(self, params):
        _load_param(self.attn_pre_norm.scale, params["input_layernorm"]["weight"])
        _load_param(self.attn_post_norm.scale, params["post_attention_layernorm"]["weight"])
        self.attn.load_params(params["self_attn"])
        self.mlp.load_params(params["mlp"])


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

        self.embeddings = nnx.Embed(config.vocab_size, config.embed, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=rngs)

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
        self.lm_head = nnx.Linear(
            config.embed, config.vocab_size, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=rngs
        )
        
    def load_params(self, params: dict[str, Any]):
        embed_params = jnp.asarray(
            params['model']['embed_tokens']['weight'],
            device=self.embeddings.embedding.value.device
        )
        print(embed_params.shape)
        print(self.embeddings.embedding.value.shape)
        assert embed_params.shape == self.embeddings.embedding.value.shape

        self.embeddings.embedding.value = embed_params

        for i, layer in enumerate(self.layers):
            layer_params = params["model"]["layers"][f"{i}"]
            layer.load_params(layer_params)
        
        _load_param(self.final_norm.scale, params["model"]["norm"]["weight"])
        _load_param(self.lm_head.kernel, params["lm_head"]["weight"].T)

    def initialize_carry(self, batch_size: int, rngs):
        return tuple(layer.initialize_carry(batch_size, rngs) for layer in self.layers)

    def __call__(
        self, tokens: jax.Array, carry=None
    ): # -> tuple[jax.Array, tfd.Distribution, tuple[KVCache, ...] | None]:
        x = self.embeddings(tokens)

        positions = jnp.repeat(jnp.arange(tokens.shape[1])[None, :], tokens.shape[0], axis=0)
        sin, cos = generate_pos_embeddings(positions, self._head_dim, self._rope_theta)  # [B, T, head_dim]

        if carry is not None:
            out_carry = []
            for layer, _carry in zip(self.layers, carry):
                x, _carry = layer(x, _carry)
                out_carry.append(_carry)
            carry = tuple(out_carry)
        else:
            for layer in self.layers:
                x, _ = layer(x, sin, cos)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        # probs = tfd.Categorical(logits=logits)

        return logits

        # if carry is not None:
        #     out_carry = []
        #     for layer, _carry in zip(self.layers, carry):
        #         x, _carry = layer(x, ts.time, _carry)
        #         out_carry.append(_carry)
        #     carry = tuple(out_carry)
        # else:
        #     for layer in self.layers:
        #         x, _ = layer(x, ts.time)

        # x = self.output_norm(x)

        # action_logits = self.action_embedder.decode(x)

        # action_logits = action_logits.astype(jnp.float32)

        # policy = tfd.Categorical(logits=action_logits)

        # return policy, carry

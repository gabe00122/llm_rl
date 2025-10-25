from typing import Any
import jax
from jax import numpy as jnp
from flax import nnx
from llmrl2.config import Config

import tensorflow_probability.substrates.jax.distributions as tfd


def _load_param(target: nnx.Param[jax.Array], value):
    value = jnp.asarray(value, device=target.value.device)
    assert value.shape == target.value.shape
    target.value = value


class MlpLayer(nnx.Module):
    def __init__(self, config: Config, *, rngs: nnx.Rngs):
        super().__init__()

        self.up_gate = nnx.Linear(
            config.embed,
            config.mlp_ffw_size,
            use_bias=False,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            config.embed,
            config.mlp_ffw_size,
            use_bias=False,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            config.mlp_ffw_size,
            config.embed,
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
            use_bias=False,
            rngs=rngs,
        )

        self.value_proj = nnx.LinearGeneral(
            in_features=config.embed,
            out_features=(config.kv_heads, config.head_dim),
            use_bias=False,
            rngs=rngs,
        )

        self.query_proj = nnx.LinearGeneral(
            in_features=config.embed,
            out_features=(config.q_heads, config.head_dim),
            use_bias=False,
            rngs=rngs,
        )

        self.out = nnx.LinearGeneral(
            in_features=(config.q_heads, config.head_dim),
            out_features=config.embed,
            axis=(-2, -1),
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, inputs) -> Any:
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)
        query = self.query_proj(inputs)

        x = jax.nn.dot_product_attention(
            query,
            key,
            value,
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

    def __call__(self, x) -> Any:
        attn_out = self.attn(x)
        x = x + attn_out

        ff_out = self.mlp(x)
        x = x + ff_out

        return x, None

    def load_params(self, params):
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
        self.embeddings = nnx.Embed(config.vocab_size, config.embed, rngs=rngs)

        layers = []
        for _ in range(config.num_layers):
            layers.append(
                Qwen3Layer(
                    config=config,
                    rngs=rngs,
                )
            )
        self.layers = nnx.List(layers)
        
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

    def initialize_carry(self, batch_size: int, rngs):
        return tuple(layer.initialize_carry(batch_size, rngs) for layer in self.layers)

    def __call__(
        self, tokens: jax.Array, carry=None
    ): # -> tuple[jax.Array, tfd.Distribution, tuple[KVCache, ...] | None]:
        x = self.embeddings(tokens)

        if carry is not None:
            out_carry = []
            for layer, _carry in zip(self.layers, carry):
                x, _carry = layer(x, _carry)
                out_carry.append(_carry)
            carry = tuple(out_carry)
        else:
            for layer in self.layers:
                x, _ = layer(x)

        return x

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

import json
import os
from pathlib import Path
import jax
from jax import numpy as jnp
from typing import Any, NamedTuple


class Config(NamedTuple):
    embed: int
    q_heads: int
    kv_heads: int
    num_layers: int
    head_dim: int
    vocab_size: int
    max_seq_len: int
    # Attention
    causal: bool
    # MoE
    moe_ffw_size: int
    moe_experts_per_tok: int
    moe_num_experts: int
    moe_gate_dtype: "jnp.dtype" = jnp.float32
    ep_strategy: str = "decode"
    # MLP
    mlp_ffw_size: int = -1
    mlp_layer_idxs: list[int] = []
    # kernel config
    use_prefill_attn_kernel: bool = False
    use_decode_attn_kernel: bool = False
    use_ragged_dot_kernel: bool = False
    dtype: "jnp.dtype" = jnp.bfloat16
    norm_eps: float = 1e-6
    # sharding
    mesh: jax.sharding.Mesh | None = None
    rope_theta: float = 500000.0
    quant_moe: bool = False
    quant_mlp: bool = False
    quant_attn: bool = False
    quant_cache: bool = True
    quant_scale_dtype: "jnp.dtype" = jnp.bfloat16


def hf_to_jax_config(hf_config: Any | dict[str, Any]) -> "Config":
    def _get(x, k, default=None):
        return (
            getattr(x, k, default)
            if not isinstance(hf_config, dict)
            else hf_config.get(k, default)
        )

    return Config(
        embed=_get(hf_config, "hidden_size"),
        mlp_ffw_size=_get(hf_config, "intermediate_size", -1),
        moe_ffw_size=_get(hf_config, "moe_intermediate_size", -1),
        mlp_layer_idxs=_get(hf_config, "mlp_only_layers", []),
        q_heads=_get(hf_config, "num_attention_heads"),
        kv_heads=_get(hf_config, "num_key_value_heads"),
        num_layers=_get(hf_config, "num_hidden_layers"),
        head_dim=_get(hf_config, "head_dim"),
        vocab_size=_get(hf_config, "vocab_size"),
        norm_eps=_get(hf_config, "rms_norm_eps"),
        moe_experts_per_tok=_get(hf_config, "num_experts_per_tok"),
        moe_num_experts=_get(hf_config, "num_experts"),
        max_seq_len=128,
        dtype=jnp.bfloat16,
        causal=True,
        use_prefill_attn_kernel=False,
        use_decode_attn_kernel=False,
        rope_theta=_get(hf_config, "rope_theta"),
    )


def load_config(config_path: str | os.PathLike[str] | Path) -> "Config":
    return hf_to_jax_config(json.loads(Path(config_path).read_text()))


class SamplingConfig(NamedTuple):
    temperature: float
    top_k: int
    top_p: float


def load_sampling_config(config_path: str | os.PathLike[str] | Path) -> SamplingConfig:
    with open(config_path, "r") as f:
        data: dict[str, Any] = json.load(f)

    temperature = data.get("temperature", 1.0)
    top_k = data.get("top_k", 20)
    top_p = data.get("top_p", 1.0)

    return SamplingConfig(temperature, top_k, top_p)


class LoraConfig(NamedTuple):
    mlp_lora: bool
    attn_lora: bool
    rank: int

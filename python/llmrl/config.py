import json
import os
from pathlib import Path
from jax import numpy as jnp
from typing import Any, NamedTuple


class LLMConfig(NamedTuple):
    embed: int
    q_heads: int
    kv_heads: int
    num_layers: int
    head_dim: int
    vocab_size: int
    # MLP
    mlp_ffw_size: int = -1
    # kernel config
    norm_eps: float = 1e-6
    rope_theta: float = 500000.0


def parse_hf_llm_config(hf_config: Any | dict[str, Any]) -> "LLMConfig":
    def _get(x, k, default=None):
        return (
            getattr(x, k, default)
            if not isinstance(hf_config, dict)
            else hf_config.get(k, default)
        )

    return LLMConfig(
        embed=_get(hf_config, "hidden_size"),
        mlp_ffw_size=_get(hf_config, "intermediate_size", -1),
        q_heads=_get(hf_config, "num_attention_heads"),
        kv_heads=_get(hf_config, "num_key_value_heads"),
        num_layers=_get(hf_config, "num_hidden_layers"),
        head_dim=_get(hf_config, "head_dim"),
        vocab_size=_get(hf_config, "vocab_size"),
        norm_eps=_get(hf_config, "rms_norm_eps"),
        rope_theta=_get(hf_config, "rope_theta"),
    )


def load_hf_llm_config(config_path: str | os.PathLike[str] | Path) -> "LLMConfig":
    return parse_hf_llm_config(json.loads(Path(config_path).read_text()))


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


class LoggerConfig(NamedTuple):
    project_name: str = "llmrl"
    use_tb: bool = False
    use_console: bool = True
    use_wandb: bool = False

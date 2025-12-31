import json
import random
from typing import Literal
from pydantic import BaseModel, ConfigDict

# Base Model Config
class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
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


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    temperature: float
    top_k: int
    top_p: float

# experiment config
class LoraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mlp: bool = False
    attn: bool = False
    rank: int = 0


class LoggerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project_name: str = "llmrl"
    use_tb: bool = False
    use_console: bool = True
    use_wandb: bool = False


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int | Literal["random"] = "random"
    base_model: str
    lora: LoraConfig
    logger: LoggerConfig

    num_env: int


def load_config(json_config: str) -> Config:
    config = Config.model_validate(json.loads(json_config), strict=True)
    if config.seed == "random":
        config = config.model_copy(update={"seed": random.getrandbits(31)})

    return config

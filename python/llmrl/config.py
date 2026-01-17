import json
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Environment Config


class ArithmeticEnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Literal["arithmetic"] = "arithmetic"
    max_x: int
    max_y: int


class WordleEnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Literal["wordle"] = "wordle"
    max_guesses: int


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


class ValueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    width: int


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    eval_env: int
    update_envs: int
    seq_length: int


class LoggerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project_name: str = "llmrl"
    use_tb: bool = False
    use_console: bool = True
    use_wandb: bool = False


class AdamWConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["adamw"] = "adamw"
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01


class WarmupCosineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["warmup_cosine"] = "warmup_cosine"
    warmup_ratio: float = 0.1


OptimizerConfig = AdamWConfig
ScheduleConfig = WarmupCosineConfig | None


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    gae_lambda: float
    gae_discount: float
    vf_coef: float
    pg_clip_high: float
    pg_clip_low: float


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int | Literal["random"] = "random"
    base_model: str
    lora: LoraConfig
    logger: LoggerConfig
    optimizer: OptimizerConfig
    schedule: ScheduleConfig = None
    multi_step: int | None = None
    loss: LossConfig
    env: ArithmeticEnvConfig | WordleEnvConfig = Field(discriminator="name")

    eval_envs: int
    update_envs: int
    max_seq_length: int
    total_update_episodes: int
    checkpoint_every: int


def load_config(json_config: str) -> Config:
    config = Config.model_validate(json.loads(json_config), strict=True)
    if config.seed == "random":
        config = config.model_copy(update={"seed": random.getrandbits(31)})

    return config

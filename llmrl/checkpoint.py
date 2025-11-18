from __future__ import annotations
from pathlib import Path

from safetensors import safe_open
from rich.progress import track

from flax import nnx

from llmrl.config import LoraConfig, load_config, load_sampling_config
from llmrl.model import Qwen3
from llmrl.util import load_tokenizer


def _put_path(data: dict, path: list[str], value) -> None:
    """Insert `value` into a nested dict following `path` segments."""
    head, *tail = path
    if not tail:
        data[head] = value
        return
    child = data.setdefault(head, {})
    _put_path(child, tail, value)


def load_param_dict(params: dict[str, object], file_path: Path):
    """Load a safetensors checkpoint into a nested python dict."""
    with safe_open(file_path, framework="np") as f:
        for key in track(f.keys(), description="Loading weights"):
            key_path = key.split(".")
            value = f.get_tensor(key)
            _put_path(params, key_path, value)

def load_safetensors(file_path: str):
    params: dict[str, object] = {}

    files = list(Path(file_path).glob("**/*safetensors"))
    for file in files:
        load_param_dict(params, file)

    return params

def load_model(model_path: str, lora_config: LoraConfig, rngs: nnx.Rngs):
    model_path = "./base-models/Qwen3-4B-Instruct-2507"

    # model_path = "./base-models/qwen3-0.6b"
    config = load_config(f"{model_path}/config.json")
    params = load_safetensors(model_path)
    tokenizer = load_tokenizer(model_path)
    sampling = load_sampling_config(f"{model_path}/generation_config.json")

    model = Qwen3(config, lora_config, rngs=rngs)
    model.load_params(params)
    
    return model, tokenizer, sampling
from __future__ import annotations

from pathlib import Path

import jax
import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx.filterlib import Filter
from jax.sharding import Mesh
from rich.progress import track
from safetensors import safe_open

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


class Checkpointer:
    def __init__(self, directory: str):
        if not directory.startswith("gs://"):
            directory = Path(directory).absolute().as_posix()
        self.mngr = ocp.CheckpointManager(directory)

    def save(self, model: object, global_step: int, param_filter: Filter = nnx.Param):
        state = nnx.state(model, param_filter)
        self.mngr.save(global_step, args=ocp.args.StandardSave(state))

    def restore[T](self, model: T, step: int, param_filter: Filter = nnx.Param) -> T:
        device = jax.devices()[0]
        mesh = Mesh((device,), ("batch",))

        target_state = nnx.state(model, param_filter)
        abstract_state = jax.tree.map(
            lambda x, s: jax.ShapeDtypeStruct(
                shape=x.shape, 
                dtype=x.dtype, 
                sharding=s
            ),
            target_state, nnx.get_named_sharding(target_state, mesh)
        )

        restored_state = self.mngr.restore(
            step, args=ocp.args.StandardRestore(abstract_state)
        )

        nnx.update(model, restored_state)
        
        return model

    def restore_latest[T](self, model: T, param_filter: Filter = nnx.Param) -> T:
        return self.restore(model, self.mngr.latest_step() or 0, param_filter)

    def close(self):
        self.mngr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

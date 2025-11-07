from __future__ import annotations
from pathlib import Path

from safetensors import safe_open
from rich.progress import track


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

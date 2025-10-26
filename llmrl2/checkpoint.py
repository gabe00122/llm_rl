from __future__ import annotations

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


def load_param_dict(file_path: str) -> dict:
    """Load a safetensors checkpoint into a nested python dict."""
    params: dict[str, object] = {}
    with safe_open(file_path, framework="np") as f:
        for key in track(f.keys(), description="Loading weights"):
            key_path = key.split(".")
            value = f.get_tensor(key)
            _put_path(params, key_path, value)
    return params

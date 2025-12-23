import json
import os
import jax
from jax import numpy as jnp

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from zipp import Path


def load_tokenizer(
    tokenizer_path: str | os.PathLike[str] | Path,
) -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(tokenizer_path)


# ideally this could be tested and used as a general scatter for updating data in the KV cache and the rollout, or something similar to it
def batched_put(
    target: jax.Array, indices: jax.Array, values: jax.Array
) -> jax.Array:
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=range(2, values.ndim),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,),
        operand_batching_dims=(0,),
        scatter_indices_batching_dims=(0,),
    )

    return jax.lax.scatter(
        target,
        indices[..., None],
        values,
        dnums,
        indices_are_sorted=True,
        unique_indices=True,
    )

def batched_get():
    pass

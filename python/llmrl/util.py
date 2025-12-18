import json
import os
import jax

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from zipp import Path


def load_tokenizer(
    tokenizer_path: str | os.PathLike[str] | Path
) -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(tokenizer_path)


# ideally this could be tested and used as a general scatter for updating data in the KV cache and the rollout, or something similar to it
def batched_scatter(target: jax.Array, indices: jax.Array, update: jax.Array) -> jax.Array:
    update_dims = tuple(range(1, len(update.shape)))
    
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=update_dims,
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,),
        operand_batching_dims=(0,),
        scatter_indices_batching_dims=(0,),
    )

    return jax.lax.scatter(
        target,
        indices,
        update.squeeze(1), # maybe this should happen in the caller
        dnums,
        unique_indices=True,
    )


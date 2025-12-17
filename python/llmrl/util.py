import json
import os
import jax

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from zipp import Path


def load_tokenizer(
    tokenizer_path: str | os.PathLike[str] | Path
) -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(tokenizer_path)


def batched_scatter(target: jax.Array, indices: jax.Array, update: jax.Array) -> jax.Array:
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,),
        operand_batching_dims=(0,),
        scatter_indices_batching_dims=(0,),
    )

    return jax.lax.scatter(
        target,
        indices,
        update.squeeze(1),
        dnums,
        unique_indices=True,
    )


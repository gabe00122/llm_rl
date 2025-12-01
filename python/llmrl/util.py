import json
import os

from transformers import AutoTokenizer
from zipp import Path


def load_tokenizer(
    tokenizer_path: str | os.PathLike[str] | Path
) -> "PreTrainedTokenizerFast":  # noqa: F821
    
    return AutoTokenizer.from_pretrained(tokenizer_path)

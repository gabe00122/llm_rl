import json
import os

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from zipp import Path


def load_tokenizer(
    tokenizer_path: str | os.PathLike[str] | Path
) -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(tokenizer_path)

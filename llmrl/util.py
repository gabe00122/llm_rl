import json
import os

from transformers import AutoTokenizer
from zipp import Path


def load_tokenizer(
    tokenizer_path: str | os.PathLike[str] | Path
) -> "PreTrainedTokenizerFast":  # noqa: F821
    
    return AutoTokenizer.from_pretrained(tokenizer_path)
    # from transformers import PreTrainedTokenizerFast, AddedToken

    # config = json.loads(Path(tokenizer_config_path).read_text())
    # config = {
    #     k: AddedToken(**v) if isinstance(v, dict) and str(k).endswith("token") else v for (k, v) in config.items()
    # }
    # config["added_tokens_decoder"] = {
    #     int(k): AddedToken(**v) for (k, v) in config.get("added_tokens_decoder", dict()).items()
    # }
    # return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **config)

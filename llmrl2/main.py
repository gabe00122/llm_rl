from safetensors import safe_open
from rich.progress import track
import ml_dtypes
# from transformers import AutoTokenizer, AutoModelForCausalLM
from jax import numpy as jnp
import jax
import numpy as np
from flax import nnx

from llmrl2.config import load_config
from llmrl2.model import Qwen3
from llmrl2.util import load_tokenizer

PAD_ID = 151643

def put_path(data, p: list[str], value):
    first, *rest = p

    if len(rest) == 0:
        data[first] = value
    else:
        data = data.setdefault(first, {})
        put_path(data, rest, value)


def load_param_dict(file_path):
    params = {}

    with safe_open(file_path, framework="np") as f:
        for key in track(f.keys()):
            key_path = key.split('.')
            value = f.get_tensor(key)
            put_path(params, key_path, value)

            print(f"{key}, {value.shape}")
    
    return params


def encode_input(tokenizer, texts, pad_id: int = PAD_ID):
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
        for text in texts
    ]
    max_len = max([len(x) for x in inputs])
    inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
    return jnp.array(inputs)


def main():
    model_path = "./base-models/qwen3-0.6b"
    config = load_config(f"{model_path}/config.json")
    params = load_param_dict(f"{model_path}/model.safetensors")
    tokenizer = load_tokenizer(model_path)

    print(config)
    rngs = nnx.Rngs(0)
    model = Qwen3(config, rngs=rngs)
    model.load_params(params)
    del params

    tokens = encode_input(tokenizer, ["Hello World"])
    print(tokens)

    x = nnx.jit(model)(tokens)
    print(x)
    # output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(output_text)




if __name__ == "__main__":
    main()

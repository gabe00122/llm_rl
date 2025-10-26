from jax import numpy as jnp
import jax
from flax import nnx

from llmrl2.config import load_config
from llmrl2.model import Qwen3
from llmrl2.util import load_tokenizer
from llmrl2.checkpoint import load_param_dict

PAD_ID = 151643


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

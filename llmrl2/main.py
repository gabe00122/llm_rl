from jax import numpy as jnp
import jax
from flax import nnx

from llmrl2.config import load_config
from llmrl2.model import Qwen3
from llmrl2.util import load_tokenizer
from llmrl2.checkpoint import load_safetensors

PAD_ID = 151643


def encode_input(tokenizer, texts, pad_size: int, pad_id: int = PAD_ID):
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
        for text in texts
    ]
    lengths = [len(x) for x in inputs]
    inputs = [x + (pad_size - length) * [pad_id] for x, length in zip(inputs, lengths)]
    return jnp.array(inputs), jnp.array(lengths)


def main():
    model_path = "./base-models/qwen3-0.6b"
    config = load_config(f"{model_path}/config.json")
    params = load_safetensors(model_path)
    tokenizer = load_tokenizer(model_path)

    print(config)
    rngs = nnx.Rngs(0)
    model = Qwen3(config, rngs=rngs)
    model.load_params(params)
    del params

    tokens, lengths = encode_input(tokenizer, ["What is the capital of france?"], 256)
    print(tokens)

    @nnx.jit
    def apply(model, tokens):
        return model(tokens)

    for _ in range(256 - lengths[0].item()):
        out = apply(model, tokens)
        logits = out[:, lengths[0] - 1, :]

        sample = jax.random.categorical(rngs.sample(), logits)
        tokens = tokens.at[0, lengths[0]].set(sample[0])
        lengths = lengths + 1

    print(tokenizer.decode(tokens[0].tolist()))
    # output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(output_text)




if __name__ == "__main__":
    main()

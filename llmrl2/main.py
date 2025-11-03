import time
from turtle import position
from typing import NamedTuple
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
        tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
        for text in texts
    ]
    lengths = [len(x) for x in inputs]
    inputs = [x + (pad_size - length) * [pad_id] for x, length in zip(inputs, lengths)]
    return jnp.array(inputs), jnp.array(lengths)


class SamplingConfig(NamedTuple):
    temperature: float
    top_k: int
    top_p: float


def sample(logits, rng_key):
    temperature = 0.6
    top_k = 20
    top_p = 0.95

    logits /= temperature
    probs = nnx.softmax(logits)

    top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
    top_p_probs = probs[top_k_indices]

    cumsum_top_p = jnp.cumsum(top_p_probs) - top_p_probs
    top_k_logits = jnp.where(cumsum_top_p < top_p, top_k_logits, -jnp.inf)

    sample_index = jax.random.categorical(rng_key, top_k_logits)
    return top_k_indices[sample_index]


@nnx.jit
def generate(model: Qwen3, prompt: jax.Array, prompt_length: jax.Array, rng_key):
    batch_size, seq_size = prompt.shape

    kv_cache = model.initialize_carry(batch_size, seq_size)

    @nnx.scan(in_axes=(0, 1, 0, nnx.Carry), out_axes=(1, nnx.Carry))
    def _step(i, prompt_token, rng_key, carry):
        kv_cache, tokens = carry

        positions = jnp.full((batch_size, 1), i, dtype=jnp.int32) # is this slow?

        logits, kv_cache = model(tokens, positions, kv_cache)
        sample_tokens = sample(logits.squeeze(), rng_key)[None, None] #jax.random.categorical(rng_key, logits)

        next_inputs =  jnp.where((i + 1 < prompt_length)[:, None], prompt_token[:, None], sample_tokens)

        carry = kv_cache, next_inputs
        return tokens, carry
    
    output, _ = _step(
        jnp.arange(seq_size),
        jnp.roll(prompt, -1),
        jax.random.split(rng_key, seq_size),
        (kv_cache, prompt[:,:1])
    )

    return output


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

    seq_length = 4096

    while True:
        prompt = input("Prompt: ")
        prompt_tokens, lengths = encode_input(tokenizer, [prompt], seq_length)
        output = generate(model, prompt_tokens, lengths, rngs.sample())

        output_text: str = tokenizer.decode(output[0].squeeze().tolist())
        print(output_text.split("<|im_end|>")[1])


if __name__ == "__main__":
    main()

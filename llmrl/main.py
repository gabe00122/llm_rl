from functools import partial
import time
from typing import NamedTuple
from jax import numpy as jnp
import jax
from flax import nnx

from llmrl.config import load_config
from llmrl.model import Qwen3
from llmrl.util import load_tokenizer
from llmrl.checkpoint import load_safetensors

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


# def sample(logits, rng_key):
#     temperature = 0.7
#     top_k = 20
#     top_p = 0.8

#     logits /= temperature
#     probs = nnx.softmax(logits)

#     top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
#     top_p_probs = probs[top_k_indices]

#     cumsum_top_p = jnp.cumsum(top_p_probs) - top_p_probs
#     top_k_logits = jnp.where(cumsum_top_p < top_p, top_k_logits, -jnp.inf)

#     sample_index = jax.random.categorical(rng_key, top_k_logits)
#     return top_k_indices[sample_index]

def sample(logits, rng_key):
    """
    logits: (..., V)
    rng_key: jax.random.PRNGKey
    returns: (...) integer token indices
    """
    temperature = 0.7
    top_k = 20
    top_p = 0.8

    V = logits.shape[-1]
    k = min(top_k, V)

    logits = logits / temperature
    probs = jax.nn.softmax(logits, axis=-1)

    topk_logits, topk_idx = jax.lax.top_k(logits, k)  # (..., k)

    topk_probs = jnp.take_along_axis(probs, topk_idx, axis=-1)  # (..., k)

    cumsum = jnp.cumsum(topk_probs, axis=-1) - topk_probs
    masked_topk_logits = jnp.where(cumsum < top_p, topk_logits, -jnp.inf)  # (..., k)

    sample_in_topk = jax.random.categorical(rng_key, masked_topk_logits, axis=-1)  # (...)

    sample_in_topk = jnp.expand_dims(sample_in_topk, axis=-1)                      # (..., 1)
    sampled_ids = jnp.take_along_axis(topk_idx, sample_in_topk, axis=-1).squeeze(-1)  # (...)

    return sampled_ids

@partial(nnx.jit, donate_argnums=(0, 1, 2))
def generate_single(model: Qwen3, kv_cache, tokens: jax.Array, i: jax.Array, prompt: jax.Array, prompt_length: jax.Array, rng_key: jax.Array):
    batch_size, _ = prompt.shape

    positions = jnp.full((batch_size, 1), i, dtype=jnp.int32) # is this slow?

    logits, kv_cache = model(tokens, positions, kv_cache)
    # sample_tokens = sample(logits.squeeze(axis=1), rng_key)[:, None]
    sample_tokens = jax.random.categorical(rng_key, logits)

    next_inputs =  jax.lax.cond(
        i < prompt_length,
        lambda: prompt[:, i][:, None],
        lambda: sample_tokens
    )

    return next_inputs, kv_cache

@partial(nnx.jit, donate_argnums=(0,))
def generate(model: Qwen3, prompt: jax.Array, prompt_length: jax.Array, rng_key):
    batch_size, seq_size = prompt.shape

    kv_cache = model.initialize_carry(batch_size, seq_size)

    @nnx.scan(in_axes=(0, 1, 0, nnx.Carry), out_axes=(1, nnx.Carry))
    def _step(i, prompt_token, rng_key, carry):
        kv_cache, tokens = carry

        positions = jnp.full((batch_size, 1), i, dtype=jnp.int32) # is this slow?

        logits, kv_cache = model(tokens, positions, kv_cache)
        # sample_tokens = sample(logits.squeeze(axis=1), rng_key)[:, None]
        sample_tokens = jax.random.categorical(rng_key, logits)

        # next_inputs =  jnp.where((i + 1 < prompt_length)[:, None], prompt_token[:, None], sample_tokens)
        next_inputs =  jax.lax.cond(
            i < prompt_length,
            lambda: prompt[:, i][:, None],
            lambda: sample_tokens
        )

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
    model_path = "./base-models/Qwen3-4B-Instruct-2507"

    # model_path = "./base-models/qwen3-0.6b"
    config = load_config(f"{model_path}/config.json")
    params = load_safetensors(model_path)
    tokenizer = load_tokenizer(model_path)

    print(config)
    rngs = nnx.Rngs(0)
    model = Qwen3(config, rngs=rngs)
    model.load_params(params)
    del params

    batch_size = 1 #128 // 2
    seq_length = 4096 * 4 #512 * 2

    while True:
        prompt = input("Prompt: ")
        prompt_tokens, lengths = encode_input(tokenizer, [prompt] * batch_size, seq_length)
        start_time = time.time()
        output = generate(model, prompt_tokens, lengths[0], rngs.sample())

        for b in range(1):
            output_text: str = tokenizer.decode(output[b].squeeze().tolist())
            print(output_text.split("<|im_end|>")[1])
        
        stop_time = time.time()
        delta_time = stop_time - start_time
        print(f"TPS: {(batch_size * seq_length) // delta_time}")


def main2():
    model_path = "./base-models/Qwen3-4B-Instruct-2507"

    # model_path = "./base-models/qwen3-0.6b"
    config = load_config(f"{model_path}/config.json")
    params = load_safetensors(model_path)
    tokenizer = load_tokenizer(model_path)

    print(config)
    rngs = nnx.Rngs(0)
    model = Qwen3(config, rngs=rngs)
    model.load_params(params)
    del params

    batch_size = 96
    seq_length = 512

    while True:
        prompt = input("Prompt: ")
        prompt_tokens, lengths = encode_input(tokenizer, [prompt] * batch_size, seq_length)
        start_time = time.time()

        kv_cache = model.initialize_carry(batch_size, seq_length)
        token_input = prompt_tokens[:, 0][:, None]
        
        for i in range(seq_length):
            token_input, kv_cache = generate_single(model, kv_cache, token_input, jnp.int32(i+1), prompt_tokens, lengths[0], rngs.sample())

        token_input.block_until_ready()
        # output_text: str = tokenizer.decode(output[0].squeeze().tolist())
        stop_time = time.time()
        # print(output_text.split("<|im_end|>")[1])

        delta_time = stop_time - start_time
        print(f"TPS: {(batch_size * seq_length) // delta_time}")

if __name__ == "__main__":
    main()

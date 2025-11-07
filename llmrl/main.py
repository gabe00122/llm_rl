from functools import partial
import time
from typing import Any, NamedTuple
from jax import numpy as jnp
import jax
from flax import nnx
# from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from llmrl.config import SamplingConfig, load_config, load_sampling_config
from llmrl.model import Qwen3
from llmrl.util import load_tokenizer
from llmrl.checkpoint import load_safetensors

PAD_ID = 151643


def encode_input(tokenizer: PreTrainedTokenizerFast, texts: list[str], pad_size: int, pad_id: int = PAD_ID):
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
        for text in texts
    ]
    lengths = [len(x) for x in inputs]
    inputs = [x + (pad_size - length) * [pad_id] for x, length in zip(inputs, lengths)]
    return jnp.array(inputs), jnp.array(lengths)


def sample(config: SamplingConfig, logits, rng_key):
    V = logits.shape[-1]
    k = min(config.top_k, V)

    logits = logits / config.temperature
    topk_logits, topk_idx = jax.lax.top_k(logits, k)  # (..., k)

    topk_probs = jax.nn.softmax(topk_logits, axis=-1)  # (..., k)

    cumsum = jnp.cumsum(topk_probs, axis=-1) - topk_probs
    masked_topk_logits = jnp.where(cumsum < config.top_p, topk_logits, -jnp.inf)

    sample_in_topk = jax.random.categorical(rng_key, masked_topk_logits, axis=-1)
    sample_in_topk = jnp.expand_dims(sample_in_topk, axis=-1)
    sampled_ids = jnp.take_along_axis(topk_idx, sample_in_topk, axis=-1).squeeze(-1)

    return sampled_ids


def generate_some(
        model: Qwen3,
        sampling_config: SamplingConfig,
        kv_cache,
        positions: jax.Array,
        prompt: jax.Array,
        prompt_lengths: jax.Array,
        rng_key: jax.Array
    ):

    B = prompt.shape[0]

    class GenerateCarry(NamedTuple):
        tokens: jax.Array
        kv_cache: Any
        positions: jax.Array
        rng_key: jax.Array
        finished: jax.Array

    init_carry = GenerateCarry(
        prompt[:, 0][:, None],
        kv_cache,
        positions,
        rng_key,
        jnp.zeros((B,), jnp.bool_)
    )
    
    def cond(carry: GenerateCarry):
        return jnp.logical_not(jnp.all(carry.finished))
    
    def body(carry: GenerateCarry):
        logits, kv_cache = model(carry.tokens, positions, carry.kv_cache)

        sample_key, rng_key = jax.random.split(carry.rng_key)
        sample_tokens = sample(sampling_config, logits.squeeze(-1), sample_key)



        return carry

    out = nnx.while_loop(cond, body, init_carry)

@partial(nnx.jit, donate_argnums=(0,), static_argnums=(1,))
def generate(model: Qwen3, sampling: SamplingConfig, prompt: jax.Array, prompt_length: jax.Array, rng_key):
    batch_size, seq_size = prompt.shape

    kv_cache = model.initialize_carry(batch_size, seq_size)

    @nnx.scan(in_axes=(0, 1, 0, nnx.Carry), out_axes=(1, nnx.Carry))
    def _step(i, prompt_token, rng_key, carry):
        kv_cache, tokens = carry

        positions = jnp.full((batch_size, 1), i, dtype=jnp.int32) # is this slow?

        logits, kv_cache = model(tokens, positions, kv_cache)
        sample_tokens = sample(sampling, logits.squeeze(axis=1), rng_key)[:, None]
        # sample_tokens = jax.random.categorical(rng_key, logits)

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
    model_path = "./base-models/Qwen3-4B-Instruct-2507"

    # model_path = "./base-models/qwen3-0.6b"
    config = load_config(f"{model_path}/config.json")
    params = load_safetensors(model_path)
    tokenizer = load_tokenizer(model_path)
    sampling = load_sampling_config(f"{model_path}/generation_config.json")

    print(config)
    rngs = nnx.Rngs(0)
    model = Qwen3(config, rngs=rngs)
    model.load_params(params)
    del params

    batch_size = 1
    seq_length = 512

    while True:
        prompt = input("Prompt: ")
        prompt_tokens, lengths = encode_input(tokenizer, [prompt] * batch_size, seq_length)
        start_time = time.time()
        output = generate(model, sampling, prompt_tokens, lengths, rngs.sample())

        for b in range(1):
            output_text: str = tokenizer.decode(output[b].squeeze().tolist())
            print(output_text.split("<|im_end|>")[1])
        
        stop_time = time.time()
        delta_time = stop_time - start_time
        print(f"TPS: {(batch_size * seq_length) // delta_time}")


if __name__ == "__main__":
    main()

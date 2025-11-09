from functools import partial
import time
from typing import Any, NamedTuple
from jax import numpy as jnp
import numpy as np
import jax
from flax import nnx
import optax
# from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from llmrl.config import SamplingConfig, load_config, load_sampling_config
from llmrl.model import Qwen3
from llmrl.util import load_tokenizer
from llmrl.checkpoint import load_safetensors

PAD_ID = 151643
# END_TOKEN = 151644


def encode_input(tokenizer: PreTrainedTokenizerFast, conversations: list[list[dict]], pad_size: int):
    # assert isinstance(texts, list)
    inputs = tokenizer.apply_chat_template(
        conversations,
        padding='max_length',
        max_length=pad_size,
        add_generation_prompt=True,
        return_tensors='np'
    )
    return jnp.array(inputs)


def get_last_turn(text: str) -> str:
    llm_response = text.split("<|im_start|>assistant\n")[-1]
    return llm_response.split("<|im_end|>")[0]


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

@partial(nnx.jit, donate_argnums=(0,2,3,4,5), static_argnums=(1,))
def generate_some(
        model: Qwen3,
        sampling_config: SamplingConfig,
        kv_cache,
        positions: jax.Array,
        prompt: jax.Array,
        rng_key: jax.Array
    ):

    B, seq_length = prompt.shape

    class GenerateCarry(NamedTuple):
        next_tokens: jax.Array
        context: jax.Array
        kv_cache: Any
        positions: jax.Array
        rng_key: jax.Array
        finished: jax.Array
    
    def cond(carry: GenerateCarry):
        return jnp.logical_not(jnp.all(carry.finished))
    
    def body(carry: GenerateCarry):
        logits, kv_cache = model(carry.next_tokens[..., None], carry.positions[..., None], carry.kv_cache)

        sample_key, rng_key = jax.random.split(carry.rng_key)
        sample_tokens = sample(sampling_config, logits.squeeze(axis=-2), sample_key)
        # sample_tokens = jax.random.categorical(sample_key, logits)

        batch_index = jnp.arange(B, dtype=jnp.int32)

        next_positions = jnp.where(carry.finished, carry.positions, carry.positions + 1)

        prompt_tokens = carry.context[batch_index, next_positions]
        next_tokens =  jnp.where(
            jnp.logical_or(prompt_tokens != PAD_ID, carry.finished),
            prompt_tokens,
            sample_tokens
        )

        next_context = carry.context.at[batch_index, next_positions].set(next_tokens)
        next_finished = jnp.logical_or(carry.finished, next_positions >= seq_length)
        next_finished = jnp.logical_or(next_finished, next_tokens == PAD_ID)

        return GenerateCarry(
            next_tokens,
            next_context,
            kv_cache,
            next_positions,
            rng_key,
            next_finished
        )
    
    init_carry = GenerateCarry(
        prompt[:, 0],
        prompt,
        kv_cache,
        positions,
        rng_key,
        jnp.zeros((B,), jnp.bool_)
    )

    out = nnx.while_loop(cond, body, init_carry)

    return out.kv_cache, out.positions, out.context, out.rng_key

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

    # tx = optax.adam(0.01)
    # optimizer = nnx.Optimizer(model, tx, wrt=nnx.LoRAParam)

    batch_size = 1
    seq_length = 4096

    rng_key = rngs.sample()
    kv_cache = model.initialize_carry(batch_size, seq_length)
    positions = jnp.zeros((batch_size,), jnp.int32)

    conversations = [[]]

    while True:
        prompt = input("Prompt: ")
        conversations[0].append({"role": "user", "content": prompt})
        prompt_tokens = encode_input(tokenizer, conversations, seq_length)
        start_time = time.time()
        start_pos = positions[0].item()

        kv_cache, positions, output, rng_key = generate_some(model, sampling, kv_cache, positions, prompt_tokens, rng_key)

        output_text: list[str] = tokenizer.batch_decode(np.asarray(output))
        out = get_last_turn(output_text[0])
        print(out)
        conversations[0].append({"role": "assistant", "content": out})
        
        stop_time = time.time()
        delta_time = stop_time - start_time
        end_pos = positions[0].item()
        total_tokens = end_pos - start_pos
        print(f"TPS: {total_tokens // delta_time}")
        print(f"Context: {end_pos}/{seq_length}")


if __name__ == "__main__":
    main()

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

from llmrl.config import LoraConfig, SamplingConfig, load_config, load_sampling_config
from llmrl.model import Qwen3
from llmrl.util import load_tokenizer
from llmrl.checkpoint import load_model, load_safetensors

PAD_ID = 151643
EOS_1 = 151645
EOS_2 = 151643
# END_TOKEN = 151644


def encode_input(tokenizer: PreTrainedTokenizerFast, conversations: list[list[dict]] | list[dict], pad_size: int):
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

    topk_logits, topk_idx = jax.lax.top_k(logits, k)  # (..., k)
    topk_logits = topk_logits / config.temperature

    topk_probs = jax.nn.softmax(topk_logits, axis=-1)  # (..., k)

    cumsum = jnp.cumsum(topk_probs, axis=-1) - topk_probs
    masked_topk_logits = jnp.where(cumsum < config.top_p, topk_logits, -jnp.inf)

    sample_in_topk = jax.random.categorical(rng_key, masked_topk_logits, axis=-1)
    sample_in_topk = jnp.expand_dims(sample_in_topk, axis=-1)
    sampled_ids = jnp.take_along_axis(topk_idx, sample_in_topk, axis=-1).squeeze(-1)

    return sampled_ids

@partial(nnx.jit, donate_argnums=(0,2,3,4,5), static_argnums=(1,))
def generate(
        model: Qwen3,
        sampling_config: SamplingConfig,
        kv_cache,
        positions: jax.Array,
        prompt: jax.Array,
        rng_key: jax.Array
    ):

    B, seq_length = prompt.shape
    batch_index = jnp.arange(B, dtype=jnp.int32)

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

        if sampling_config.temperature == 0.0:
            sample_tokens = jnp.argmax(logits.squeeze(-2), -1)
        else:
            sample_tokens = sample(sampling_config, logits.squeeze(axis=-2), sample_key)

        next_positions = jnp.where(carry.finished, carry.positions, carry.positions + 1)

        prompt_tokens = carry.context[batch_index, next_positions]
        use_sample = jnp.logical_and(jnp.logical_or(prompt_tokens == PAD_ID, carry.finished), jnp.logical_not(carry.finished))
        next_tokens =  jnp.where(
            use_sample,
            sample_tokens,
            prompt_tokens
        )

        next_context = carry.context.at[batch_index, next_positions].set(next_tokens)
        next_finished = jnp.logical_or(carry.finished, next_positions >= seq_length - 1)

        # we check to make sure we aren't reading the prompt and that the model sampled a end token
        next_finished = jnp.logical_or(next_finished, jnp.logical_and(sample_tokens == EOS_1, use_sample))
        # jax.debug.print("{}", next_tokens[0])

        return GenerateCarry(
            next_tokens,
            next_context,
            kv_cache,
            next_positions,
            rng_key,
            next_finished
        )
    

    init_carry = GenerateCarry(
        prompt[batch_index, positions],
        prompt,
        kv_cache,
        positions,
        rng_key,
        jnp.zeros((B,), jnp.bool_)
    )

    out = nnx.while_loop(cond, body, init_carry)

    return out.kv_cache, out.positions, out.context, out.rng_key


def chat(model: Qwen3, tokenizer, sampling, batch_size: int, seq_length: int, rngs: nnx.Rngs):
    rng_key = rngs.sample()
    kv_cache = model.initialize_carry(batch_size, seq_length)
    positions = jnp.zeros((batch_size,), jnp.int32)

    conversations = []
    for _ in range(batch_size):
        conversations.append([])

    while True:
        prompt = input("Prompt: ")

        if prompt == "/clear":
            positions = jnp.zeros((batch_size,), jnp.int32)
            conversations = []
            for _ in range(batch_size):
                conversations.append([])
            continue

        for conv in conversations:
            conv.append({"role": "user", "content": prompt})
        prompt_tokens = encode_input(tokenizer, conversations, seq_length)
        # print(prompt_tokens.shape)
        start_time = time.time()
        start_pos = positions.copy()

        kv_cache, positions, output, rng_key = generate(model, sampling, kv_cache, positions, prompt_tokens, rng_key)

        output_text: list[str] = tokenizer.batch_decode(np.asarray(output))
        for conv, out in zip(conversations, output_text):
            assistant = get_last_turn(out)
            print("--------")
            print(assistant)
            conv.append({"role": "assistant", "content": assistant})
        
        stop_time = time.time()
        delta_time = stop_time - start_time
        total_tokens = jnp.sum(positions - start_pos).item()
        print(f"TPS: {total_tokens // delta_time}")
        print(f"Context: {positions[0].item()}/{seq_length}")



def main():
    # model_path = "./base-models/qwen3-0.6b"
    model_path = "./base-models/Qwen3-4B-Instruct-2507"
    lora_config = LoraConfig(False, False, 0)
    rngs = nnx.Rngs(0)
    model, tokenizer, sampling = load_model(model_path, lora_config, rngs)

    batch_size = 1
    seq_length = 16384 #512
    chat(model, tokenizer, sampling, batch_size, seq_length, rngs)


if __name__ == "__main__":
    main()

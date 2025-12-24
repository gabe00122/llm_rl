from typing import Any, Iterable, override

import jax
import numpy as np
from jax import numpy as jnp
from flax import nnx

from llmrl.agent.base import Agent
from llmrl.chat import (
    create_generation_state,
    encode_input,
    decode_responses,
    generate,
    reset_generation_state,
    append_prompt_tokens,
    append_user_prompts
)
from llmrl.config import SamplingConfig
from llmrl.model import Qwen3
from transformers import PreTrainedTokenizerFast


class LocalAgent(Agent):
    def __init__(
        self,
        model_def,
        model_state,
        tokenizer: PreTrainedTokenizerFast,
        agent_count: int,
        max_context_length: int,
        instructions: str,
        rng_key: jax.Array,
    ):
        self._model_def = model_def
        self._model_state = model_state
        self._tokenizer = tokenizer
        self._agent_count = agent_count
        self._max_context_length = max_context_length
        self._instructions = instructions

        self._rng_key = rng_key

        shape = (self._agent_count, self._max_context_length)
        kv_cache = nnx.merge(self._model_def, self._model_state).initialize_carry(*shape)
        self._gen = create_generation_state(
            kv_cache, self._agent_count, self._max_context_length, self._rng_key
        )

        self._instruction_tokens = encode_input(
            tokenizer,
            [
                [{"role": "system", "content": self._instructions}]
                for _ in range(agent_count)
            ],
            False,
        )

        self._gen = append_prompt_tokens(
            self._gen,
            jnp.arange(self._agent_count, dtype=jnp.int32),
            self._instruction_tokens,
        )

    @override
    def reset(self):
        self._gen = reset_generation_state(self._gen)
        self._gen = append_prompt_tokens(
            self._gen,
            jnp.arange(self._agent_count, dtype=jnp.int32),
            self._instruction_tokens,
        )

    @override
    def act(
        self, batch_indices: jax.Array, obs: list[str], rewards: jax.Array
    ) -> tuple[jax.Array, list[str]]:
        self._gen = append_user_prompts(self._gen, batch_indices, self._tokenizer, obs)
        self._gen = generate(self._model_def, self._model_state, "simple", self._gen)

        response_indices, response = decode_responses(self._tokenizer, self._gen)
        return response_indices, response

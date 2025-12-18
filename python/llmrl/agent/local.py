from typing import Any, Iterable, override

import jax
import numpy as np
from jax import numpy as jnp
from flax import nnx

from llmrl.agent.base import Agent
from llmrl.chat import encode_input, generate, get_last_turn
from llmrl.config import SamplingConfig
from llmrl.model import Qwen3
from transformers import PreTrainedTokenizerFast


class LocalAgent(Agent):
    def __init__(
        self,
        model: Qwen3,
        sampling_config: SamplingConfig,
        tokenizer: PreTrainedTokenizerFast,
        agent_count: int,
        max_context_length: int,
        instructions: str,
        rng_key: jax.Array,
    ):
        self._model = model
        self._sampling_config = sampling_config
        self._tokenizer = tokenizer
        self._agent_count = agent_count
        self._max_context_length = max_context_length
        self._instructions = instructions

        self._rng_key = rng_key

        shape = (self._agent_count, self._max_context_length)
        self._context = jnp.zeros(shape, dtype=jnp.int32)
        self._is_prompt = jnp.zeros(shape, dtype=jnp.bool_)
        self._log_probs = jnp.zeros(shape, dtype=jnp.float32)
        self._values = jnp.zeros(shape, jnp.float32)
        self._rewards = np.zeros(shape, np.float32)

        self._kv_cache = self._model.initialize_carry(*shape)

    @override
    def reset(self):
        self._messages: list[list[Any]] = [[] for _ in range(self._agent_count)]
        self._positions = jnp.zeros((self._agent_count,), dtype=jnp.int32)

    @override
    def act(self, obs: Iterable[str]) -> list[str]:
        for messages, o in zip(self._messages, obs):
            messages.append({"role": "user", "content": o})

        prompts = encode_input(
            self._tokenizer, self._messages, self._max_context_length
        )

        self._kv_cache, self._positions, self._context, self._rng_key = generate(
            self._model,
            self._sampling_config,
            self._kv_cache,
            self._positions,
            prompts,
            self._rng_key,
        )

        response: list[str] = self._tokenizer.batch_decode(np.asarray(self._context))

        for messages, res in zip(self._messages, response):
            messages.append({"role": "assistant", "content": get_last_turn(res)})

        return response

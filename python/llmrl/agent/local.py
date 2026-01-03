from llmrl.checkpointer import Checkpointer
from llmrl.config import Config
import time
from typing import override

from idna import decode
import jax
from jax import numpy as jnp
from flax import nnx

from llmrl.agent.base import Agent
from llmrl.chat import (
    create_generation_state,
    encode_input,
    decode_responses,
    generate,
    reset_episodes,
    reset_generation_state,
    append_prompt_tokens,
    append_user_prompts,
    GenerationState
)
from llmrl.buffer import UpdateBuffer
from llmrl.update_step import update_step
from llmrl.logger import BaseLogger
import numpy as np
import optax
from transformers import PreTrainedTokenizerFast


class LocalAgent(Agent):
    def __init__(
        self,
        model_def,
        model_state,
        opt_def,
        opt_state,
        tokenizer: PreTrainedTokenizerFast,
        checkpointer: Checkpointer,
        config: Config,
        instructions: str,
        logger: BaseLogger,
        rng_key: jax.Array,
    ):
        self._model_def = model_def
        self._model_state = model_state
        self._opt_def = opt_def
        self._opt_state = opt_state
        self._tokenizer = tokenizer
        self._checkpointer = checkpointer
        self._config = config
        self._instructions = instructions
        self._logger = logger

        self._update_step = 0

        self._rng_key = rng_key

        shape = (self._config.eval_envs, self._config.max_seq_length)
        kv_cache = nnx.merge(self._model_def, self._model_state).initialize_carry(*shape)
        self._gen = create_generation_state(
            kv_cache, self._config.eval_envs, self._config.max_seq_length, self._rng_key
        )

        # v this count be wrapped as a convenience function
        instruction_tokens = encode_input(
            tokenizer,
            [
                [{"role": "system", "content": self._instructions}]
                for _ in range(self._config.update_envs)
            ],
            False,
        )
        self._gen = append_prompt_tokens(
            self._gen,
            np.arange(self._config.update_envs, dtype=jnp.int32),
            instruction_tokens,
        )
        self._gen = self._gen._replace(
            env_instruction_length=self._gen.context_length.copy(),
        )
        self._rewards = np.zeros(shape, dtype=np.float32)

        self._buffer = UpdateBuffer(self._config.update_envs * 4, self._config.update_envs, self._config.max_seq_length)

        self._reset_time = 0.0
        self._append_time = 0.0
        self._decode_time = 0.0
        self._gen_time = 0.0


    @override
    def reset(self):
        # self._gen = reset_generation_state(self._gen)
        # self._gen = append_prompt_tokens(
        #     self._gen,
        #     jnp.arange(self._agent_count, dtype=jnp.int32),
        #     self._instruction_tokens,
        # )
        pass

    @override
    def act(
        self, batch_indices: np.ndarray, obs: list[str], rewards: np.ndarray, dones: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        kv_cache_lengths = np.array(self._gen.kv_cache_length)
        self._rewards[batch_indices, kv_cache_lengths[batch_indices]] = rewards

        done_idx = batch_indices[np.where(dones)]

        if dones.any():
            self._buffer.store(
                done_idx,
                np.array(self._gen.context),
                kv_cache_lengths,
                self._rewards,
                np.array(self._gen.values),
                np.array(self._gen.log_probs),
                np.array(self._gen.policy_mask)
            )
            while self._buffer.has_batch:
                ub = self._buffer.take_batch()
                self._opt_state, self._model_state, metrics = update_step(self._opt_def, self._opt_state, self._model_def, self._model_state, ub)

                metrics["rewards"] = ub.rewards.sum() / ub.rewards.shape[0]

                self._logger.log_dict(metrics, self._update_step)
                self._update_step += 1
                # first_ctx = self._tokenizer.decode(ub.context[0, :ub.kv_cache_lengths[0]])
                # print(first_ctx)

            reset_start = time.perf_counter()
            done_mask = np.zeros((self._agent_count,), np.bool_)
            done_mask[batch_indices] = dones
            self._gen: GenerationState = reset_episodes(self._gen, done_mask)
            self._reset_time += time.perf_counter() - reset_start
            
            self._rewards[done_idx] = 0.0
        
        append_start = time.perf_counter()
        self._gen = append_user_prompts(self._gen, batch_indices, self._tokenizer, obs)
        self._append_time += time.perf_counter() - append_start
        
        gen_start = time.perf_counter()
        self._gen = generate(self._model_def, self._model_state, "simple", self._gen, 4)
        self._gen_time += time.perf_counter() - gen_start

        decode_start = time.perf_counter()
        response_indices, response = decode_responses(self._tokenizer, self._gen)
        self._decode_time += time.perf_counter() - decode_start

        return response_indices, response

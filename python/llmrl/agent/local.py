from typing import Protocol, override
import os

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from jax import numpy as jnp
from llmrl.agent.base import Agent
from llmrl.buffer import UpdateBatch, UpdateBuffer
from llmrl.chat import (
    GenerationState,
    append_prompt_tokens,
    append_user_prompts,
    create_generation_state,
    decode_responses,
    encode_input,
    generate,
    reset_episodes,
)
from llmrl.checkpointer import Checkpointer
from llmrl.config import Config
from llmrl.logger import BaseLogger
from llmrl.model.qwen3 import Qwen3
from llmrl.update_step import update_step
from llmrl.utils.performance import PerformanceTracker
from transformers import PreTrainedTokenizerFast


class EpisodeListener(Protocol):
    def on_episodes(self, batch: UpdateBatch): ...


class ModelProvider(Protocol):
    model_def: nnx.GraphDef
    model_state: nnx.State


class Trainer(EpisodeListener):
    def __init__(
        self,
        model_provider: ModelProvider,
        optimizer: nnx.Optimizer,
        checkpointer: Checkpointer,
        performance: PerformanceTracker,
        logger: BaseLogger,
        config: Config,
    ):
        self._model_provider = model_provider
        self._opt_def, self._opt_state = nnx.split(optimizer)

        self._checkpointer = checkpointer
        self._performance = performance
        self._logger = logger
        self._config = config
        self._update_step = 0

    def save_checkpoint(self):
        opt = nnx.merge(self._opt_def, self._opt_state)
        model = nnx.merge(
            self._model_provider.model_def, self._model_provider.model_state
        )
        self._checkpointer.save(
            {"opt": opt, "model": model}, self._update_step, opt.wrt
        )

    def restore_checkpoint(self, *, checkpointer: Checkpointer | None = None, wrt: nnx.filterlib.Filter | None = None):        
        opt = nnx.merge(self._opt_def, self._opt_state)
        model = nnx.merge(
            self._model_provider.model_def, self._model_provider.model_state
        )

        if checkpointer is None:
            step = self._checkpointer.restore_latest({"opt": opt, "model": model}, wrt or opt.wrt)
            self._update_step = step
            self._opt_state = nnx.state(opt)
        else:
            checkpointer.restore_latest(
                {"opt": ocp.PLACEHOLDER, "model": model}, wrt or opt.wrt
            )

        self._model_provider.model_state = nnx.state(model)

    @property
    def progress(self) -> float:
        return self._update_step / self._config.total_update_episodes

    def on_episodes(self, batch: UpdateBatch):
        with self._performance.time("update_step"):
            self._opt_state, new_model_state, metrics = update_step(
                self._opt_def,
                self._opt_state,
                self._model_provider.model_def,
                self._model_provider.model_state,
                batch,
                self._config.loss,
                False,
            )

        self._model_provider.model_state = new_model_state

        metrics["rewards"] = batch.rewards.sum() / batch.rewards.shape[0]
        metrics["performance"] = self._performance.total_time_percentages()
        self._performance.reset()

        self._logger.log_dict(metrics, self._update_step)
        self._update_step += 1

        if self._update_step % self._config.checkpoint_every == 0:
            self.save_checkpoint()


class EpisodeSaver(EpisodeListener):
    def __init__(self, directory: str):
        self._directory = directory
        self.chunk_num = 0

    def on_episodes(self, batch: UpdateBatch):
        file_name = os.path.join(self._directory, f"episodes_{self.chunk_num}.npz")
        batch.save_npz(file_name, compressed=False)
        self.chunk_num += 1


class MultiEpisodeListener(EpisodeListener):
    def __init__(self, listeners: list[EpisodeListener]):
        self._listeners = listeners

    def on_episodes(self, batch: UpdateBatch):
        for listener in self._listeners:
            listener.on_episodes(batch)


class BufferedEpisodeListener(EpisodeListener):
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        seq_length: int,
        listener: EpisodeListener,
    ):
        self._listener = listener
        self._buffer = UpdateBuffer(buffer_size, batch_size, seq_length)

    def on_episodes(self, batch: UpdateBatch):
        print(f"Storing {batch.rewards.shape[0]} episodes")
        self._buffer.store(batch)
        while self._buffer.has_batch:
            self._listener.on_episodes(self._buffer.take_batch())


class LocalAgent(Agent, ModelProvider):
    def __init__(
        self,
        model: Qwen3,
        tokenizer: PreTrainedTokenizerFast,
        config: Config,
        logger: BaseLogger,
        performance_tracker: PerformanceTracker,
        rng_key: jax.Array,
    ):
        self.episode_listener: EpisodeListener | None = None

        self.model_def, self.model_state = nnx.split(model)
        self._tokenizer = tokenizer
        self._config = config
        self._logger = logger
        self._performance_tracker = performance_tracker

        self._rng_key = rng_key

        shape = (self._config.eval_envs, self._config.max_seq_length)
        kv_cache = model.initialize_carry(*shape)
        self._gen = create_generation_state(
            kv_cache, self._config.eval_envs, self._config.max_seq_length, self._rng_key
        )

        self._rewards = np.zeros(
            (self._config.eval_envs, self._config.max_seq_length), dtype=np.float32
        )

    def set_episode_instructions(self, instructions: str):
        instruction_tokens = encode_input(
            self._tokenizer,
            [
                [{"role": "system", "content": instructions}]
                for _ in range(self._config.eval_envs)
            ],
            False,
        )
        self._gen = append_prompt_tokens(
            self._gen,
            np.arange(self._config.eval_envs, dtype=np.int32),
            instruction_tokens,
        )
        self._gen = self._gen._replace(
            env_instruction_length=self._gen.context_length.copy(),
        )

    @override
    def reset(self) -> None:
        pass

    @override
    def act(
        self,
        batch_indices: np.ndarray,
        obs: list[str],
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        kv_cache_lengths = np.array(self._gen.kv_cache_length)
        self._rewards[batch_indices, kv_cache_lengths[batch_indices]] = rewards

        done_idx = batch_indices[np.where(dones)]

        if dones.any():
            if self.episode_listener is not None:
                self.episode_listener.on_episodes(
                    UpdateBatch(
                        np.array(self._gen.context)[done_idx],
                        kv_cache_lengths[done_idx],
                        np.array(self._gen.log_probs)[done_idx],
                        np.array(self._gen.values)[done_idx],
                        self._rewards[done_idx],
                        np.array(self._gen.policy_mask)[done_idx],
                    )
                )

            with self._performance_tracker.time("reset"):
                done_mask = np.zeros((self._config.eval_envs,), np.bool_)
                done_mask[batch_indices] = dones
                self._gen: GenerationState = reset_episodes(self._gen, done_mask)
                self._rewards[done_idx] = 0.0

        with self._performance_tracker.time("encode"):
            self._gen = append_user_prompts(
                self._gen, batch_indices, self._tokenizer, obs
            )

        with self._performance_tracker.time("generate"):
            self._gen = generate(
                self.model_def, self.model_state, "simple", self._gen, 1
            )

        with self._performance_tracker.time("decode"):
            response_indices, response = decode_responses(self._tokenizer, self._gen)

        return response_indices, response

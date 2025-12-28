from typing import NamedTuple
import jax
from jax import numpy as jnp
import numpy as np

class UpdateBatch(NamedTuple):
    context: jax.Array
    context_lengths: jax.Array
    log_probs: jax.Array
    values: jax.Array
    rewards: jax.Array
    prompt_mask: jax.Array



class Rollout:
    def __init__(self, batch_size: int, seq_length: int) -> None:
        self._batch_size = batch_size

        self.context = np.zeros((batch_size, seq_length), dtype=np.int32)
        self.context_lengths = np.zeros((batch_size,), dtype=np.int32)
        self.log_probs = np.zeros((batch_size, seq_length), dtype=np.float32)
        self.values = np.zeros((batch_size, seq_length), dtype=np.float32)
        self.rewards = np.zeros((batch_size, seq_length), dtype=np.float32)
        self.prompt_mask = np.zeros((batch_size, seq_length), dtype=np.bool_)

        self._batch_index = 0

    @property
    def is_full(self):
        return self._batch_index >= self._batch_size
    
    def store(self, done_indices: np.ndarray, context: np.ndarray, context_lengths: np.ndarray, rewards: np.ndarray, values: np.ndarray, log_probs: np.ndarray, prompt_mask: np.ndarray):
        if self.is_full:
            breakpoint()
            return
        
        store_size = done_indices.shape[0]
        end_index = self._batch_index + store_size

        if end_index > self._batch_size:
            trim_amount = end_index - self._batch_size
            done_indices = done_indices[:-trim_amount] # todo: we are throwing away good data here
            end_index = min(end_index, self._batch_size)

        store_range = range(self._batch_index, end_index)

        self.context[store_range] = context[done_indices]
        self.context_lengths[store_range] = context_lengths[done_indices]
        self.log_probs[store_range] = log_probs[done_indices]
        self.values[store_range] = values[done_indices]
        self.rewards[store_range] = rewards[done_indices]
        self.prompt_mask[store_range] = prompt_mask[done_indices]

        self._batch_index = end_index

    def create_update_batch(self) -> UpdateBatch:
        return UpdateBatch(
            context=jnp.array(self.context),
            context_lengths=jnp.array(self.context_lengths),
            log_probs=jnp.array(self.log_probs),
            values=jnp.array(self.values),
            rewards=jnp.array(self.rewards),
            prompt_mask=jnp.array(self.prompt_mask),
        )
    # def calculate_advantages(self):


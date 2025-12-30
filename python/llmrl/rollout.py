from typing import NamedTuple
import jax
from jax import numpy as jnp
import numpy as np

class UpdateBatch(NamedTuple):
    context: jax.Array
    kv_cache_lengths: jax.Array
    log_probs: jax.Array
    values: jax.Array
    rewards: jax.Array
    policy_mask: jax.Array



class UpdateBuffer:
    def __init__(self, buffer_size: int, batch_size: int, seq_length: int) -> None:
        self._buffer_size = buffer_size
        self._batch_size = batch_size

        self.context = np.zeros((buffer_size, seq_length), dtype=np.int32)
        self.kv_cache_lengths = np.zeros((buffer_size,), dtype=np.int32)
        self.log_probs = np.zeros((buffer_size, seq_length), dtype=np.float32)
        self.values = np.zeros((buffer_size, seq_length), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, seq_length), dtype=np.float32)
        self.policy_mask = np.zeros((buffer_size, seq_length), dtype=np.bool_)

        self._index = 0

    @property
    def has_batch(self):
        return self._index >= self._batch_size - 1
    
    def store(self, done_indices: np.ndarray, context: np.ndarray, kv_cache_lengths: np.ndarray, rewards: np.ndarray, values: np.ndarray, log_probs: np.ndarray, policy_mask: np.ndarray):
        if self.has_batch:
            return
        
        store_size = done_indices.shape[0]
        end_index = self._index + store_size

        if end_index > self._buffer_size:
            trim_amount = end_index - self._buffer_size
            done_indices = done_indices[:-trim_amount] # todo: we are throwing away good data here
            end_index = min(end_index, self._buffer_size)

        store_range = range(self._index, end_index)

        self.context[store_range] = context[done_indices]
        self.kv_cache_lengths[store_range] = kv_cache_lengths[done_indices]
        self.log_probs[store_range] = log_probs[done_indices]
        self.values[store_range] = values[done_indices]
        self.rewards[store_range] = rewards[done_indices]
        self.policy_mask[store_range] = policy_mask[done_indices]

        self._index = end_index

    def take_batch(self) -> UpdateBatch:
        ub = UpdateBatch(
            context=self.context[:self._batch_size],
            kv_cache_lengths=self.kv_cache_lengths[:self._batch_size],
            log_probs=self.log_probs[:self._batch_size],
            values=self.values[:self._batch_size],
            rewards=self.rewards[:self._batch_size],
            policy_mask=self.policy_mask[:self._batch_size],
        )

        self.context = np.roll(self.context, -self._batch_size, axis=0)
        self.kv_cache_lengths = np.roll(self.kv_cache_lengths, -self._batch_size, axis=0)
        self.log_probs = np.roll(self.log_probs, -self._batch_size, axis=0)
        self.values = np.roll(self.values, -self._batch_size, axis=0)
        self.rewards = np.roll(self.rewards, -self._batch_size, axis=0)
        self.policy_mask = np.roll(self.policy_mask, -self._batch_size, axis=0)

        self._index -= self._batch_size

        return ub


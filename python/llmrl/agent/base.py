from typing import Iterable, Protocol
import jax


class Agent(Protocol):
    def reset(self): ...
    def act(
        self, batch_indices: jax.Array, obs: list[str], rewards: jax.Array
    ) -> tuple[jax.Array, list[str]]: ...

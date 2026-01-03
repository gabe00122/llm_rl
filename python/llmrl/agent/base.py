from typing import Iterable, Protocol
import jax
import numpy as np 

class Agent(Protocol):
    def reset(self): ...
    def act(
        self, batch_indices: np.ndarray, obs: list[str], rewards: np.ndarray, dones: np.ndarray
    ) -> tuple[np.ndarray, list[str]]: ...

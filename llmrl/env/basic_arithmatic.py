import numpy as np
import numpy.typing as npt

import random
from typing import NamedTuple


class TimeStep(NamedTuple):
    obs: list[str]
    reward: npt.NDArray[np.float32]


class BasicArithmeticEnv:
    def __init__(self):
        self._max = 10000
        self._min = 0

    def instructions(self) -> str:
        return """
Please complete the following math expression, you can show your work if needed but the final output should simply be the result on a new line of it's own.
"""

    def step(self, action: str):
        try:
            answer = action.split(" ")[-1]
            correct = int(answer) == self._result
        except ValueError:
            correct = False
        except IndexError:
            correct = False
        
        reward = 1.0 if correct else 0.0

        return TimeStep([""], np.array(reward))

    
    def reset(self, seed: int) -> TimeStep:
        random.seed(seed)
        a = random.randint(0, 10000)
        b = random.randint(0, 10000)
        self._result = a + b

        obs = f"{a} + {b} = ..."
        
        return TimeStep([obs], np.array(0.0))

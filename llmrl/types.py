from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class TimeStep(NamedTuple):
    obs: list[str]
    reward: npt.NDArray[np.float32]


class Action(NamedTuple):
    text: list[str]

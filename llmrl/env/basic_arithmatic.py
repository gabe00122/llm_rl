import random

import numpy as np

from llmrl.types import Action, TimeStep


class BasicArithmeticEnv:
    def __init__(self):
        self._max = 10000
        self._min = 0
        self._rngs = random.Random()

    def instructions(self) -> str:
        return """
Please complete the following math expression, you can show your work if needed but the final output should simply be the result on a new line of it's own.
"""

    def step(self, action: Action):
        rewards = np.zeros((len(action.text),), dtype=np.float32)

        for i, action_text in enumerate(action.text):
            try:
                print(action_text)
                print("-----")
                answer = action_text.split()[-1].replace(",", "").replace(".", "")
                correct = int(answer) == self._result

                print(f"{self._result} == {answer}")
            except ValueError:
                correct = False
                print("Invalid input")
            except IndexError:
                correct = False
                print("Invalid input")

            reward = 1.0 if correct else 0.0
            rewards[i] = reward

        return TimeStep([""], rewards)

    def reset(self) -> TimeStep:
        a = self._rngs.randint(0, 10000)
        b = self._rngs.randint(0, 10000)
        self._result = a + b

        obs = f"{a} + {b} = ..."

        return TimeStep([obs], np.array(0.0))

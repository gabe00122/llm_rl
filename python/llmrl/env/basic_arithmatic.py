import math
import random
import re

import numpy as np
from llmrl.types import Action, TimeStep


class BasicArithmeticEnv:
    def __init__(self):
        self._max = 10000
        self._min = 0
        self._rngs = random.Random()
        self._ops = ["+", "-", "*", "/"]

    def instructions(self) -> str:
        return """
Solve the arithmetic expression using +, -, * or /. Show your work if needed, but end with only the numeric result on its own line.
"""

    def step(self, action: Action):
        rewards = np.zeros((len(action.text),), dtype=np.float32)

        for i, action_text in enumerate(action.text):
            try:
                print(action_text)
                print("-----")
                answer = self._parse_answer(action_text)
                correct = math.isclose(answer, self._result, rel_tol=1e-4, abs_tol=1e-4)
                print(f"{self._result} == {answer}")
            except ValueError:
                correct = False
                print("Invalid input")

            reward = 1.0 if correct else 0.0
            rewards[i] = reward

        return TimeStep([""], rewards)

    def _parse_answer(self, action_text: str) -> float:
        # Take the last numeric token so the agent can show work before giving the answer.
        numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", action_text)
        if not numbers:
            raise ValueError("No numeric answer found")

        normalized = numbers[-1].replace(",", "")
        return float(normalized)

    def _sample_problem(self) -> tuple[int, int, str, float]:
        op = self._rngs.choice(self._ops)

        if op == "+":
            a = self._rngs.randint(self._min, self._max)
            b = self._rngs.randint(self._min, self._max)
            result = a + b
        elif op == "-":
            a = self._rngs.randint(self._min, self._max)
            b = self._rngs.randint(self._min, self._max)
            result = a - b
        elif op == "*":
            a = self._rngs.randint(self._min, self._max)
            b = self._rngs.randint(self._min, self._max)
            result = a * b
        else:  # Division
            b = self._rngs.randint(1, 12)
            max_result = max(self._min, self._max // b)
            result = self._rngs.randint(self._min, max_result)
            a = result * b

        return a, b, op, float(result)

    def reset(self) -> TimeStep:
        a, b, op, result = self._sample_problem()
        self._result = result

        obs = f"{a} {op} {b} = ..."

        return TimeStep([obs], np.array(0.0))

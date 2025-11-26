from typing import Any

from litellm import completion

from llmrl.rl_types import Action, TimeStep


class LiteAgent:
    def __init__(self, model: str, agent_count: int, instructions: str) -> None:
        self._model = model
        self._agent_count = agent_count
        self._instruction = instructions

        self.reset()

    def reset(self):
        self._first_turn = True
        self._messages: list[list[Any]] = [[] for _ in range(self._agent_count)]

    def act(self, ts: TimeStep) -> Action:
        obs = ts.obs

        if self._first_turn:
            obs = [f"{self._instruction}\n---\n{o}" for o in obs]
            self._first_turn = False

        action_text = []

        for messages, obs in zip(self._messages, obs):
            messages.append({"role": "user", "content": obs})

            response = completion(
                model=self._model,
                messages=messages,
                # base_url="http://127.0.0.1:1234/v1",
            )
            response_message = response.choices[0].message

            messages.append(response_message)
            action_text.append(response_message["content"])

        return Action(action_text)

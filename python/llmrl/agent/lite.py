from typing import Any, Iterable

from litellm import completion


class LiteAgent:
    def __init__(
        self,
        model: str,
        agent_count: int,
        instructions: str,
        *,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._agent_count = agent_count
        self._instruction = instructions
        self._base_url = base_url

        self.reset()

    def reset(self):
        self._first_turn = True
        self._messages: list[list[Any]] = [[] for _ in range(self._agent_count)]

    def act(self, obs: Iterable[str]) -> list[str]:
        if self._first_turn:
            obs = [f"{self._instruction}\n---\n{o}" for o in obs]
            self._first_turn = False

        action_text: list[str] = []

        for messages, obs in zip(self._messages, obs):
            messages.append({"role": "user", "content": obs})

            response = completion(
                model=self._model,
                messages=messages,
                base_url=self._base_url,
            )
            response_message = response.choices[0].message

            messages.append(response_message)
            action_text.append(response_message["content"])

        return action_text

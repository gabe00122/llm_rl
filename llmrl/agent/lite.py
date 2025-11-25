from typing import Any, NamedTuple
from litellm import completion

from llmrl.env.basic_arithmatic import TimeStep


class Action(NamedTuple):
    text: list[str]


class LiteAgent:
    def __init__(self, model: str, agent_num: int, instruction: str) -> None:
        self._agent_num = agent_num
        self._instruction = instruction

        self.reset()

    def reset(self):
        self._first_turn = True
        self._messages: list[list[Any]] = [[] for _ in range(self._agent_num)]
    
    def act(self, ts: TimeStep) -> Action:
        obs = ts.obs

        if self._first_turn:
            obs = [f"{self._instruction}\n---\n{o}" for o in obs]
            self._first_turn = False

        action_text = []

        for messages, obs in zip(self._messages, obs):
            messages.append({
                "role": "user",
                "content": obs
            })
        
            response = completion(model="", messages=messages)
            response_message = response.choices[0].message

            messages.append({
                "rope": "assistant",
                "content": response_message
            })

            action_text.append(response_message)
        
        return Action(action_text)



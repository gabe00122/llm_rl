from typing import Any, override

import numpy as np
from litellm import completion

from llmrl.agent.base import Agent


class LiteAgent(Agent):
    """
    A lightweight agent that uses LiteLLM to call base models from OpenRouter
    or other API providers. This agent is designed for testing base models
    against RL environments to establish baselines.
    """

    def __init__(
        self,
        model: str,
        agent_count: int,
        *,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._agent_count = agent_count
        self._base_url = base_url

        self._instructions: str | None = None
        self._messages: list[list[dict[str, Any]]] = []

        self.reset()

    def set_episode_instructions(self, instructions: str) -> None:
        """Set system instructions that will be prepended to each conversation."""
        self._instructions = instructions

    @override
    def reset(self) -> None:
        """Reset all agent conversations to initial state."""
        self._messages = [[] for _ in range(self._agent_count)]

        if self._instructions is not None:
            for messages in self._messages:
                messages.append({"role": "system", "content": self._instructions})

    @override
    def act(
        self,
        batch_indices: np.ndarray,
        obs: list[str],
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Process observations and return actions for the specified batch indices.

        Args:
            batch_indices: Indices of agents that need to act
            obs: List of observation strings for each agent in batch_indices
            rewards: Rewards from the previous step (used for tracking, not decision making)
            dones: Boolean array indicating which episodes are done

        Returns:
            Tuple of (response_indices, response_texts) where response_indices
            are the batch indices that produced responses
        """
        # Reset conversations for done episodes
        done_indices = batch_indices[dones]
        for idx in done_indices:
            self._messages[idx] = []
            if self._instructions is not None:
                self._messages[idx].append(
                    {"role": "system", "content": self._instructions}
                )

        # Process each agent in the batch
        action_texts: list[str] = []
        for idx, observation in zip(batch_indices, obs):
            self._messages[idx].append({"role": "user", "content": observation})

            response = completion(
                model=self._model,
                messages=self._messages[idx],
                base_url=self._base_url,
            )
            # litellm.completion returns ModelResponse when stream=False (default)
            content = response.choices[0].message.content or ""  # type: ignore[union-attr]

            self._messages[idx].append({"role": "assistant", "content": content})
            action_texts.append(content)

        return batch_indices.copy(), action_texts

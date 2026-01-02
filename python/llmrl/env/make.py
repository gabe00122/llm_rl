from llmrl.env.base import Env
from llmrl._envs import ArithmeticEnv, WordleEnv


def make(env_name: str, num_agents: int, seed: int, settings: dict) -> Env:
    if env_name == "arithmetic":
        return ArithmeticEnv(num_agents, seed, settings)
    elif env_name == "wordle":
        return WordleEnv(num_agents, seed, settings)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

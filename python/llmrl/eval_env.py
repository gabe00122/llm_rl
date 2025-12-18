import os
import time

import numpy as np
from flax import nnx

from rich.progress import track
from rich.console import Console

from llmrl._envs import ArithmeticEnv
from llmrl.env.base import Env
from llmrl.agent.lite import LiteAgent

# from llmrl.env.basic_arithmetic import BasicArithmeticEnv
from llmrl.agent.local import LocalAgent
from llmrl.checkpoint import load_model
from llmrl.config import LoraConfig


def main():
    console = Console()

    model_path = "./base-models/Qwen3-4B-Instruct-2507"
    lora_config = LoraConfig(False, False, 0)
    rngs = nnx.Rngs(0)
    model, tokenizer, sampling = load_model(model_path, lora_config, rngs)
    sampling = sampling._replace(temperature=0.0)

    batch_size = 64 // 1
    seq_length = 1024 * 1  # 16384 #512

    env: Env = ArithmeticEnv(batch_size)
    # print(tokenizer.vocab_size)

    # os.environ["LM_STUDIO_API_BASE"] = ""

    # env = BasicArithmeticEnv()
    # agent = LiteAgent(
    #     model="lm_studio/qwen/qwen3-30b-a3b-thinking-2507",  # "openrouter/openai/gpt-oss-20b",
    #     agent_count=1,
    #     instructions=env.instructions(),
    # )
    agent = LocalAgent(
        model,
        sampling,
        tokenizer,
        batch_size,
        seq_length,
        env.instructions(),
        rngs.agent(),
    )

    total_reward = 0
    iterations = 20  # 128 // batch_size

    total_tokens = 0

    start_time = time.time()
    for _ in track(range(iterations), console=console):
        agent.reset()
        obs = env.reset()

        console.print(obs)
        actions = agent.act(obs)
        console.print(actions)
        total_tokens += sum([len(action) for action in actions])

        obs, rewards = env.step(actions)
        total_reward += sum(rewards)

        console.print("\n---\n")
    stop_time = time.time()
    delta_time = stop_time - start_time

    console.print(f"Total reward: {total_reward / (iterations * batch_size)}")
    console.print(f"Tokens per second: {total_tokens / delta_time}")
    console.print(f"Turns per second: {(iterations * batch_size) / delta_time}")


if __name__ == "__main__":
    main()

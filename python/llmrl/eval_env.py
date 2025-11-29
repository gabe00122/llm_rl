import os

import numpy as np
from llmrl._envs import ArithmaticEnv
from llmrl.agent.lite import LiteAgent
from llmrl.env.basic_arithmatic import BasicArithmeticEnv
from llmrl.types import TimeStep


def main():
    env = ArithmaticEnv()

    os.environ["LM_STUDIO_API_BASE"] = ""

    # env = BasicArithmeticEnv()
    agent = LiteAgent(
        model="lm_studio/qwen/qwen3-30b-a3b-thinking-2507",  # "openrouter/openai/gpt-oss-20b",
        agent_count=1,
        instructions=env.instructions(),
    )

    total_reward = 0
    iterations = 100

    for _ in range(iterations):
        agent.reset()
        ts = env.reset()

        print(ts)
        action = agent.act(TimeStep([ts], np.array(0.0, dtype=np.float32)))
        print(action.text[0])

        ts, reward = env.step(action.text[0])
        total_reward += reward

        print("\n---\n")

    print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    main()

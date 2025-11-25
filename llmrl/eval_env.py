from litellm.integrations.arize.arize_phoenix import os

from llmrl.agent.lite import LiteAgent
from llmrl.env.basic_arithmatic import BasicArithmeticEnv


def main():
    os.environ["LM_STUDIO_API_BASE"] = ""

    env = BasicArithmeticEnv()
    agent = LiteAgent(
        model="openrouter/openai/gpt-oss-20b",
        agent_count=1,
        instructions=env.instructions(),
    )

    total_reward = 0
    iterations = 100

    for _ in range(iterations):
        agent.reset()
        ts = env.reset()

        action = agent.act(ts)

        ts = env.step(action)
        total_reward += ts.reward.item()

    print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    main()

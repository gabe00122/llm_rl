import os
import time

from jax import numpy as jnp
import numpy as np
from flax import nnx

from llmrl._envs import ArithmeticEnv
from llmrl.env.base import Env
from llmrl.agent.lite import LiteAgent

# from llmrl.env.basic_arithmetic import BasicArithmeticEnv
from llmrl.agent.local import LocalAgent
from llmrl.checkpoint import load_model
from llmrl.config import LoraConfig
from llmrl.model import Qwen3


def main():
    model_path = "./base-models/Qwen3-4B-Instruct-2507"
    lora_config = LoraConfig(False, False, 0)
    rngs = nnx.Rngs(0)
    model, tokenizer, sampling = load_model(model_path, lora_config, rngs)

    batch_size = 64 // 1
    seq_length = 1024 * 1  # 16384 #512

    env: Env = ArithmeticEnv(batch_size)

    print(env.step(np.array([0], dtype=np.int32), ["55"]))

    # os.environ["LM_STUDIO_API_BASE"] = ""

    # env = BasicArithmeticEnv()
    # agent = LiteAgent(
    #     model="lm_studio/qwen/qwen3-30b-a3b-thinking-2507",  # "openrouter/openai/gpt-oss-20b",
    #     agent_count=1,
    #     instructions=env.instructions(),
    # )
    model_def, model_state = nnx.split(model)

    agent = LocalAgent(
        model_def,
        model_state,
        tokenizer,
        batch_size,
        seq_length,
        env.instructions(),
        rngs.agent(),
    )
    
    env_indices = jnp.arange(batch_size, dtype=np.int32)
    rewards = jnp.zeros((batch_size,), dtype=np.float32)

    obs = env.reset(np.asarray(env_indices))

    for _ in range(5):
        env_indices, response = agent.act(env_indices, obs, rewards)

        print(env_indices)
        print(response)

    # total_reward = 0
    # iterations = 128 // batch_size

    # total_tokens = 0

    # start_time = time.time()
    # for _ in range(iterations):
    #     agent.reset()
    #     obs = env.reset()

    #     print(obs)
    #     actions = agent.act(obs)
    #     print(actions)
    #     total_tokens += sum([len(action) for action in actions])

    #     obs, rewards = env.step(actions)
    #     total_reward += sum(rewards)

    #     print("\n---\n")
    # stop_time = time.time()
    # delta_time = stop_time - start_time

    # print(f"Total reward: {total_reward / (iterations * batch_size)}")
    # print(f"Tokens per second: {total_tokens / delta_time}")
    # print(f"Turns per second: {(iterations * batch_size) / delta_time}")


if __name__ == "__main__":
    main()

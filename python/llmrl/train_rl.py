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

    batch_size = 128
    seq_length = 512  # 16384 #512

    env: Env = ArithmeticEnv(batch_size)

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
    
    env_indices = np.arange(batch_size, dtype=np.int32)
    rewards = jnp.zeros((batch_size,), dtype=np.float32)
    dones = jnp.zeros((batch_size,), dtype=jnp.bool_)

    obs = env.reset(env_indices)

    correct_count = 0
    total_count = 0

    env_time = 0.0

    start = time.time()
    for _ in range(1000):
        env_indices, actions = agent.act(env_indices, obs, rewards, dones)

        env_start = time.perf_counter()
        obs, rewards, dones = env.step(env_indices, actions)
        env_time += time.perf_counter() - env_start

        correct_count += rewards.sum().item()
        total_count += rewards.size

    total_time = time.time() - start
    print(f"Perfect Correct: {correct_count / total_count:.2%}")

    print(f"Delta: {total_time}")
    print(f"Total Episodes: {total_count}")
    print(f"Reset Time: {agent._reset_time / total_time:.2%}")
    print(f"Gen Time: {agent._gen_time / total_time:.2%}")
    print(f"Decode Time: {agent._decode_time / total_time:.2%}")
    print(f"Append Time: {agent._append_time / total_time:.2%}")
    print(f"Env Time: {env_time / total_time:.2%}")
    print(f"Accounted Time: {1 - (total_time - agent._reset_time - agent._gen_time - agent._decode_time - agent._append_time - env_time) / total_time:.2%}")
    print(f"Turns per second: {total_count / total_time}")


if __name__ == "__main__":
    main()

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

    correct_count = 0
    total_count = 0

    start = time.time()
    for _ in range(50):
        env_indices, actions = agent.act(env_indices, obs, rewards)

        obs, t_rewards, done = env.step(np.asarray(env_indices), actions)
        print(t_rewards)
        print(agent._gen.kv_cache_length.max())

        correct_count += t_rewards.sum().item()
        total_count += t_rewards.size

    end = time.time()

    delta = end - start
    print(100 / delta)
    print(agent._gen.kv_cache_length.sum().item() / delta)
    print(correct_count / total_count)

    print(delta)
    print(total_count)


if __name__ == "__main__":
    main()

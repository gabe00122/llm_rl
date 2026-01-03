from llmrl.checkpointer import Checkpointer
from llmrl.experiement import Experiment
import time

from jax import numpy as jnp
from llmrl.model.value_network import ValueParam
import numpy as np
from flax import nnx

from llmrl.agent.local import LocalAgent
from llmrl.base_model_loader import load_base_model
from llmrl.logger import create_logger
from llmrl.env.make import make_env
from rich.console import Console
import optax


def main():
    experiment = Experiment.from_config_file("configs/test.json")
    config = experiment.config
    console = Console()
    logger = create_logger(config, experiment.unique_token, console)

    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    rngs = nnx.Rngs(0)
    model, tokenizer, sampling = load_base_model(model_name, rngs)

    checkpointer = Checkpointer(experiment.checkpoints_url)

    eval_batch_size = config.eval_envs
    env = make_env(config.env.name, eval_batch_size, experiment.environments_seed, config.env)

    opt = nnx.Optimizer(model=model, tx=optax.adamw(config.optimizer.lr), wrt=nnx.Any(ValueParam, nnx.LoRAParam))
    model_def, model_state = nnx.split(model)
    opt_def, opt_state = nnx.split(opt)

    agent = LocalAgent(
        model_def,
        model_state,
        opt_def,
        opt_state,
        tokenizer,
        checkpointer,
        config,
        env.instructions(),
        logger,
        rngs.agent(),
    )

    env_indices = np.arange(eval_batch_size, dtype=np.int32)
    rewards = np.zeros((eval_batch_size,), dtype=np.float32)
    dones = np.zeros((eval_batch_size,), dtype=jnp.bool_)

    obs = env.reset(env_indices)

    correct_count = 0
    total_count = 0

    env_time = 0.0

    start = time.time()
    while agent.update_episodes < config.total_update_episodes:
        env_indices, actions = agent.act(env_indices, obs, rewards, dones)

        env_start = time.perf_counter()
        obs, rewards, dones = env.step(env_indices, actions)
        env_time += time.perf_counter() - env_start

        correct_count += rewards.sum().item()
        total_count += rewards.size
        # print(f"Score: {rewards.sum().item() / rewards.size:.2%}")

    total_time = time.time() - start
    print(f"Percent Correct: {correct_count / total_count:.2%}")

    print(f"Delta: {total_time}")
    print(f"Total Episodes: {total_count}")
    print(f"Reset Time: {agent._reset_time / total_time:.2%}")
    print(f"Gen Time: {agent._gen_time / total_time:.2%}")
    print(f"Decode Time: {agent._decode_time / total_time:.2%}")
    print(f"Append Time: {agent._append_time / total_time:.2%}")
    print(f"Env Time: {env_time / total_time:.2%}")
    print(f"Accounted Time: {1 - (total_time - agent._reset_time - agent._gen_time - agent._decode_time - agent._append_time - env_time) / total_time:.2%}")
    print(f"Turns per second: {total_count / total_time}")

    checkpointer.close()


if __name__ == "__main__":
    main()

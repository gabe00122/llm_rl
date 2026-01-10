from llmrl.agent.local import Trainer
from llmrl.agent.local import BufferedEpisodeListener
from llmrl.utils.performance import PerformanceTracker
from llmrl.checkpointer import Checkpointer
from llmrl.experiement import Experiment

from llmrl.model.value_network import ValueParam
import numpy as np
from flax import nnx

from llmrl.agent.local import LocalAgent
from llmrl.base_model_loader import load_base_model
from llmrl.logger import create_logger
from llmrl.env.make import make_env
from rich.console import Console
import optax

def _make_optimizer(model, config) -> nnx.Optimizer:
    lr = optax.warmup_constant_schedule(0, config.optimizer.lr, (100000//4)//10)
    return nnx.Optimizer(
        model=model,
        tx=optax.MultiSteps(
            optax.adam(lr, b1=0.9, b2=0.95), every_k_schedule=4
        ), wrt=nnx.Any(ValueParam, nnx.LoRAParam)
    )


def train_cli(config_url: str):
    experiment = Experiment.from_config_file(config_url)
    config = experiment.config
    console = Console()
    preformace_tracker = PerformanceTracker()
    logger = create_logger(config, experiment.unique_token, console)

    rngs = nnx.Rngs(experiment.params_seed)
    model, tokenizer, sampling = load_base_model(config.base_model, rngs)
    model.initialize_lora(config.lora, rngs=rngs)

    checkpointer = Checkpointer(experiment.checkpoints_url)

    eval_batch_size = config.eval_envs
    env = make_env(config.env.name, eval_batch_size, experiment.environments_seed, config.env)

    opt = _make_optimizer(model, config)

    agent = LocalAgent(
        model,
        tokenizer,
        config,
        logger,
        preformace_tracker,
        rngs.agent(),
    )

    agent.set_episode_instructions(env.instructions())

    trainer = Trainer(
        agent,
        opt,
        checkpointer,
        logger,
        config,
    )
    agent.episode_listener = BufferedEpisodeListener(
        config.eval_envs,
        config.update_envs,
        config.max_seq_length,
        trainer,
    )

    env_indices = np.arange(eval_batch_size, dtype=np.int32)
    rewards = np.zeros((eval_batch_size,), dtype=np.float32)
    dones = np.zeros((eval_batch_size,), dtype=np.bool_)

    obs = env.reset(env_indices)

    while trainer.progress < 1.0:
        env_indices, actions = agent.act(env_indices, obs, rewards, dones)
        with preformace_tracker.time("env_step"):
            obs, rewards, dones = env.step(env_indices, actions)

    checkpointer.close()

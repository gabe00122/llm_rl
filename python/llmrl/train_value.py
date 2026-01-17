from llmrl.update_step import update_step
from llmrl.buffer import UpdateBuffer
from llmrl.buffer import UpdateBatch

from jax import numpy as jnp
import numpy as np
import optax
from flax import nnx
from llmrl.agent.local import (
    BufferedEpisodeListener,
    EpisodeSaver,
    LocalAgent,
    MultiEpisodeListener,
    Trainer,
)
from llmrl.base_model_loader import load_base_model
from llmrl.checkpointer import Checkpointer
from llmrl.config import Config
from llmrl.env.make import make_env
from llmrl.experiement import Experiment
from llmrl.logger import create_logger
from llmrl.model.value_network import ValueParam
from llmrl.utils.performance import PerformanceTracker
from rich.console import Console


def _make_optimizer(model, config: Config) -> nnx.Optimizer:
    return nnx.Optimizer(
        model=model,
        tx=optax.adamw(config.optimizer.lr, b1=0.9, b2=0.95),
        wrt=nnx.Any(ValueParam, nnx.LoRAParam),
    )


def train_value_cli(config_url: str):
    experiment = Experiment.from_config_file(config_url)
    config = experiment.config
    console = Console()
    # performance_tracker = PerformanceTracker()
    logger = create_logger(config, experiment.unique_token, console)

    rngs = nnx.Rngs(experiment.params_seed)
    model, tokenizer, sampling = load_base_model(config.base_model, rngs)
    # model.initialize_lora(config.lora, rngs=rngs)

    checkpointer = Checkpointer(experiment.checkpoints_url)

    eval_batch_size = config.eval_envs
    env = make_env(
        config.env.name, eval_batch_size, experiment.environments_seed, config.env
    )

    opt = _make_optimizer(model, config)

    opt_def, opt_state = nnx.split(opt)
    model_def, model_state = nnx.split(model)

    buffer = UpdateBuffer(100 * 100, 16, config.max_seq_length)
    for i in range(100):
        episode_data = UpdateBatch.load_npz(f"./offline_data/episodes_{i}.npz")
        buffer.store(episode_data)

    step = 0
    while buffer.has_batch:
        batch = buffer.take_batch()
        opt_state, model_state, metrics = update_step(
            opt_def,
            opt_state,
            model_def,
            model_state,
            batch,
            config.loss,
            jnp.array(0.0),
        )
        metrics["reward"] = batch.rewards.sum(axis=1).mean()
        logger.log_dict(metrics, step)
        step += 1

    logger.close()
    checkpointer.close()

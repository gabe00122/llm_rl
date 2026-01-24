from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from rich.console import Console
from pathlib import Path

from llmrl.model.value_network import ValueParam
from llmrl.update_step import update_step
from llmrl.buffer import UpdateBuffer
from llmrl.buffer import UpdateBatch
from llmrl.base_model_loader import load_base_model
from llmrl.checkpointer import Checkpointer
from llmrl.experiement import Experiment
from llmrl.logger import create_logger
from llmrl.utils.optimizer import make_optimizer


@jax.jit(static_argnames=('model_def'))
def calculate_values(model_def, model_state, context: jax.Array):
    model = nnx.merge(model_def, model_state)
    positions = jnp.arange(context.shape[0])
    _, values, _ = model(context[None, :], positions[None, :], None)
    return jnp.squeeze(values, 0)


def train_value_cli(config_url: str, offline_data_url: str):
    experiment = Experiment.from_config_file(config_url)

    config = experiment.config
    console = Console()
    logger = create_logger(config, experiment.unique_token, console)

    rngs = nnx.Rngs(experiment.params_seed)
    model, tokenizer, sampling = load_base_model(config.base_model, rngs)

    data_dir = Path(offline_data_url)
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"offline_data_url {offline_data_url} does not exist or is not a directory.")

    data_files = sorted(list(data_dir.glob("*.npz")))
    if not data_files:
        raise ValueError(f"No .npz files found in {offline_data_url}")

    first_batch = UpdateBatch.load_npz(data_files[0])
    num_episodes_per_file = first_batch.context.shape[0]
    buffer_size = config.update_envs + num_episodes_per_file

    buffer = UpdateBuffer(buffer_size, config.update_envs, config.max_seq_length)
    buffer.store(first_batch)

    total_updates = (len(data_files) * num_episodes_per_file) // config.update_envs
    opt = make_optimizer(model, config, total_updates)
    opt_def, opt_state = nnx.split(opt)
    model_def, model_state = nnx.split(model)

    ref_context = first_batch.context[0]
    output_values = np.zeros((total_updates, config.max_seq_length))
    
    step = 0
    file_idx = 1
    while file_idx < len(data_files) or buffer.has_batch:
        # Load more data if buffer doesn't have a batch and there are more files
        if not buffer.has_batch and file_idx < len(data_files):
            batch_data = UpdateBatch.load_npz(data_files[file_idx])
            buffer.store(batch_data)
            file_idx += 1
            # Check again if we have enough after loading
            if not buffer.has_batch:
                continue

        if not buffer.has_batch:
            break

        batch = buffer.take_batch()
        opt_state, model_state, metrics = update_step(
            opt_def,
            opt_state,
            model_def,
            model_state,
            batch,
            config.loss,
            True,
        )
        values = calculate_values(model_def, model_state, ref_context)
        output_values[step] = np.array(values)

        metrics["reward"] = batch.rewards.sum(axis=1).mean()
        logger.log_dict(metrics, step)
        step += 1
    
    np.save("./values", output_values)

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        opt = nnx.merge(opt_def, opt_state)
        model = nnx.merge(model_def, model_state)
        checkpointer.save({"opt": opt, "model": model}, step, ValueParam)

    logger.close()

from flax import nnx
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


def train_value_cli(config_url: str, offline_data_url: str):
    experiment = Experiment.from_config_file(config_url)

    config = experiment.config
    console = Console()
    logger = create_logger(config, experiment.unique_token, console)

    rngs = nnx.Rngs(experiment.params_seed)
    model, tokenizer, sampling = load_base_model(config.base_model, rngs)

    opt = make_optimizer(model, config, config.total_update_episodes)
    opt_def, opt_state = nnx.split(opt)
    model_def, model_state = nnx.split(model)

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
        metrics["reward"] = batch.rewards.sum(axis=1).mean()
        logger.log_dict(metrics, step)
        step += 1

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        opt = nnx.merge(opt_def, opt_state)
        model = nnx.merge(model_def, model_state)
        checkpointer.save({"opt": opt, "model": model}, step, nnx.filterlib.Any(nnx.OptState, ValueParam))

    logger.close()

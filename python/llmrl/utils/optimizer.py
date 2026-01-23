from llmrl.config import SGDConfig
import optax
from flax import nnx
from llmrl.config import Config, WarmupCosineConfig, AdamWConfig
from llmrl.model.value_network import ValueParam

def make_optimizer(model: nnx.Module, config: Config, total_steps: int) -> nnx.Optimizer:
    """
    Creates an nnx.Optimizer based on the provided configuration.
    
    Args:
        model: The model to optimize.
        config: The execution configuration.
        total_steps: Total number of steps for scheduling.
    """
    
    if config.schedule is None:
        tx_lr = config.optimizer.lr
    elif isinstance(config.schedule, WarmupCosineConfig):
        warmup_steps = int(total_steps * config.schedule.warmup_ratio)
        tx_lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.optimizer.lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
        )
    else:
        raise ValueError(f"Unsupported schedule type: {type(config.schedule)}")

    if isinstance(config.optimizer, SGDConfig):
        tx = optax.sgd(
            learning_rate=tx_lr,
        )
    elif isinstance(config.optimizer, AdamWConfig):
        tx = optax.adamw(
            learning_rate=tx_lr,
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
            weight_decay=config.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {type(config.optimizer)}")

    if config.multi_step is not None:
        tx = optax.MultiSteps(tx, every_k_schedule=config.multi_step)

    return nnx.Optimizer(
        model=model,
        tx=tx,
        wrt=nnx.Any(ValueParam, nnx.LoRAParam),
    )

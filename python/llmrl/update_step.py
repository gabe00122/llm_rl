from typing import NamedTuple
import jax
from jax import numpy as jnp
from flax import nnx
from llmrl.rollout import Rollout, UpdateBatch
from einops import rearrange


class UpdateSettings(NamedTuple):
    mini_batch_count: int


def calculate_advantages(
    rewards: jax.Array, values: jax.Array, discount: float, gae_lambda: float, norm_adv: bool
) -> tuple[jax.Array, jax.Array]:
    def _body(acc, xs):
        rewards, discount, v_tp1 = xs
        acc = rewards + discount * ((1 - gae_lambda) * v_tp1 + gae_lambda * acc)
        return acc, acc

    # swap to time major
    _, targets = jax.lax.scan(
        _body,
        jnp.array(0.0, dtype=jnp.float32),
        (jnp.swapaxes(rewards, 0, 1), jnp.swapaxes(values, 0, 1)),
        reverse=True,
    )
    targets = jnp.swapaxes(targets, 0, 1)

    advantages = targets - values

    # rollout norm
    if norm_adv:
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

    return advantages, targets

def loss():
    pass

@jax.jit(static_argnames=('opt_def', 'model_def', 'settings'), donate_argnames=('opt_state', 'model_state'))
def update_step(opt_def, opt_state, model_def, model_state, rollout: UpdateBatch, settings: UpdateSettings):
    opt = nnx.merge(opt_def, opt_state)
    model = nnx.merge(model_def, model_state)

    batch_len, seq_len = rollout.context.shape

    # batch_range = jnp.arange(batch_len, dtype=jnp.int32)
    seq_range = jnp.arange(seq_len, dtype=jnp.int32)
    policy_mask = jnp.logical_and(rollout.prompt_mask, seq_range[None, :] < rollout.context_lengths[:, None])

    values = jnp.where(policy_mask, rollout.values, 0.0)

    advantages, targets = calculate_advantages(rollout.rewards, values, 0.99, 0.9, False)

    update_data = (rollout, advantages, targets)

    update_data = jax.tree.map()

    return nnx.state(opt), nnx.state(model)

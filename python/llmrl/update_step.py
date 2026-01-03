from llmrl.config import LossConfig
from typing import NamedTuple
import distrax
import jax
from jax import numpy as jnp
from flax import nnx
from llmrl.model.qwen3 import Qwen3
from llmrl.model.value_network import ValueParam
from llmrl.buffer import UpdateBatch


class UpdateSettings(NamedTuple):
    mini_batch_count: int


def calculate_advantages(
    rewards: jax.Array, values: jax.Array, discount: float, gae_lambda: float, norm_adv: bool
) -> tuple[jax.Array, jax.Array]:
    def _body(acc, xs):
        rewards, v_tp1 = xs
        acc = rewards + discount * ((1 - gae_lambda) * v_tp1 + gae_lambda * acc)
        return acc, acc

    rolled_values = jnp.roll(values, -1, axis=1).at[:, -1].set(0)

    # swap to time major
    _, targets = jax.lax.scan(
        _body,
        jnp.zeros((rewards.shape[0],), dtype=jnp.float32),
        (jnp.swapaxes(rewards, 0, 1), jnp.swapaxes(rolled_values, 0, 1)),
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

def loss_fn(model: Qwen3, rollout: UpdateBatch, advantages: jax.Array, targets: jax.Array, config: LossConfig) -> tuple[jax.Array, dict[str, jax.Array]]:
    batch_len, seq_len = rollout.context.shape
    
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    positions = jnp.repeat(positions, batch_len, 0)

    logits, values, _ = model(jnp.asarray(rollout.context), positions)
    policy = distrax.Categorical(logits=logits[:, :-1, :])

    log_prob = policy.log_prob(rollout.context[:, 1:])

    value_loss = 0.5 * config.vf_coef * jnp.square(values - targets).mean(where=rollout.policy_mask)
    actor_loss = -(log_prob * advantages[:, :-1]).mean(where=rollout.policy_mask[:, :-1])
    loss = value_loss + actor_loss

    # entropy_loss = -0.002 * policy.entropy().mean(where=rollout.policy_mask[:, :-1])

    metrics = {
        'value_loss': value_loss,
        'actor_loss': actor_loss
    }

    return loss, metrics


@jax.jit(static_argnames=('opt_def', 'model_def', 'config'), donate_argnames=('opt_state', 'model_state'))
def update_step(opt_def, opt_state, model_def, model_state, rollout: UpdateBatch, config: LossConfig):
    opt = nnx.merge(opt_def, opt_state)
    model = nnx.merge(model_def, model_state)

    batch_len, seq_len = rollout.context.shape

    seq_range = jnp.arange(seq_len, dtype=jnp.int32)
    policy_mask = jnp.logical_and(rollout.policy_mask, seq_range[None, :] < rollout.kv_cache_lengths[:, None])

    values = jnp.where(policy_mask, rollout.values, 0.0)

    rollout = rollout._replace(
        policy_mask=policy_mask,
        values=values
    )

    advantages, targets = calculate_advantages(jnp.asarray(rollout.rewards), values, config.gea_discount, config.gea_lambda, False)

    # do the update
    diff = nnx.DiffState(0, opt.wrt)
    grad, metrics = nnx.grad(loss_fn, argnums=diff, has_aux=True)(model, rollout, advantages, targets, config)

    opt.update(model, grad)

    return nnx.state(opt), nnx.state(model), metrics

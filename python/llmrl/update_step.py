from llmrl.config import LossConfig
import distrax
import jax
from jax import numpy as jnp
from flax import nnx
from llmrl.model.qwen3 import Qwen3
from llmrl.buffer import UpdateBatch


def calculate_advantages(
    rewards: jax.Array, values: jax.Array, discount: float, gae_lambda: float
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

    # advantages = (advantages - advantages.mean()) / (
    #     advantages.std() + 1e-8
    # )

    return advantages, targets

def loss_fn(model: Qwen3, rollout: UpdateBatch, advantages: jax.Array, targets: jax.Array, config: LossConfig, bounds_mask: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    batch_len, seq_len = rollout.context.shape

    # jax.debug.print("last target: {target}", target=targets[:, -1])

    policy_mask = rollout.policy_mask[:, :-1]
    advantages = advantages[:, :-1]
    
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    positions = jnp.repeat(positions, batch_len, 0)

    logits, values, _ = model(jnp.asarray(rollout.context), positions)
    policy = distrax.Categorical(logits=logits[:, :-1])

    log_prob = policy.log_prob(rollout.context[:, 1:])

    value_loss = 0.5 * jnp.square(values - targets).mean(where=bounds_mask)
    # actor_loss = -(log_prob * advantages).mean(where=policy_mask)

    pg_ratio = jnp.exp(log_prob - rollout.log_probs[:, :-1])
    pg_loss1 = pg_ratio * advantages
    pg_loss2 = (
        jnp.clip(pg_ratio, 1.0 - config.pg_clip_low, 1.0 + config.pg_clip_high) * advantages
    )
    actor_loss = -jnp.minimum(pg_loss1, pg_loss2).mean(where=policy_mask)
    
    # entropy_loss = -0.0001 * policy.entropy().mean(where=policy_mask)
    loss = config.vf_coef * value_loss + actor_loss # + entropy_loss

    metrics = {
        'value_loss': value_loss,
        'actor_loss': actor_loss,
    }

    return loss, metrics


@jax.jit(static_argnames=('opt_def', 'model_def', 'config'), donate_argnames=('opt_state', 'model_state'))
def update_step(opt_def, opt_state, model_def, model_state, rollout: UpdateBatch, config: LossConfig, progress):
    opt = nnx.merge(opt_def, opt_state)
    model = nnx.merge(model_def, model_state)

    batch_len, seq_len = rollout.context.shape

    seq_range = jnp.arange(seq_len, dtype=jnp.int32)
    bounds_mask = seq_range[None, :] < rollout.kv_cache_lengths[:, None]

    policy_mask = jnp.logical_and(rollout.policy_mask, bounds_mask)

    values = jnp.where(bounds_mask, rollout.values, 0.0)

    rollout = rollout._replace(
        policy_mask=policy_mask,
        values=values
    )

    advantages, targets = calculate_advantages(jnp.asarray(rollout.rewards), values, config.gae_discount, config.gae_lambda)

    # do the update
    diff = nnx.DiffState(0, opt.wrt)
    grad, metrics = nnx.grad(loss_fn, argnums=diff, has_aux=True)(model, rollout, advantages, targets, config, bounds_mask)

    opt.update(model, grad)

    metrics['value'] = values.mean(where=bounds_mask)
    metrics['episode_length'] = rollout.kv_cache_lengths.mean()

    return nnx.state(opt), nnx.state(model), metrics

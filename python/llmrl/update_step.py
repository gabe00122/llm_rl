import distrax
import jax
from flax import nnx
from jax import numpy as jnp
from llmrl.buffer import UpdateBatch
from llmrl.config import LossConfig
from llmrl.model.qwen3 import Qwen3


def calculate_advantages(
    rewards: jax.Array, values: jax.Array, discount: float, gae_lambda: float
) -> tuple[jax.Array, jax.Array]:
    def _body(acc, xs):
        rewards, v_tp1 = xs
        acc = rewards + discount * ((1 - gae_lambda) * v_tp1 + gae_lambda * acc)
        return acc, acc

    # swap to time major
    _, targets = jax.lax.scan(
        _body,
        jnp.zeros((rewards.shape[0],), dtype=jnp.float32),
        (jnp.swapaxes(rewards[:, 1:], 0, 1), jnp.swapaxes(values[:, 1:], 0, 1)),
        reverse=True,
    )
    targets = jnp.swapaxes(targets, 0, 1)

    advantages = targets - values[:, :-1]

    # advantages = (advantages - advantages.mean()) / (
    #     advantages.std() + 1e-8
    # )

    return advantages, targets


def loss_fn(
    model: Qwen3,
    rollout: UpdateBatch,
    advantages: jax.Array,
    targets: jax.Array,
    config: LossConfig,
    bounds_mask: jax.Array,
    value_only: bool,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    batch_len, seq_len = rollout.context.shape

    policy_mask = rollout.policy_mask

    positions = jnp.repeat(jnp.arange(seq_len, dtype=jnp.int32)[None, :], batch_len, 0)

    logits, values_logits, _ = model(jnp.asarray(rollout.context), positions)
    values = model.get_value(values_logits)
    policy = distrax.Categorical(logits=logits[:, :-1])

    log_prob = policy.log_prob(rollout.context[:, 1:])

    # temp
    if value_only:
        # use fresh values for the target
        _, targets = calculate_advantages(
            jnp.asarray(rollout.rewards), jax.lax.stop_gradient(values), config.gae_discount, config.gae_lambda
        )
    # temp

    value_loss = model.get_value_loss(values_logits[:, :-1], targets).mean(
        where=bounds_mask[:, :-1]
    )

    loss = config.vf_coef * value_loss
    metrics = {
        "value_loss": value_loss,
        "value": values.mean(where=bounds_mask),
    }

    if not value_only:
        log_prob = policy.log_prob(rollout.context[:, 1:])
        # actor_loss: jax.Array = -(log_prob * advantages).mean(where=policy_mask[:, :-1])
        pg_ratio = jnp.exp(log_prob - rollout.log_probs)
        pg_loss1 = pg_ratio * advantages
        pg_loss2 = (
            jnp.clip(pg_ratio, 1.0 - config.pg_clip_low, 1.0 + config.pg_clip_high)
            * advantages
        )
        actor_loss = -jnp.minimum(pg_loss1, pg_loss2).mean(where=policy_mask[:, :-1])

        metrics = {**metrics, "actor_loss": actor_loss}
        loss = loss + actor_loss
    else:
        _, true_targets = calculate_advantages(
            jnp.asarray(rollout.rewards), jax.lax.stop_gradient(values), config.gae_discount, 1.0
        )
        true_value_loss = 0.5 * jnp.square(values[:, :-1] - true_targets).mean(
            where=bounds_mask[:, :-1]
        )
        metrics = {**metrics, "true_value_loss": true_value_loss}

    return loss, metrics


@jax.jit(
    static_argnames=("policy_opt_def", "value_opt_def", "model_def", "config", "value_only"),
    donate_argnames=("policy_opt_state", "value_opt_state", "model_state"),
)
def update_step(
    policy_opt_def,
    policy_opt_state,
    value_opt_def,
    value_opt_state,
    model_def,
    model_state,
    rollout: UpdateBatch,
    config: LossConfig,
    value_only: bool,
):
    policy_opt: nnx.Optimizer = nnx.merge(policy_opt_def, policy_opt_state)
    value_opt: nnx.Optimizer = nnx.merge(value_opt_def, value_opt_state)
    model: Qwen3 = nnx.merge(model_def, model_state)

    batch_len, seq_len = rollout.context.shape

    seq_range = jnp.arange(seq_len, dtype=jnp.int32)
    bounds_mask = seq_range[None, :] < rollout.kv_cache_lengths[:, None]
    policy_mask = jnp.logical_and(rollout.policy_mask, bounds_mask)

    values = jnp.where(bounds_mask, rollout.values, 0.0)

    rollout = rollout._replace(policy_mask=policy_mask, values=values)

    advantages, targets = calculate_advantages(
        jnp.asarray(rollout.rewards), values, config.gae_discount, config.gae_lambda
    )

    # do the update
    diff = nnx.DiffState(0, nnx.Any(policy_opt.wrt, value_opt.wrt))
    grad, metrics = nnx.grad(loss_fn, argnums=diff, has_aux=True)(
        model, rollout, advantages, targets, config, bounds_mask, value_only
    )

    policy_opt.update(model, grad)
    value_opt.update(model, grad)

    # metrics["value"] = values.mean(where=bounds_mask)
    metrics["episode_length"] = rollout.kv_cache_lengths.mean()

    return nnx.state(policy_opt), nnx.state(value_opt), nnx.state(model), metrics

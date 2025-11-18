from datasets import load_dataset
import jax
from jax import numpy as jnp
from flax import nnx
import optax

from llmrl.checkpoint import load_model
from llmrl.config import LoraConfig
from llmrl.main import PAD_ID, chat, encode_input
from llmrl.model import Qwen3

def create_conversation(tokenizer, sample, pad_size: int):
    messages = [[
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ] for user, assistant in zip(sample["player"], sample["alien"])]

    tokens = encode_input(tokenizer, messages, pad_size)
    return {"messages": tokens}


def loss_fn(model: Qwen3, tokens: jax.Array):
    batch, seq_len = tokens.shape
    positions = jnp.repeat(jnp.arange(seq_len)[None, :], batch, axis=0)

    logits, _ = model(tokens, positions)
    logits = logits[:, :-1, :]
    labels = tokens[:, 1:]

    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, labels),
        where=labels != PAD_ID,
    )

    return loss, loss


@nnx.jit(donate_argnums=(0, 1))
def update_step(optimizer: nnx.Optimizer, model: Qwen3, tokens: jax.Array):
    diff = nnx.DiffState(0, nnx.LoRAParam)
    grad, loss = nnx.grad(loss_fn, argnums=diff, has_aux=True)(model, tokens)
    optimizer.update(model, grad)

    return loss


def main():
    epochs = 6
    batch_size = 5
    seq_length = 256

    # model_path = "./base-models/qwen3-0.6b"
    model_path = "./base-models/Qwen3-4B-Instruct-2507"
    rngs = nnx.Rngs(0)
    lora_config = LoraConfig(True, True, 16)
    model, tokenizer, sampling = load_model(model_path, lora_config, rngs)

    npc_type = "venusian"
    dataset = load_dataset("bebechien/MobileGameNPC", npc_type, split="train")
    dataset = dataset.map(lambda sample: create_conversation(tokenizer, sample, seq_length), remove_columns=dataset.features, batched=True)
    dataset = dataset.with_format("jax")

    # lr = optax.linear_schedule(0.0002, 0.0, )
    total_steps = epochs * len(dataset)
    warmup_steps = total_steps // 10
    learning_rate = optax.warmup_cosine_decay_schedule(
        0,
        0.0002,
        warmup_steps,
        total_steps,
    )
    tx = optax.adamw(learning_rate, weight_decay=0.01)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.LoRAParam)

    for epoch in range(epochs):
        dataset = dataset.shuffle()
        for batch in dataset.iter(batch_size, drop_last_batch=True):
            tokens = batch["messages"]
            loss = update_step(optimizer, model, tokens)
            print(loss)

    del optimizer
    batch_size = 1
    seq_length = 2048
    chat(model, tokenizer, sampling, batch_size, seq_length, rngs)



if __name__ == "__main__":
    main()

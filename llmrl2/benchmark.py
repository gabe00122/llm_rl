from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import jax
from jax import numpy as jnp
from flax import nnx
from rich.console import Console
from rich.table import Table

from llmrl2.checkpoint import load_param_dict
from llmrl2.config import load_config, Config
from llmrl2.model import Qwen3


console = Console()


@dataclass
class BenchmarkConfig:
    model_path: str
    batch_sizes: tuple[int, ...]
    sequence_lengths: tuple[int, ...]
    warmup_steps: int
    steps: int
    seed: int


def _parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Simple JAX benchmark for the Qwen3 model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./base-models/qwen3-0.6b",
        help="Path to the HuggingFace style checkpoint (config + safetensors).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=(1, 2, 4),
        help="Batch sizes to benchmark. Provide one or more integers.",
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=(64, 128, 256),
        help="Sequence lengths to benchmark. Provide one or more integers.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Additional warmup runs after compilation.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of timed iterations per configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for RNG and model initialization.",
    )

    args = parser.parse_args()
    return BenchmarkConfig(
        model_path=args.model_path,
        batch_sizes=tuple(args.batch_sizes),
        sequence_lengths=tuple(args.sequence_lengths),
        warmup_steps=args.warmup,
        steps=args.steps,
        seed=args.seed,
    )


def _prepare_model(cfg: BenchmarkConfig) -> tuple[Qwen3, Config]:
    config = load_config(f"{cfg.model_path}/config.json")
    params = load_param_dict(f"{cfg.model_path}/model.safetensors")
    model = Qwen3(config, rngs=nnx.Rngs(cfg.seed))
    model.load_params(params)
    del params
    return model, config


def _make_tokens(config: Config, batch: int, seq: int, seed: int) -> jax.Array:
    key = jax.random.PRNGKey(seed)
    tokens = jax.random.randint(
        key, (batch, seq), minval=0, maxval=config.vocab_size, dtype=jnp.int32
    )
    return tokens


def _block_forward(fn, tokens: jax.Array) -> None:
    logits = fn(tokens).logits_parameter()
    jax.block_until_ready(logits)


def benchmark_shape(
    fn,
    config: Config,
    batch: int,
    seq: int,
    warmup_steps: int,
    steps: int,
    seed: int,
) -> tuple[float, float, float, float]:
    tokens = _make_tokens(config, batch, seq, seed)

    compile_start = time.perf_counter()
    _block_forward(fn, tokens)
    compile_time = time.perf_counter() - compile_start

    for _ in range(max(0, warmup_steps)):
        _block_forward(fn, tokens)

    start = time.perf_counter()
    for _ in range(steps):
        _block_forward(fn, tokens)
    run_time = time.perf_counter() - start

    avg_latency_ms = (run_time / steps) * 1_000
    tokens_per_second = (batch * seq * steps) / run_time
    samples_per_second = (batch * steps) / run_time
    # Return compile time so the caller can display it, but also throughput metrics.
    return compile_time, avg_latency_ms, tokens_per_second, samples_per_second


def run_benchmark(cfg: BenchmarkConfig) -> None:
    model, config = _prepare_model(cfg)
    compiled = nnx.jit(model)

    table = Table(title="Qwen3 JAX Benchmark")
    table.add_column("Batch", justify="right")
    table.add_column("Seq Len", justify="right")
    table.add_column("Compile (s)", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Tokens/s", justify="right")
    table.add_column("Samples/s", justify="right")

    for batch in cfg.batch_sizes:
        for seq in cfg.sequence_lengths:
            compile_time, latency_ms, tok_s, samp_s = benchmark_shape(
                compiled,
                config,
                batch,
                seq,
                cfg.warmup_steps,
                cfg.steps,
                cfg.seed,
            )
            table.add_row(
                str(batch),
                str(seq),
                f"{compile_time:.2f}",
                f"{latency_ms:.2f}",
                f"{tok_s:,.0f}",
                f"{samp_s:,.2f}",
            )

    console.print(table)


def main() -> None:
    cfg = _parse_args()
    run_benchmark(cfg)


if __name__ == "__main__":
    main()

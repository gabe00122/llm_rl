from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any

from rich.console import Console
import jax
import numpy as np
from tensorboardX import SummaryWriter
import wandb

from llmrl.config import LoggerConfig

Metrics = dict[str, jax.Array | float | int]


def json_normalize(data: dict, sep: str = ".") -> dict:
    out = {}

    def flatten(x, name=""):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + sep)
        else:
            if isinstance(x, jax.Array):
                x = x.item()

            out[name[:-len(sep)]] = x

    flatten(data)
    return out

class BaseLogger(ABC):
    @abstractmethod
    def __init__(self, unique_token: str):
        pass

    def log_dict(self, data: Metrics, step: int) -> None:
        pass

    def close(self) -> None:
        pass


class MultiLogger(BaseLogger):
    def __init__(self, loggers: list[BaseLogger]) -> None:
        self.loggers = loggers

    def log_dict(self, data: dict, step: int) -> None:
        for logger in self.loggers:
            logger.log_dict(data, step)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


class TensorboardLogger(BaseLogger):
    def __init__(self, unique_token: str) -> None:
        log_path = Path("./logs/tensorboard") / unique_token
        os.makedirs(log_path, exist_ok=True)
        self.writer = SummaryWriter(log_path.as_posix())

    def log_dict(self, data: Metrics, step: int) -> None:
        data = json_normalize(data, sep="/")

        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()


class ConsoleLogger(BaseLogger):
    def __init__(self, unique_token: str, console: Console) -> None:
        self._console = console

    def log_dict(self, data: Metrics, step: int) -> None:
        data = json_normalize(data, sep=".")

        keys = data.keys()
        values = []
        for value in data.values():
            if hasattr(value, "item"):
                 value = value.item()
            values.append(f"{value:.3f}" if isinstance(value, float) else value)

        log_str = "\n".join([f"{key}: {value}" for key, value in zip(keys, values)])
        log_str = f"step: {step}\n{log_str}"
        self._console.print(log_str)


class WandbLogger(BaseLogger):
    def __init__(self, unique_token: str, settings: LoggerConfig):
        wandb.init(project=settings.project_name, name=unique_token)

    def log_dict(self, data: Metrics, step: int) -> None:
        normalized_data = json_normalize(data)

        wandb.log(normalized_data, step=step)

    def close(self) -> None:
        wandb.finish()


def create_logger(settings: LoggerConfig, unique_token: str, console: Console) -> BaseLogger:
    loggers: list[BaseLogger] = []

    if settings.use_tb:
        loggers.append(TensorboardLogger(unique_token))
    if settings.use_console:
        loggers.append(ConsoleLogger(unique_token, console))
    if settings.use_wandb:
        loggers.append(WandbLogger(unique_token, settings))

    return MultiLogger(loggers)

import typer
from llmrl.buffer import UpdateBatch
from llmrl.experiement import Experiment
from llmrl.train_rl import train_cli
from llmrl.util import load_tokenizer
from llmrl.utils.episode_to_jsonl import episode_to_jsonl as episode_to_jsonl_fn

app = typer.Typer(pretty_exceptions_show_locals=False)

# @app.command()
# def chat(config_url: str):


@app.command()
def train(config_url: str):
    train_cli(config_url)


@app.command()
def episode_to_jsonl(config_url: str):
    experiment = Experiment.load(config_url)
    tokenizer = load_tokenizer(experiment.config.base_model)
    episode_to_jsonl_fn(
        "./episode_viewer/episodes.jsonl",
        UpdateBatch.load_npz("./episode_viewer/episodes.npz"),
        tokenizer,
    )


if __name__ == "__main__":
    app()

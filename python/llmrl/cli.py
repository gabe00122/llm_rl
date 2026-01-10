from llmrl.train_rl import train_cli
import typer

app = typer.Typer()

# @app.command()
# def chat(config_url: str):
    
@app.command()
def train(config_url: str):
    train_cli(config_url)


@app.command()
def chat(config_url: str):


if __name__ == "__main__":
    app()

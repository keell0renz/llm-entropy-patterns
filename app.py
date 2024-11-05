from utils.hugginface import upload_missing_files, download_missing_files
from utils.health import health_check
from evals.creative_factual import generate_creative_factual_dataset
from dotenv import load_dotenv
import typer

load_dotenv()

app = typer.Typer()


@app.command()
def health():
    """
    Check the health of the environment.
    """

    if not health_check():
        typer.Exit(code=1)


@app.command()
def upload():
    """
    Upload missing files to Hugging Face.
    """

    upload_missing_files(local_dir="files", hf_subdir="files")


@app.command()
def download():
    """
    Download missing files from Hugging Face.
    """

    download_missing_files(local_dir="files", hf_subdir="files")


@app.command()
def generate_dataset(type: str, N: int = 50):
    """
    Generate a creative or factual dataset.

    Args:
        type (str): The type of dataset to generate, either 'creative' or 'factual'.
        N (int): The number of prompts to generate. Default is 50.
    """
    if type not in ["creative", "factual"]:
        typer.echo(
            "[bold red]Invalid type. Choose either 'creative' or 'factual'[/bold red]."
        )
        raise typer.Exit(code=1)

    generate_creative_factual_dataset(type, N)
    typer.echo(
        f"[bold green]{type.capitalize()} dataset with {N} prompts generated successfully.[/bold green]"
    )


if __name__ == "__main__":
    app()

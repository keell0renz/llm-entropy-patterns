from utils.hugginface import upload_missing_files, download_missing_files
from utils.health import health_check
from creative_factual.generate import generate_creative_factual_dataset
from inference.generate import generate_data
from dotenv import load_dotenv
from rich import print
from typer import Argument, Option
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
def generate_dataset(
    type: str = Argument(..., help="The type of dataset to generate."),
    N: int = Option(50, help="The number of prompts to generate."),
):
    """
    Generate a creative or factual dataset.

    Args:
        type (str): The type of dataset to generate, either 'creative' or 'factual'.
        N (int): The number of prompts to generate. Default is 50.
    """
    if type not in ["creative", "factual"]:
        print(
            "[bold red]Invalid type. Choose either 'creative' or 'factual'[/bold red]."
        )
        raise typer.Exit(code=1)

    generate_creative_factual_dataset(type, N)
    print(
        f"[bold green]{type.capitalize()} dataset with {N} prompts generated successfully.[/bold green]"
    )


@app.command()
def inference(
    model_hf: str = Argument(..., help="The Hugging Face model identifier."),
    prompts_path: str = Argument(..., help="The path to the prompts file."),
    output_path: str = Argument(..., help="The path to save the output file."),
):
    """
    Run inference on a set of prompts using a specified model.

    Args:
        model_hf (str): The Hugging Face model identifier.
        prompts_path (str): The path to the prompts file.
        output_path (str): The path to save the output file.
    """
    generate_data(model_hf, prompts_path, output_path)
    print(
        f"[bold green]Inference completed and results saved to {output_path}.[/bold green]"
    )


if __name__ == "__main__":
    app()

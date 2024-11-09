from creative_factual.prompts import (
    creative_prompt,
    factual_prompt,
)
from typing import List, Literal
from pydantic import BaseModel
from openai import OpenAI
import instructor
import csv
import os


class Output(BaseModel):
    prompts: List[str]


def generate_creative_factual_dataset(
    type: Literal["creative", "factual"], N: int = 50
):
    client = instructor.from_openai(OpenAI())
    if type == "creative":
        prompt = creative_prompt(N)
    else:
        prompt = factual_prompt(N)

    output = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_model=Output,
    )

    output_dir = f"./files/creative_factual"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/{type}_{N}_prompts.csv"

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt"])
        for prompt in output.prompts:
            writer.writerow([str(prompt).replace('"', "")])

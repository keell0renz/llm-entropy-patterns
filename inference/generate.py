from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from typing import List, Tuple
import warnings
import pickle
import torch
import tqdm
import csv
import os

from inference.statistics import calculate_entropies

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


def generate_data(model_hf: str, prompts_path: str, output_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_hf)
    model = AutoModelForCausalLM.from_pretrained(model_hf)

    # Add device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    prompts = []
    with open(prompts_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            prompts.append(row[0])

    entropies = []

    for i, prompt in enumerate(tqdm.tqdm(prompts)):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You must always respond in 200 words, with allowed +- 10 words deviation.",
            },
            {"role": "user", "content": prompt},
        ]

        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

        # Move inputs to GPU
        inputs = inputs.to(device)

        generation_output = model.generate(
            inputs,
            max_new_tokens=1024,
            return_dict_in_generate=True,
            output_attentions=True,
            use_cache=True,
        )

        response = (
            tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
            .split("assistant")[-1]
            .strip()
        )

        entropies.append((calculate_entropies(generation_output.attentions), response))

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(entropies, f)

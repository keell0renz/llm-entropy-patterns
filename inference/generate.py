from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
import pickle
import torch
import tqdm
import csv
import os


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

    outputs: List[Tuple[str, Tuple[Tuple[torch.FloatTensor]]]] = []

    for prompt in tqdm.tqdm(prompts):
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

        outputs = model.generate(
            inputs,
            max_new_tokens=1024,
            return_dict_in_generate=True,
            output_attentions=True,
            use_cache=False,
        )

        response = (
            tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            .split("assistant")[-1]
            .strip()
        )

        outputs.append((response, outputs.attentions))

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(outputs, f)
import torch


def calculate_entropies(attention_matrices: tuple) -> tuple:
    """
    Calculate the entropy and varentropy for each head in each attention matrix using GPU operations.

    Args:
        attention_matrices (tuple): A tuple of tuples of torch.Tensors, where each tensor has shape [1, H, N, N] or [1, H, 1, N_2].

    Returns:
        tuple: A tuple of tuples of tuples of size [step][layer][H, 2], where [0] is entropy and [1] is varentropy.
    """
    results = []

    for step in attention_matrices:
        step_results = []
        for layer in step:
            # Process all heads in parallel
            # Reshape to [H, N, N] or [H, 1, N_2]
            heads = layer.squeeze(0)

            # Calculate row-wise entropies for each head [H, N]
            row_entropies = -(heads * (heads + 1e-12).log()).sum(dim=-1)

            # Calculate mean entropy per head [H]
            mean_entropies = row_entropies.mean(dim=-1)

            # Calculate varentropy per head [H]
            varentropies = ((row_entropies - mean_entropies.unsqueeze(-1)) ** 2).mean(
                dim=-1
            )

            # Convert to tuples of (entropy, varentropy) for each head
            layer_results = [
                (float(e.item()), float(v.item()))
                for e, v in zip(mean_entropies, varentropies)
            ]
            step_results.append(tuple(layer_results))
        results.append(tuple(step_results))

    return tuple(results)

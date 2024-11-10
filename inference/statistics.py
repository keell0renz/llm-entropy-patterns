from scipy.stats import entropy as scipy_entropy
import torch


def entropy_varentropy_per_head(tensor: torch.Tensor) -> tuple[float, float]:
    """
    Calculate the average entropy and varentropy of an attention matrix.
    Handles both full attention (NxN) and KV-cached attention (1xN) matrices.

    Args:
        tensor (torch.Tensor): Input attention tensor after softmax (probabilities sum to 1 per row).

    Returns:
        tuple[float, float]: (average entropy, varentropy) across all attention distributions.
    """
    # Ensure tensor is 2D
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)

    # Calculate entropy for each row (each attention distribution)
    row_entropies = []
    for row in tensor:
        # Each row should already sum to 1 (post-softmax)
        row_probs = row.cpu().numpy()
        row_entropies.append(scipy_entropy(row_probs))

    # Calculate mean entropy and varentropy
    mean_entropy = float(sum(row_entropies) / len(row_entropies))
    varentropy = float(
        sum((e - mean_entropy) ** 2 for e in row_entropies) / len(row_entropies)
    )

    return mean_entropy, varentropy


def calculate_entropies(attention_matrices: tuple) -> tuple:
    """
    Calculate the entropy and varentropy for each head in each attention matrix.

    Args:
        attention_matrices (tuple): A tuple of tuples of torch.Tensors, where each tensor has shape [1, H, N, N] or [1, H, 1, N_2].

    Returns:
        tuple: A tuple of tuples of tuples of size [step][layer][H, 2], where [0] is entropy and [1] is varentropy.
    """
    results = []

    for step in attention_matrices:
        step_results = []
        for layer in step:
            layer_results = []
            for head in range(layer.size(1)):
                head_tensor = layer[0, head]
                entropy, varentropy = entropy_varentropy_per_head(head_tensor)
                layer_results.append((entropy, varentropy))
            step_results.append(tuple(layer_results))
        results.append(tuple(step_results))

    return tuple(results)

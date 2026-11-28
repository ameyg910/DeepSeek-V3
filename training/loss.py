import torch
import torch.nn.functional as F


def next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    logits:  (B, L, V)
    targets: (B, L)

    targets contain token IDs, with -100 used for ignored positions.
    Returns scalar cross-entropy loss.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, L, V)")

    if targets.ndim != 2:
        raise ValueError("targets must have shape (B, L)")

    if logits.shape[:2] != targets.shape:
        raise ValueError("logits and targets must match in batch and sequence dimensions")

    vocab_size = logits.size(-1)

    # Flatten batch and sequence dimensions.
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)

    loss = F.cross_entropy(
        logits,
        targets,
        ignore_index=-100,
    )

    return loss
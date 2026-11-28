import torch
from torch import nn


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return optimizer
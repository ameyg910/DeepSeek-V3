import torch
from torch import nn

from training.loss import next_token_loss


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    gradient_clip: float = 1.0,
) -> float:

    model.train()

    input_ids = batch["input_ids"].to(device)
    targets = batch["labels"].to(device)

    attention_mask = batch.get("attention_mask")

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Clear gradients from the previous update.
    optimizer.zero_grad(set_to_none=True)

    # Forward pass.
    if attention_mask is not None:
        logits = model(
            input_ids,
            attention_mask=attention_mask,
        )
    else:
        logits = model(input_ids)

    # Compute next-token prediction loss.
    loss = next_token_loss(logits, targets)

    # Backpropagation.
    loss.backward()

    # Clip gradients to stabilize training.
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=gradient_clip,
    )

    # Update model parameters.
    optimizer.step()

    return loss.item()

def train_accumulated_step(
    model,
    optimizer,
    batches,
    device,
    gradient_clip=1.0,
):
    if not batches:
        raise ValueError("batches cannot be empty")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    accumulation_steps = len(batches)

    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        targets = batch["labels"].to(device)

        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if attention_mask is not None:
            logits = model(
                input_ids,
                attention_mask=attention_mask,
            )
        else:
            logits = model(input_ids)

        loss = next_token_loss(logits, targets)

        # Average the gradients across micro-batches.
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=gradient_clip,
    )

    optimizer.step()

    return total_loss / accumulation_steps
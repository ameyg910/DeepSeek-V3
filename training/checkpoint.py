import torch
from pathlib import Path


def save_checkpoint(
    model,
    optimizer,
    step: int,
    loss: float,
    path: str,
):
    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)


def load_checkpoint(
    model,
    optimizer,
    path: str,
    device: torch.device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return checkpoint["step"], checkpoint["loss"]
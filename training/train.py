
import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from data.tokenizer import MathTokenizer
from data.dataset import MathTextDataset
from models.deepseek_v3 import DeepSeekV3

from training.optimizer import create_optimizer
from training.trainer import train_accumulated_step
from training.checkpoint import save_checkpoint, load_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_training_data(
    tokenizer_path: str,
    textbook_path: str,
    seq_len: int,
):
    tokenizer = MathTokenizer.load(tokenizer_path)

    text = Path(textbook_path).read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("Textbook is empty")

    # Encode without automatically adding BOS/EOS.
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    # Add document boundary tokens.
    token_ids = [
        tokenizer.token_id(tokenizer.BOS),
        *token_ids,
        tokenizer.token_id(tokenizer.EOS),
    ]

    dataset = MathTextDataset(
        token_ids=token_ids,
        seq_len=seq_len,
        pad_token_id=tokenizer.token_id(tokenizer.PAD),
    )

    return tokenizer, dataset


def set_learning_rate(
    optimizer,
    base_learning_rate: float,
    step: int,
    warmup_steps: int,
):
    """
    Linear learning-rate warmup.

    step is zero-indexed.
    After warmup, the learning rate stays at its base value.
    """
    if warmup_steps > 0:
        scale = min((step + 1) / warmup_steps, 1.0)
    else:
        scale = 1.0

    learning_rate = base_learning_rate * scale

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/tiny_v3.yaml",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from",
    )

    args = parser.parse_args()

    # 1. Load configuration.
    config = load_config(args.config)
    training_config = config.training
    model_config = config.model

    set_seed(training_config.seed)

    # 2. Select device.
    device = get_device()
    print(f"Using device: {device}")

    # 3. Load tokenizer and dataset.
    tokenizer, dataset = load_training_data(
        tokenizer_path="models/tokenizer.json",
        textbook_path="data/textbook.txt",
        seq_len=model_config.max_seq_len,
    )

    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({tokenizer.vocab_size}) "
            f"does not match model vocabulary "
            f"({model_config.vocab_size})"
        )

    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Dataset samples: {len(dataset):,}")

    # 4. Create DataLoader.
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    if len(dataloader) == 0:
        raise ValueError(
            "Dataset is too small for the configured batch size"
        )

    # 5. Initialize model.
    pad_token_id = tokenizer.token_id(tokenizer.PAD)

    model = DeepSeekV3(
        config=model_config,
        pad_token_id=pad_token_id,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")

    # 6. Initialize optimizer.
    optimizer = create_optimizer(
        model=model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    # 7. Optionally resume training.
    start_step = 0
    last_loss = 0.0

    if args.resume is not None:
        start_step, last_loss = load_checkpoint(
            model=model,
            optimizer=optimizer,
            path=args.resume,
            device=device,
        )

        print(
            f"Resuming from step {start_step}, "
            f"previous loss: {last_loss:.4f}"
        )

    # 8. Training loop.
    model.train()

    data_iterator = iter(dataloader)

    accumulation_steps = (
        training_config.gradient_accumulation_steps
    )

    checkpoint_path = "checkpoints/latest.pt"

    for step in range(start_step, training_config.max_steps):

        # Collect micro-batches.
        batches = []

        for _ in range(accumulation_steps):
            try:
                batch = next(data_iterator)

            except StopIteration:
                # Start another epoch when the current one ends.
                data_iterator = iter(dataloader)
                batch = next(data_iterator)

            batches.append(batch)

        # Set learning rate for this optimizer update.
        set_learning_rate(
            optimizer=optimizer,
            base_learning_rate=training_config.learning_rate,
            step=step,
            warmup_steps=training_config.warmup_steps,
        )

        # Perform one optimizer update.
        loss = train_accumulated_step(
            model=model,
            optimizer=optimizer,
            batches=batches,
            device=device,
            gradient_clip=training_config.gradient_clip,
        )

        completed_step = step + 1

        # Logging.
        if completed_step % 10 == 0 or completed_step == 1:
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Step [{completed_step}/{training_config.max_steps}] "
                f"Loss: {loss:.4f} "
                f"LR: {current_lr:.6e}"
            )

        # Periodic checkpoint.
        if completed_step % 100 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=completed_step,
                loss=loss,
                path=checkpoint_path,
            )

            print(f"Checkpoint saved at step {completed_step}")

    # 9. Save final checkpoint.
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=training_config.max_steps,
        loss=loss,
        path=checkpoint_path,
    )

    print("Training completed.")
    print(f"Final loss: {loss:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn

from training.trainer import train_step, train_accumulated_step


class ToyLanguageModel(nn.Module):
    def __init__(self, vocab_size=20, hidden_dim=8):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # (B, L) -> (B, L, H)
        logits = self.lm_head(x)       # (B, L, H) -> (B, L, V)

        return logits


def test_train_step_updates_parameters():
    model = ToyLanguageModel(vocab_size=20)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
    )

    batch = {
        "input_ids": torch.randint(0, 20, (4, 5)),
        "labels": torch.randint(0, 20, (4, 5)),
    }

    before = model.lm_head.weight.detach().clone()

    loss = train_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        device=torch.device("cpu"),
    )

    after = model.lm_head.weight.detach()

    assert isinstance(loss, float)
    assert loss > 0
    assert not torch.equal(before, after)

def test_train_accumulated_step():

    model = ToyLanguageModel(vocab_size=20)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
    )

    batches = [
        {
            "input_ids": torch.randint(0, 20, (4, 5)),
            "labels": torch.randint(0, 20, (4, 5)),
        }
        for _ in range(3)
    ]

    before = model.lm_head.weight.detach().clone()

    loss = train_accumulated_step(
        model=model,
        optimizer=optimizer,
        batches=batches,
        device=torch.device("cpu"),
        gradient_clip=1.0,
    )

    after = model.lm_head.weight.detach()

    assert isinstance(loss, float)
    assert loss > 0
    assert not torch.equal(before, after)
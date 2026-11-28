import torch

from training.optimizer import create_optimizer


def test_optimizer_creation():
    model = torch.nn.Linear(10, 5)

    optimizer = create_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    assert isinstance(optimizer, torch.optim.AdamW)


def test_optimizer_updates_parameters():
    model = torch.nn.Linear(10, 5)

    optimizer = create_optimizer(
        model,
        learning_rate=0.01,
        weight_decay=0.0,
    )

    x = torch.randn(4, 10)
    target = torch.randn(4, 5)

    before = model.weight.detach().clone()

    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()

    optimizer.step()

    after = model.weight.detach()

    assert not torch.equal(before, after)
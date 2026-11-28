import torch

from training.loss import next_token_loss


def test_loss_is_scalar():
    logits = torch.randn(2, 4, 100)
    targets = torch.randint(0, 100, (2, 4))

    loss = next_token_loss(logits, targets)

    assert loss.ndim == 0


def test_loss_is_finite():
    logits = torch.randn(2, 4, 100)
    targets = torch.randint(0, 100, (2, 4))

    loss = next_token_loss(logits, targets)

    assert torch.isfinite(loss)


def test_ignored_targets():
    logits = torch.randn(2, 4, 100)
    targets = torch.randint(0, 100, (2, 4))

    targets[:, -1] = -100

    loss = next_token_loss(logits, targets)

    assert torch.isfinite(loss)


def test_loss_has_gradients():
    logits = torch.randn(2, 4, 100, requires_grad=True)
    targets = torch.randint(0, 100, (2, 4))

    loss = next_token_loss(logits, targets)
    loss.backward()

    assert logits.grad is not None
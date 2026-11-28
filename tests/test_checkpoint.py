
import torch
import torch.nn as nn

from training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


class ToyLanguageModel(nn.Module):
    def __init__(self, vocab_size=20, hidden_dim=8):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        return self.lm_head(x)


def test_save_and_load_checkpoint(tmp_path):
    model = ToyLanguageModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
    )

    # Perform one optimizer step so the optimizer has state.
    input_ids = torch.randint(0, 20, (2, 5))
    targets = torch.randint(0, 20, (2, 5))

    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, 20),
        targets.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    expected_model_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    expected_step = 100
    expected_loss = 2.5

    checkpoint_path = tmp_path / "checkpoints" / "test.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=expected_step,
        loss=expected_loss,
        path=str(checkpoint_path),
    )

    assert checkpoint_path.exists()

    # Create fresh objects to simulate restarting training.
    new_model = ToyLanguageModel()
    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=0.01,
    )

    step, restored_loss = load_checkpoint(
        model=new_model,
        optimizer=new_optimizer,
        path=str(checkpoint_path),
        device=torch.device("cpu"),
    )

    assert step == expected_step
    assert restored_loss == expected_loss

    # Verify model parameters were restored.
    for key, value in new_model.state_dict().items():
        assert torch.equal(
            value,
            expected_model_state[key],
        )

    # Verify optimizer state was restored.
    original_optimizer_state = optimizer.state_dict()["state"]
    restored_optimizer_state = new_optimizer.state_dict()["state"]

    assert original_optimizer_state.keys() == restored_optimizer_state.keys()

    for param_id in original_optimizer_state:
        original_state = original_optimizer_state[param_id]
        restored_state = restored_optimizer_state[param_id]

        assert original_state.keys() == restored_state.keys()

        for key in original_state:
            original_value = original_state[key]
            restored_value = restored_state[key]

            if isinstance(original_value, torch.Tensor):
                assert torch.equal(original_value, restored_value)
            else:
                assert original_value == restored_value

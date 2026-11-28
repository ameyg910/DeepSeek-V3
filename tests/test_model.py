import torch

from src.config import load_config
from models.deepseek_v3 import DeepSeekV3


def test_model_output_shape():
    config = load_config("configs/tiny_v3.yaml")

    model = DeepSeekV3(config.model)

    input_ids = torch.randint(
        0,
        config.model.vocab_size,
        (2, 16),
    )

    logits = model(input_ids)

    assert logits.shape == (2, 16, config.model.vocab_size)


def test_model_forward_with_attention_mask():
    config = load_config("configs/tiny_v3.yaml")

    model = DeepSeekV3(config.model)

    input_ids = torch.randint(
        0,
        config.model.vocab_size,
        (2, 16),
    )

    attention_mask = torch.ones(2, 16, dtype=torch.bool)

    logits = model(
        input_ids,
        attention_mask=attention_mask,
    )

    assert logits.shape == (2, 16, config.model.vocab_size)


def test_model_has_gradients():
    config = load_config("configs/tiny_v3.yaml")

    model = DeepSeekV3(config.model)

    input_ids = torch.randint(
        0,
        config.model.vocab_size,
        (2, 16),
    )

    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()

    assert model.embedding.embedding.weight.grad is not None
import pytest
import torch

from models.moe import DeepSeekMoE, SwiGLUExpert

def make_moe():
    return DeepSeekMoE(
        hidden_dim=64,
        expert_dim=128,
        num_experts=8,
        top_k=2,
        num_shared_experts=1,
    )

def test_expert_output_shape():
    expert = SwiGLUExpert(
        hidden_dim=64,
        expert_dim=128,
    )

    x = torch.randn(2, 10, 64)
    y = expert(x)

    assert y.shape == x.shape

def test_moe_output_shape():
    moe = make_moe()

    x = torch.randn(2, 16, 64)
    y = moe(x)

    assert y.shape == x.shape

def test_moe_has_gradients():
    moe = make_moe()

    x = torch.randn(2, 8, 64, requires_grad=True)
    y = moe(x)

    y.sum().backward()

    assert x.grad is not None
    assert moe.router.weight.grad is not None

def test_moe_output_is_finite():
    moe = make_moe()

    x = torch.randn(2, 8, 64)
    y = moe(x)

    assert torch.isfinite(y).all()

def test_invalid_top_k():
    with pytest.raises(ValueError):
        DeepSeekMoE(
            hidden_dim=64,
            expert_dim=128,
            num_experts=8,
            top_k=9,
        )

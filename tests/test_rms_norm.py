import torch 

from models.rms_norm import RMSNorm

def test_rms_norm_output_shape():
    norm = RMSNorm(hidden_dim=512)

    x = torch.randn(2, 128, 512)
    y = norm(x)

    assert y.shape == x.shape

def test_rms_norm_unit_rms_when_weight_is_one():
    norm = RMSNorm(hidden_dim=4)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = norm(x)

    rms = torch.sqrt(y.pow(2).mean(dim=-1))

    assert torch.allclose(
        rms,
        torch.ones_like(rms),
        atol=1e-5,
    )

def test_rms_norm_matches_manual_calculation():
    norm = RMSNorm(hidden_dim=2, eps=1e-6)

    x = torch.tensor([[3.0, 4.0]])

    # RMS = sqrt((9 + 16) / 2) = sqrt(12.5)
    expected = x / torch.sqrt(torch.tensor(12.5 + 1e-6))

    y = norm(x)

    assert torch.allclose(y, expected, atol=1e-6)

def test_rms_norm_has_gradients():
    norm = RMSNorm(hidden_dim=8)

    x = torch.randn(2, 3, 8, requires_grad=True)
    y = norm(x)

    y.sum().backward()

    assert x.grad is not None
    assert norm.weight.grad is not None

def test_rms_norm_preserves_input_dtype():
    norm = RMSNorm(hidden_dim=8)

    x = torch.randn(2, 3, 8, dtype=torch.float16)
    y = norm(x)

    assert y.dtype == x.dtype

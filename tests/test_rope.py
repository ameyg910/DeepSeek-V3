import pytest
import torch
from models.rope import RotaryEmbedding

def test_rope_preserves_shape():
    rope = RotaryEmbedding(head_dim=32)

    x = torch.randn(2, 8, 128, 32)
    y = rope(x)

    assert y.shape == x.shape

def test_rope_position_zero_is_identity():
    rope = RotaryEmbedding(head_dim=4)

    x = torch.randn(2, 5, 4)
    y = rope(x)

    # Position 0 has angle 0, so rotation is identity.
    assert torch.allclose(y[:, 0, :], x[:, 0, :], atol=1e-6)

def test_rope_changes_nonzero_positions():
    rope = RotaryEmbedding(head_dim=4)

    x = torch.ones(1, 3, 4)
    y = rope(x)

    assert not torch.allclose(y[:, 1, :], x[:, 1, :])

def test_rope_preserves_vector_norm():
    rope = RotaryEmbedding(head_dim=32)

    x = torch.randn(2, 5, 32)
    y = rope(x)

    x_norm = torch.linalg.vector_norm(x, dim=-1)
    y_norm = torch.linalg.vector_norm(y, dim=-1)

    assert torch.allclose(x_norm, y_norm, atol=1e-5)

def test_rope_supports_custom_positions():
    rope = RotaryEmbedding(head_dim=8)

    x = torch.randn(1, 3, 8)
    positions = torch.tensor([10, 11, 12])

    y = rope(x, position_ids=positions)

    assert y.shape == x.shape

def test_rope_requires_even_dimension():
    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=7)



import torch

from models.mla import MultiHeadLatentAttention


def make_mla():
    return MultiHeadLatentAttention(
        hidden_dim=512,
        num_heads=8,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
    )


def test_mla_output_shape():
    mla = make_mla()

    x = torch.randn(2, 16, 512)
    y = mla(x)

    assert y.shape == (2, 16, 512)


def test_mla_supports_attention_mask():
    mla = make_mla()

    x = torch.randn(2, 16, 512)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[1, -4:] = False

    y = mla(x, attention_mask=mask)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mla_has_gradients():
    mla = make_mla()

    x = torch.randn(2, 8, 512, requires_grad=True)
    y = mla(x)

    y.sum().backward()

    assert x.grad is not None
    assert mla.q_down.weight.grad is not None
    assert mla.kv_down.weight.grad is not None
    assert mla.o_proj.weight.grad is not None


def test_mla_different_sequence_lengths():
    mla = make_mla()

    for seq_len in [1, 4, 16]:
        x = torch.randn(1, seq_len, 512)
        y = mla(x)

        assert y.shape == (1, seq_len, 512)

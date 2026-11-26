import torch 
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()

        if head_dim <= 0 or head_dim % 2 != 0: 
            raise ValueError("head_dim must be a positive even number")

        self.head_dim = head_dim

        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0/(base**(exponent/head_dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor: 
        seq_len = x.shape[-2]

        if position_ids is None: 
            position_ids = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        else:
            position_ids = position_ids.to(
                device=x.device,
                dtype=torch.float32,
            )
        # Angles shape: (seq_len, head_dim // 2)
        angles = torch.outer(position_ids, self.inv_freq)
        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)
        
        # Split the final dimension into two halves.
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]

        while cos.ndim < x1.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        return torch.cat([rotated_x1, rotated_x2], dim=-1)

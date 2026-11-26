import torch 
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6): 
        super().__init__()

        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_float = x.float()

        mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)

        scale = torch.rsqrt(mean_square + self.eps)

        output = x_float * scale * self.weight.float()

        return output.to(dtype=x.dtype)

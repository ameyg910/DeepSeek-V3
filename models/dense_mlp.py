import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
    ):
        super().__init__()

        self.w1 = nn.Linear(
            hidden_dim,
            intermediate_dim,
            bias=False,
        )

        self.w2 = nn.Linear(
            intermediate_dim,
            hidden_dim,
            bias=False,
        )

        self.w3 = nn.Linear(
            hidden_dim,
            intermediate_dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., hidden_dim)

        gate = F.silu(self.w1(x))
        value = self.w3(x)

        return self.w2(gate * value)
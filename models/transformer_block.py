import torch 
import torch.nn as nn

from models.rms_norm import RMSNorm

class DeepSeekTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, attention: nn.Module, feed_forward: nn.Module,):
        super().__init__()

        self.attention = attention 
        self.feed_forward = feed_forward

        self.attention_norm = RMSNorm(hidden_dim)
        self.moe_norm = RMSNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_input = self.attention_norm(x) 
        attention_output = self.attention(
            attention_input,
            attention_mask=attention_mask,
        )
        x = x + attention_output

        moe_input = self.moe_norm(x)
        feed_forward_output = self.feed_forward(moe_input)

        x = x + feed_forward_output

        return x 

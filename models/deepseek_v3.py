import torch
import torch.nn as nn

from src.config import ModelConfig

from models.embeddings import TokenEmbedding
from models.rms_norm import RMSNorm
from models.mla import MultiHeadLatentAttention
from models.dense_mlp import DenseMLP
from models.moe import DeepSeekMoE
from models.transformer_block import DeepSeekTransformerBlock


class DeepSeekV3(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.config = config

        self.embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            pad_token_id=pad_token_id,
        )

        self.layers = nn.ModuleList()

        for layer_idx in range(config.num_layers):

            attention = MultiHeadLatentAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
            )

            if layer_idx < config.num_dense_layers:
                feed_forward = DenseMLP(
                    hidden_dim=config.hidden_dim,
                    intermediate_dim=config.dense_intermediate_dim,
                )
            else:
                feed_forward = DeepSeekMoE(
                    hidden_dim=config.hidden_dim,
                    expert_dim=config.expert_intermediate_dim,
                    num_experts=config.num_experts,
                    top_k=config.num_experts_per_token,
                    num_shared_experts=config.num_shared_experts,
                )

            block = DeepSeekTransformerBlock(
                hidden_dim=config.hidden_dim,
                attention=attention,
                feed_forward=feed_forward,
            )

            self.layers.append(block)

        self.final_norm = RMSNorm(config.hidden_dim)

        self.lm_head = nn.Linear(
            config.hidden_dim,
            config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # input_ids: (B, L)
        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum {self.config.max_seq_len}"
            )

        # (B, L) -> (B, L, D)
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(
                x,
                attention_mask=attention_mask,
            )

        x = self.final_norm(x)

        # (B, L, D) -> (B, L, vocab_size)
        logits = self.lm_head(x)

        return logits
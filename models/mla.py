
import torch
from torch import nn
import torch.nn.functional as F

from models.rms_norm import RMSNorm
from models.rope import RotaryEmbedding


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # Query compression and expansion.
        self.q_down = nn.Linear(
            hidden_dim, q_lora_rank, bias=False
        )
        self.q_norm = RMSNorm(q_lora_rank)
        self.q_up = nn.Linear(
            q_lora_rank,
            num_heads * self.qk_head_dim,
            bias=False,
        )

        # Compress shared key/value information.
        self.kv_down = nn.Linear(
            hidden_dim,
            kv_lora_rank + qk_rope_head_dim,
            bias=False,
        )
        self.kv_norm = RMSNorm(kv_lora_rank)

        # Expand latent KV into per-head non-rotary keys and values.
        self.kv_up = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        self.rope = RotaryEmbedding(qk_rope_head_dim)

        # Merge all head outputs back into hidden_dim.
        self.o_proj = nn.Linear(
            num_heads * v_head_dim,
            hidden_dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, L, hidden_dim)
        batch_size, seq_len, _ = x.shape

        # ----- Query path -----
        q_latent = self.q_down(x)
        q_latent = self.q_norm(q_latent)

        q = self.q_up(q_latent)
        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.qk_head_dim,
        )
        q = q.transpose(1, 2)
        # q: (B, H, L, qk_head_dim)

        q_nope, q_rope = torch.split(
            q,
            [
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
            ],
            dim=-1,
        )

        q_rope = self.rope(q_rope)

        # ----- Key/value compression -----
        kv_compressed = self.kv_down(x)

        kv_latent, k_rope = torch.split(
            kv_compressed,
            [
                self.kv_down.out_features
                - self.qk_rope_head_dim,
                self.qk_rope_head_dim,
            ],
            dim=-1,
        )

        kv_latent = self.kv_norm(kv_latent)

        # Shared rotary key: one per token, then shared across heads.
        k_rope = k_rope.unsqueeze(1)
        k_rope = self.rope(k_rope)

        # Expand latent into per-head non-rotary keys and values.
        kv = self.kv_up(kv_latent)
        kv = kv.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        kv = kv.transpose(1, 2)
        # kv: (B, H, L, nope_dim + value_dim)

        k_nope, v = torch.split(
            kv,
            [
                self.qk_nope_head_dim,
                self.v_head_dim,
            ],
            dim=-1,
        )

        # Concatenate non-rotary and rotary parts.
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat(
            [k_nope, k_rope.expand(
                -1, self.num_heads, -1, -1
            )],
            dim=-1,
        )

        # ----- Causal attention -----
        if attention_mask is None:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
            )
        else:
            # attention_mask: (B, L), True means valid token.
            causal = torch.tril(
                torch.ones(
                    seq_len,
                    seq_len,
                    dtype=torch.bool,
                    device=x.device,
                )
            )

            valid_keys = attention_mask[:, None, None, :].bool()
            attn_mask = causal[None, None, :, :] & valid_keys

            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=False,
            )

        # (B, H, L, V) -> (B, L, H * V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(
            batch_size,
            seq_len,
            self.num_heads * self.v_head_dim,
        )

        return self.o_proj(attn_output)

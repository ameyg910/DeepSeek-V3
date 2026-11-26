import torch 
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, pad_token_id: int,):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, #no. of possible token IDS (12k)
            embedding_dim=hidden_dim, #sz of each token's vector (512)
            padding_idx=pad_token_id, #tells torch not to update padding token's emb through gradients
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)
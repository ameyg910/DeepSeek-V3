import torch 
import torch.nn as nn
import torch.nn.functional as F 

class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_dim: int, expert_dim: int):
        super().__init__()

        self.w1 = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.w2 = nn.Linear(expert_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, expert_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        value = self.w3(x) 

        return self.w2(gate*value)

class DeepSeekMoE(nn.Module):
    def __init__(self, hidden_dim: int, expert_dim: int, num_experts: int, top_k: int, num_shared_experts: int =1):
        super().__init__()

        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(hidden_dim, expert_dim)
                for _ in range(num_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                SwiGLUExpert(hidden_dim, expert_dim)
                for _ in range(num_shared_experts)
            ]
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        #top_k experts for each token 

        topk_probs, topk_indices = torch.topk(
            router_probs, 
            k=self.top_k,
            dim=-1,
        )

        # Renormalize selected weights to sum to 1.
        topk_weights = topk_probs / topk_probs.sum(
            dim=-1,
            keepdim=True,
        )

        # Accumulate sparse routed expert outputs.
        routed_output = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            token_indices, topk_slots = torch.where(
                topk_indices == expert_id
            )
            if token_indices.numel() == 0:
                continue

            expert_input = x_flat[token_indices]
            expert_output = expert(expert_input)
            weights = topk_weights[token_indices, topk_slots].unsqueeze(-1)
            weighted_output = expert_output * weights

            routed_output = routed_output.index_add(
                0,
                token_indices,
                weighted_output,
            )
        # Shared experts always process every token.
        shared_output = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x_flat)

        output = routed_output + shared_output
        return output.reshape(batch_size, seq_len, hidden_dim)



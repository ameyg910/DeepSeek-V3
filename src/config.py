from dataclasses import dataclass 
from pathlib import Path 

import yaml 

@dataclass 
class ModelConfig: 
    vocab_size: int 
    hidden_dim: int
    num_layers: int
    num_heads: int
    max_seq_len: int

    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int 
    v_head_dim: int

    num_dense_layers: int
    num_experts: int
    num_experts_per_token: int 
    num_shared_experts: int 
    dense_intermediate_dim: int
    expert_intermediate_dim: int


    def validate(self): 
        assert self.hidden_dim > 0
        assert self.num_layers > 0
        assert self.num_heads > 0

        assert self.hidden_dim % self.num_heads == 0

        assert self.num_dense_layers <= self.num_layers

        assert self.num_experts_per_token <= self.num_experts

        assert self.q_lora_rank > 0
        assert self.kv_lora_rank > 0

        assert self.dense_intermediate_dim > 0
        assert self.expert_intermediate_dim > 0

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int
    max_steps: int
    warmup_steps: int
    gradient_clip: float
    seed: int

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str) -> Config:
    path = Path(path)

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    model_config = ModelConfig(**raw["model"])
    training_config = TrainingConfig(**raw["training"])

    model_config.validate()

    return Config(
        model=model_config,
        training=training_config,
    )
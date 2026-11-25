from src.config import load_config 

def test_load_config():
    config = load_config("configs/tiny_v3.yaml")

    assert config.model.hidden_dim == 512
    assert config.model.num_layers == 8
    assert config.model.num_experts == 8


def test_config_dimensions():
    config = load_config("configs/tiny_v3.yaml")

    assert(
        config.model.hidden_dim%config.model.num_heads == 0 
    )

def test_expert_routing():
    config = load_config("configs/tiny_v3.yaml")

    assert(
        config.model.num_experts_per_token <= config.model.num_experts
    )
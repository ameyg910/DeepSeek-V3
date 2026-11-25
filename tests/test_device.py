import torch 

from src.utils import get_device, set_seed

def test_device():
    device = get_device()

    x = torch.randn(4, 4, device=device)
    y = x @ x 

    assert y.shape == (4, 4)

def test_seed():
    set_seed(42)
    a = torch.randn(5)

    set_seed(42)
    b = torch.randn(5)

    assert torch.equal(a, b)
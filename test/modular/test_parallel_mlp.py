"""CPU checks for the dense (ungated) tensor-parallel MLP and the relu2 activation."""

import torch
import torch.nn.functional as F

from mstar.model.components.distributed import ParallelMLP
from mstar.model.components.mlp import _resolve_activation


def test_relu2_activation_is_squared_relu() -> None:
    x = torch.randn(64)
    for name in ("relu2", "relu_squared"):
        out = _resolve_activation(name)(x)
        assert torch.equal(out, torch.square(F.relu(x)))
    assert torch.all(out >= 0)


def test_parallel_mlp_matches_dense_reference() -> None:
    torch.manual_seed(0)
    mlp = ParallelMLP(hidden_size=16, intermediate_size=40, activation="relu2")
    # The parallel linears allocate uninitialized storage; give them values.
    for p in mlp.parameters():
        torch.nn.init.normal_(p, std=0.1)
    assert set(mlp.state_dict()) == {"up_proj.weight", "down_proj.weight"}
    assert mlp.up_proj.weight.shape == (40, 16)
    assert mlp.down_proj.weight.shape == (16, 40)

    x = torch.randn(5, 16)
    ref = F.linear(torch.square(F.relu(F.linear(x, mlp.up_proj.weight))), mlp.down_proj.weight)
    assert torch.allclose(mlp(x), ref, atol=1e-6)


def test_parallel_mlp_gelu_and_bias() -> None:
    torch.manual_seed(1)
    mlp = ParallelMLP(hidden_size=8, intermediate_size=12, activation="gelu_tanh", bias=True)
    for p in mlp.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x = torch.randn(3, 8)
    h = F.gelu(F.linear(x, mlp.up_proj.weight, mlp.up_proj.bias), approximate="tanh")
    ref = F.linear(h, mlp.down_proj.weight, mlp.down_proj.bias)
    assert torch.allclose(mlp(x), ref, atol=1e-6)

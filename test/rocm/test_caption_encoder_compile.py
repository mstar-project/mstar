"""The compiled Qwen3 caption encoder against the eager forward on the GPU, at several widths and batch sizes."""

from __future__ import annotations

import pytest
import torch

from mstar.model.components.diffusion.qwen3.encoder import Qwen3HiddenStateEncoder, compiled_encoder_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def encoder():
    torch.manual_seed(0)
    model = Qwen3HiddenStateEncoder(
        vocab_size=512, hidden_size=256, intermediate_size=512, num_heads=8, num_kv_heads=2, head_dim=32,
        rms_norm_eps=1e-6, rope_theta=10000.0, tap_layers=(3,),
    )
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.02)
    return model.to("cuda", torch.bfloat16).eval().requires_grad_(False)


@pytest.mark.parametrize(("rows", "width"), [(1, 32), (1, 64), (1, 128), (2, 64), (4, 32), (4, 128)])
def test_compiled_forward_matches_eager(encoder, rows, width):
    forward = compiled_encoder_forward(encoder)
    ids = torch.randint(1, 512, (rows, width), device="cuda")
    mask = torch.ones_like(ids)
    real = width - 5
    mask[:, real:] = 0
    with torch.no_grad():
        eager = encoder(ids, mask).float()[:, :real]
        compiled = forward(ids, mask).float()[:, :real]
    error = ((compiled - eager).norm() / eager.norm()).item()
    # rounding only: the served encoder lands 2e-3 from eager on real prompts
    assert error < 5e-2, error

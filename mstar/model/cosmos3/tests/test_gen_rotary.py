"""The generation tower's mRoPE table is built eagerly inside a compiled denoise step.

Inductor fuses the interleaved mRoPE into one pointwise kernel that Triton 3.7.1
miscompiles on sm_103 (B300) when a thread holds two elements, and pointwise
autotuning picks such a config for a 121-frame 832x480 clip, which scrambles the
generated video. Needs CUDA, since the CPU backend does not generate Triton.
"""

import pytest
import torch

from mstar.model.cosmos3.components.packing import build_static_inputs
from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer, Cosmos3RotaryEmbedding
from mstar.model.cosmos3.config import Cosmos3Config

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (inductor's Triton backend)")


def _edge_rotary_transformer(cfg: Cosmos3Config) -> Cosmos3OmniTransformer:
    # `_gen_rotary` reads only the rotary module
    tf = Cosmos3OmniTransformer.__new__(Cosmos3OmniTransformer)
    torch.nn.Module.__init__(tf)
    tf.rotary_emb = Cosmos3RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.rope_axes_dim)
    return tf


@needs_cuda
@torch.no_grad()
def test_gen_rotary_is_the_eager_table_under_compile() -> None:
    cfg = Cosmos3Config(rope_theta=1e8)  # Edge's mRoPE: theta 1e8, sections (24, 20, 20), fps-modulated time
    tf = _edge_rotary_transformer(cfg)
    # the served Edge video: 832x480, 121 frames -> 31 x 15 x 26 generation tokens after a short prompt
    pos = build_static_inputs(list(range(40)), (1, 48, 31, 30, 52), cfg, 4, 24.0, "cuda")["vision_mrope_ids"]

    def table(p):
        return tf._gen_rotary(p, p.device, torch.bfloat16)

    cos_e, sin_e = table(pos)
    torch._dynamo.reset()
    cos_c, sin_c = torch.compile(table, dynamic=False)(pos)
    assert torch.equal(cos_c, cos_e) and torch.equal(sin_c, sin_e), (
        f"compiled generation mRoPE differs from eager by {(cos_c.float() - cos_e.float()).abs().max().item():.4f}"
    )

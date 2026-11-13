"""Wan2.2's DiT carries each block's FFN residual into the next block's first norm (``Wan22DiTBlock.stream``).
The stack must give what the blocks' own forwards give, one after another, bit for bit, eager and compiled."""

import pytest
import torch

from mstar.model.wan22.components.dit import Wan22DiT
from mstar.model.wan22.config import Wan22Config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _blockwise(dit: Wan22DiT, hidden, timestep, text):
    """``Wan22DiT.forward`` before the carry: each block's own forward, then the output norm alone."""
    from mstar.model.components.diffusion.modulation import ada_layer_norm

    batch, _, frames, height, width = hidden.shape
    p_t, p_h, p_w = dit.config.patch_size
    grid = (frames // p_t, height // p_h, width // p_w)
    rotary = tuple(t.contiguous() for t in dit.rope(grid, hidden.device))
    x = dit.patch_embedding(hidden).flatten(2).transpose(1, 2)
    temb, tproj, text = dit.condition_embedder(timestep, text)
    for block in dit.blocks:
        x = block(x, text, tproj, rotary)
    shift, scale = (dit.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).unbind(2)
    _, x = ada_layer_norm(x, shift, scale, eps=dit.norm_out.eps)
    x = dit.proj_out(x).reshape(batch, *grid, p_t, p_h, p_w, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
    return x.flatten(6, 7).flatten(4, 5).flatten(2, 3)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("per_sample", [False, True], ids=["per-frame", "per-sample"])
@pytest.mark.parametrize("batch", [2, 1], ids=["cfg-batch", "one-branch"])
def test_the_carried_residual_is_the_blocks_own(compiled, per_sample, batch):
    torch.manual_seed(0)
    cfg = Wan22Config(num_attention_heads=4, attention_head_dim=64, num_layers=3, ffn_dim=512, in_channels=16,
                      out_channels=16, text_dim=64, freq_dim=32)
    dit = Wan22DiT(cfg)
    for p in dit.parameters():
        torch.nn.init.normal_(p, std=0.05)
    dit = dit.cuda().cast_serving_dtypes().eval()
    # batch 1 at one timestep group is a CFG-parallel rank's A14B step (size-1 dims' strides)
    hidden = torch.randn(batch, 16, 3, 8, 12, device="cuda", dtype=torch.bfloat16)
    timestep = torch.tensor([[900.0]] * batch if per_sample else [[900.0, 800.0, 700.0]] * batch, device="cuda")
    text = torch.randn(batch, 7, 64, device="cuda", dtype=torch.bfloat16)
    forward, blockwise = dit.forward, _blockwise
    if compiled:
        torch._dynamo.reset()
        forward = torch.compile(dit.forward, fullgraph=False)
        blockwise = torch.compile(_blockwise, fullgraph=False)
    with torch.no_grad():
        got = forward(hidden, timestep, text)
        want = blockwise(dit, hidden, timestep, text)
    assert torch.equal(got, want)

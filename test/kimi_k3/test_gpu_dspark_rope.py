"""GPU: the one-launch yarn rope (``yarn_rope_fused``) is bit-identical to the torch path of
``YarnRotary.apply`` on 2-D and 3-D inputs, bf16 and fp32, int32 and int64 positions."""
import pytest
import torch

from mstar.model.kimi_k3.dspark.config import YarnParams
from mstar.model.kimi_k3.dspark.rope import YarnRotary
from mstar.model.kimi_k3.dspark.rope_kernel import yarn_rope_fused

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")


def torch_apply(rope, x, positions):
    cos, sin = rope.cos[positions], rope.sin[positions]
    if x.dim() == 3:
        cos, sin = cos[:, None, :], sin[:, None, :]
    xf = x.float()
    x1, x2 = xf[..., ::2], xf[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return (xf * cos + rotated * sin).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(37, 64), (37, 8, 64), (5, 3, 32), (1, 64)])
@pytest.mark.parametrize("pos_dtype", [torch.int64, torch.int32])
def test_fused_rope_is_bit_identical(dtype, shape, pos_dtype):
    torch.manual_seed(0)
    params = YarnParams(original_max_position_embeddings=2048, factor=4.0)
    rope = YarnRotary(shape[-1], params, max_positions=4096).to(DEV)
    x = torch.randn(*shape, device=DEV).to(dtype)
    positions = torch.randint(0, 4096, (shape[0],), device=DEV, dtype=pos_dtype)
    want = torch_apply(rope, x, positions)
    got = yarn_rope_fused(x, rope.cos, rope.sin, positions)
    assert got.shape == want.shape and got.dtype == dtype
    assert torch.equal(got, want)
    assert torch.equal(rope.apply(x, positions), want), "apply takes the fused path on CUDA"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_rope_reads_split_views_in_place(dtype):
    """The rope part of a query ``[T, H, nope + rope]`` or a key ``[T, latent + rope]`` is a strided
    view: the kernel reads it where it is (no copy) and matches the torch path bit for bit."""
    torch.manual_seed(3)
    rope = YarnRotary(64, YarnParams(original_max_position_embeddings=2048, factor=4.0), max_positions=4096).to(DEV)
    positions = torch.randint(0, 4096, (11,), device=DEV)
    q = torch.randn(11, 4, 128 + 64, device=DEV).to(dtype)
    q_pe = q.split([128, 64], dim=-1)[1]
    assert not q_pe.is_contiguous()
    assert torch.equal(rope.apply(q_pe, positions), torch_apply(rope, q_pe.contiguous(), positions))
    kv = torch.randn(11, 512 + 64, device=DEV).to(dtype)
    k_pe = kv.split([512, 64], dim=-1)[1]
    assert not k_pe.is_contiguous()
    got = rope.apply(k_pe, positions)
    assert got.is_contiguous() and torch.equal(got, torch_apply(rope, k_pe.contiguous(), positions))


def test_fused_rope_is_capturable():
    rope = YarnRotary(64, YarnParams(original_max_position_embeddings=2048, factor=4.0), max_positions=4096).to(DEV)
    x = torch.randn(16, 4, 64, device=DEV).to(torch.bfloat16)
    positions = torch.arange(100, 116, device=DEV)
    want = yarn_rope_fused(x, rope.cos, rope.sin, positions)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = yarn_rope_fused(x, rope.cos, rope.sin, positions)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(got, want)

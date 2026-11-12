"""The opt-in FP8 FA4 self-attention (``RaggedAttnBackend.FA4_FP8``): the per-(segment, head)
e4m3 quantization it feeds the kernel, and the kernel's output against an fp32 reference,
eager and replayed from a CUDA graph. Needs a datacenter Blackwell GPU with flash-attn-4.
"""

from __future__ import annotations

import pytest
import torch
from mstar.engine.resources.attn.ragged.fp8_quant import E4M3_MAX, quantize_per_segment_head

from mstar.engine.resources import (
    AttentionStep,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedAttnBackend,
    Segment,
    StepContext,
)
from mstar.engine.resources.attn.ragged.cudnn_wrappers import Fa4UniformPrefillWrapper
from mstar.engine.resources.attn.ragged.dense import fa4_unavailable_reason
from mstar.engine.resources.base import EngineResourceInfo, build_resource

DEVICE = torch.device("cuda", 0)
_REASON = fa4_unavailable_reason(DEVICE) if torch.cuda.is_available() else "no CUDA"
pytestmark = pytest.mark.skipif(_REASON is not None, reason=f"FA4 unavailable: {_REASON}")

D = 128
# e4m3 q / k / v with one scale per (segment, head): ~3-4% rel L2 from fp32 on unit-normal
# inputs (bf16 kernels: 0.23%); the bound catches a wrong scale or layout, which lands near 100%
FP8_REL_L2 = 6e-2


def qkv(n: int, seq: int, heads: int, seed: int = 0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return tuple(torch.randn(n * seq, heads, D, device=DEVICE, dtype=torch.bfloat16, generator=gen) for _ in range(3))


def reference(q, k, v, n: int) -> torch.Tensor:
    seq = q.shape[0] // n
    out = torch.empty(q.shape, dtype=torch.float32, device=DEVICE)
    for i in range(n):
        rows = slice(i * seq, (i + 1) * seq)
        qs, ks, vs = (t[rows].float().transpose(0, 1) for t in (q, k, v))
        p = torch.softmax(qs @ ks.transpose(-1, -2) * D ** -0.5, dim=-1)
        out[rows] = (p @ vs).transpose(0, 1)
    return out


def rel_l2(got, want) -> float:
    return float((got.float() - want).norm() / want.norm())


def test_each_segment_and_head_gets_its_own_scale():
    n, seq, heads = 2, 1000, 3
    x = torch.randn(n * seq, heads, D, device=DEVICE, dtype=torch.bfloat16)
    x[seq:, 1] *= 50  # segment 1, head 1 spans a far wider range than the rest
    y, descale = quantize_per_segment_head(x, n)
    assert y.dtype == torch.float8_e4m3fn and y.shape == x.shape and descale.shape == (n, heads)
    want = x.float().view(n, seq, heads, D).abs().amax(dim=(1, 3)) / E4M3_MAX
    torch.testing.assert_close(descale, want)
    back = (y.float().view(n, seq, heads, D) * descale[:, None, :, None]).view_as(x)
    # e4m3 keeps 3 mantissa bits: half an ulp is 1/16 of the value, plus the subnormal floor
    err = (back - x.float()).abs()
    assert bool((err <= x.float().abs() / 16 + descale.max() * 2 ** -9).all())


def test_quantize_takes_strided_views():
    seq, heads = 512, 4
    fused = torch.randn(seq, 3, heads, D, device=DEVICE, dtype=torch.bfloat16)
    q = fused[:, 0]
    y, descale = quantize_per_segment_head(q, 1)
    y2, descale2 = quantize_per_segment_head(q.contiguous(), 1)
    assert torch.equal(y.view(torch.uint8), y2.view(torch.uint8)) and torch.equal(descale, descale2)


@pytest.mark.parametrize(("n", "seq", "heads"), [(1, 4608, 24), (2, 4096, 40)])
def test_fp8_matches_fp32_within_its_quantization(n, seq, heads):
    q, k, v = qkv(n, seq, heads)
    want = reference(q, k, v, n)
    cu = torch.arange(0, (n + 1) * seq, seq, dtype=torch.int32)
    kw = dict(num_qo_heads=heads, num_kv_heads=heads, head_dim=D, device=DEVICE, q_data_type=torch.bfloat16,
              precision="fp8")
    eager = Fa4UniformPrefillWrapper(**kw)
    eager.plan(cu)
    out = eager.run(q, k, v)
    assert out.dtype == torch.bfloat16
    assert rel_l2(out, want) < FP8_REL_L2
    bf16 = Fa4UniformPrefillWrapper(**{**kw, "precision": "bf16"})
    bf16.plan(cu)
    assert rel_l2(bf16.run(q, k, v), want) < 5e-3

    captured = Fa4UniformPrefillWrapper(**kw, accelerator_graph=True, max_num_segments=n, max_total_tokens=n * seq)
    captured.plan(cu)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        captured.run(q, k, v)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = captured.run(q, k, v)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, out)


def test_the_backend_builds_and_runs_fp8():
    heads, seq = 24, 2048
    spec = RaggedAttentionSpec(resource_key="dit_attn", nodes={"dit"}, config=RaggedAttentionConfig(
        num_qo_heads=heads, num_kv_heads=heads, head_dim=D, dtype=torch.bfloat16, backend=RaggedAttnBackend.FA4_FP8))
    manager = build_resource(spec, EngineResourceInfo(device=DEVICE, kv_dtype=torch.bfloat16))
    step = AttentionStep(segments=tuple(Segment(request_id=f"r{i}", label="main", span=seq) for i in range(2)),
                         causal=False)
    manager.plan(step, StepContext(request_ids=["r0", "r1"], graph_walk="video_gen", slot=0, capture=False))
    wrapper = manager._current_plan_states["main"]
    assert type(wrapper) is Fa4UniformPrefillWrapper and wrapper.precision == "fp8"
    q, k, v = qkv(2, seq, heads)
    assert rel_l2(manager.run(q, k, v, label="main"), reference(q, k, v, 2)) < FP8_REL_L2

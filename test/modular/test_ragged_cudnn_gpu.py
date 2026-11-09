"""The cacheless self-attention kernels against an fp32 reference at the image
DiTs' sizes (klein 4B / 9B, Z-Image Turbo at 1024x1024), eager and replayed
from a CUDA graph, and the resource running the kernel its table picks.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from mstar.engine.resources import (
    AttentionStep,
    BucketKey,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    Segment,
    SlotLease,
    StepContext,
)
from mstar.engine.resources.attn.ragged.cudnn_wrappers import CudnnRaggedPrefillWrapper
from mstar.engine.resources.attn.ragged.wrappers import FlashInferRaggedPrefillWrapper
from mstar.engine.resources.attn.workspace import Workspace
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DEVICE = torch.device("cuda", 0)
DTYPE = torch.bfloat16
D = 128
# heads, segment length: klein 4B / 9B joint attention (512 text + 4096 image tokens), Z-Image's
# main, image and caption spans (its captions are padded to a multiple of 32), and Wan2.2-5B's
# self-attention at 832x480x33, the other dit whose geometry the sm_103 table covers
SIZES = [(24, 4608), (32, 4608), (30, 4192), (30, 4096), (30, 96), (24, 3510)]
# bf16 kernels sit at ~0.23% rel L2 from fp32 here; the fp32 reference rounded to bf16 is 0.17%
REL_L2 = 5e-3


def cu(n: int, seq: int) -> torch.Tensor:
    return torch.arange(0, (n + 1) * seq, seq, dtype=torch.int32)


def qkv(total: int, heads: int, seed: int = 0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return tuple(torch.randn(total, heads, D, device=DEVICE, dtype=DTYPE, generator=gen) for _ in range(3))


def reference(q, k, v, cu_seqlens, lse=False):
    out = torch.zeros(q.shape, dtype=torch.float32, device=DEVICE)
    lses = torch.zeros(q.shape[:2], dtype=torch.float32, device=DEVICE)
    bounds = cu_seqlens.tolist()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for s, e in zip(bounds[:-1], bounds[1:], strict=True):
            qs, ks, vs = (t[s:e].float().transpose(0, 1) for t in (q, k, v))
            with sdpa_kernel(SDPBackend.MATH):
                out[s:e] = F.scaled_dot_product_attention(qs, ks, vs).transpose(0, 1)
            if lse:
                lses[s:e] = torch.logsumexp(qs @ ks.transpose(-1, -2) * D ** -0.5, dim=-1).transpose(0, 1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return (out, lses) if lse else out


def rel_l2(got, want) -> float:
    return float((got.float() - want).norm() / want.norm())


def make(kernel: str, heads: int, n: int, seq: int, graph: bool, rows: int | None = None):
    kw = dict(num_qo_heads=heads, num_kv_heads=heads, head_dim=D, device=DEVICE, q_data_type=DTYPE)
    if graph:
        kw.update(accelerator_graph=True, max_num_segments=n, max_total_tokens=rows or n * seq)
    if kernel == "cudnn":
        return CudnnRaggedPrefillWrapper(**kw)
    ws = Workspace(torch.empty(256 << 20, dtype=torch.uint8, device=DEVICE), "test")
    return FlashInferRaggedPrefillWrapper(ws, backend=kernel.removeprefix("flashinfer_"), **kw)


def replay(fn):
    """``fn()`` captured into a CUDA graph and replayed."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return out


KERNELS = ["cudnn", "flashinfer_fa2", "flashinfer_cudnn"]


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(("heads", "seq"), SIZES, ids=[f"{h}h_{s}" for h, s in SIZES])
@pytest.mark.parametrize("n", [1, 2])
def test_kernel_matches_fp32_at_the_dit_sizes(kernel, heads, seq, n):
    q, k, v = qkv(n * seq, heads)
    want = reference(q, k, v, cu(n, seq))
    eager = make(kernel, heads, n, seq, graph=False)
    eager.plan(cu(n, seq))
    assert rel_l2(eager.run(q, k, v), want) < REL_L2
    captured = make(kernel, heads, n, seq, graph=True)
    captured.plan(cu(n, seq))
    assert rel_l2(replay(lambda: captured.run(q, k, v)), want) < REL_L2


def test_cudnn_lse_matches_the_reference():
    q, k, v = qkv(2 * 4608, 24)
    want, want_lse = reference(q, k, v, cu(2, 4608), lse=True)
    w = make("cudnn", 24, 2, 4608, graph=True)
    w.plan(cu(2, 4608))
    out, lse = replay(lambda: w.run(q, k, v, return_lse=True))
    assert rel_l2(out, want) < REL_L2
    assert lse.shape == want_lse.shape and lse.dtype == torch.float32
    torch.testing.assert_close(lse, want_lse, atol=2e-2, rtol=0)


def test_cudnn_rows_past_the_layout_read_zero():
    """A graph wrapper sized for more rows than its layout fills: the tail reads back zero
    whatever the inputs hold there."""
    heads, seq = 24, 512
    q, k, v = qkv(3 * seq, heads)
    for t in (q, k, v):
        t[2 * seq:] = float("nan")
    w = make("cudnn", heads, 2, seq, graph=True, rows=3 * seq)
    w.plan(cu(2, seq))
    out = replay(lambda: w.run(q, k, v))
    assert out.shape == q.shape
    assert rel_l2(out[: 2 * seq], reference(q[: 2 * seq], k[: 2 * seq], v[: 2 * seq], cu(2, seq))) < REL_L2
    assert bool((out[2 * seq:] == 0).all())


def test_cudnn_takes_strided_views():
    """q / k / v sliced out of a fused projection."""
    heads, seq = 30, 4192
    qkv_fused = torch.randn(seq, 3, heads, D, device=DEVICE, dtype=DTYPE)
    q, k, v = qkv_fused.unbind(1)
    w = make("cudnn", heads, 1, seq, graph=False)
    w.plan(cu(1, seq))
    assert rel_l2(w.run(q, k, v), reference(q, k, v, cu(1, seq))) < REL_L2


def test_eager_cudnn_runs_unequal_segments_on_its_fallback():
    heads = 24
    lens = torch.tensor([0, 4608, 4608 + 1024], dtype=torch.int32)
    q, k, v = qkv(4608 + 1024, heads)
    ws = Workspace(torch.empty(256 << 20, dtype=torch.uint8, device=DEVICE), "test")
    w = CudnnRaggedPrefillWrapper(
        num_qo_heads=heads, num_kv_heads=heads, head_dim=D, device=DEVICE, q_data_type=DTYPE,
        fallback=lambda: FlashInferRaggedPrefillWrapper(ws, num_qo_heads=heads, num_kv_heads=heads, head_dim=D,
                                                        device=DEVICE, q_data_type=DTYPE),
    )
    w.plan(lens)
    assert w._on_fallback
    assert rel_l2(w.run(q, k, v), reference(q, k, v, lens)) < REL_L2


@pytest.mark.skipif(not torch.cuda.is_available() or current_platform.arch() != "sm_103",
                    reason="the shipped table covers sm_103")
@pytest.mark.parametrize("captured", [False, True], ids=["eager", "captured"])
@pytest.mark.parametrize(("heads", "seq"), [(24, 4608), (30, 96)])
def test_the_resource_runs_the_tables_kernel(heads, seq, captured):
    spec = RaggedAttentionSpec(resource_key="dit_attn", nodes={"dit"}, config=RaggedAttentionConfig(
        num_qo_heads=heads, num_kv_heads=heads, head_dim=D, dtype=DTYPE))
    manager = build_resource(spec, EngineResourceInfo(device=DEVICE, kv_dtype=DTYPE))
    step = AttentionStep(segments=tuple(Segment(request_id=f"r{i}", label="main", span=seq) for i in range(2)),
                         causal=False, fixed_layout=True)
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="image_gen", bs=2, num_tokens=2 * seq)) if captured else None
    manager.plan(step, StepContext(request_ids=["r0", "r1"], graph_walk="image_gen", slot=0, capture=False,
                                   is_preplan=False, slot_lease=lease))
    wrapper = manager._current_plan_states["main"]
    assert type(wrapper) is CudnnRaggedPrefillWrapper and wrapper.accelerator_graph is captured
    q, k, v = qkv(2 * seq, heads)
    out = replay(lambda: manager.run(q, k, v, label="main")) if captured else manager.run(q, k, v, label="main")
    assert rel_l2(out, reference(q, k, v, cu(2, seq))) < REL_L2

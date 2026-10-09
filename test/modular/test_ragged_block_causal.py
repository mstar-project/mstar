"""Block-causal cacheless attention: the page-list layout, the FlashInfer
wrapper eager and under CUDA-graph replay, and the resource that plans it.

The reference is per-segment SDPA with a dense block-causal mask: a token in
block ``b`` of its segment attends keys of blocks ``0..b`` of that segment.
Runs at head_dim=72, which FlashInfer has no kernel for, so the pad-to-128
path is exercised too.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch
import torch.nn.functional as F

from mstar.engine.resources import (
    AttentionStep,
    BucketKey,
    RaggedAttentionConfig,
    RaggedBlockCausalAttentionSpec,
    Segment,
    SlotLease,
    StepContext,
)
from mstar.engine.resources.attn.ragged.block_causal import (
    RaggedBlockCausalWrapper,
    block_causal_layout,
    max_blocks,
    max_prefix_tokens,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer requires CUDA")

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
HEADS, HEAD_DIM, BLOCK = 4, 72, 8
MAX_SEGMENTS, MAX_TOKENS = 4, 96
TOL = 3e-2  # bf16 vs an fp32 reference
ATTN = "block_causal"

LAYOUTS = {
    "aligned": [16, 32],
    "partial_blocks": [13, 21, 5],
    "shorter_than_a_block": [3],
    "with_padding_rows": [20, 0, 9, 0],
    "one_long": [MAX_TOKENS],
}


@pytest.fixture(autouse=True)
def _small_workspaces(monkeypatch):
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "64")


def qkv(total: int, seed: int = 0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return tuple(
        torch.randn(total, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE, generator=gen)
        for _ in range(3)
    )


def ref(q, k, v, seg_lens):
    out = torch.zeros_like(q)
    off = 0
    for n in seg_lens:
        if n:
            blk = torch.arange(n, device=q.device) // BLOCK
            mask = blk[None, :] <= blk[:, None]
            qs, ks, vs = (t[off:off + n].transpose(0, 1).float() for t in (q, k, v))
            o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask, scale=HEAD_DIM ** -0.5)
            out[off:off + n] = o.transpose(0, 1).to(out.dtype)
        off += n
    return out


def close(got, want, tol=TOL):
    err = (got.float() - want.float()).abs().max().item()
    assert err < tol, f"max_abs_err={err} exceeds {tol}"


def wrapper(**overrides) -> RaggedBlockCausalWrapper:
    kwargs = dict(
        workspace_buffer=torch.empty(64 << 20, dtype=torch.uint8, device=DEVICE),
        num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM, block_size=BLOCK,
        device=DEVICE, q_data_type=DTYPE,
    )
    kwargs.update(overrides)
    return RaggedBlockCausalWrapper(**kwargs)


def graph_wrapper() -> RaggedBlockCausalWrapper:
    return wrapper(max_num_segments=MAX_SEGMENTS, max_total_tokens=MAX_TOKENS, use_cuda_graph=True)


# --- the layout (host only) ------------------------------------------------

def test_layout_lists_each_blocks_prefix():
    qo, kv, idx = block_causal_layout([5, 3], block_size=2)
    # span 0: blocks [0,2) [2,4) [4,5); span 1 (offset 5): [5,7) [7,8)
    assert qo == [0, 2, 4, 5, 7, 8]
    assert kv == [0, 2, 6, 11, 13, 16]
    assert idx.tolist() == [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 5, 6, 7]


def test_layout_of_nothing_is_empty():
    qo, kv, idx = block_causal_layout([0, 0], block_size=4)
    assert qo == [0] and kv == [0] and idx.numel() == 0


@pytest.mark.parametrize("seg_lens", list(LAYOUTS.values()), ids=list(LAYOUTS))
def test_graph_ceilings_cover_any_layout_in_the_bucket(seg_lens):
    qo, kv, _ = block_causal_layout(seg_lens, BLOCK)
    assert len(qo) - 1 <= max_blocks(MAX_SEGMENTS, MAX_TOKENS, BLOCK)
    assert kv[-1] <= max_prefix_tokens(MAX_TOKENS, BLOCK)


# --- the wrapper -----------------------------------------------------------

@cuda
@pytest.mark.parametrize("seg_lens", list(LAYOUTS.values()), ids=list(LAYOUTS))
def test_eager_matches_reference(seg_lens):
    q, k, v = qkv(sum(seg_lens))
    w = wrapper()
    w.plan(seg_lens)
    close(w.run(q, k, v), ref(q, k, v, seg_lens))


@cuda
def test_one_block_is_bidirectional_and_blocks_are_causal():
    """The two halves of the mask, checked directly: perturbing a later block
    leaves earlier blocks unchanged, perturbing a token's own block does not."""
    seg = [2 * BLOCK]
    q, k, v = qkv(sum(seg))
    w = wrapper()
    w.plan(seg)
    base = w.run(q, k, v)
    v2 = v.clone()
    v2[-1] += 10.0  # last token of block 1
    out = w.run(q, k, v2)
    assert torch.equal(out[:BLOCK], base[:BLOCK])
    assert not torch.allclose(out[BLOCK], base[BLOCK])  # first token of block 1 sees the last


def capture(w):
    static = tuple(torch.zeros(MAX_TOKENS, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) for _ in range(3))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            w.run(*static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = w.run(*static)
    torch.cuda.synchronize()
    return graph, static, out


@cuda
def test_graph_replay_matches_reference_across_changing_layouts():
    w = graph_wrapper()
    graph, static, out = capture(w)
    for i, seg_lens in enumerate(list(LAYOUTS.values()) * 2):
        total = sum(seg_lens)
        q, k, v = qkv(total, seed=i)
        w.plan(seg_lens)
        for buf, real in zip(static, (q, k, v), strict=True):
            buf.zero_()
            buf[:total].copy_(real)
        graph.replay()
        torch.cuda.synchronize()
        close(out[:total], ref(q, k, v, seg_lens))


@cuda
def test_graph_mode_rejects_what_the_bucket_cannot_hold():
    with pytest.raises(ValueError, match="segments exceeds"):
        graph_wrapper().plan([1] * (MAX_SEGMENTS + 1))
    with pytest.raises(ValueError, match="tokens exceeds"):
        graph_wrapper().plan([MAX_TOKENS, 1])


# --- the resource ----------------------------------------------------------

def manager():
    spec = RaggedBlockCausalAttentionSpec(
        resource_key=ATTN, nodes={"encoder"}, block_size=BLOCK,
        config=RaggedAttentionConfig(
            num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM,
            max_segments_per_request=MAX_SEGMENTS,
        ),
    )
    return build_resource(spec, EngineResourceInfo(device=DEVICE, kv_dtype=DTYPE))


def ctx(n, lease=None) -> StepContext:
    return StepContext(
        request_ids=list(range(n)), graph_walk="encode", slot=0, capture=False, slot_lease=lease,
    )


def step(seg_lens, causal=False, label="main"):
    return AttentionStep(
        segments=tuple(Segment(request_id=f"r{i}", label=label, span=n) for i, n in enumerate(seg_lens)),
        causal=causal,
    )


@cuda
def test_plan_then_run_matches_reference():
    seg_lens = [13, 21, 5]
    m = manager()
    m.plan(step(seg_lens), ctx(len(seg_lens)))
    q, k, v = qkv(sum(seg_lens))
    close(m.run(q, k, v, label="main"), ref(q, k, v, seg_lens))


@cuda
def test_a_causal_step_is_refused():
    with pytest.raises(ValueError, match="causal=False"):
        manager().plan(step([8], causal=True), ctx(1))


@cuda
def test_lease_plans_a_graph_wrapper_sized_by_the_bucket():
    m = manager()
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="encode", bs=1, num_tokens=MAX_TOKENS))
    m.plan(step([20, 9]), ctx(2, lease=lease))
    w = m._wrapper("main")
    assert w.use_cuda_graph and w.max_total_tokens == MAX_TOKENS
    assert w.max_blocks == max_blocks(MAX_SEGMENTS, MAX_TOKENS, BLOCK)

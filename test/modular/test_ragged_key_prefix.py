"""Key-prefix cacheless attention: every token of a segment queries, and attends the
first ``key_lens[i]`` tokens of its own segment (a key padding mask).

The host layout is checked on the CPU; the wrappers (FlashInfer, and FA3 where it
runs) eager and under CUDA-graph replay, and the resource that plans them, against
per-segment SDPA over the key prefix. Runs at head_dim=72, which FlashInfer has no
kernel for, so the pad-to-128 path is exercised too.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mstar.engine.resources import (
    BucketKey,
    KeyPrefixAttentionStep,
    RaggedAttentionConfig,
    RaggedKeyPrefixAttentionSpec,
    Segment,
    SlotLease,
    StepContext,
)
from mstar.engine.resources.attn.ragged.config import RaggedAttnBackend
from mstar.engine.resources.attn.ragged.wrappers import FlashInferKeyPrefixPrefillWrapper, key_prefix_layout
from mstar.engine.resources.attn.workspace import Workspace
from mstar.engine.resources.base import EngineResourceInfo, build_resource

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer requires CUDA")

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
HEADS, HEAD_DIM = 4, 72
MAX_SEGMENTS, MAX_TOKENS = 4, 96
TOL = 3e-2  # bf16 vs an fp32 reference
ATTN = "key_prefix"

# (spans, key_lens)
LAYOUTS = {
    "one_masked_key_each": ([16, 31], [15, 30]),
    "two_masked_and_none": ([13, 21, 5], [11, 21, 5]),
    "with_padding_rows": ([20, 0, 9, 0], [18, 0, 7, 0]),
    "a_single_key": ([7], [1]),
    "one_long": ([MAX_TOKENS], [MAX_TOKENS - 2]),
}


@pytest.fixture(autouse=True)
def _small_workspaces(monkeypatch):
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "64")


def qkv(total: int, seed: int = 0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return tuple(
        torch.randn(total, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE, generator=gen) for _ in range(3)
    )


def ref(q, k, v, spans, key_lens):
    out = torch.zeros_like(q)
    off = 0
    for n, kl in zip(spans, key_lens, strict=True):
        if n:
            qs = q[off:off + n].transpose(0, 1).float()
            ks, vs = (t[off:off + kl].transpose(0, 1).float() for t in (k, v))
            out[off:off + n] = F.scaled_dot_product_attention(qs, ks, vs, scale=HEAD_DIM ** -0.5).transpose(0, 1)
        off += n
    return out


def close(got, want, tol=TOL):
    err = (got.float() - want.float()).abs().max().item()
    assert err < tol, f"max_abs_err={err} exceeds {tol}"


def cu(spans) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(spans).cumsum(0).tolist()], dtype=torch.int32)


# --- the layout (host only) ------------------------------------------------

def test_layout_packs_each_segments_key_prefix():
    kv_cu, index = key_prefix_layout([4, 0, 3], [2, 0, 3])
    assert kv_cu.tolist() == [0, 2, 2, 5]
    assert index.tolist() == [0, 1, 4, 5, 6]


@pytest.mark.parametrize(("spans", "key_lens"), [([4], [5]), ([4], [0]), ([4, 2], [1])])
def test_layout_refuses_keys_outside_their_segment(spans, key_lens):
    with pytest.raises(ValueError):
        key_prefix_layout(spans, key_lens)


# --- the wrapper -----------------------------------------------------------

def wrapper(**overrides) -> FlashInferKeyPrefixPrefillWrapper:
    kwargs = dict(
        workspace_buffer=Workspace(torch.empty(64 << 20, dtype=torch.uint8, device=DEVICE), "test"),
        num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM, device=DEVICE, q_data_type=DTYPE,
    )
    kwargs.update(overrides)
    return FlashInferKeyPrefixPrefillWrapper(**kwargs)


@cuda
@pytest.mark.parametrize(("spans", "key_lens"), list(LAYOUTS.values()), ids=list(LAYOUTS))
def test_eager_matches_reference(spans, key_lens):
    q, k, v = qkv(sum(spans))
    w = wrapper()
    w.plan_key_prefix(cu(spans), key_lens)
    close(w.run(q, k, v), ref(q, k, v, spans, key_lens))


@cuda
def test_masked_keys_are_not_attended_but_still_query():
    spans, key_lens = [12], [10]
    q, k, v = qkv(12)
    w = wrapper()
    w.plan_key_prefix(cu(spans), key_lens)
    base = w.run(q, k, v)
    k2, v2 = k.clone(), v.clone()
    k2[10:] += 50.0
    v2[10:] += 50.0
    assert torch.equal(w.run(q, k2, v2), base)  # the masked keys change nothing
    q2 = q.clone()
    q2[11] += 5.0
    assert not torch.equal(w.run(q2, k, v)[11], base[11])  # but a masked token is still a query


def capture(w, spans, key_lens):
    static = tuple(torch.zeros(MAX_TOKENS, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) for _ in range(3))
    w.plan_key_prefix(cu(spans), key_lens)
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
    w = wrapper(max_num_segments=MAX_SEGMENTS, max_total_tokens=MAX_TOKENS, accelerator_graph=True)
    graph, static, out = capture(w, [MAX_TOKENS], [MAX_TOKENS])
    for i, (spans, key_lens) in enumerate(list(LAYOUTS.values()) * 2):
        total = sum(spans)
        q, k, v = qkv(total, seed=i)
        w.plan_key_prefix(cu(spans), key_lens)
        for buf, real in zip(static, (q, k, v), strict=True):
            buf.zero_()
            buf[:total].copy_(real)
        graph.replay()
        torch.cuda.synchronize()
        close(out[:total], ref(q, k, v, spans, key_lens))
        assert (out[total:] == 0).all()


# --- the resource ----------------------------------------------------------

def manager(backend=RaggedAttnBackend.FLASHINFER):
    spec = RaggedKeyPrefixAttentionSpec(
        resource_key=ATTN, nodes={"encoder"},
        config=RaggedAttentionConfig(
            num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM,
            max_segments_per_request=MAX_SEGMENTS, backend=backend,
        ),
    )
    return build_resource(spec, EngineResourceInfo(device=DEVICE, kv_dtype=DTYPE))


def ctx(n, lease=None) -> StepContext:
    return StepContext(request_ids=list(range(n)), graph_walk="encode", slot=0, capture=False, slot_lease=lease)


def step(spans, key_lens, label="main"):
    return KeyPrefixAttentionStep(
        segments=tuple(Segment(request_id=f"r{i}", label=label, span=n) for i, n in enumerate(spans)),
        key_lens=tuple(key_lens),
    )


@cuda
@pytest.mark.parametrize("backend", [RaggedAttnBackend.FLASHINFER, RaggedAttnBackend.DENSE])
def test_plan_then_run_matches_reference(backend):
    spans, key_lens = LAYOUTS["two_masked_and_none"]
    m = manager(backend)
    m.plan(step(spans, key_lens), ctx(len(spans)))
    q, k, v = qkv(sum(spans))
    close(m.run(q, k, v), ref(q, k, v, spans, key_lens))


@cuda
def test_a_captured_bucket_plans_a_graph_wrapper():
    m = manager()
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="encode", bs=1, num_tokens=MAX_TOKENS))
    spans, key_lens = LAYOUTS["with_padding_rows"]
    m.plan(step(spans, key_lens), ctx(len(spans), lease=lease))
    w = m._wrapper("main")
    assert w.accelerator_graph and w.max_total_tokens == MAX_TOKENS
    # replayed padded inside a graph (test_graph_replay_matches_reference_across_changing_layouts)
    assert w._kv_index.numel() == MAX_TOKENS


@cuda
def test_key_lengths_must_match_the_segments():
    with pytest.raises(ValueError, match="key lengths for"):
        manager().plan(KeyPrefixAttentionStep(
            segments=(Segment(request_id="r0", label="main", span=4),), key_lens=(4, 4),
        ), ctx(1))

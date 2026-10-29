"""A GDN prefill saves its state at the page boundaries its row keeps, into the recurrent
pool's snapshot slots, where a later prompt restores it.

The saved state must be the one a prefill ending at that boundary leaves in its slot: the
recurrent state from FlashInfer's checkpoint there, and the conv window of the inputs before
it. A restore from anything else decodes the later prompt from the wrong state.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.linear_attn import LinearAttnConfig, LinearAttnManager, LinearAttnSpec, LinearAttnStep
from mstar.engine.resources.linear_attn.config import LinearAttnVariant
from mstar.engine.resources.recurrent import (
    DeltaNetGeometry,
    RecurrentStateConfig,
    RecurrentStatePool,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="runs the GDN kernels; needs a GPU")

POOL, GDN, PAGE = "gdn_state", "linear_attn", 128
GEOMETRY = DeltaNetGeometry(num_k_heads=2, num_v_heads=4, head_k_dim=128, head_v_dim=128, conv_kernel_size=4)


class _StubChains:
    """Keys every whole page of a request as its index."""

    config = type("Config", (), {"page_size": PAGE})()

    def keyed_label(self, rid, node_name, graph_walk):
        return "main"

    def keyed_walks(self, label):
        return frozenset({"prefill"})

    def chain_key(self, rid, label, pages):
        return f"{rid}/{pages}".encode()

    def chain_keys(self, rid, label, pages):
        return [self.chain_key(rid, label, n) for n in range(1, pages + 1)]


def _build(device):
    spec = RecurrentStateSpec(POOL, {"llm"}, RecurrentStateConfig(
        num_layers=1, blocks=GEOMETRY.to_blocks(), max_slots=6, prefix_slots=4, kv_cache="kv",
    ))
    kv = KVSpec("kv", {"llm"}, PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4 * PAGE, page_size=PAGE,
    ))
    pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=device, dependencies={"kv": kv}))
    pool.bind_kv(_StubChains())
    manager = LinearAttnManager.build(
        LinearAttnSpec(GDN, {"llm"}, LinearAttnConfig(recurrent_state=POOL, variant=LinearAttnVariant.GDN)),
        EngineResourceInfo(device=device, dependencies={POOL: spec}),
    )
    manager.bind_dependencies({POOL: pool})
    return pool, manager


def _prefill(pool, manager, rows: dict[str, dict], shared: dict) -> None:
    """Admit, plan and run one prefill step of ``rows``, each request's own inputs."""
    segments = tuple(Segment(rid, "main", x["q"].shape[0]) for rid, x in rows.items())
    for rid in rows:
        pool.ingest_request(rid)
        pool.resolve_cached_prefix(rid, "llm", "prefill", None)
    step = RecurrentStep(segments=segments)
    ctx = StepContext(request_ids=tuple(rows), graph_walk="prefill", slot=0, capture=False)
    assert pool.admit(step, ctx).ok
    ctx.plan_results[POOL] = pool.plan(step, ctx)
    manager.plan(LinearAttnStep(segments=segments), ctx)
    x = {name: torch.cat([inputs[name] for inputs in rows.values()]) for name in ("conv", "q", "k", "v", "a", "b")}
    manager.run_conv(x["conv"], pool.block("conv", 0), shared["weight"], label="main")
    manager.run(x["q"], x["k"], x["v"], x["a"], x["b"], pool.block("state", 0),
                shared["a_log"], shared["dt_bias"], label="main")
    pool.commit(step, ctx)


def _shared(device) -> dict:
    hv = GEOMETRY.num_v_heads
    return {
        "weight": torch.randn(GEOMETRY.conv_dim, GEOMETRY.conv_kernel_size, device=device, dtype=torch.bfloat16),
        "a_log": torch.randn(hv, device=device),
        "dt_bias": torch.randn(hv, device=device),
    }


def _inputs(device, n: int) -> dict:
    h, hv, k = GEOMETRY.num_k_heads, GEOMETRY.num_v_heads, GEOMETRY.head_k_dim
    return {
        "conv": torch.randn(n, GEOMETRY.conv_dim, device=device, dtype=torch.bfloat16),
        "q": torch.randn(n, h, k, device=device, dtype=torch.bfloat16),
        "k": torch.randn(n, h, k, device=device, dtype=torch.bfloat16),
        "v": torch.randn(n, hv, k, device=device, dtype=torch.bfloat16) * 0.5,
        "a": torch.randn(n, hv, device=device, dtype=torch.bfloat16),
        "b": torch.randn(n, hv, device=device, dtype=torch.bfloat16),
    }


@pytest.mark.parametrize("lengths", [{"a": 300}, {"a": 300, "b": 200}])
def test_a_prefill_saves_the_state_a_prefill_ending_at_its_boundary_leaves(lengths):
    device = torch.device("cuda")
    torch.manual_seed(0)
    pool, manager = _build(device)
    shared = _shared(device)
    rows = {rid: _inputs(device, n) for rid, n in lengths.items()}
    _prefill(pool, manager, rows, shared)
    conv, state = pool.block("conv", 0), pool.block("state", 0)
    for rid, n in lengths.items():
        boundary = n // PAGE * PAGE
        snap = pool._snap_by_key[f"{rid}/{boundary // PAGE}".encode()]
        _prefill(pool, manager, {f"{rid}-ref": {k: v[:boundary] for k, v in rows[rid].items()}}, shared)
        ref = pool._slots[f"{rid}-ref"]["main"].index
        torch.cuda.synchronize()
        assert torch.equal(conv[snap], conv[ref]), f"{rid}'s conv window is not the one a prefill to {boundary} leaves"
        # FlashInfer's context-parallel kernel, which a step of a few rows takes, rounds a boundary
        # state by how it splits the row. The reference prefill splits the row otherwise, so the
        # two states differ by about 1e-3.
        assert (state[snap] - state[ref]).abs().max().item() < 1e-2, f"{rid}'s saved state is not its checkpoint"


def test_a_prefill_past_the_largest_bucket_saves_the_state_at_its_widened_boundary():
    """Past the largest captured bucket a step checkpoints less often, so it fits the shared
    buffer, and keeps its boundary on the wider interval: 768 for a 1,000-token row here."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    pool, manager = _build(device)
    for resource in (pool, manager):
        resource.build_accelerator_graph_buffers([], max_bs=1, max_seq_len=512)
    shared = _shared(device)
    rows = _inputs(device, 1000)
    _prefill(pool, manager, {"a": rows}, shared)
    assert manager.current_plan("main")._num_checkpoints <= (512 + 1) // PAGE + 1
    snap = pool._snap_by_key[b"a/6"]
    _prefill(pool, manager, {"a-ref": {k: v[:768] for k, v in rows.items()}}, shared)
    ref = pool._slots["a-ref"]["main"].index
    conv, state = pool.block("conv", 0), pool.block("state", 0)
    torch.cuda.synchronize()
    assert torch.equal(conv[snap], conv[ref]), "the widened step saved another conv window"
    assert (state[snap] - state[ref]).abs().max().item() < 1e-2, "the widened step saved another checkpoint"


def test_a_captured_prefill_saves_the_state_an_eager_prefill_ending_at_its_boundary_leaves():
    """A replay addresses its snapshots through buffers fixed at capture: the shared checkpoint
    buffer, and a snapshot index whose padding entries write the sink slot."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    pool, manager = _build(device)
    bucket = 512
    manager.build_accelerator_graph_buffers([], max_bs=1, max_seq_len=bucket)
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="prefill", bs=1, num_tokens=bucket))
    shared = _shared(device)
    static = {name: torch.zeros_like(x) for name, x in _inputs(device, bucket).items()}
    conv, state = pool.block("conv", 0), pool.block("state", 0)

    def plan(rid: str, span: int):
        pool.ingest_request(rid)
        pool.resolve_cached_prefix(rid, "llm", "prefill", None)
        step = RecurrentStep(segments=(Segment(rid, "main", span),))
        ctx = StepContext(request_ids=(rid,), graph_walk="prefill", slot=0, capture=False, slot_lease=lease)
        assert pool.admit(step, ctx).ok
        ctx.plan_results[POOL] = pool.plan(step, ctx)
        manager.plan(LinearAttnStep(segments=step.segments), ctx)
        return step, ctx

    def forward():
        manager.run_conv(static["conv"], conv, shared["weight"], label="main")
        manager.run(static["q"], static["k"], static["v"], static["a"], static["b"], state,
                    shared["a_log"], shared["dt_bias"], label="main")

    step, ctx = plan("__cg_dummy__", bucket)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        forward()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    pool.commit(step, ctx)
    pool.remove_request("__cg_dummy__")

    rows = _inputs(device, 300)
    step, ctx = plan("a", 300)
    for name, x in rows.items():
        static[name][:300].copy_(x)
    graph.replay()
    pool.commit(step, ctx)
    snap = pool._snap_by_key[b"a/2"]
    _prefill(pool, manager, {"a-ref": {k: v[:256] for k, v in rows.items()}}, shared)
    ref = pool._slots["a-ref"]["main"].index
    torch.cuda.synchronize()
    assert torch.equal(conv[snap], conv[ref]), "the replay saved another conv window"
    assert (state[snap] - state[ref]).abs().max().item() < 1e-2, "the replay saved another checkpoint"

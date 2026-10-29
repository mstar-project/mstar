"""A GDN draft-verify step (``GDNVerifyWrapper``): blocks of tokens run from each slot without
writing it, then the verdict settles each row's state after its accepted prefix.

On a GPU, against the decode path a token at a time (FlashInfer's one-token kernel and the
conv update kernel): the verify step's outputs are the decode path's over the whole block,
and the settled conv window and state are the decode path's after the accepted prefix, for
every accepted count in one batch, a padding row included; a captured verify step replays
as the eager one runs. On CPU, against the reference recurrence over the prefix alone: the
closed gates leave the state exactly where the prefix left it.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.linear_attn import (
    LinearAttnConfig,
    LinearAttnManager,
    LinearAttnSpec,
    LinearAttnStep,
)
from mstar.engine.resources.linear_attn.config import LinearAttnVariant
from mstar.engine.resources.linear_attn.wrappers import _gated_delta_rule_reference
from mstar.engine.resources.recurrent import (
    DeltaNetGeometry,
    RecurrentStateConfig,
    RecurrentStatePool,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext

GDN_STATE, LINEAR_ATTN = "gdn_state", "linear_attn"
LAYERS = 2
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer's GDN kernels need a GPU")


def _build(device, geometry: DeltaNetGeometry):
    pool_spec = RecurrentStateSpec(
        GDN_STATE, {"llm"},
        RecurrentStateConfig(num_layers=LAYERS, blocks=geometry.to_blocks(), max_slots=8),
    )
    pool = RecurrentStatePool.build(pool_spec, EngineResourceInfo(device=device))
    manager = LinearAttnManager.build(
        LinearAttnSpec(
            LINEAR_ATTN, {"llm"}, LinearAttnConfig(recurrent_state=GDN_STATE, variant=LinearAttnVariant.GDN),
        ),
        EngineResourceInfo(device=device, dependencies={GDN_STATE: pool_spec}),
    )
    return pool, manager


def _plan(pool, manager, rids, span, verify, lease=None, padded=None):
    ctx = StepContext(request_ids=list(rids), graph_walk="decode", slot=0, capture=False, slot_lease=lease)
    if padded is not None:
        ctx.set_padded_rids(padded)
    segments = tuple(Segment(rid, "main", span) for rid in (padded or rids))
    rstep = RecurrentStep(segments=segments)
    assert pool.admit(rstep, ctx).ok
    ctx.plan_results[GDN_STATE] = pool.plan(rstep, ctx)
    manager.plan(LinearAttnStep(segments=segments, verify=verify), ctx)
    return rstep, ctx


class _Inputs:
    """One step's per-layer inputs for ``rows`` blocks of ``block`` tokens."""

    def __init__(self, geometry, rows, block, device, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        g, n = geometry, rows * block

        def r(*shape, scale=1.0):
            return (torch.randn(*shape, generator=gen) * scale).to(device)

        self.x = [r(n, g.conv_dim).bfloat16() for _ in range(LAYERS)]
        self.q = [r(n, g.num_k_heads, g.head_k_dim).bfloat16() for _ in range(LAYERS)]
        self.k = [r(n, g.num_k_heads, g.head_k_dim).bfloat16() for _ in range(LAYERS)]
        self.v = [r(n, g.num_v_heads, g.head_v_dim).bfloat16() for _ in range(LAYERS)]
        self.a = [r(n, g.num_v_heads).bfloat16() for _ in range(LAYERS)]
        self.b = [r(n, g.num_v_heads).bfloat16() for _ in range(LAYERS)]
        # the layers' weights, the same whatever the inputs' seed
        wgen = torch.Generator(device="cpu").manual_seed(1234)

        def w(*shape):
            return (torch.randn(*shape, generator=wgen) * 0.5).to(device)

        self.weight = [w(g.conv_dim, g.conv_kernel_size).bfloat16() for _ in range(LAYERS)]
        self.a_log = [w(g.num_v_heads).float() for _ in range(LAYERS)]
        self.dt_bias = [w(g.num_v_heads).float() for _ in range(LAYERS)]

    def tokens(self, rows, block, t):
        """The inputs of token ``t`` of every block, as a one-token step's."""
        def at(x):
            return x.view(rows, block, *x.shape[1:])[:, t].contiguous()
        return {name: [at(x) for x in getattr(self, name)] for name in ("x", "q", "k", "v", "a", "b")}


def _run(pool, manager, inp, x, q, k, v, a, b):
    outs = []
    for layer in range(LAYERS):
        manager.set_default_layer_idx(layer)
        conv = manager.run_conv(x[layer], conv_layer=pool.block("conv", layer), weight=inp.weight[layer])
        core = manager.run(q[layer], k[layer], v[layer], a[layer], b[layer], state_layer=pool.block("state", layer),
                           a_log=inp.a_log[layer], dt_bias=inp.dt_bias[layer])
        outs.append((conv, core))
    return outs


def _snapshot(pool):
    return {name: pool._blocks[name].clone() for name in ("state", "conv")}


def _restore(pool, snap):
    for name, value in snap.items():
        pool._blocks[name].copy_(value)


def _setup(device, geometry, rows, seed=0):
    pool, manager = _build(device, geometry)
    rids = list(range(rows))
    for rid in rids:
        pool.ingest_request(rid)
    # a first step takes the slots and marks them as holding state; then fill them
    rstep, ctx = _plan(pool, manager, rids, 1, verify=False)
    pool.commit(rstep, ctx)
    gen = torch.Generator(device="cpu").manual_seed(seed + 100)
    for name in ("state", "conv"):
        block = pool._blocks[name]
        block.copy_((torch.randn(block.shape, generator=gen) * 0.3).to(block.dtype).to(device))
    return pool, manager, rids


GEOMETRY_GPU = DeltaNetGeometry(2, 4, 128, 128, 4)
GEOMETRY_CPU = DeltaNetGeometry(2, 4, 16, 16, 4)


@CUDA
@pytest.mark.parametrize("block", [2, 5])
def test_verify_then_settle_is_decode_over_the_accepted_prefix(block):
    device = torch.device("cuda")
    rows = block + 1  # every accepted count, 0 .. block - 1, and one more row
    pool, manager, rids = _setup(device, GEOMETRY_GPU, rows)
    inp = _Inputs(GEOMETRY_GPU, rows, block, device, seed=1)
    start = _snapshot(pool)
    slots = [pool._slots[rid]["main"].index for rid in rids]

    # the decode path, a token at a time, keeping the state after each
    after: list[dict] = []
    decode_out = []
    for t in range(block):
        rstep, ctx = _plan(pool, manager, rids, 1, verify=False)
        tok = inp.tokens(rows, block, t)
        decode_out.append(_run(pool, manager, inp, **tok))
        pool.commit(rstep, ctx)
        after.append(_snapshot(pool))

    _restore(pool, start)
    accepted = torch.tensor([i % block for i in range(rows)], device=device)
    rstep, ctx = _plan(pool, manager, rids, block, verify=True)
    outs = _run(pool, manager, inp, inp.x, inp.q, inp.k, inp.v, inp.a, inp.b)
    manager.settle_verify(accepted, "main")
    pool.commit(rstep, ctx)

    for layer in range(LAYERS):
        conv, core = outs[layer]
        want_conv = torch.stack([decode_out[t][layer][0] for t in range(block)], 1)
        want_core = torch.stack([decode_out[t][layer][1] for t in range(block)], 1)
        torch.testing.assert_close(conv.view(rows, block, -1), want_conv, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            core.view(rows, block, *core.shape[1:]).float(), want_core.float(), atol=3e-2, rtol=3e-2,
        )
    for i, slot in enumerate(slots):
        a = int(accepted[i])
        for name, tol in (("conv", 0.0), ("state", 2e-3)):
            got = pool._blocks[name][:, slot].float()
            want = after[a][name][:, slot].float()
            torch.testing.assert_close(got, want, atol=tol, rtol=tol)


@CUDA
def test_a_captured_verify_step_replays_as_the_eager_one():
    device = torch.device("cuda")
    block, rows, bucket = 4, 3, 4
    pool, manager, rids = _setup(device, GEOMETRY_GPU, rows)
    manager.build_accelerator_graph_buffers([], max_bs=bucket, max_seq_len=bucket * block)
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=bucket, num_tokens=bucket * block))
    padded = rids + [-7]  # a padding row: the sink
    static = _Inputs(GEOMETRY_GPU, bucket, block, device, seed=2)
    accepted = torch.zeros(bucket, dtype=torch.int64, device=device)
    start = _snapshot(pool)

    _plan(pool, manager, rids, block, verify=True, lease=lease, padded=padded)
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _run(pool, manager, static, static.x, static.q, static.k, static.v, static.a, static.b)
        manager.settle_verify(accepted, "main")
    torch.cuda.current_stream().wait_stream(side)
    _restore(pool, start)
    _plan(pool, manager, rids, block, verify=True, lease=lease, padded=padded)
    with torch.cuda.graph(graph):
        outs = _run(pool, manager, static, static.x, static.q, static.k, static.v, static.a, static.b)
        manager.settle_verify(accepted, "main")

    for seed, counts in ((3, [0, 3, 1, 2]), (4, [2, 0, 3, 3])):
        fresh = _Inputs(GEOMETRY_GPU, bucket, block, device, seed=seed)
        _restore(pool, start)
        _plan(pool, manager, rids, block, verify=True)
        eager = _run(pool, manager, fresh, *(
            [t[: rows * block] for t in getattr(fresh, name)] for name in ("x", "q", "k", "v", "a", "b")))
        manager.settle_verify(torch.tensor(counts[:rows], device=device), "main")
        want = _snapshot(pool)

        _restore(pool, start)
        for name in ("x", "q", "k", "v", "a", "b"):
            for dst, src in zip(getattr(static, name), getattr(fresh, name), strict=True):
                dst.copy_(src)
        accepted.copy_(torch.tensor(counts, device=device))
        _plan(pool, manager, rids, block, verify=True, lease=lease, padded=padded)
        graph.replay()
        torch.cuda.synchronize()
        for layer in range(LAYERS):
            for got, ref in zip(outs[layer], eager[layer], strict=True):
                torch.testing.assert_close(got[: rows * block], ref)
        slots = [pool._slots[rid]["main"].index for rid in rids]
        for name in ("state", "conv"):
            torch.testing.assert_close(pool._blocks[name][:, slots], want[name][:, slots])


@pytest.mark.parametrize("block", [2, 4])
def test_closed_gates_leave_the_state_where_the_prefix_left_it(block):
    """CPU, the reference recurrence: settling a block with ``accepted`` drafts is running its
    first ``accepted + 1`` tokens alone."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    rows = block
    pool, manager, rids = _setup(device, GEOMETRY_CPU, rows)
    inp = _Inputs(GEOMETRY_CPU, rows, block, device, seed=5)
    start = _snapshot(pool)
    accepted = torch.arange(rows) % block
    rstep, ctx = _plan(pool, manager, rids, block, verify=True)
    _run(pool, manager, inp, inp.x, inp.q, inp.k, inp.v, inp.a, inp.b)
    manager.settle_verify(accepted, "main")
    slots = torch.tensor([pool._slots[rid]["main"].index for rid in rids], dtype=torch.int32)
    taps = GEOMETRY_CPU.conv_kernel_size - 1
    for i in range(rows):
        keep = int(accepted[i]) + 1
        def view(x, i=i, keep=keep):
            return x.view(rows, block, *x.shape[1:])[i:i + 1, :keep]

        for layer in range(LAYERS):
            state = start["state"][layer].clone()
            _gated_delta_rule_reference(
                view(inp.q[layer]), view(inp.k[layer]), view(inp.v[layer]), view(inp.a[layer]), view(inp.b[layer]),
                state, slots[i:i + 1], inp.a_log[layer], inp.dt_bias[layer], None, True, update=True,
            )
            torch.testing.assert_close(pool._blocks["state"][layer, slots[i]], state[slots[i]], atol=1e-6, rtol=1e-5)
            window = torch.cat([start["conv"][layer, slots[i]], view(inp.x[layer])[0].T], 1)
            torch.testing.assert_close(pool._blocks["conv"][layer, slots[i]], window[:, -taps:])

"""AITER's GDN kernels behind the linear-attention resource, on an AMD GPU.

Everything goes through the resource as the engine drives it (pool plan, then
the backend's plan, then ``GDNManager.run`` per layer) and is checked against
a sequential fp32 torch reference of the gated delta rule:

    S <- exp(g) S;  v' = beta (v - S k);  S <- S + v' k^T;  o = S (scale q)

with q and k L2-normalised, the state K-last ([HV, V, K]) as the pool holds
it, and v-head h reading k-head h // (HV / HK).

The eager chunked prefill, the sequential prefill a captured bucket runs, and
both decode kernels must each match the reference, and so each other. Capture
tests record a bucket against the runner's dummy rows and replay it on layouts
it never saw: fewer rows (padding rows on the sink), slots handed back and
reused, a batch that fills the bucket exactly.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.accelerator_graph_config import PackedAcceleratorGraphConfig
from mstar.engine.resources.base import CGSlotSpec, EngineResourceInfo
from mstar.engine.resources.linear_attn import (
    LinearAttnConfig,
    LinearAttnManager,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
)
from mstar.engine.resources.linear_attn.gdn_kernels_aiter import (
    AiterChunkedPrefill,
    AiterGDNKernels,
    AiterSequentialDecode,
    AiterSequentialPrefill,
    FlyDSLDecode,
)
from mstar.engine.resources.linear_attn.wrappers import GDNMixedWrapper
from mstar.engine.resources.recurrent import (
    DeltaNetGeometry,
    RecurrentStateConfig,
    RecurrentStatePool,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import SINK_SLOT
from mstar.engine.resources.spec import CombinedSplit
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform.is_rocm()),
    reason="AITER's GDN kernels need an AMD GPU",
)

DEV = torch.device("cuda")
POOL, BACKEND = "gdn_state", "linear_attn"
MAX_SLOTS = 12
# K = V = 128: Qwen3.5's head dims, and the HIP hidden-state stage's only shape
GQA = DeltaNetGeometry(num_k_heads=4, num_v_heads=8, head_k_dim=128, head_v_dim=128, conv_kernel_size=4)
MHA = DeltaNetGeometry(num_k_heads=4, num_v_heads=4, head_k_dim=128, head_v_dim=128, conv_kernel_size=4)

# bf16 activations against an fp32 reference
OUT_TOL = dict(atol=2e-2, rtol=2e-2)
STATE_TOL = {torch.float32: dict(atol=1e-2, rtol=1e-2), torch.bfloat16: dict(atol=6e-2, rtol=6e-2)}


# --- the reference -----------------------------------------------------------------


def l2norm(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


@pytest.fixture(autouse=True)
def exact_fp32_matmul():
    """The reference's einsums are fp32 matmuls, which the engine's
    process-wide 'high' precision runs as XF32 on these GPUs (about 1e-2
    relative); keep the reference exact."""
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(precision)


def reference_row(q, k, v, a, b, a_log, dt_bias, h0, scale):
    """One row: q/k [T, HK, K], v [T, HV, V], a/b [T, HV], h0 [HV, V, K].
    Returns (o [T, HV, V], final state [HV, V, K]), fp32."""
    rep = v.shape[1] // q.shape[1]
    g = -torch.exp(a_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias.float())
    beta = torch.sigmoid(b.float())
    qn, kn = l2norm(q).repeat_interleave(rep, 1), l2norm(k).repeat_interleave(rep, 1)
    h = h0.float().clone()
    out = torch.empty(v.shape, dtype=torch.float32, device=v.device)
    for t in range(v.shape[0]):
        h = h * torch.exp(g[t])[:, None, None]
        delta = (v[t].float() - torch.einsum("hvk,hk->hv", h, kn[t])) * beta[t][:, None]
        h = h + delta[:, :, None] * kn[t][:, None, :]
        out[t] = torch.einsum("hvk,hk->hv", h, qn[t] * scale)
    return out, h


# --- the resource, driven as the engine drives it ---------------------------------


def build(geometry=GQA, state_dtype=torch.float32, **config):
    spec = RecurrentStateSpec(
        POOL, {"llm"},
        RecurrentStateConfig(
            num_layers=1, blocks=geometry.to_blocks(state_dtype=state_dtype),
            max_slots=MAX_SLOTS,
        ),
    )
    pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=DEV))
    manager = LinearAttnManager.build(
        LinearAttnSpec(
            BACKEND, {"llm"},
            LinearAttnConfig(recurrent_state=POOL, variant=LinearAttnVariant.GDN, **config),
        ),
        EngineResourceInfo(device=DEV, dependencies={POOL: spec}),
    )
    assert isinstance(manager._kernels, AiterGDNKernels)
    return pool, manager


def plan_step(pool, manager, rows, *, walk, lease=None, capture=False, padding=(),
              padding_span=0):
    """``rows`` are (rid, span) for real requests; ``padding`` the rids a
    replay pads with, at the span `pad_inputs` gives them: 0 in a packed
    prefill bucket, the template's 1 in a decode bucket."""
    real = [rid for rid, _ in rows]
    ctx = StepContext(
        request_ids=real, graph_walk=walk, slot=0, capture=capture, slot_lease=lease,
    )
    if padding:
        ctx.set_padded_rids([*real, *padding])
    segments = (
        *(Segment(rid, "main", span) for rid, span in rows),
        *(Segment(rid, "main", padding_span) for rid in padding),
    )
    rstep = RecurrentStep(segments=segments)
    if not capture:
        for rid in real:
            pool.ingest_request(rid)
    assert pool.admit(rstep, ctx).ok
    ctx.plan_results[POOL] = pool.plan(rstep, ctx)
    manager.plan(LinearAttnStep(segments=segments), ctx)
    return rstep, ctx


def inputs(tokens: int, geometry=GQA, seed: int = 0):
    gen = torch.Generator(device=DEV).manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=DEV) * scale).to(torch.bfloat16)

    hk, hv, kd, vd = (
        geometry.num_k_heads, geometry.num_v_heads, geometry.head_k_dim, geometry.head_v_dim,
    )
    return dict(
        q=randn(tokens, hk, kd), k=randn(tokens, hk, kd), v=randn(tokens, hv, vd),
        a=randn(tokens, hv), b=randn(tokens, hv),
    )


def weights(geometry=GQA, seed: int = 1):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    hv = geometry.num_v_heads
    # fp32, as `GatedDeltaNet` keeps them; dt_bias holds bf16 values, as
    # Qwen3.5's checkpoint does
    a_log = torch.randn(hv, generator=gen, device=DEV) * 0.5
    dt_bias = torch.randn(hv, generator=gen, device=DEV).to(torch.bfloat16).float()
    return a_log, dt_bias


def run(manager, pool, x, w):
    a_log, dt_bias = w
    return manager.run(
        x["q"], x["k"], x["v"], x["a"], x["b"],
        state_layer=pool.block("state", 0), a_log=a_log, dt_bias=dt_bias,
    )


def check_rows(out, state, before, rows, slots, carried, x, w, state_dtype, scale):
    """Each real row's output and final state against the reference, from the
    state its slot held before the step (or zeros where it starts fresh)."""
    start = 0
    for (_, span), slot, has in zip(rows, slots, carried, strict=True):
        if span == 0:
            continue
        sl = slice(start, start + span)
        h0 = before[slot] if has else torch.zeros_like(before[slot])
        ref_out, ref_state = reference_row(
            x["q"][sl], x["k"][sl], x["v"][sl], x["a"][sl], x["b"][sl], *w, h0, scale,
        )
        torch.testing.assert_close(out[sl].float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(state[slot].float(), ref_state, **STATE_TOL[state_dtype])
        start += span


def slots_of(pool, rids):
    return [pool._slots[rid]["main"].index for rid in rids]


# --- FlyDSL's decode and a -1 slot ------------------------------------------------


def test_flydsl_decode_zeroes_a_negative_slots_output_and_leaves_the_pool_alone():
    """Settles what the kernel does with -1: AITER v0.1.25's source says it
    zeroes the row's output and skips the state; an older comment said it
    skips no slot. The resource keeps a sink either way."""
    from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode

    x = inputs(4)
    a_log, dt_bias = weights()
    state = torch.randn(6, 8, 128, 128, device=DEV)
    before = state.clone()
    out = torch.full((4, 1, 8, 128), float("nan"), dtype=torch.bfloat16, device=DEV)
    slots = torch.tensor([3, -1, 5, -1], dtype=torch.int32, device=DEV)
    flydsl_gdr_decode(
        x["q"].unsqueeze(1), x["k"].unsqueeze(1), x["v"].unsqueeze(1),
        x["a"].unsqueeze(1), x["b"].unsqueeze(1), dt_bias.to(torch.bfloat16), a_log,
        slots, state, out, use_qk_l2norm=True, need_shuffle_state=False,
    )
    torch.cuda.synchronize()
    assert torch.equal(out[[1, 3]], torch.zeros_like(out[[1, 3]]))
    untouched = [0, 1, 2, 4]
    assert torch.equal(state[untouched], before[untouched])
    for row, slot in ((0, 3), (2, 5)):
        ref_out, ref_state = reference_row(
            *(x[n][row:row + 1] for n in ("q", "k", "v", "a", "b")), a_log, dt_bias,
            before[slot], 128 ** -0.5,
        )
        torch.testing.assert_close(out[row, 0].float(), ref_out[0], **OUT_TOL)
        torch.testing.assert_close(state[slot], ref_state, **STATE_TOL[torch.float32])


# --- eager prefill ----------------------------------------------------------------


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("geometry", [GQA, MHA], ids=["gqa", "mha"])
def test_eager_prefill_matches_the_reference(geometry, state_dtype):
    """Two prefills of the same requests: the first starts from zeros, the
    second carries the state the first left (chunked prefill's shape)."""
    pool, manager = build(geometry, state_dtype)
    w = weights(geometry)
    rows = [("r0", 37), ("r1", 1), ("r2", 130), ("r3", 64)]
    for step in range(2):
        rstep, ctx = plan_step(pool, manager, rows, walk="prefill")
        assert isinstance(manager.current_plan()._kernel, AiterChunkedPrefill)
        x = inputs(sum(s for _, s in rows), geometry, seed=step)
        state = pool.block("state", 0)
        before = state.clone()
        out = run(manager, pool, x, w)
        torch.cuda.synchronize()
        slots = slots_of(pool, [r for r, _ in rows])
        check_rows(out, state, before, rows, slots, [step > 0] * len(rows), x, w,
                   state_dtype, geometry.head_k_dim ** -0.5)
        pool.commit(rstep, ctx)


def test_eager_prefill_reads_zeros_for_a_fresh_row_whatever_its_slot_holds():
    """The in-place kernels read every row's slot; a fresh row must not."""
    pool, manager = build()
    w = weights()
    rows = [("r0", 20), ("r1", 9)]
    plan_step(pool, manager, rows, walk="prefill")
    state = pool.block("state", 0)
    for slot in slots_of(pool, ["r0", "r1"]):
        state[slot].normal_()
    before = state.clone()
    x = inputs(29)
    out = run(manager, pool, x, w)
    torch.cuda.synchronize()
    check_rows(out, state, before, rows, slots_of(pool, ["r0", "r1"]), [False, False],
               x, w, torch.float32, 128 ** -0.5)


def test_eager_prefill_with_a_zero_span_row():
    """A row with nothing to extend gets one borrowed token on the sink; the
    other rows and every other slot are unaffected. Such rows trail the rest,
    as padding rows do: the borrowed tokens sit past the real ones."""
    pool, manager = build()
    w = weights()
    rows = [("r0", 12), ("r2", 7), ("r1", 0)]
    plan_step(pool, manager, rows, walk="prefill")
    state = pool.block("state", 0)
    before = state.clone()
    x = inputs(19)
    out = run(manager, pool, x, w)
    torch.cuda.synchronize()
    assert out.shape[0] == 19
    live = slots_of(pool, ["r0", "r2"])
    check_rows(out, state, before, [("r0", 12), ("r2", 7)], live, [False, False],
               {n: t for n, t in x.items()}, w, torch.float32, 128 ** -0.5)
    others = [s for s in range(MAX_SLOTS) if s not in live and s != SINK_SLOT]
    assert torch.equal(state[others], before[others])


# --- the sequential prefill a captured bucket runs -------------------------------


def test_sequential_prefill_agrees_with_chunked_and_the_reference():
    rows = [("r0", 100), ("r1", 3), ("r2", 120), ("r3", 13)]
    tokens = sum(s for _, s in rows)  # 236 of the bucket's 256
    w = weights()
    x = inputs(tokens)
    outs, states = [], []
    for captured in (False, True):
        pool, manager = build()
        lease = (
            SlotLease(slot=0, bucket=BucketKey("prefill", bs=4, num_tokens=256))
            if captured else None
        )
        plan_step(pool, manager, rows, walk="prefill", lease=lease)
        kernel = manager.current_plan()._kernel
        assert isinstance(kernel, AiterSequentialPrefill if captured else AiterChunkedPrefill)
        state = pool.block("state", 0)
        before = state.clone()
        # a captured bucket's inputs are its whole token width; the tail is
        # stale, and must not reach the output
        xin = dict(x)
        if captured:
            xin = {n: torch.cat((t, torch.full_like(t[:256 - tokens], 1e4))) for n, t in x.items()}
        out = run(manager, pool, xin, w)
        torch.cuda.synchronize()
        slots = slots_of(pool, [r for r, _ in rows])
        check_rows(out, state, before, rows, slots, [False] * 4, x, w, torch.float32, 128 ** -0.5)
        if captured:
            assert torch.equal(out[tokens:], torch.zeros_like(out[tokens:]))
        outs.append(out[:tokens].float())
        states.append(torch.stack([state[s] for s in slots]))
    torch.testing.assert_close(outs[0], outs[1], **OUT_TOL)
    torch.testing.assert_close(states[0], states[1], **STATE_TOL[torch.float32])


def test_a_prefill_bucket_past_the_bound_is_refused():
    pool, manager = build(max_captured_prefill_tokens=64)
    config = PackedAcceleratorGraphConfig(
        capture_graph_walk="prefill", capture_token_lengths=[64, 128], make_node_input=None,
    )
    for tokens, refused in ((64, False), (128, True)):
        spec = CGSlotSpec(BucketKey("prefill", bs=1, num_tokens=tokens), 0, config)
        assert (manager.capture_refusal(spec) is not None) is refused


# --- capture and replay -------------------------------------------------------------


def capture(fn):
    """Warm up on a side stream, then record, as the runner does."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def test_a_captured_prefill_bucket_replays_layouts_it_never_saw():
    bs, tokens = 4, 64
    pool, manager = build()
    w = weights()
    lease = SlotLease(slot=0, bucket=BucketKey("prefill", bs=bs, num_tokens=tokens))
    static = inputs(tokens, seed=99)
    dummies = [f"__cg_{i}" for i in range(bs)]
    for rid in dummies:
        pool.ingest_request(rid)
    plan_step(pool, manager, [(rid, tokens // bs) for rid in dummies],
              walk="prefill", lease=lease, capture=True)
    graph, out = capture(lambda: run(manager, pool, static, w))
    torch.cuda.synchronize()

    layouts = [
        [("a", 20), ("b", 9)],                      # padding rows on the sink
        [("c", 16), ("d", 16), ("e", 16), ("f", 16)],  # fills the bucket exactly
        [("g", 64)],                                 # one row takes it all
        [("h", 5), ("i", 1), ("j", 30)],
    ]
    for step, rows in enumerate(layouts):
        padding = dummies[len(rows):]
        rstep, ctx = plan_step(pool, manager, rows, walk="prefill", lease=lease, padding=padding)
        x = inputs(tokens, seed=step)
        for name, t in static.items():
            t.copy_(x[name])
        state = pool.block("state", 0)
        before = state.clone()
        graph.replay()
        torch.cuda.synchronize()
        real = sum(s for _, s in rows)
        live = slots_of(pool, [r for r, _ in rows])
        check_rows(out, state, before, rows, live, [False] * len(rows), x, w,
                   torch.float32, 128 ** -0.5)
        assert torch.equal(out[real:], torch.zeros_like(out[real:])), "tail must be zero"
        others = [s for s in range(MAX_SLOTS) if s not in live and s != SINK_SLOT]
        assert torch.equal(state[others], before[others])
        pool.commit(rstep, ctx)
        for rid, _ in rows:
            pool.remove_request(rid)


@pytest.mark.parametrize("flydsl", [True, False], ids=["flydsl", "sequential"])
def test_captured_decode_replays_with_slot_reuse(flydsl):
    bs = 4
    pool, manager = build()
    manager._kernels.flydsl_decode = flydsl
    w = weights()
    lease = SlotLease(slot=0, bucket=BucketKey("decode", bs=bs, num_tokens=bs))
    static = inputs(bs, seed=99)
    dummies = [f"__cg_{i}" for i in range(bs)]
    for rid in dummies:
        pool.ingest_request(rid)
    plan_step(pool, manager, [(rid, 1) for rid in dummies], walk="decode", lease=lease, capture=True)
    kernel = manager.current_plan()._kernel
    assert isinstance(kernel, FlyDSLDecode if flydsl else AiterSequentialDecode)
    graph, out = capture(lambda: run(manager, pool, static, w))
    torch.cuda.synchronize()

    state = pool.block("state", 0)
    # each request's prefill leaves a state; decode carries it
    live: list[str] = []
    for step, rows in enumerate([["a", "b"], ["a", "b", "c", "d"], ["d", "e"], ["e", "f", "a"]]):
        for rid in rows:
            if rid not in live:
                pool.ingest_request(rid)
                live.append(rid)
        for rid in list(live):
            if rid not in rows:
                pool.remove_request(rid)  # slot handed back, zeroed, reused
                live.remove(rid)
        rstep, ctx = plan_step(
            pool, manager, [(rid, 1) for rid in rows], walk="decode", lease=lease,
            padding=dummies[len(rows):], padding_span=1,
        )
        slots = slots_of(pool, rows)
        for rid, slot in zip(rows, slots, strict=True):
            if not pool._slots[rid]["main"].has_state:
                # what a prefill would have left there
                state[slot].normal_(0, 0.1)
        x = inputs(bs, seed=step)
        for name, t in static.items():
            t.copy_(x[name])
        before = state.clone()
        graph.replay()
        torch.cuda.synchronize()
        check_rows(out, state, before, [(r, 1) for r in rows], slots, [True] * len(rows),
                   x, w, torch.float32, 128 ** -0.5)
        others = [s for s in range(MAX_SLOTS) if s not in slots and s != SINK_SLOT]
        assert torch.equal(state[others], before[others])
        pool.commit(rstep, ctx)


# --- a combined walk's step: decode rows beside prompts ---------------------------


@pytest.mark.parametrize("bucket", [False, True], ids=["eager", "bucket"])
def test_a_split_walks_mixed_step_matches_the_reference(bucket):
    """Decode rows that carry a state beside fresh prompts, in one step of a
    split walk: the decode kernel takes the one-token rows, the prefill kernel
    the prompts. Each row matches the reference and no other slot moves. The
    bucket variant pads rows and tokens as a replay would."""
    split = CombinedSplit("mixed", prefill_walks=("prefill",))
    pool, manager = build(splits=(split,))
    w = weights()
    # a prefill leaves each decode row a state to carry
    first = [("d0", 5), ("d1", 7), ("d2", 3)]
    rstep, ctx = plan_step(pool, manager, first, walk="prefill")
    run(manager, pool, inputs(15, seed=7), w)
    pool.commit(rstep, ctx)

    rows = [("d0", 1), ("p0", 23), ("d1", 1), ("p1", 9), ("d2", 1)]
    real = sum(s for _, s in rows)
    lease, padding, width = None, (), real
    if bucket:
        lease = SlotLease(slot=0, bucket=BucketKey("mixed", bs=8, num_tokens=64))
        padding, width = tuple(f"__cg_{i}" for i in range(8 - len(rows))), 64
        for rid in padding:
            pool.ingest_request(rid)
    plan_step(pool, manager, rows, walk="mixed", lease=lease, padding=padding)
    assert isinstance(manager.current_plan(), GDNMixedWrapper)
    x = inputs(real, seed=8)
    xin = {n: torch.cat((t, torch.full_like(t[:width - real], 1e4))) for n, t in x.items()}
    state = pool.block("state", 0)
    before = state.clone()
    out = run(manager, pool, xin, w)
    torch.cuda.synchronize()
    slots = slots_of(pool, [r for r, _ in rows])
    carried = [r.startswith("d") for r, _ in rows]
    check_rows(out[:real], state, before, rows, slots, carried, x, w, torch.float32, 128 ** -0.5)
    others = [s for s in range(MAX_SLOTS) if s not in slots and s != SINK_SLOT]
    assert torch.equal(state[others], before[others])

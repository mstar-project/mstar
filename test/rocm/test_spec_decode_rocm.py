"""A speculative verify step's device side on an AMD GPU.

* GDN through the pool and the manager as the engine drives them: a verify row
  of k + 1 tokens starts from the slot its last step accepted up to and leaves
  the state after every token in its own slot, against the fp32 reference run
  token by token, including a step that starts after a rejected tail.
* The step's device work (the speculative resource's in-graph operations, the
  GDN verify kernels and a whole draft pass's attention) issues no host sync:
  run under ``torch.cuda.set_sync_debug_mode("error")``.
* A captured step follows the device registers: replayed after the registers
  move, its positions, KV slots and lengths move with them, with no re-plan.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.attn.aiter import AiterPagedAttentionManager, _Staged
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.manager import KVPlanState
from mstar.engine.resources.linear_attn import (
    LinearAttnConfig,
    LinearAttnManager,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
)
from mstar.engine.resources.recurrent import (
    DeltaNetGeometry,
    RecurrentStateConfig,
    RecurrentStatePool,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.spec_decode.config import SpecPhase
from mstar.engine.resources.spec_decode.resource import SpecDecodePlan, SpecDecodeResource
from mstar.engine.resources.step import Segment, StepContext
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform.is_rocm()),
    reason="AITER's GDN kernels need an AMD GPU",
)

DEV = torch.device("cuda")
POOL, BACKEND, SPEC = "gdn_state", "linear_attn", "spec"
GEO = DeltaNetGeometry(num_k_heads=4, num_v_heads=8, head_k_dim=128, head_v_dim=128, conv_kernel_size=4)
OUT_TOL = dict(atol=2e-2, rtol=2e-2)
STATE_TOL = dict(atol=1e-2, rtol=1e-2)
PAGE = 16


@pytest.fixture(autouse=True)
def exact_fp32_matmul():
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(precision)


def l2norm(x):
    x = x.float()
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def reference(q, k, v, a, b, a_log, dt_bias, h0, scale):
    """Token by token from ``h0``; the output and the state after each token."""
    rep = v.shape[1] // q.shape[1]
    g = -torch.exp(a_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias.float())
    beta = torch.sigmoid(b.float())
    qn, kn = l2norm(q).repeat_interleave(rep, 1), l2norm(k).repeat_interleave(rep, 1)
    h = h0.float().clone()
    outs, states = [], []
    for t in range(v.shape[0]):
        h = h * torch.exp(g[t])[:, None, None]
        delta = (v[t].float() - torch.einsum("hvk,hk->hv", h, kn[t])) * beta[t][:, None]
        h = h + delta[:, :, None] * kn[t][:, None, :]
        outs.append(torch.einsum("hvk,hk->hv", h, qn[t] * scale))
        states.append(h.clone())
    return torch.stack(outs), torch.stack(states)


def build(k: int, rows: int):
    spec = RecurrentStateSpec(POOL, {"llm"}, RecurrentStateConfig(
        num_layers=1, blocks=GEO.to_blocks(), max_slots=1 + rows * (k + 1), spec_tokens=k,
    ))
    pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=DEV))
    manager = LinearAttnManager.build(
        LinearAttnSpec(BACKEND, {"llm"}, LinearAttnConfig(
            recurrent_state=POOL, variant=LinearAttnVariant.GDN, spec_decode=SPEC,
        )),
        EngineResourceInfo(device=DEV, dependencies={POOL: spec}),
    )
    return pool, manager


def inputs(tokens: int, seed: int):
    gen = torch.Generator(device=DEV).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=DEV).to(torch.bfloat16)

    return dict(q=randn(tokens, GEO.num_k_heads, GEO.head_k_dim), k=randn(tokens, GEO.num_k_heads, GEO.head_k_dim),
                v=randn(tokens, GEO.num_v_heads, GEO.head_v_dim), a=randn(tokens, GEO.num_v_heads),
                b=randn(tokens, GEO.num_v_heads))


def step(pool, manager, rids, span, num_accepted=None):
    segs = tuple(Segment(r, "main", span) for r in rids)
    ctx = StepContext(request_ids=rids, graph_walk="decode" if num_accepted is not None else "prefill",
                      slot=0, capture=False)
    for r in rids:
        pool.ingest_request(r)
    rstep = RecurrentStep(segments=segs)
    assert pool.admit(rstep, ctx).ok
    ctx.plan_results[POOL] = pool.plan(rstep, ctx)
    if num_accepted is not None:
        ctx.plan_results[SPEC] = SimpleNamespace(phase=SpecPhase.VERIFY, num_accepted=num_accepted)
    manager.plan(LinearAttnStep(segments=segs), ctx)
    pool.commit(rstep, ctx)


def run(manager, pool, x, w):
    return manager.run(x["q"], x["k"], x["v"], x["a"], x["b"], state_layer=pool.block("state", 0),
                       a_log=w[0], dt_bias=w[1])


@pytest.mark.parametrize("k", [1, 3, 4])
def test_verify_rows_resume_from_the_accepted_slot(k):
    rows, t, prompt = 3, k + 1, 9
    pool, manager = build(k, rows)
    rids = list(range(rows))
    gen = torch.Generator(device=DEV).manual_seed(5)
    w = (torch.randn(GEO.num_v_heads, generator=gen, device=DEV) * 0.5,
         torch.randn(GEO.num_v_heads, generator=gen, device=DEV).to(torch.bfloat16).float())
    scale = GEO.head_k_dim ** -0.5

    # the prompts land in each request's first slot, as a prefill leaves them
    xp = inputs(rows * prompt, seed=1)
    step(pool, manager, rids, prompt)
    run(manager, pool, xp, w)
    state = pool.block("state", 0)
    groups = [pool._slots[r]["main"].group for r in rids]
    starts = [state[g[0]].clone() for g in groups]

    accepted = torch.ones(rows, dtype=torch.int32, device=DEV)
    for round_ in range(2):
        x = inputs(rows * t, seed=10 + round_)
        step(pool, manager, rids, t, num_accepted=accepted)
        out = run(manager, pool, x, w)
        torch.cuda.synchronize()
        next_starts = []
        for r, g in enumerate(groups):
            sl = slice(r * t, (r + 1) * t)
            ref_out, ref_states = reference(x["q"][sl], x["k"][sl], x["v"][sl], x["a"][sl], x["b"][sl], *w,
                                            starts[r], scale)
            torch.testing.assert_close(out[sl].float(), ref_out, **OUT_TOL)
            for i in range(t):
                torch.testing.assert_close(state[g[i]].float(), ref_states[i], **STATE_TOL)
            # the next step accepted a row-dependent prefix of this one
            n = (r + round_) % t
            next_starts.append(ref_states[n])
        accepted = torch.tensor([(r + round_) % t + 1 for r in range(rows)], dtype=torch.int32, device=DEV)
        starts = next_starts


def test_the_verify_kernel_replays_in_a_graph():
    """FlyDSL's chain kernel from a captured graph: a real row as in eager, a
    padding row (slot -1) skipped, its pool slots and output left alone."""
    k, rows, t = 2, 2, 3
    pool, manager = build(k, rows)
    rids = list(range(rows))
    w = (torch.zeros(GEO.num_v_heads, device=DEV), torch.zeros(GEO.num_v_heads, device=DEV))
    step(pool, manager, rids, 5)
    run(manager, pool, inputs(rows * 5, seed=1), w)
    accepted = torch.ones(rows, dtype=torch.int32, device=DEV)
    step(pool, manager, rids, t, num_accepted=accepted)
    state = pool.block("state", 0)
    x = inputs(rows * t, seed=2)
    # the second row is a padding row from here on
    wrapper = manager._current["main"]
    wrapper._plan_state.slot_map[1] = -1
    groups = [pool._slots[r]["main"].group for r in rids]
    before = state.clone()
    eager = run(manager, pool, x, w).clone()
    after = state[groups[0]].clone()
    torch.cuda.synchronize()

    state.copy_(before)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run(manager, pool, x, w)
    state.copy_(before)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, rtol=0, atol=0)
    torch.testing.assert_close(state[groups[0]], after, rtol=0, atol=0)
    assert torch.equal(state[groups[1]], before[groups[1]])
    assert torch.isfinite(out[t:].float()).all()


def _attention(rows: int, width: int, table: torch.Tensor) -> AiterPagedAttentionManager:
    """The real manager's kernel call over a plan with this block table."""
    manager = AiterPagedAttentionManager.__new__(AiterPagedAttentionManager)
    manager._device, manager._dtype, manager._scale = DEV, torch.bfloat16, 256 ** -0.5
    manager._row_starts = None
    manager._reserve_row_starts(rows)
    manager._current_plans = {"main": _Staged(
        manager._row_starts[: rows + 1], torch.ones(rows, dtype=torch.int32, device=DEV), table, 1, 32768, rows,
    )}
    return manager


def test_one_query_a_row_is_its_slot_of_a_verify_row():
    """A draft pass attends one query a row off the verify's block table: what
    the slot would read in a row of k + 1 queries, up to the kernel's rounding."""
    t, rows, heads, kv_heads, dim, page = 3, 2, 16, 4, 256, 128
    gen = torch.Generator(device=DEV).manual_seed(3)
    base = torch.tensor([100, 201], dtype=torch.int32, device=DEV)
    pages = 8
    layer = torch.randn(pages * rows, 2, page, kv_heads, dim, device=DEV, generator=gen).to(torch.bfloat16)
    table = torch.arange(pages * rows, dtype=torch.int32, device=DEV).view(rows, pages)
    manager = _attention(rows, pages, table)
    multi = _Staged(
        torch.arange(rows + 1, dtype=torch.int32, device=DEV) * t, base + t, table, t, 1024, rows * t,
    )
    q = torch.randn(rows * t, heads, dim, device=DEV, generator=gen).to(torch.bfloat16)
    full = manager._unified(q, multi, layer)
    for n in range(t):
        keep = torch.arange(rows, device=DEV) * t + n
        one = manager.attend_rows(q[keep], layer, base + n + 1)
        torch.testing.assert_close(one.float(), full[keep].float(), rtol=2e-2, atol=2e-2)


def _spec(k: int, rows: int, width: int):
    """The speculative resource with a verify plan over fake attention and KV
    buffers: a block table of distinct pages per row."""
    spec = SpecDecodeResource(device=DEV, num_draft_tokens=k, max_requests=rows, page_size=PAGE,
                              kv_key="kv", attn_key="attn", stop_token_ids=(7,), sampling=True)
    for r in range(rows):
        spec.ingest_request(r)
    t = k + 1
    image, _ = spec._image_for(StepContext(request_ids=list(range(rows)), graph_walk="decode", slot=0,
                                           capture=False), rows)
    image.rows[:rows].copy_(torch.tensor([spec._row_of[r] for r in range(rows)], device=DEV))
    image.real[:rows].fill_(1)
    table = (1 + torch.arange(rows * width, device=DEV, dtype=torch.int32)).view(rows, width)
    kv_lens = torch.zeros(rows, dtype=torch.int32, device=DEV)
    state = KVPlanState(token_to_page=torch.zeros(rows * t, dtype=torch.long, device=DEV),
                        token_to_cache=torch.zeros(rows * t, dtype=torch.long, device=DEV))
    spec._current = SpecDecodePlan(
        phase=SpecPhase.VERIFY, num_rows=rows, image=image, block_table=table, kv_lens=kv_lens,
        write_pages=state.token_to_page, write_offsets=state.token_to_cache,
    )
    regs = slice(1, rows + 1)
    spec._seq_len[regs] = torch.tensor([30 + 17 * r for r in range(rows)], dtype=torch.int32, device=DEV)
    spec._tokens[:, regs] = torch.arange(rows * t, device=DEV).view(t, rows) % 5
    spec._num_acc[regs] = 1
    spec._budget[regs] = 1 << 20
    spec._stop_on[regs] = 1
    return spec


def _verify_step(spec, logits_rows, vocab, k, sample=False):
    """Every device operation of a verify step and its draft passes, as the
    forward issues them, with stand-ins for the layers."""
    ids, positions = spec.verify_inputs()
    # a target that predicts each token's successor
    logits = logits_rows.scatter(1, ((ids + 1) % vocab)[:, None], 1.0)
    out = spec.accept(logits, sample=sample)
    drafted = out.target.reshape(-1)[spec.draft_slots()]
    rows = drafted.shape[0]
    for j in range(k):
        spec.draft_address(j)
        drafted = spec.record_draft(j, logits_rows[:rows].scatter(1, ((drafted + 1) % vocab)[:, None], 1.0))
    return out


@pytest.mark.parametrize("k", [1, 3])
def test_a_verify_step_issues_no_host_sync(k):
    rows, vocab = 4, 64
    pool, manager = build(k, rows)
    spec = _spec(k, rows, width=8)
    rids = list(range(rows))
    t = k + 1
    step(pool, manager, rids, 5)
    xp = inputs(rows * 5, seed=2)
    w = (torch.zeros(GEO.num_v_heads, device=DEV), torch.zeros(GEO.num_v_heads, device=DEV))
    run(manager, pool, xp, w)
    step(pool, manager, rids, t, num_accepted=spec._current.num_accepted)
    noise = torch.zeros(rows * t, vocab, device=DEV)
    x = inputs(rows * t, seed=3)
    conv_x = torch.randn(rows * t, GEO.conv_dim, device=DEV).to(torch.bfloat16)
    conv_w = torch.randn(GEO.conv_dim, GEO.conv_kernel_size, device=DEV).to(torch.bfloat16)
    head_dim, heads, kv_heads = 256, 16, 4
    layer = torch.zeros(64, 2, PAGE, kv_heads, head_dim, dtype=torch.bfloat16, device=DEV)
    spec._kv = SimpleNamespace(layer_view=lambda idx: layer)
    spec._attn = _attention(rows, width=8, table=spec._current.block_table)
    packed = (rows * t, (2 * heads + 2 * kv_heads) * head_dim)
    qkv = [torch.randn(*packed, device=DEV).to(torch.bfloat16),
           torch.randn(rows, packed[1], device=DEV).to(torch.bfloat16)]
    weight = torch.zeros(head_dim, dtype=torch.bfloat16, device=DEV)
    freqs = torch.arange(1024, device=DEV).float()[:, None] * torch.linspace(0, 1, 32, device=DEV)[None, :]
    cos, sin = freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)
    layout = (heads, kv_heads, head_dim, 1e-6, True, 0)

    def step_with_attention():
        out = _verify_step(spec, noise, vocab, k)
        for j in range(k):
            spec.attend_draft(j, qkv[min(j, 1)], weight, weight, cos, sin, *layout)
        return out

    # compile the kernels before the tripwire: a first launch syncs
    step_with_attention()
    torch.cuda.synchronize()
    spec._seq_len[1:rows + 1] = torch.tensor([30 + 17 * r for r in range(rows)], dtype=torch.int32, device=DEV)
    spec._num_acc[1:rows + 1] = 1
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step_with_attention()
        manager.run_conv(conv_x, conv_layer=pool.block("conv", 0), weight=conv_w)
        run(manager, pool, x, w)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def test_a_captured_step_follows_the_registers():
    k, rows, vocab, width = 3, 2, 64, 24
    spec = _spec(k, rows, width=width)
    t = k + 1
    noise = torch.zeros(rows * t, vocab, device=DEV)
    plan = spec._current
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _verify_step(spec, noise, vocab, k)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _verify_step(spec, noise, vocab, k)
    for base in ([100, 7], [129, 263]):
        spec._seq_len[1:rows + 1] = torch.tensor(base, dtype=torch.int32, device=DEV)
        spec._num_acc[1:rows + 1] = 1
        spec._tokens[:, 1:rows + 1] = torch.tensor([[5, 6, 7, 8], [1, 2, 9, 9]], device=DEV).T
        graph.replay()
        torch.cuda.synchronize()
        # The toy target predicts each token's successor: row 0's drafts are
        # all right (the stream grows by 4) but it emits through the stop id 7
        # only; row 1's second draft is wrong.
        assert out.accepted.tolist() == [4, 2]
        assert out.tokens.tolist() == [[6, 7, -1, -1], [2, 3, -1, -1]]
        assert out.done.tolist() == [True, False]
        assert spec._seq_len[1:rows + 1].tolist() == [base[0] + 4, base[1] + 2]
        # draft pass j addressed slot n at P + n + j through the table
        for j in range(k):
            where = spec.draft_address(j)
            for r in range(rows):
                n = int(out.accepted[r]) - 1
                pos = base[r] + n + j
                assert int(where.kv_lens[r]) == pos + 1
                if j:
                    assert int(where.positions[r]) == pos
                    assert int(where.slots[r]) == int(plan.block_table[r, pos // PAGE]) * PAGE + pos % PAGE


def test_a_captured_sampled_step_draws_fresh_tokens_each_replay():
    """The sampled verification (FlashInfer's top-k/top-p sampler with device
    seeds and offsets) captures, draws anew on every replay, and keeps greedy
    rows on the argmax."""
    k, rows, vocab = 2, 2, 64
    spec = _spec(k, rows, width=24)
    t = k + 1
    regs = slice(1, rows + 1)
    # row 0 samples flat logits; row 1 is greedy
    spec._temperature[regs] = torch.tensor([1.0, 0.0], device=DEV)
    # top-k and top-p cut by FlashInfer's sampler inside the graph
    spec._top_p[regs], spec._top_k[regs] = 0.9, 40
    spec._seed[regs] = torch.tensor([11, 12], device=DEV)
    flat = torch.zeros(rows * t, vocab, device=DEV)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _verify_step(spec, flat, vocab, k, sample=True)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _verify_step(spec, flat, vocab, k, sample=True)
    drawn = set()
    for _ in range(20):
        spec._seq_len[regs] = 50
        spec._num_acc[regs] = 1
        spec._tokens[:, regs] = torch.tensor([[5, 6, 7], [1, 2, 3]], device=DEV).T
        graph.replay()
        torch.cuda.synchronize()
        drawn.add(int(out.target[0, 0]))
        # the greedy row's target is each token's successor, so it accepts all three
        assert out.target[1].tolist() == [2, 3, 4] and int(out.accepted[1]) == 3
    assert len(drawn) > 5, drawn

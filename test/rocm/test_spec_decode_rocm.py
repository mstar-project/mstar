"""A speculative verify step's device side on an AMD GPU.

* GDN through the pool and the manager as the engine drives them: a verify row
  of k + 1 tokens starts from the slot its last step accepted up to and leaves
  the state after every token in its own slot, against the fp32 reference run
  token by token, including a step that starts after a rejected tail.
* The step's device work (the speculative resource's in-graph operations and
  the GDN verify kernels) issues no host sync: run under
  ``torch.cuda.set_sync_debug_mode("error")``.
* A captured step follows the device registers: replayed after the registers
  move, its positions, KV slots and lengths move with them, with no re-plan.
* The sampled leg: FlashInfer's filter and accept chain against their torch
  references, exact distributions at a small vocabulary, and a captured sampled
  step that draws afresh on each replay and equals the eager step bit for bit
  at the same registers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

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
from mstar.engine.resources.spec_decode import sampled
from mstar.engine.resources.spec_decode.config import SpecPhase
from mstar.engine.resources.spec_decode.resource import RNG_START, SpecDecodePlan, SpecDecodeResource
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
VOCAB = sampled.CANDIDATES


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
    spec._tokens[regs] = torch.arange(rows * t, device=DEV).view(rows, t) % 5
    spec._num_acc[regs] = 1
    spec._budget[regs] = 1 << 20
    spec._stop_on[regs] = 1
    spec._rng_offset[regs] = RNG_START
    return spec


def _verify_step(spec, logits_rows, vocab, k, sample=False):
    """Every device operation of a verify step and its draft passes, as the
    forward issues them, with stand-ins for the layers."""
    ids, positions = spec.verify_inputs()
    # a target that predicts each token's successor
    logits = logits_rows.scatter(1, ((ids + 1) % vocab)[:, None], 1.0)
    out = spec.accept(logits, sample=sample)
    drafted = out.target.reshape(-1)
    for j in range(k):
        spec.begin_draft_pass(j)
        successor = ((drafted + 1) % vocab)[spec.draft_slots()]
        if sample:
            drafted_logits = torch.zeros(successor.shape[0], vocab, device=DEV, dtype=logits_rows.dtype)
            successor = spec.draw_draft(j, drafted_logits.scatter(1, successor[:, None], 1.0))
        drafted = spec.record_draft(j, successor)
    return out


@pytest.mark.parametrize(("k", "sample"), [(1, False), (3, False), (2, True)])
def test_a_verify_step_issues_no_host_sync(k, sample):
    rows, vocab = 4, VOCAB
    pool, manager = build(k, rows)
    spec = _spec(k, rows, width=8)
    spec._temperature[1:rows + 1] = 0.7
    spec._top_p[1:rows + 1], spec._top_k[1:rows + 1] = 0.9, 40
    spec._draft_probs[1:rows + 1] = 1.0 / vocab
    spec._draft_ids[1:rows + 1] = torch.arange(vocab, device=DEV)
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
    # autotune and JIT happen here, as in the eager forward the engine runs before a capture
    if sample:
        one, ones = torch.ones(rows * t, device=DEV), torch.ones(rows * t, device=DEV, dtype=torch.long)
        sampled.draw(sampled.filtered_probs(noise, one * 0.7, torch.full_like(ones, 40).int(), one * 0.9), ones, ones)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _verify_step(spec, noise, vocab, k, sample=sample)
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
        spec._tokens[1:rows + 1] = torch.tensor([[5, 6, 7, 8], [1, 2, 9, 9]], device=DEV)
        graph.replay()
        torch.cuda.synchronize()
        # The toy target predicts each token's successor: row 0's drafts are
        # all right (the stream grows by 4) but it emits through the stop id 7
        # only; row 1's second draft is wrong.
        assert out.accepted.tolist() == [4, 2]
        assert out.tokens.tolist() == [[6, 7, -1, -1], [2, 3, -1, -1]]
        assert out.done.tolist() == [True, False]
        assert spec._seq_len[1:rows + 1].tolist() == [base[0] + 4, base[1] + 2]
        # the last draft pass addressed slot n at P + n + (k - 1) through the table
        for r in range(rows):
            n = int(out.accepted[r]) - 1
            pos = base[r] + n + (k - 1)
            page = int(plan.block_table[r, pos // PAGE])
            assert int(plan.write_pages[r * t + n]) == page
            assert int(plan.write_offsets[r * t + n]) == pos % PAGE
            assert int(plan.kv_lens[r]) == base[r] + t + (k - 1)


def _sampled_step(rows: int = 2, k: int = 2):
    """Row 0 samples a flat target at 0.9 / 40; row 1 is greedy."""
    t = k + 1
    spec = _spec(k, rows, width=24)
    regs = slice(1, rows + 1)
    spec._temperature[regs] = torch.tensor([1.0, 0.0], device=DEV)
    spec._top_p[regs], spec._top_k[regs] = 0.9, 40
    spec._seed[regs] = torch.tensor([11, 12], device=DEV)
    flat = torch.zeros(rows * t, VOCAB, device=DEV)
    return spec, flat, regs


def _reset(spec, regs, offset):
    """Registers as a seed leaves them: each draft a point mass, so the sampling
    row mostly rejects and draws from what is left."""
    tokens = torch.tensor([[5, 6, 7], [1, 2, 3]], device=DEV)
    spec._seq_len[regs] = 50
    spec._num_acc[regs] = 1
    spec._tokens[regs] = tokens
    spec._draft_ids[regs] = tokens[:, 1:, None].expand(2, 2, sampled.CANDIDATES)
    spec._draft_probs[regs] = torch.tensor([1.0] + [0.0] * (sampled.CANDIDATES - 1), device=DEV)
    spec._rng_offset[regs] = offset


def _capture(spec, flat, regs, k=2):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _reset(spec, regs, RNG_START)
        _verify_step(spec, flat, VOCAB, k, sample=True)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _verify_step(spec, flat, VOCAB, k, sample=True)
    return graph, out


def test_a_captured_sampled_step_draws_fresh_tokens_each_replay():
    """The sampled verification (the filter, the accept chain and the draft
    draws, all with device seeds and offsets) captures, draws anew on every
    replay, and keeps greedy rows on the argmax."""
    spec, flat, regs = _sampled_step()
    graph, out = _capture(spec, flat, regs)
    drawn = set()
    for i in range(20):
        _reset(spec, regs, RNG_START + 5 * i)
        graph.replay()
        torch.cuda.synchronize()
        drawn.add(int(out.target[0, 0]))
        # the greedy row's target is each token's successor, so it accepts all three
        assert out.target[1].tolist() == [2, 3, 4] and int(out.accepted[1]) == 3
    assert len(drawn) > 5, drawn


def test_a_replay_equals_the_eager_step_at_the_same_registers():
    """Same registers, same seeds and offsets: the tokens, counts, drafts and
    stored draft distributions of a replay and of the step run eagerly."""
    spec, flat, regs = _sampled_step()
    graph, out = _capture(spec, flat, regs)
    for i in range(8):
        offset = RNG_START + 1000 + 5 * i
        _reset(spec, regs, offset)
        eager = _verify_step(spec, flat, VOCAB, 2, sample=True)
        torch.cuda.synchronize()
        want = tuple(x.clone() for x in eager) + tuple(
            r[regs].clone() for r in (spec._tokens, spec._draft_ids, spec._draft_probs)
        )
        _reset(spec, regs, offset)
        graph.replay()
        torch.cuda.synchronize()
        got = tuple(out) + tuple(r[regs] for r in (spec._tokens, spec._draft_ids, spec._draft_probs))
        for a, b in zip(got, want, strict=True):
            assert torch.equal(a, b)
        assert spec._rng_offset[1:3].tolist() == [offset + sampled.words_per_step(2)] * 2


@pytest.mark.parametrize("settings", [(0.7, 40, 0.9), (0.7, 0, 0.9), (1.0, 5, 1.0), (0.0, 0, 1.0), (0.8, 3, 0.5),
                                      (3.0, 0, 0.9), (1.0, 100, 1.0), (1.0, 0, 1.0)])
def test_the_filter_matches_its_reference(settings):
    """The kernels and the torch path keep the same tokens and renormalise alike,
    whether the filter's kept set fits the candidates or not (the last three). The
    vocabulary is small enough to read the references in float32."""
    temperature, top_k, top_p = settings
    rows, vocab = 8, 512
    logits = (torch.randn(rows, vocab, generator=torch.Generator().manual_seed(2)) * 3).to(DEV)
    args = (torch.full((rows,), float(temperature), device=DEV),
            torch.full((rows,), top_k or vocab, device=DEV, dtype=torch.int32),
            torch.full((rows,), float(top_p), device=DEV))
    got = sampled.filtered_probs(logits, *args).cpu()
    want = sampled.filtered_probs(logits.cpu(), *(a.cpu() for a in args))
    assert torch.equal(got > 0, want > 0)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-3)


def test_tokens_after_a_rejection_follow_the_target():
    """Steps of a chain whose tokens depend on the one before, through the kernels
    and their word streams: the pairs of emitted tokens have the target's joint
    distribution, across the step boundary. A draft that shares a Philox word with
    the replacement before it fails this."""
    k, rows, vocab = 2, 30000, VOCAB
    gen = torch.Generator().manual_seed(21)
    target = (torch.randn(vocab, vocab, generator=gen) * 2).to(DEV)
    draft = target + (torch.randn(vocab, vocab, generator=gen).to(DEV))
    temperature = 0.8
    args = (torch.full((vocab,), temperature, device=DEV), torch.full((vocab,), vocab, device=DEV, dtype=torch.int32),
            torch.ones(vocab, device=DEV))
    q = sampled.filtered_probs(target, *args)
    spec = SpecDecodeResource(device=DEV, num_draft_tokens=k, max_requests=rows, page_size=PAGE, kv_key="kv",
                              attn_key="attn", sampling=True)
    for r in range(rows):
        spec.ingest_request(r)
    image = spec._new_image(rows)
    image.rows[:rows].copy_(torch.tensor([spec._row_of[r] for r in range(rows)], device=DEV))
    image.real[:rows].fill_(1)
    table = (1 + torch.arange(rows * 8, device=DEV, dtype=torch.int32)).view(rows, 8)
    state = torch.zeros(rows * (k + 1), dtype=torch.long, device=DEV)
    spec._current = SpecDecodePlan(phase=SpecPhase.VERIFY, num_rows=rows, image=image, block_table=table,
                                   kv_lens=torch.zeros(rows, dtype=torch.int32, device=DEV), write_pages=state,
                                   write_offsets=state.clone())
    regs = slice(1, rows + 1)
    spec._temperature[regs] = temperature
    spec._top_p[regs] = 1.0
    spec._seed[regs] = torch.arange(rows, device=DEV)
    spec._rng_offset[regs] = RNG_START
    spec._budget[regs] = 1 << 20
    spec._seq_len[regs] = 10
    spec._num_acc[regs] = 1
    spec._tokens[regs] = torch.tensor([3, 7, 7], device=DEV)
    spec._draft_ids[regs] = 7
    spec._draft_probs[regs] = 0.0
    spec._draft_probs[1:rows + 1, :, 0] = 1.0
    outs = []
    for _ in range(2):
        spec._seq_len[regs] = 10
        ids, _ = spec.verify_inputs()
        out = spec.accept(target[ids], sample=True)
        last = out.target.gather(1, (out.accepted - 1).long()[:, None])[:, 0]
        for j in range(k):
            spec.begin_draft_pass(j)
            last = spec.draw_draft(j, draft[last])
            spec.record_draft(j, last)
        outs.append(out.tokens.clone())
    torch.cuda.synchronize()
    tokens = torch.cat(outs, dim=1).cpu()
    # each row's first two emitted tokens, past the -1s
    keep = (tokens >= 0).long().argsort(dim=1, descending=True, stable=True)
    first_two = tokens.gather(1, keep)[:, :2]
    keys = first_two[:, 0] * vocab + first_two[:, 1]
    pairs = torch.bincount(keys, minlength=vocab * vocab).float().view(vocab, vocab)
    want = (q[3][:, None] * q).cpu()
    # against what a sample of the target itself reads at this size
    drawn = torch.multinomial(want.reshape(-1), rows, replacement=True, generator=gen)
    iid = torch.bincount(drawn, minlength=vocab * vocab).float().view(vocab, vocab)
    noise = 0.5 * (iid / rows - want).abs().sum()
    tv = 0.5 * (pairs / rows - want).abs().sum()
    assert tv < 2 * noise, (tv, noise)


def _renormed(logits, temperature, top_k, top_p):
    """FlashInfer's vocabulary-wide filter, the distribution the plain sampler draws from."""
    fi = sampled.sampling
    probs = fi.fused_temperature_softmax(logits, temperature, include_greedy=True)
    return fi._top_p_renorm_probs(fi._top_k_renorm_probs(probs, top_k), top_p)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("settings", [(0.7, 40, 0.9), (0.7, 0, 0.9), (1.0, 5, 1.0), (1.0, 0, 0.95), (2.0, 0, 0.9),
                                      (1.0, 100, 0.9), (1.0, 0, 1.0)])
def test_the_filter_is_flashinfers_at_a_real_vocabulary(settings, dtype):
    """Noise under a few dozen tokens across tiles, and the flat rows that run past
    the candidates: the same tokens kept, in the same proportions. Float32 sums over
    100k tokens round differently, so a flat row's cut can land a token or two away."""
    temperature, top_k, top_p = settings
    rows, vocab = 6, 248320
    g = torch.Generator().manual_seed(11)
    logits = torch.randn(rows, vocab, generator=g) * 1.5
    logits.scatter_add_(1, torch.randint(0, vocab, (rows, 64), generator=g), 6 + torch.rand(rows, 64, generator=g) * 14)
    logits = logits.to(DEV, dtype=dtype)
    args = (torch.full((rows,), float(temperature), device=DEV),
            torch.full((rows,), top_k or vocab, device=DEV, dtype=torch.int32),
            torch.full((rows,), float(top_p), device=DEV))
    got, want = sampled.filtered_probs(logits, *args), _renormed(logits, *args)
    differ = ((got > 0) != (want > 0)).sum(dim=-1)
    assert (differ <= 2).all(), differ
    if int((want > 0).sum(dim=-1).max()) < 1000:
        assert not differ.any()
    assert (0.5 * (got - want).abs().sum(dim=-1) < 1e-4).all()


def test_more_kept_tokens_in_a_tile_than_it_hands_up():
    """A row whose kept tokens crowd one tile says it isn't complete, and the
    target is exact all the same; a row they don't crowd is left to its candidates."""
    rows, vocab = 2, 8192
    logits = torch.full((rows, vocab), -30.0)
    logits[0, :40] = 5.0 + torch.arange(40) * 0.01      # one tile
    logits[1, torch.arange(40) * 200] = 5.0 + torch.arange(40) * 0.01      # many
    logits = logits.to(DEV)
    args = (torch.ones(rows, device=DEV), torch.full((rows,), vocab, device=DEV, dtype=torch.int32),
            torch.full((rows,), 0.9, device=DEV))
    cand = sampled.filtered_candidates(logits, *args)
    assert cand.complete.tolist() == [0, 1]
    got, want = sampled.filtered_probs(logits, *args), _renormed(logits, *args)
    assert torch.equal(got > 0, want > 0)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-3)
    # past the candidates a draft is still a distribution to draw from
    assert torch.allclose(cand.probs.sum(dim=-1), torch.ones(rows, device=DEV))


@pytest.mark.parametrize(("top_k", "complete"), [(40, 1), (100, 1), (12, 1), (0, 1)])
def test_a_top_k_cut_out_of_sight_leaves_the_kept_tokens_in_sight(top_k, complete):
    """Frequent tokens crowd a tile past what it hands up, so a top-k of 40 can't see
    its own last token. What top-p keeps of a steep row doesn't depend on that."""
    rows, vocab = 2, 8192
    logits = torch.full((rows, vocab), -30.0)
    logits[:, :100] = 10.0 - torch.arange(100) * 0.25
    logits = logits.to(DEV)
    args = (torch.full((rows,), 0.7, device=DEV), torch.full((rows,), top_k or vocab, device=DEV, dtype=torch.int32),
            torch.full((rows,), 0.9, device=DEV))
    assert sampled.filtered_candidates(logits, *args).complete.tolist() == [complete] * rows
    got, want = sampled.filtered_probs(logits, *args), _renormed(logits, *args)
    assert torch.equal(got > 0, want > 0)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-3)


def test_a_cut_that_depends_on_top_ks_mass_takes_the_exact_row():
    """Top-p's cut moves with a mass out of sight: not complete, and exact. A row where
    the two ends of that mass put the cut on different tokens."""
    rows, vocab = 2, 8192
    logits = torch.full((rows, vocab), -30.0)
    # flat, so 90% of a top-k's 40 is 36 tokens and 90% of the row's 100 is 90
    logits[:, :100] = 5.0 - torch.arange(100) * 1e-3
    logits = logits.to(DEV)
    args = (torch.ones(rows, device=DEV), torch.full((rows,), 40, device=DEV, dtype=torch.int32),
            torch.full((rows,), 0.9, device=DEV))
    assert sampled.filtered_candidates(logits, *args).complete.tolist() == [0, 0]
    got, want = sampled.filtered_probs(logits, *args), _renormed(logits, *args)
    assert torch.equal(got > 0, want > 0), ((got > 0).sum(-1), (want > 0).sum(-1))
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-3)


def test_the_filter_matches_the_plain_sampler():
    """Draws of the plain step's kernel fall on exactly what the filter keeps, in
    its proportions."""
    rows, vocab, top_k, top_p = 20000, 128, 12, 0.8
    logits = (torch.randn(1, vocab, generator=torch.Generator().manual_seed(3)) * 2).repeat(rows, 1).to(DEV)
    temperature = torch.full((rows,), 0.7, device=DEV)
    kept = torch.full((rows,), top_k, device=DEV, dtype=torch.int32)
    mass = torch.full((rows,), top_p, device=DEV)
    probs = sampled.sampling.fused_temperature_softmax(logits, temperature, include_greedy=True)
    drawn = sampled.sampling._top_k_top_p_sampling_from_probs(
        probs, kept, mass, deterministic=True, seed=torch.arange(rows, device=DEV),
        offset=torch.zeros(rows, device=DEV, dtype=torch.long),
    )
    drawn = drawn[0] if isinstance(drawn, tuple) else drawn
    freq = torch.bincount(drawn.long().cpu(), minlength=vocab).float() / rows
    q = sampled.filtered_probs(logits[:1], temperature[:1], kept[:1], mass[:1])[0].cpu()
    assert not freq[q == 0].any()
    assert 0.5 * (freq - q).abs().sum() < 0.02


def test_the_chain_kernel_is_speculative_sampling():
    """FlashInfer's chain at a small vocabulary against the closed form: what a row
    emits first follows q0 whatever p is, and it keeps a draft sum min(p, q) of
    the time. Every row gets its own stream."""
    vocab, k, rows = 32, 2, 100_000
    g = torch.Generator().manual_seed(5)
    q = torch.softmax(torch.randn(k + 1, vocab, generator=g) * 2, -1).to(DEV)
    p = torch.softmax(torch.randn(k, vocab, generator=g) * 2, -1).to(DEV)
    d = torch.multinomial(p.cpu(), rows, replacement=True, generator=g).T.contiguous().to(DEV)
    tokens, kept = sampled.accept_chain(
        torch.arange(vocab, device=DEV).expand(rows, k, vocab).contiguous(),
        p[None].expand(rows, k, vocab).contiguous(), d, q[None].expand(rows, k + 1, vocab).contiguous(),
        torch.arange(rows, device=DEV), torch.full((rows,), RNG_START, device=DEV),
    )
    torch.cuda.synchronize()
    freq = torch.bincount(tokens[:, 0].long().cpu(), minlength=vocab).float() / rows
    assert 0.5 * (freq - q[0].cpu()).abs().sum() < 0.015
    assert abs(float((kept >= 1).float().mean()) - float(torch.minimum(p[0], q[0]).sum())) < 0.01
    assert ((tokens == -1).sum(dim=1) == k - kept.clamp_max(k)).all()


def test_the_margin_leaves_a_residual_for_every_rejection():
    """Point masses on both sides, as a confident top-p leaves them: the draft is
    kept, and a wrong one is replaced by the target's token, never an unset id."""
    vocab, k, rows = 64, 1, 1024
    q = torch.zeros(rows, k + 1, vocab, device=DEV)
    q[:, 0, 9], q[:, 1, 3] = 1.0, 1.0
    right = torch.arange(rows, device=DEV) % 2 == 0
    d = torch.where(right, 9, 8)[:, None]
    ids, mass = d[:, :, None].expand(rows, k, 4).contiguous(), torch.zeros(rows, k, 4, device=DEV)
    mass[:, :, 0] = 1.0
    tokens, kept = sampled.accept_chain(ids, mass, d, q, torch.arange(rows, device=DEV),
                                        torch.zeros(rows, device=DEV, dtype=torch.long))
    assert (tokens[:, 0] == 9).all()
    assert (kept[right] == 1).all() and (kept[~right] == 0).all()
    assert (tokens[right, 1] == 3).all()

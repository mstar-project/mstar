"""The greedy capture (`GreedySample`): which steps the engine leases it for,
what its sampler draws, and that a node's rows keep their RNG offsets, last
tokens and loop-back reads when its steps flip between it and the full
capture, resident or host-planned, pre-planned or not.

On the CPU the captured sampler's ops run eagerly, as a replay runs what its
capture recorded, the full sampler's draw is replaced by one that returns a
token of the row's seed and offset, and the argmax is torch's. The Triton
argmax and its agreement with the full sampler run on CUDA.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.resident import ResidentPlanner
from mstar.engine.resources.sampler import utils as sampler_utils
from mstar.engine.resources.sampler.config import GreedySample, SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource, greedy_samplers
from mstar.engine.resources.sampler.utils import (
    FUSED_SAMPLE_MAX_VOCAB,
    SamplingConfig,
    _rng_offset_stride,
    greedy_tokens,
    samples_greedy,
)
from mstar.engine.resources.step import (
    BucketKey,
    Segment,
    SlotLease,
    StepContext,
    SubmoduleStep,
)

CPU = torch.device("cpu")
V = 64
WIDE = FUSED_SAMPLE_MAX_VOCAB + 1
WALK = "decode"
STRIDE = _rng_offset_stride("cpu", V)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cfg(temperature: float, seed: int = 0, **kw) -> SamplingReqConfig:
    cfg = SamplingReqConfig(temperature=temperature, **kw)
    cfg.apply_conductor_config(seed=seed)
    return cfg


# ---- what greedy means ----

@pytest.mark.parametrize("kw,greedy", [
    (dict(temperature=0.0), True),
    # below GREEDY_TEMPERATURE reads as 0
    (dict(temperature=1e-6), True),
    # the argmax is the first token every filter keeps
    (dict(temperature=0.0, top_p=0.5, top_k=5, min_p=0.3), True),
    (dict(temperature=0.7), False),
    # top-k 1 breaks exact ties by the seed
    (dict(temperature=1.0, top_k=1), False),
    # these move the logits before the argmax, or score the draw
    (dict(temperature=0.0, repetition_penalty=1.1), False),
    (dict(temperature=0.0, repetition_penalty=1.1, repetition_window=8), False),
    (dict(temperature=0.0, min_tokens=4), False),
    (dict(temperature=0.0, return_logprobs=True), False),
])
def test_what_samples_greedily(kw, greedy):
    assert samples_greedy(SamplingConfig(**kw)) is greedy


def test_only_a_vocabulary_past_the_one_kernel_draw_gets_a_greedy_capture():
    wide = SamplerResource(vocab_size=WIDE, enable_repetion_penalty=True, device=CPU)
    narrow = SamplerResource(vocab_size=FUSED_SAMPLE_MAX_VOCAB, enable_repetion_penalty=True, device=CPU)
    unsized = SamplerResource(vocab_size=None, enable_repetion_penalty=False, device=CPU)
    assert greedy_samplers({"a": wide, "b": narrow, "c": unsized, "d": object()}) == {"a": wide}


def test_the_argmax_takes_the_first_of_equal_maxima_on_the_cpu():
    logits = torch.zeros(3, 10)
    logits[0, [7, 2, 9]] = 5.0
    logits[1] = -float("inf")
    logits[2, 4] = 1.0
    assert greedy_tokens(logits).tolist() == [2, 0, 4]


# ---- the plan's check ----

@pytest.fixture
def wide():
    return SamplerResource(vocab_size=WIDE, enable_repetion_penalty=True, device=CPU)


def test_the_plan_refuses_a_greedy_lease_over_a_sampled_row(wide):
    wide.ingest_request(1, _cfg(0.0))
    wide.ingest_request(2, _cfg(0.7))
    lease = SlotLease(0, BucketKey(WALK, 2, 2, GreedySample()))
    ctx = StepContext(request_ids=(1, 2), graph_walk=WALK, slot=0, capture=False, slot_lease=lease)
    with pytest.raises(RuntimeError, match="don't sample greedily"):
        wide.plan(SamplerStep(), ctx)
    # a capture's rows are padding rows, whatever their config
    ctx = StepContext(request_ids=(1, 2), graph_walk=WALK, slot=0, capture=True, slot_lease=lease)
    wide.plan(SamplerStep(), ctx)
    assert wide._greedy_this_step
    ctx = StepContext(request_ids=(1, 2), graph_walk=WALK, slot=0, capture=False,
                      slot_lease=SlotLease(0, BucketKey(WALK, 2, 2)))
    wide.plan(SamplerStep(), ctx)
    assert not wide._greedy_this_step


# ---- steps flipping between the two captures ----

@pytest.fixture(params=[True, False], ids=["ingraph_scatter", "scatter_after_replay"])
def ingraph(request, monkeypatch) -> bool:
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "1" if request.param else "0")
    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "1")
    return request.param


@pytest.fixture
def drawn(monkeypatch):
    """The full draws, each a list of its rows' temperatures; a token of the
    row's seed and offset, so a token says which draw it was."""
    calls: list[list[float]] = []

    def captured(logits, temperature, top_k, top_p, seed, offset, **kwargs):
        calls.append(temperature.tolist())
        return (seed * 7 + offset // STRIDE) % V

    monkeypatch.setattr(sampler_utils, "sample_accelerator_graphable_gpu", captured)
    return calls


class _Node:
    """A sampler with or without a planner, its decode steps captured per
    (bucket, slot) and replayed as the engine drives them."""

    def __init__(self, planner: bool):
        # a wide vocabulary in name, so the node captures greedy; the logits stay V wide
        self.sampler = SamplerResource(vocab_size=WIDE, enable_repetion_penalty=True, device=CPU)
        self.sampler.enable_device_loopback_reader()
        resources = {"sampler": self.sampler}
        self.planner = ResidentPlanner("n", {WALK}, resources, CPU, max_bs=4) if planner else None
        self.sampler.build_accelerator_graph_buffers(
            [SimpleNamespace(slot=0), SimpleNamespace(slot=1)], max_bs=4, max_seq_len=1,
        )
        self.bufs = self.sampler._cg_buffers
        self.captured: dict = {}
        self.tokens: dict[int, list[int]] = {}
        self.gen = torch.Generator().manual_seed(0)

    def ingest(self, rid: int, cfg: SamplingReqConfig) -> None:
        self.sampler.ingest_request(rid, cfg)
        if self.bufs.ingraph_scatter:
            assert self.sampler.reserve(rid, "n", WALK, None) is None

    def offset(self, rid: int) -> int:
        return int(self.bufs.offset.master[self.bufs.slot_of(rid)])

    def decode(self, rids, slot: int, bs: int = 4, preplan: bool = True):
        """One captured step on the bucket the engine's choice names."""
        greedy = self.sampler.runs_greedy(rids)
        padded = tuple(rids) + tuple(-(i + 1) for i in range(bs - len(rids)))
        bucket = BucketKey(graph_walk=WALK, bs=bs, num_tokens=bs, cg_key_info=GreedySample() if greedy else None)
        lease = SlotLease(slot=slot, bucket=bucket)
        segments = tuple(Segment(r, "main", 1) for r in padded)
        smp = SamplerStep()
        step = SubmoduleStep(steps={"sampler": smp}, segments=list(segments))
        ctx = StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=slot, capture=False, slot_lease=lease)
        ctx.set_padded_rids(padded)
        key = (bucket, slot)
        if key not in self.captured:
            for rid in padded[len(rids):]:
                if rid not in self.sampler._sampler._sampling_config:
                    self.sampler.ingest_capture_row(rid)
            cap = StepContext(request_ids=padded, graph_walk=WALK, slot=slot, capture=True, slot_lease=lease)
            self.sampler.plan(smp, cap)
            rec = self.planner.capture_step(bucket, slot, step) if self.planner is not None else None
            if rec is not None:
                self.planner.gather(rec)
            # what the capture's sample ran: the variant is baked in
            self.captured[key] = (rec, self.sampler._cg_sampler, self.sampler._greedy_this_step)
        rec, graph_sampler, captured_greedy = self.captured[key]
        assert captured_greedy == greedy
        if self.planner is not None:
            ctx.resident = self.planner
        if preplan:
            ctx.is_preplan = True
            if self.planner is not None:
                self.planner.plan(step, ctx)
            self.sampler.plan(smp, ctx)
            ctx.is_preplan = False
        if self.planner is not None:
            self.planner.plan(step, ctx)
        self.sampler.plan(smp, ctx)
        assert self.sampler._cg_sampler is graph_sampler and self.sampler._greedy_this_step == greedy
        ids = self.sampler.loopback_tokens(list(padded))
        if self.planner is not None:
            self.planner.flush(ctx.resident_plan)
            self.sampler._cells.gather(rec)
        logits = torch.randn(bs, V, generator=self.gen)
        out = graph_sampler.sample(list(padded), logits, greedy=captured_greedy)
        if self.planner is not None:
            self.planner.commit_in_graph(rec)
        self.sampler.commit(smp, ctx)
        if self.planner is not None:
            self.planner.commit(step, ctx)
        n = len(rids)
        for rid, token in zip(rids, out[:n].tolist(), strict=True):
            self.tokens.setdefault(rid, []).append(token)
        return SimpleNamespace(ids=ids[:n].tolist(), out=out[:n], argmax=logits[:n].argmax(-1), greedy=greedy)


@pytest.mark.parametrize("planner", [True, False], ids=["resident", "host_planned"])
@pytest.mark.parametrize("preplan", [True, False], ids=["preplanned", "inline"])
def test_rows_keep_their_offsets_tokens_and_loopback_across_flips(drawn, ingraph, planner, preplan):
    node = _Node(planner)
    for rid in (1, 2):
        node.ingest(rid, _cfg(0.0, seed=rid))
    steps = [((1, 2), True), ((1, 2), True), ((1, 2, 3), False), ((1, 2, 3), False), ((1, 2), True),
             ((2, 1), True), ((1, 2, 3), False)]
    last: dict[int, int] = {}
    for i, (rids, greedy) in enumerate(steps):
        if i == 2:
            # a sampled request arrives mid-run
            node.ingest(3, _cfg(0.7, seed=3))
        before = len(drawn)
        s = node.decode(list(rids), slot=i % 2, preplan=preplan)
        assert s.greedy == greedy
        # a greedy step draws nothing in full; a full one draws once over its rows
        assert len(drawn) == before + (not greedy)
        if greedy:
            assert torch.equal(s.out, s.argmax)
        else:
            assert drawn[-1][:3] == pytest.approx([0.0, 0.0, 0.7])
        # each row reads back the token its own last step wrote, whichever capture wrote it
        for rid, read in zip(rids, s.ids, strict=True):
            if rid in last:
                assert read == last[rid]
        last.update(zip(rids, s.out.tolist(), strict=True))
    # every step advances its rows' offsets, on either capture
    assert [node.offset(rid) for rid in (1, 2, 3)] == [7 * STRIDE, 7 * STRIDE, 3 * STRIDE]


def test_a_greedy_step_leaves_the_seen_token_masks_alone(ingraph):
    """Its rows' penalties are all 1.0: no staging, gather or sync of masks,
    even with a penalised request resident."""
    node = _Node(planner=True)
    node.ingest(1, _cfg(0.0))
    node.ingest(9, _cfg(0.7, repetition_penalty=1.3))
    assert node.sampler._penalty_live
    staged = []
    node.bufs.stage_seen_token_masks = lambda request_ids, seen_masks: staged.append(request_ids)
    node.decode([1], slot=0)
    assert not node.sampler._penalty_needed_this_step and staged == []


# ---- on CUDA ----

@cuda
@pytest.mark.parametrize("bs", [1, 3, 16, 64, 128])
@pytest.mark.parametrize("vocab", [248320, 151936, 5003])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_triton_argmax_is_torchs(bs, vocab, dtype):
    gen = torch.Generator(device="cuda").manual_seed(bs * vocab)
    logits = (torch.randn(bs, vocab, device="cuda", generator=gen) * 4).to(dtype)
    # exact ties at each row's max, the later index first in some rows; a -inf row
    for r in range(bs):
        at = torch.randint(0, vocab, (3,), device="cuda", generator=gen)
        logits[r, at] = 30.0
    logits[0, :] = -float("inf")
    assert torch.equal(greedy_tokens(logits), torch.argmax(logits, dim=-1))


@cuda
@pytest.mark.parametrize("bs", [1, 16, 64])
def test_the_greedy_capture_draws_what_the_full_sampler_draws_at_temperature_0(bs):
    """On the same logits: the full sampler's one-hot at temperature 0 and the
    argmax, ties included, and the greedy draw captured and replayed."""
    vocab = 248320
    gen = torch.Generator(device="cuda").manual_seed(bs)
    logits = (torch.randn(bs, vocab, device="cuda", generator=gen) * 4).to(torch.bfloat16)
    for r in range(0, bs, 2):
        logits[r, torch.randint(0, vocab, (2,), device="cuda", generator=gen)] = 30.0
    zeros = dict(device="cuda")
    full = sampler_utils.sample_accelerator_graphable_gpu(
        logits, torch.zeros(bs, **zeros), torch.zeros(bs, dtype=torch.int32, **zeros), torch.ones(bs, **zeros),
        torch.arange(bs, dtype=torch.long, **zeros), torch.zeros(bs, dtype=torch.long, **zeros),
    )
    sampler = sampler_utils.AcceleratorGraphableSampler(
        temperature_buf=torch.zeros(bs, **zeros), top_k_buf=torch.zeros(bs, dtype=torch.int32, **zeros),
        top_p_buf=torch.ones(bs, **zeros), seed_buf=torch.zeros(bs, dtype=torch.long, **zeros),
        offset_buf=torch.zeros(bs, dtype=torch.long, **zeros),
        last_token_buf=torch.zeros(bs, dtype=torch.long, **zeros),
    )
    static = logits.clone()
    sampler.sample(list(range(bs)), static, greedy=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        sampler.sample(list(range(bs)), static, greedy=True)
    torch.cuda.current_stream().wait_stream(stream)
    sampler.last_token_buf.zero_()
    sampler.offset_buf.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(full, torch.argmax(logits, dim=-1))
    assert torch.equal(sampler.last_token_buf, full)
    assert sampler.offset_buf.tolist() == [_rng_offset_stride("cuda", vocab)] * bs

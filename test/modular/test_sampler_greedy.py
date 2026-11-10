"""The sampler's lean step (`SamplerResource.lean_for`): which steps take each row's
argmax, in the twin of a batched capture or eagerly, which buckets get that twin, the
serving opt-out, what the lean step draws, and that a node's rows keep their RNG
offsets, last tokens and loop-back reads when its steps flip between the twin and the
plain capture, resident or host-planned, pre-planned or not.

On the CPU the captured sampler's ops run eagerly, as a replay runs what its
capture recorded, the full sampler's draw is replaced by one that returns a
token of the row's seed and offset, and the argmax is torch's. The Triton
argmax and its agreement with the full sampler run on CUDA.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig, PackedAcceleratorGraphConfig
from mstar.engine.resources.base import CGSlotSpec, EngineResourceInfo, Resource, build_resource
from mstar.engine.resources.resident import ResidentPlanner
from mstar.engine.resources.sampler import utils as sampler_utils
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.sampler.utils import (
    FUSED_SAMPLE_MAX_VOCAB,
    SamplingConfig,
    _rng_offset_stride,
    greedy_tokens,
    samples_greedy,
)
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import (
    BucketKey,
    Segment,
    SlotLease,
    StepContext,
    SubmoduleStep,
)
from mstar.model.submodule_base import ARNodeInputs
from mstar.testing_utils.engine import make_engine, make_mgmt

CPU = torch.device("cpu")
V = 64
WIDE = FUSED_SAMPLE_MAX_VOCAB + 1
WALK = "decode"
STRIDE = _rng_offset_stride("cpu", V)
GREEDY = frozenset({"sampler"})
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cfg(temperature: float, seed: int = 0, **kw) -> SamplingReqConfig:
    cfg = SamplingReqConfig(temperature=temperature, **kw)
    cfg.apply_conductor_config(seed=seed)
    return cfg


def _sampler(vocab_size=WIDE, **kw) -> SamplerResource:
    """A sampler built as the engine builds it, under the key "sampler"."""
    spec = SamplerSpec(resource_key="sampler", nodes={"LLM"}, vocab_size=vocab_size, **kw)
    return build_resource(spec, EngineResourceInfo(device=CPU))


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


@pytest.mark.parametrize("vocab_size,greedy_capture,lean", [
    (WIDE, None, True),
    (FUSED_SAMPLE_MAX_VOCAB, None, False),
    (None, None, False),
    # the opt-out, and its opposite where the vocabulary alone wouldn't
    (WIDE, False, False),
    (FUSED_SAMPLE_MAX_VOCAB, True, True),
])
def test_a_sampler_runs_lean_by_its_vocabulary_unless_its_spec_says_otherwise(vocab_size, greedy_capture, lean):
    sampler = _sampler(vocab_size, greedy_capture=greedy_capture, enable_repetition_penalty=vocab_size is not None)
    assert sampler.lean_capture is lean and sampler.resource_key == "sampler"


def test_a_serving_config_turns_the_greedy_twins_off():
    spec = SamplerSpec(resource_key="sampler", nodes={"LLM"}, vocab_size=WIDE)
    apply_yaml_overrides([spec], {"resources": {"sampler": {"greedy_capture": False}}})
    assert spec.greedy_capture is False and not build_resource(spec, EngineResourceInfo(device=CPU)).lean_capture
    with pytest.raises(ValueError, match="greedy_capture"):
        apply_yaml_overrides([spec], {"resources": {"sampler": {"greedy_capture": "off"}}})


def test_the_argmax_takes_the_first_of_equal_maxima_on_the_cpu():
    logits = torch.zeros(3, 10)
    logits[0, [7, 2, 9]] = 5.0
    logits[1] = -float("inf")
    logits[2, 4] = 1.0
    assert greedy_tokens(logits).tolist() == [2, 0, 4]


# ---- which steps lease the twin ----

class _Lora(Resource):
    """A stand-in for the LoRA pool: lean for rows that name no adapter."""
    build = None
    lean_capture = True

    def __init__(self, adapter_rows=()):
        self.adapter_rows = set(adapter_rows)

    def lean_for(self, request_ids) -> bool:
        return self.adapter_rows.isdisjoint(request_ids)


def _bucket(*lean) -> BucketKey:
    return BucketKey(WALK, 4, 4, None, frozenset(lean))


def _engine(resources, captured):
    """An engine whose node keys its captures None and holds ``captured``, leased by the runner's own code."""
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner

    config = BatchedAcceleratorGraphConfig(
        capture_graph_walk=WALK, capture_batch_sizes=[4], single_request_inputs=ARNodeInputs(input_seq_len=1),
    )
    runner = object.__new__(AcceleratorGraphRunner)
    runner._buckets, runner._twins = {}, {}
    for bucket in captured:
        for slot in range(2):
            runner._register_slot(CGSlotSpec(bucket, slot, config, 0), object())
    return make_engine(_submodules={"LLM": make_mgmt(
        SimpleNamespace(cg_key_info=lambda walk, info, **kw: None), resources=resources,
        accelerator_graph_runner=runner,
    )})


def _leased(engine, rids):
    """The bucket a step of ``rids`` leases (None: eager), and what it runs lean."""
    batch = SimpleNamespace(
        node_name="LLM", slot=0, request_ids=list(rids), inputs=None, cg_key_info=None,
        per_request_info_wrapped={}, per_request_input_metadata={},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
    )
    batch.lease_slot = lambda lease: setattr(batch.step_context, "slot_lease", lease)
    lease = engine.reserve_replay_slot(batch)
    return (None if lease is None else lease.bucket), batch.step_context.lean


@pytest.fixture
def wide():
    return _sampler()


def test_a_step_takes_the_greedy_twin_only_when_every_row_samples_greedily(wide):
    for rid, t in ((1, 0.0), (2, 0.0), (3, 0.7), (4, 0.9)):
        wide.ingest_request(rid, _cfg(t))
    engine = _engine({"sampler": wide}, [_bucket(), _bucket("sampler")])
    assert _leased(engine, [1, 2]) == (_bucket("sampler"), GREEDY)
    # some rows sample, or none is greedy: the plain bucket
    assert _leased(engine, [1, 3]) == (_bucket(), frozenset())
    assert _leased(engine, [3, 4]) == (_bucket(), frozenset())


def test_the_twin_follows_the_batch_not_who_is_resident(wide):
    """A sampled row arriving mid-run sends the steps it joins to the plain bucket; the
    greedy rows' steps without it, and after it leaves, go back."""
    engine = _engine({"sampler": wide}, [_bucket(), _bucket("sampler")])
    for rid in (1, 2):
        wide.ingest_request(rid, _cfg(0.0))
    assert _leased(engine, [1, 2])[1] == GREEDY
    wide.ingest_request(3, _cfg(0.7))
    assert _leased(engine, [1, 2])[1] == GREEDY
    assert _leased(engine, [1, 2, 3]) == (_bucket(), frozenset())
    wide.remove_request(3)
    assert _leased(engine, [1, 2])[1] == GREEDY
    # a re-ingest replaces the config both ways
    wide.ingest_request(2, _cfg(0.7))
    assert _leased(engine, [1, 2])[1] == frozenset()
    wide.ingest_request(2, _cfg(0.0))
    assert _leased(engine, [1, 2])[1] == GREEDY


def test_no_greedy_twin_falls_back_to_the_plain_bucket(wide):
    wide.ingest_request(1, _cfg(0.0))
    assert _leased(_engine({"sampler": wide}, [_bucket()]), [1]) == (_bucket(), frozenset())


def test_greedy_and_base_only_nest_through_the_set(wide):
    """Greedy rows with no adapter take the twin with both lean; an adapter row takes the
    greedy one alone, and a sampled row the base-only one, each falling back through the
    sets the node's declaration order ranks."""
    for rid, t in ((1, 0.0), (2, 0.0), (3, 0.7)):
        wide.ingest_request(rid, _cfg(t))
    resources = {"lora": _Lora(adapter_rows=[2]), "sampler": wide}
    every = [_bucket(), _bucket("lora"), _bucket("sampler"), _bucket("lora", "sampler")]
    engine = _engine(resources, every)
    assert _leased(engine, [1]) == (_bucket("lora", "sampler"), {"lora", "sampler"})
    assert _leased(engine, [1, 2]) == (_bucket("sampler"), GREEDY)
    assert _leased(engine, [1, 3]) == (_bucket("lora"), {"lora"})
    assert _leased(engine, [2, 3]) == (_bucket(), frozenset())
    # the nested twin not captured: the larger single set by declaration order, the pool's
    assert _leased(_engine(resources, every[:3]), [1]) == (_bucket("lora"), {"lora"})
    assert _leased(_engine(resources, every[:1] + every[2:3]), [1]) == (_bucket("sampler"), GREEDY)


def test_a_node_whose_sampler_opted_out_never_runs_it_lean():
    off = _sampler(greedy_capture=False)
    off.ingest_request(1, _cfg(0.0))
    assert _leased(_engine({"sampler": off}, [_bucket(), _bucket("sampler")]), [1]) == (_bucket(), frozenset())


# ---- which buckets get the twin ----

def _listed(resources, steps=None):
    """The (walk, lean) of the buckets a runner over a decode and a prefill config captures,
    whose steps drive ``steps[walk]`` (every resource by default)."""
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner

    configs = [
        BatchedAcceleratorGraphConfig(capture_graph_walk=WALK, capture_batch_sizes=[1, 4],
                                      single_request_inputs=ARNodeInputs(input_seq_len=1)),
        PackedAcceleratorGraphConfig(capture_graph_walk="prefill", capture_token_lengths=[16],
                                     capture_batch_sizes=[1], make_node_input=lambda n: ARNodeInputs(input_seq_len=n)),
    ]

    def declare_step(graph_walk, **kwargs):
        keys = resources.keys() if steps is None else steps[graph_walk]
        return SimpleNamespace(steps=dict.fromkeys(keys, object()), segments=None)

    submodule = SimpleNamespace(get_accelerator_graph_configs=lambda *args: configs, declare_step=declare_step,
                                compiled_entries_per_shape=1)
    for resource in resources.values():
        # a CPU sampler can't capture; what is listed is the question here
        resource.can_capture = lambda: True
    runner = AcceleratorGraphRunner(
        submodule_name="LLM", submodule=submodule, resources=resources, step_runner=Mock(), device=CPU,
        autocast_dtype=None, joint_comm_group=SimpleNamespace(world_size=1), num_slots=2,
    )
    return {(spec.bucket.graph_walk, spec.bucket.lean) for spec in runner.prepare_for_capture()}


def test_only_batched_buckets_get_a_greedy_twin(wide):
    plain = {(WALK, frozenset()), ("prefill", frozenset())}
    assert _listed({"sampler": wide}) == plain | {(WALK, GREEDY)}
    assert _listed({"sampler": _sampler(V)}) == plain, "a narrow vocabulary draws in one kernel already"
    assert _listed({"sampler": _sampler(greedy_capture=False)}) == plain, "the opt-out"
    # a step that doesn't run the sampler gets no twin of it
    assert _listed({"sampler": wide}, steps={WALK: [], "prefill": []}) == plain
    # beside the pool: every set on decode, the pool's alone on prefill
    nested = _listed({"lora": _Lora(), "sampler": wide})
    assert nested == plain | {(WALK, frozenset({"lora", "sampler"})), (WALK, frozenset({"lora"})), (WALK, GREEDY),
                              ("prefill", frozenset({"lora"}))}


# ---- the plan's check, and the eager step ----

def test_the_plan_refuses_a_greedy_step_over_a_sampled_row(wide):
    wide.ingest_request(1, _cfg(0.0))
    wide.ingest_request(2, _cfg(0.7))
    lease = SlotLease(0, _bucket("sampler"))
    ctx = StepContext(request_ids=(1, 2), graph_walk=WALK, slot=0, capture=False, slot_lease=lease, lean=GREEDY)
    with pytest.raises(RuntimeError, match="don't sample greedily"):
        wide.plan(SamplerStep(), ctx)
    # eager too
    ctx = StepContext(request_ids=(1, 2), graph_walk=WALK, slot=0, capture=False, lean=GREEDY)
    with pytest.raises(RuntimeError, match=r"greedy step \(eager\)"):
        wide.plan(SamplerStep(), ctx)
    # a capture's rows are padding rows, whatever their config
    ctx = StepContext(request_ids=(1, 2), graph_walk=WALK, slot=0, capture=True, slot_lease=lease, lean=GREEDY)
    wide.plan(SamplerStep(), ctx)
    assert wide._greedy_this_step
    ctx = StepContext(request_ids=(1, 2), graph_walk=WALK, slot=0, capture=False, slot_lease=SlotLease(0, _bucket()))
    wide.plan(SamplerStep(), ctx)
    assert not wide._greedy_this_step


def test_an_eager_greedy_step_takes_the_argmax_and_moves_offsets_as_a_full_one(monkeypatch):
    """With no capture to lease, the engine runs the sampler lean all the same: each row
    takes its argmax, without a full draw, and its offset moves as a draw's would."""
    sampler = _sampler(V, greedy_capture=True)
    full = Mock(side_effect=AssertionError("drew in full"))
    monkeypatch.setattr(sampler_utils, "sample_tokens", full)
    for rid in (1, 2):
        sampler.ingest_request(rid, _cfg(0.0))
    sampler.ingest_request(3, _cfg(0.7))
    engine = make_engine(_submodules={"LLM": make_mgmt(
        SimpleNamespace(cg_key_info=lambda walk, info, **kw: None), resources={"sampler": sampler},
    )})
    assert _leased(engine, [1, 2]) == (None, GREEDY) and _leased(engine, [1, 3]) == (None, frozenset())

    ctx = StepContext(request_ids=(1, 2), graph_walk=WALK, slot=0, capture=False, lean=GREEDY)
    sampler.plan(SamplerStep(), ctx)
    logits = torch.randn(2, V, generator=torch.Generator().manual_seed(0))
    assert torch.equal(sampler.sample([1, 2], logits), logits.argmax(-1))
    assert [sampler._sampler._step_offset[rid] for rid in (1, 2, 3)] == [STRIDE, STRIDE, 0]
    sampler.commit(SamplerStep(), ctx)
    # with the sampled row in it, the full draw
    ctx = StepContext(request_ids=(1, 3), graph_walk=WALK, slot=0, capture=False)
    sampler.plan(SamplerStep(), ctx)
    with pytest.raises(AssertionError, match="drew in full"):
        sampler.sample([1, 3], logits)


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
        self.sampler = _sampler()
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
        greedy = self.sampler.lean_for(rids)
        padded = tuple(rids) + tuple(-(i + 1) for i in range(bs - len(rids)))
        bucket = BucketKey(graph_walk=WALK, bs=bs, num_tokens=bs, lean=GREEDY if greedy else frozenset())
        lease = SlotLease(slot=slot, bucket=bucket)
        segments = tuple(Segment(r, "main", 1) for r in padded)
        smp = SamplerStep()
        step = SubmoduleStep(steps={"sampler": smp}, segments=list(segments))
        ctx = StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=slot, capture=False, slot_lease=lease,
                          lean=bucket.lean)
        ctx.set_padded_rids(padded)
        key = (bucket, slot)
        if key not in self.captured:
            for rid in padded[len(rids):]:
                if rid not in self.sampler._sampler._sampling_config:
                    self.sampler.ingest_capture_row(rid)
            cap = StepContext(request_ids=padded, graph_walk=WALK, slot=slot, capture=True, slot_lease=lease,
                              lean=bucket.lean)
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

"""The transducer's decode loop through the engine's recurrent state pool,
driven by hand the way the engine drives a node: prepare_inputs ->
declare_step -> admit -> plan -> preprocess -> forward_batched -> commit ->
check_stop, per step.

Requests of 1, 5, 17 and 30 frames join and leave at different iterations, in
both frame buckets (small ones here), in eager steps and in steps padded to a
capture bucket as a replay is. A stopped request rides along in one more
iteration, as the async worker's speculated step does, and is dropped. Each
request's tokens must equal the one-request reference loop's and those of
#796's in-forward decoder (a frozen copy below). On CUDA a captured graph,
replayed through the engine's runner, must emit what the eager steps do.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from test_transducer import _head, _reference

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_runner import dummy_rid_handle
from mstar.engine.fp32_precision import Fp32Precision, fp32_precision_scope, node_fp32_precision
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext
from mstar.model.components.fastconformer import FastConformerConfig
from mstar.model.parakeet.config import (
    DECODE_LOOP,
    DECODE_WALK,
    NUM_FRAMES,
    PREFILL_WALK,
    TRANSDUCER_STATE,
    ParakeetModelConfig,
)
from mstar.model.parakeet.parakeet_model import ParakeetModel
from mstar.model.parakeet.submodules import TransducerSubmodule, _batch_sizes
from mstar.model.submodule_base import BatchedModelOutput, HostRows, ModelInputsFromEngine

BUCKETS = (8, 32)
# (iteration the request joins at, its frames): 1 and 5 in the narrow bucket, 17 and 30 in the wide one
ARRIVALS = ((0, 17), (0, 1), (1, 30), (2, 5), (2, 30), (3, 1), (3, 17), (5, 5), (6, 30), (8, 1))
# 5 usable, fewer than the requests: slots are freed and handed out again
MAX_SLOTS = 6


@pytest.fixture(autouse=True)
def _one_thread():
    # token lists compare exactly, and CPU GEMMs can round differently with
    # the intra-op thread count
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# -- #796's decoder at bf9f0676c, frozen -------------------------------------
# A deliberate copy of the in-forward greedy decoder the loop replaced
# (TransducerHead.start / step / advance / greedy_decode at bf9f0676c), kept as
# an oracle: the loop must emit exactly its tokens. It calls only the head's
# layers. Don't update it to follow the live code.

def _frozen_start(head, frames, lengths):
    cfg = head.config
    b = frames.shape[0]
    start = torch.full((b,), cfg.start_token_id, dtype=torch.long, device=frames.device)
    zeros = frames.new_zeros(b, cfg.num_layers, cfg.hidden_size)
    dec_out, h, c = head.decoder(start, zeros, zeros)
    return {
        "frames": frames, "lengths": lengths, "cursor": torch.zeros_like(lengths),
        "symbols": torch.zeros_like(lengths), "active": lengths > 0, "dec_out": dec_out, "h": h, "c": c,
    }


def _frozen_step(head, s):
    cfg = head.config
    frames, cursor, symbols, active, lengths = s["frames"], s["cursor"], s["symbols"], s["active"], s["lengths"]
    rows, t, hidden = frames.shape
    width = head.window
    offsets = torch.arange(width, device=frames.device)
    at = cursor[:, None] + offsets
    valid = at < lengths[:, None]
    window = frames.gather(1, at.clamp(max=t - 1)[:, :, None].expand(-1, -1, hidden))
    logits = head.joint.head(F.relu(window + s["dec_out"][:, None]))
    tokens = logits[..., :cfg.vocab_size].argmax(dim=-1)
    blank = tokens == cfg.blank_token_id
    if cfg.durations:
        skips = head.durations[logits[..., cfg.vocab_size:].argmax(dim=-1)]
    else:
        skips = torch.zeros_like(tokens)
    reach = max(cfg.durations, default=1) or 1
    nxt = torch.where(blank & valid, offsets + skips.clamp(min=1), offsets)
    nxt = torch.cat([nxt, torch.arange(width, width + reach, device=frames.device).expand(rows, -1)], dim=1)
    for _ in range(math.ceil(math.log2(width))):
        nxt = nxt.gather(1, nxt)
    first = nxt[:, 0]
    at_first = first.clamp(max=width - 1)[:, None]
    emit = active & (first < width) & (valid & ~blank).gather(1, at_first)[:, 0]
    token = tokens.gather(1, at_first)[:, 0]
    skip = skips.gather(1, at_first)[:, 0]
    count = torch.where(first > 0, 0, symbols)
    count = torch.where(emit & (skip == 0), count + 1, 0)
    forced = count >= cfg.max_symbols_per_step
    skip = skip.masked_fill(forced, 1)
    symbols.copy_(count.masked_fill(forced, 0))
    new_out, h, c = head.decoder(token, s["h"], s["c"])
    s["dec_out"].copy_(torch.where(emit[:, None], new_out, s["dec_out"]))
    s["h"].copy_(torch.where(emit[:, None, None], h, s["h"]))
    s["c"].copy_(torch.where(emit[:, None, None], c, s["c"]))
    cursor.add_(torch.where(active, torch.where(emit, first + skip, first), 0))
    active.copy_(cursor < lengths)
    return torch.where(emit, token, -1)


@torch.no_grad()
def _frozen_greedy_decode(head, encoder_states, lengths, prompt_ids=None):
    with fp32_precision_scope(Fp32Precision("highest", None)):
        frames = head.project_frames(encoder_states, prompt_ids)
        state = _frozen_start(head, frames, lengths)
        chunks = []
        for _ in range(math.ceil(head.config.max_symbols_per_step * encoder_states.shape[1] / 16)):
            tokens = [_frozen_step(head, state) for _ in range(16)]
            chunks.append(torch.stack(tokens, dim=1))
            if not bool(state["active"].any()):
                break
    rows = torch.cat(chunks, dim=1).tolist()
    return [[tok for tok in row if tok >= 0] for row in rows]


# -- heads and clips -------------------------------------------------------------

def _random(durations, arrivals=ARRIVALS):
    head = _head(durations, seed=6)
    generator = torch.Generator().manual_seed(7)
    return head, [torch.randn(n, 24, generator=generator) for _, n in arrivals]


def _never_blank(durations):
    """Every frame emits until NeMo's guard moves the cursor on: max_symbols
    tokens per frame, through TDT's duration 0 as well as RNN-T."""
    head, clips = _random(durations)
    cfg = head.config
    with torch.no_grad():
        head.joint.head.bias[cfg.blank_token_id] = -1e4
        if cfg.durations:
            head.joint.head.bias[cfg.vocab_size] = 1e4  # duration 0
    return head, clips


def _blank_runs(durations):
    """Blank (TDT: for one frame) wherever hidden unit 0 is quiet and token 4
    (TDT: for zero frames) where it is loud: emissions at a window's edges,
    after blank runs longer than a window, and at a clip's last frame."""
    head = _head(durations, seed=9)
    cfg = head.config
    with torch.no_grad():
        head.joint.head.bias[:cfg.vocab_size] = -1e4
        head.joint.head.bias[cfg.blank_token_id] = 0.0
        head.joint.head.weight[4, 0] = 1e3
        if cfg.durations:
            head.joint.head.bias[cfg.vocab_size + 1] = 50.0
            head.joint.head.weight[cfg.vocab_size, 0] = 1e3
        head.encoder_projector.weight.zero_()
        head.encoder_projector.bias.zero_()
        head.encoder_projector.weight[0, 0] = 1.0
    loud = {1: [], 5: [4], 17: [7, 8, 9, 10, 16], 30: [0, 3, 4, 11, 22, 29]}
    clips = []
    for _, n in ARRIVALS:
        clip = torch.zeros(n, 24)
        clip[loud[n], 0] = 100.0
        clips.append(clip)
    return head, clips


SCENARIOS = {"random": _random, "never_blank": _never_blank, "blank_runs": _blank_runs}


# -- the node, its pool and a step runner ------------------------------------------

def _info(rid: int, walk: str, num_frames: int, iterations: int = 0) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=f"r{rid}", graph_walk=walk, fwd_index=0, random_seed=0, max_tokens=2048, rid_handle=rid,
        step_metadata={NUM_FRAMES: num_frames}, dynamic_loop_iter_counts={DECODE_LOOP: iterations},
    )


class Harness:
    """What the engine does for one node: the model's pool declaration with
    a YAML's max_slots, a step runner over it, and the submodule."""

    def __init__(self, head, device: str = "cpu", compile_step: bool = False, max_slots: int = MAX_SLOTS):
        model = object.__new__(ParakeetModel)
        model.config = ParakeetModelConfig("parakeet_tdt", FastConformerConfig(), head.config)
        (spec,) = model.get_node_resources()
        spec.apply_yaml_overrides(max_slots=max_slots)
        self.device = torch.device(device)
        self.pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=self.device))
        self.resources = {TRANSDUCER_STATE: self.pool}
        self.runner = StepRunner(self.resources, node_resources={"transducer": [TRANSDUCER_STATE]})
        self.node = TransducerSubmodule(head, compile_step=compile_step, frame_buckets=BUCKETS)
        # as the engine does at load
        self.node.requires_grad_(False)
        self.node.bind_node_resources(self.resources)
        self.configs = {
            cfg.additional_key_info: cfg for cfg in self.node.get_accelerator_graph_configs(self.device)
        }
        self.precision = node_fp32_precision("transducer", self.node)
        self.frames: dict[int, torch.Tensor] = {}
        self.iterations: dict[int, int] = {}
        # the replay slot the next step leases, rotated as the engine does
        self.slot = 0

    def _drive(self, walk, inputs, infos, ctx: StepContext, forward=None) -> BatchedModelOutput:
        rids = list(ctx.padded_request_ids)
        step = self.node.declare_step(walk, rids, inputs, slot_lease=ctx.slot_lease, per_request_info=infos)
        step.set_ctx(ctx)
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        engine_inputs = ModelInputsFromEngine(
            request_ids=rids, per_request_info=infos, resources=self.resources,
            captured=ctx.slot_lease is not None, step=step,
        )
        preprocessed = self.node.preprocess(walk, engine_inputs, inputs)
        with fp32_precision_scope(self.precision):
            if forward is None:
                out = self.node.forward_batched(walk, engine_inputs=engine_inputs, **preprocessed)
            else:
                out = forward(preprocessed)
        self.runner.commit(step)
        return BatchedModelOutput.coerce(out)

    def join(self, rids: list[int], clips: list[torch.Tensor]) -> None:
        """Open the requests and run their prefill step, as one batch."""
        for rid in rids:
            self.runner.ingest_request(rid)
        infos = {rid: _info(rid, PREFILL_WALK, clip.shape[0]) for rid, clip in zip(rids, clips, strict=True)}
        inputs = [
            self.node.prepare_inputs(PREFILL_WALK, infos[rid], {"encoder_states": [clip.to(self.device)]})
            for rid, clip in zip(rids, clips, strict=True)
        ]
        ctx = StepContext(request_ids=list(rids), graph_walk=PREFILL_WALK, slot=0, capture=False)
        out = self._drive(PREFILL_WALK, inputs, infos, ctx)
        for rid in rids:
            assert self.node.check_stop(rid, infos[rid], out.per_rid_outputs[rid]) == set()
            frames = self.frames[rid] = out.per_rid_outputs[rid]["frames"][0]
            assert frames.is_contiguous() and frames.shape == (infos[rid].step_metadata[NUM_FRAMES], 16)
            self.iterations[rid] = 0

    def _decode_inputs(self, rids):
        infos = {
            rid: _info(rid, DECODE_WALK, self.frames[rid].shape[0], self.iterations[rid]) for rid in rids
        }
        inputs = [self.node.prepare_inputs(DECODE_WALK, infos[rid], {"frames": [self.frames[rid]]}) for rid in rids]
        return infos, inputs

    def iterate(self, rids: list[int], padded_to: int | None = None, runner=None) -> tuple[dict, set]:
        """One decode iteration of one capture group: eager, padded to a
        capture's ``padded_to`` rows as a replay is (``forward_batched`` run
        directly), or replayed by ``runner``. -> each request's tokens, and the
        requests check_stop ends."""
        infos, inputs = self._decode_inputs(rids)
        bucket = self.node.cg_key_info(DECODE_WALK, infos)
        assert bucket == inputs[0].resource_step_info
        if runner is not None:
            out = self._replay(runner, rids, infos, inputs, bucket)
        elif padded_to is not None:
            lease = SlotLease(slot=0, bucket=BucketKey(DECODE_WALK, padded_to, padded_to, bucket))
            padding = [dummy_rid_handle(f"__cg_transducer_{bucket}_slot0_{i}__") for i in range(len(rids), padded_to)]
            inputs = [*inputs, *self.configs[bucket].get_node_inputs(len(padding), 0)]
            ctx = StepContext(request_ids=list(rids), graph_walk=DECODE_WALK, slot=0, capture=False, slot_lease=lease)
            ctx.set_padded_rids([*rids, *padding])
            out = self._drive(DECODE_WALK, inputs, infos, ctx)
        else:
            ctx = StepContext(request_ids=list(rids), graph_walk=DECODE_WALK, slot=0, capture=False)
            out = self._drive(DECODE_WALK, inputs, infos, ctx)
        tokens = out.row_outputs["tokens"][:len(rids)].cpu()
        host = HostRows(request_ids=tuple(rids), buffers={"active": out.check_stop_buffers["active"].cpu()})
        stops = self.node.check_stop_batched(rids, infos, host, step_walk=DECODE_WALK)
        assert all(loops == {DECODE_LOOP} for loops in stops.values())
        for rid in rids:
            self.iterations[rid] += 1
        return dict(zip(rids, tokens.tolist(), strict=True)), set(stops)

    def _replay(self, cg, rids, infos, inputs, bucket):
        """The engine's ``_exec_single`` under a lease: pad, declare, admit, plan,
        preprocess, stage and replay, commit, release."""
        lease = cg.lease_slot(DECODE_WALK, bs=len(rids), slot=self.slot, num_tokens=len(rids), cg_key_info=bucket)
        self.slot += 1
        assert lease is not None, f"no capture for {len(rids)} rows of bucket {bucket}"
        padded = cg.pad_inputs(lease, inputs)
        ctx = StepContext(request_ids=list(rids), graph_walk=DECODE_WALK, slot=lease.slot, capture=False,
                          slot_lease=lease)
        ctx.set_padded_rids(cg.step_ids(lease, list(rids)))
        try:
            return self._drive(
                DECODE_WALK, padded, cg.step_metadata(lease, list(rids), infos), ctx,
                forward=lambda preprocessed: cg.run_forward(lease, preprocessed),
            )
        finally:
            cg.release(lease, len(rids))

    def leave(self, rid: int) -> None:
        self.runner.remove_request(rid)
        self.node.cleanup_request(rid)
        del self.frames[rid], self.iterations[rid]


def _run(harness: Harness, clips, mode, runner=None, arrivals=ARRIVALS) -> dict[int, list[int]]:
    """Every request of ``arrivals`` through prefill and its decode loop. A
    request joins when a slot is free (the conductor's queue), each
    iteration runs one step per frame bucket, and a request check_stop ends
    runs one more iteration whose tokens are dropped."""
    out: dict[int, list[int]] = {}
    waiting = list(range(len(arrivals)))
    live: list[int] = []
    stopped: set[int] = set()
    for iteration in range(200):
        joining = []
        for rid in list(waiting):
            if arrivals[rid][0] <= iteration and len(live) + len(joining) < harness.pool.config.usable_slots:
                waiting.remove(rid)
                joining.append(rid)
        if joining:
            harness.join(joining, [clips[rid] for rid in joining])
            live += joining
            out.update({rid: [] for rid in joining})
        if not live and not waiting:
            break
        groups: dict[int | None, list[int]] = {}
        for rid in live:
            groups.setdefault(harness.node._bucket(harness.frames[rid].shape[0]), []).append(rid)
        newly_stopped = set()
        for group, (bucket, rids) in enumerate(sorted(groups.items())):
            padded_to = None
            if mode == "replay" or (mode == "mixed" and (iteration + group) % 2 == 0):
                padded_to = next(bs for bs in _batch_sizes(bucket) if bs >= len(rids))
            tokens, stops = harness.iterate(rids, padded_to=padded_to, runner=runner)
            for rid in rids:
                if rid in stopped:
                    # the speculated iteration after the stop: dropped, and it emits nothing
                    assert all(token == -1 for token in tokens[rid])
                else:
                    out[rid] += [token for token in tokens[rid] if token >= 0]
            newly_stopped |= stops - stopped
        for rid in stopped:
            live.remove(rid)
            harness.leave(rid)
        stopped = newly_stopped
    assert not live and not waiting, "the loop did not finish"
    assert harness.pool.num_free_slots == harness.pool.config.usable_slots
    return out


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("durations", [[0, 1, 2, 3, 4], []], ids=["tdt", "rnnt"])
@pytest.mark.parametrize("mode", ["eager", "replay", "mixed"])
def test_loop_matches_the_reference_and_the_frozen_decoder(scenario, durations, mode):
    head, clips = SCENARIOS[scenario](durations)
    with torch.no_grad():
        got = _run(Harness(head), clips, mode)
        lengths = [clip.shape[0] for clip in clips]
        padded = torch.zeros(len(clips), max(lengths), 24)
        for i, clip in enumerate(clips):
            padded[i, :clip.shape[0]] = clip
        frozen = _frozen_greedy_decode(head, padded, torch.tensor(lengths))
        expected = [_reference(head, clip) for clip in clips]
    for rid in range(len(clips)):
        assert got[rid] == expected[rid] == frozen[rid], (rid, lengths[rid])
    assert sum(map(len, expected)) > len(clips), "the head emitted almost nothing; the test would be vacuous"
    if scenario == "never_blank":
        assert [len(tokens) for tokens in expected] == [head.config.max_symbols_per_step * n for n in lengths]


def test_loop_caps_and_stop_check_agree():
    """The loop cap is where the stop check's guard would end the loop: past
    the most steps a clip can take, so active ends it first."""
    head, clips = _never_blank([])
    cfg = head.config
    for n in (1, 5, 17, 30):
        steps = cfg.max_symbols_per_step * n  # every frame at the guard, the most it takes
        assert (cfg.decode_iterations(n) - 1) * 16 < steps <= cfg.decode_iterations(n) * 16
    model = object.__new__(ParakeetModel)
    model.config = ParakeetModelConfig("parakeet_rnnt", FastConformerConfig(), cfg)
    assert model.get_loop_caps(**{NUM_FRAMES: 30}) == {DECODE_LOOP: cfg.decode_iterations(30)}
    assert model.get_loop_caps() == {}
    node = TransducerSubmodule(head, compile_step=False, frame_buckets=BUCKETS)
    info = _info(0, DECODE_WALK, 30, iterations=cfg.decode_iterations(30) - 1)
    assert node.check_stop(0, info, {"clock": [torch.tensor([True])]}) == {DECODE_LOOP}
    info = _info(0, DECODE_WALK, 30, iterations=cfg.decode_iterations(30) - 2)
    assert node.check_stop(0, info, {"clock": [torch.tensor([True])]}) == set()
    assert node.check_stop(0, info, {"clock": [torch.tensor([False])]}) == {DECODE_LOOP}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="captures CUDA graphs")
@pytest.mark.parametrize("durations", [[0, 1, 2, 3, 4], []], ids=["tdt", "rnnt"])
@pytest.mark.parametrize("bs", [1, 3, 64])
def test_captured_iterations_match_eager(durations, bs):
    """The engine's runner captures one graph per frame bucket and batch size
    over the pool's capture addressing and replays it for batches of
    different requests: the same tokens as eager steps of the compiled step.
    ``bs`` requests of the wide bucket start together; two narrow ones join
    later."""
    import gc

    from mstar.distributed.communication import CommGroup, JointGroups
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner

    arrivals = tuple((0, 9 + 7 * i % 22) for i in range(bs)) + ((2, 5), (3, 1))
    head, clips = _random(durations, arrivals)
    head = head.cuda()
    # an earlier test's graphs freed by the collector mid-capture would void it
    gc.collect()
    with torch.no_grad():
        eager = _run(Harness(head, "cuda", compile_step=True, max_slots=bs + 3), clips, "eager", arrivals=arrivals)
        harness = Harness(head, "cuda", compile_step=True, max_slots=bs + 3)
        assert harness.node._step is not head.step, "the captures should replay the compiled step"
        cg = AcceleratorGraphRunner(
            submodule_name="transducer", submodule=harness.node, resources=harness.resources,
            step_runner=harness.runner, device=torch.device("cuda", 0), autocast_dtype=None,
            joint_comm_group=JointGroups(CommGroup.trivial(), CommGroup.trivial()), num_slots=2,
        )
        with fp32_precision_scope(harness.precision):
            cg.warmup_and_capture(None)
        assert not cg.dropped_buckets
        captured = _run(harness, clips, "replay", runner=cg, arrivals=arrivals)
    assert captured == eager
    assert sum(map(len, eager.values())) > len(eager), "the random head emitted nothing; the test would be vacuous"

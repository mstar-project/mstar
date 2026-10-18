"""CUDA graph capture leaves the largest eager step free.

Each runner used to capture every bucket it could, and its graph pool kept
whatever the captures took. Qwen3-Omni colocated on one H100 ended
capture with 3 MiB free, and every text request then failed with CUDA OOM in
the Thinker's prefill: a dropped bucket runs eager, and nothing had kept room
for it. The floor is the largest eager step of any runner on the device,
measured before any of them captures.
"""

from __future__ import annotations

import ctypes
import inspect
import sys
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.distributed.ar_fusion import GraphCaptureScope
from mstar.engine import accelerator_graph_runner
from mstar.engine.accelerator_graph_backend import AcceleratorGraphPool
from mstar.engine.accelerator_graph_runner import (
    AcceleratorGraphRunner,
    CaptureCost,
    ExecutableMemory,
    GraphMemory,
    WarmedSpec,
    plan_captures,
)
from mstar.engine.engine import Engine
from mstar.engine.resources import BucketKey, CGSlotSpec, SubmoduleStep

_GIB = 2**30
_MIB = 2**20

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="a throwaway capture needs a GPU")


def _bucket(walk: str, bs: int, tokens: int) -> BucketKey:
    return BucketKey(graph_walk=walk, bs=bs, num_tokens=tokens)


def _sized(costs: dict) -> dict:
    """``costs`` without each walk's measurement, which only the plan reads."""
    return {bucket: cost._replace(executable=None) for bucket, cost in costs.items()}


_DECODE = _bucket("thinker_decode", 32, 32)
_PREFILL = _bucket("prefill_vision", 1, 16384)


class _StubMeasuredRunner:
    """`size_captures` over stub warm-ups of ``steps``, two slots each, in capture order."""

    size_captures = AcceleratorGraphRunner.size_captures
    size_lent_captures = AcceleratorGraphRunner.size_lent_captures
    NUM_WARMUP = AcceleratorGraphRunner.NUM_WARMUP

    def __init__(self, device: torch.device, steps: dict[BucketKey, CaptureCost]):
        self._steps = steps
        self._device = device
        self._autocast_dtype = None
        self._submodule_name = "Thinker"
        self._num_slots = 2
        self._capture_costs = {}
        self._lent = set()
        self._executables = {}
        group = SimpleNamespace(world_size=1, barrier=lambda: None)
        self._comm_group = SimpleNamespace(tp_group=group, sp_group=group)
        self.warmed: list[tuple[BucketKey, int]] = []
        # buckets a resource of the node cannot run in a graph
        self.refuse: set[BucketKey] = set()
        self.forwards: list[int] = []

    def prepare_for_capture(self):
        return [CGSlotSpec(bucket=b, slot=s, config=SimpleNamespace()) for b in self._steps for s in (0, 1)]

    def _capture_refusal(self, spec):
        return "no capturable kernel" if spec.bucket in self.refuse else None

    @contextmanager
    def _warmed(self, spec, forwards):
        self.warmed.append((spec.bucket, spec.slot))
        self.forwards.append(forwards)
        step = self._steps[spec.bucket]
        yield WarmedSpec(
            run=lambda: step, static_inputs={}, static_input_keys=(), dummy_rids=[], dummy_metadata={},
            peak=step.peak, returned=step.kept,
        )


@pytest.fixture
def measured(monkeypatch):
    """Each sized bucket's cost, with the throwaway capture stubbed out."""
    runs = []

    def capture(capture, what, executable):
        _, cost = capture(pool=None)
        runs.append(cost)
        return None if cost.graph is None else GraphMemory(cost.graph, cost.outside)

    monkeypatch.setattr(accelerator_graph_runner, "_capture_thrown_away", capture)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (50 * _GIB, 80 * _GIB))
    monkeypatch.setattr(
        accelerator_graph_runner, "capture_into_graph", lambda run, pool, device, autocast_dtype: (None, run()),
    )
    return runs


def test_every_bucket_is_measured_once_before_any_capture(measured):
    """The first spec in capture order is a 32-row decode; the one-row 16k-token
    prefill behind it peaks ten times higher. Measuring only the first would
    set the floor far too low."""
    steps = {_DECODE: CaptureCost(_GIB // 10, _GIB // 5, 0, 2), _PREFILL: CaptureCost(_GIB, 2 * _GIB, 0, 2)}

    kept = _StubMeasuredRunner(torch.device("cuda", 0), steps).size_captures()

    assert measured == list(steps.values()), "each bucket is measured once, in capture order"
    assert _sized(kept) == steps, "every bucket's step must be kept to size its capture"


def test_a_bucket_a_resource_refuses_is_warmed_but_never_captured_to_size_it(measured):
    """It runs eagerly, so its warm-up peak still sets the floor. It is not
    captured to size it, and with no graph the plan never takes it."""
    steps = {_DECODE: CaptureCost(_GIB // 10, _GIB // 5, 0, 2), _PREFILL: CaptureCost(_GIB, 2 * _GIB, 0, 2)}
    runner = _StubMeasuredRunner(torch.device("cuda", 0), steps)
    runner.refuse = {_PREFILL}

    costs = _sized(runner.size_captures())

    assert measured == [steps[_DECODE]], "a refused bucket must not be captured to size it"
    assert runner.warmed.count((_PREFILL, 0)) == runner.warmed.count((_PREFILL, 1)) == 1
    assert costs[_PREFILL].graph is None and costs[_PREFILL].peak == _GIB
    plan = plan_captures({"Thinker": costs}, free=8 * _GIB)
    assert plan.buckets["Thinker"] == {_DECODE} and plan.floor == _GIB


def test_a_walks_two_largest_buckets_are_captured_to_size_the_rest(measured):
    """Bagel's 29 buckets were each captured once to be sized, and thrown away.
    A walk runs one forward at every shape, so in the pool they share the
    smaller graphs fit in the scratch the largest leaves free: 132 MiB for the
    64-row decode, where the 4-row one took 54 MiB of a pool of its own."""
    walk = {
        _bucket("decode", 64, 64): CaptureCost(93 * _MIB, 133 * _MIB, _MIB, 2, 4 * _MIB),
        _bucket("decode", 32, 32): CaptureCost(46 * _MIB, 94 * _MIB, _MIB // 2, 2, 4 * _MIB),
        _bucket("decode", 8, 8): CaptureCost(13 * _MIB, 22 * _MIB, _MIB // 8, 2),
        _bucket("decode", 4, 4): CaptureCost(7 * _MIB, 54 * _MIB, _MIB // 16, 2),
    }
    largest, second, *smaller = walk

    costs = _sized(_StubMeasuredRunner(torch.device("cuda", 0), walk).size_captures())

    assert measured == [walk[largest], walk[second]], "only the walk's two largest buckets are captured to size it"
    for bucket in smaller:
        assert costs[bucket] == walk[bucket]._replace(graph=132 * _MIB + walk[bucket].kept, outside=4 * _MIB), (
            "a smaller bucket takes the largest's scratch and executable memory, and adds its own outputs"
        )


def test_a_walk_whose_second_bucket_takes_more_than_its_largest_is_captured_whole(measured):
    """Chatterbox's CFG decode takes 24 MiB at 32 rows and 56 MiB at 16, each
    in a pool of its own. Sized by the largest, the walk would be planned at
    24 MiB."""
    walk = {
        _bucket("decode", bs, 2 * bs): CaptureCost(8 * bs * _MIB // 32, graph * _MIB, 0, 2)
        for bs, graph in ((32, 24), (16, 56), (8, 34), (4, 34))
    }

    costs = _sized(_StubMeasuredRunner(torch.device("cuda", 0), walk).size_captures())

    assert measured == list(walk.values())
    assert costs == walk, "a walk that doesn't nest keeps each bucket's own size"


def test_a_bucket_that_peaks_above_its_walks_first_is_captured_too(measured):
    """Capture order leads with batch size, not with the eager peak: Bagel's
    prefill of 4 rows comes first and peaks at 278.0 MiB, and the 1-row prefill
    of the same 2048 tokens, far down the order, peaks at 278.8."""
    walk = {
        _bucket("prefill_text", 4, 2048): CaptureCost(2780 * _MIB // 10, 336 * _MIB, 0, 2),
        _bucket("prefill_text", 4, 1024): CaptureCost(139 * _MIB, 206 * _MIB, 0, 2),
        _bucket("prefill_text", 4, 512): CaptureCost(70 * _MIB, 132 * _MIB, 0, 2),
        _bucket("prefill_text", 1, 2048): CaptureCost(2788 * _MIB // 10, 340 * _MIB, 0, 2),
        _bucket("prefill_text", 1, 1024): CaptureCost(140 * _MIB, 206 * _MIB, 0, 2),
    }
    first, second, between, higher, after = walk

    costs = _StubMeasuredRunner(torch.device("cuda", 0), walk).size_captures()

    assert measured == [walk[first], walk[second], walk[higher]], "a bucket that peaks above every one before it"
    assert costs[between].graph == walk[first].graph
    assert costs[after].graph == walk[higher].graph, "the bucket with the largest peak sizes the ones after it"


def test_a_walk_whose_largest_cannot_be_captured_is_sized_by_the_next(measured):
    """A largest bucket that runs out of memory in its capture has no scratch
    to lend. The next ones down may still fit, and are captured to find out."""
    walk = {
        _bucket("prefill_vision", 1, 16384): CaptureCost(3 * _GIB, None, 0, 2),
        _bucket("prefill_vision", 1, 4096): CaptureCost(_GIB, _GIB, 0, 2),
        _bucket("prefill_vision", 1, 2048): CaptureCost(_GIB // 2, _GIB // 2, 0, 2),
        _bucket("prefill_vision", 1, 1024): CaptureCost(_GIB // 4, _GIB // 2, 0, 2),
    }
    *captured, smallest = walk

    costs = _StubMeasuredRunner(torch.device("cuda", 0), walk).size_captures()

    assert measured == [walk[bucket] for bucket in captured]
    assert costs[smallest].graph == _GIB, "the smallest is sized by the largest bucket that did capture"


@pytest.mark.parametrize(("room", "captured", "planned"), [(8192, 2, 6), (1024, 6, 3)], ids=["roomy", "short"])
def test_a_plan_that_leaves_a_graph_out_sizes_each_bucket_by_its_own_capture(
    measured, monkeypatch, room, captured, planned,
):
    """Qwen3-Omni's Code2Wav takes 7088 MiB at 32 rows, on a GPU with 2 GiB
    for graphs. Sized by it, every smaller bucket costs as much and stays out
    of the plan with it, where the 4, 2 and 1-row graphs fit in 799 MiB."""
    walk = {
        _bucket("code2wav_chunk", bs, 0): CaptureCost(graph * _MIB // 2, graph * _MIB, bs * _MIB // 8, 1)
        for bs, graph in ((32, 7088), (16, 3168), (8, 1552), (4, 794), (2, 390), (1, 176))
    }
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: ((3544 + room) * _MIB, 80 * _GIB))
    runner = _StubMeasuredRunner(torch.device("cuda", 0), walk)
    runner._num_slots = 1
    engine = SimpleNamespace(
        _device=torch.device("cuda", 0), _submodules={"Code2Wav": None}, _gpu_memory_fraction=None,
    )

    plan = Engine._plan_captures(engine, {"Code2Wav": runner}, {"Code2Wav": {}})

    assert len(measured) == captured, "two captures size the walk while every graph fits, and each its own when not"
    assert plan.buckets["Code2Wav"] == set(list(walk)[-planned:])
    assert runner.forwards[2 * len(walk):] == [0] * (captured - 2), "a later capture needs no second warm-up"


@pytest.mark.parametrize(
    ("wanted", "peer_wanted", "captured"), [(False, False, 2), (False, True, 4), (True, False, 4)],
    ids=["neither", "the_peer", "this_rank"],
)
def test_a_tp_group_sizes_its_buckets_again_together(measured, wanted, peer_wanted, captured):
    """Whether a plan leaves a graph out depends on the memory free on its own
    rank. A rank that sized again alone would plan on each bucket's own size
    while its peer still planned on the largest's."""
    walk = {
        _bucket("decode", bs, bs): CaptureCost(bs * _MIB, 2 * bs * _MIB, 0, 2) for bs in (16, 8, 4, 2)
    }
    runner = _StubMeasuredRunner(torch.device("cpu"), walk)
    runner._comm_group.tp_group = _PeerGroup([peer_wanted])
    runner.size_captures()

    costs = _sized(runner.size_lent_captures(wanted))

    assert len(measured) == captured, "a group's ranks capture their lender-sized buckets together or not at all"
    assert (costs == walk) == (captured == 4)


def test_a_regions_inputs_and_outputs_count_outside_its_pool(monkeypatch):
    """A region's graph copies its outputs into buffers outside the pool, and
    reads inputs from there, a set per graph. The pool a capture took is all
    scratch: a smaller bucket takes it whole and adds its own buffers."""
    monkeypatch.setattr(
        accelerator_graph_runner, "_capture_thrown_away", lambda capture, what, executable: GraphMemory(60 * _MIB, 0),
    )
    captured = []

    def size(peak: int, kept: int) -> tuple[CaptureCost, bool]:
        cost, lent = accelerator_graph_runner._size_in_walk(
            captured, CaptureCost(peak * _MIB, None, kept * _MIB, 1), False, None, "", ExecutableMemory(),
        )
        return cost._replace(executable=None), lent

    assert size(peak=8, kept=4) == (CaptureCost(8 * _MIB, 64 * _MIB, 4 * _MIB, 1), False)
    assert size(peak=4, kept=2) == (CaptureCost(4 * _MIB, 62 * _MIB, 2 * _MIB, 1), False)
    assert size(peak=2, kept=1) == (CaptureCost(2 * _MIB, 61 * _MIB, _MIB, 1), True)


def test_every_slot_is_warmed_before_any_capture(measured):
    """A slot's first plan builds state of its own: FlashInfer's graph wrappers,
    and a 512 MiB workspace for the Talker's second slot. Warmed at slot 0 only,
    that came after free memory was read, and the Talker's capture took 844 MiB
    against 98 predicted."""
    runner = _StubMeasuredRunner(torch.device("cuda", 0), {_DECODE: CaptureCost(_GIB, 2 * _GIB, 0, 2)})

    runner.size_captures()

    assert runner.warmed == [(_DECODE, 0), (_DECODE, 1)], "every slot must be warmed before the plan reads free memory"


def test_a_spec_is_warmed_up_to_size_it_and_not_again_to_capture_it(measured, monkeypatch):
    """Sizing runs every spec's forward twice. Capture ran each twice more
    before its real capture, as it had to when nothing ran before it: 5 s of
    the 10 s sizing added to Bagel's 66 s boot."""
    runner = _StubMeasuredRunner(torch.device("cuda", 0), {_DECODE: CaptureCost(_GIB, 2 * _GIB, 0, 2)})
    runner._memory_pool = None
    runner._build_slot_from_capture = lambda **slot: slot
    runner.size_captures()
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        accelerator_graph_runner, "capture_into_graph", lambda run, pool, device, autocast_dtype: (None, {}),
    )

    for spec in runner.prepare_for_capture():
        AcceleratorGraphRunner._capture_one(runner, spec, forwards=0)

    assert runner.forwards == [2, 2, 0, 0], "a spec sizing warmed up is captured without another forward"


def test_a_sizing_step_that_fails_fails_the_boot(measured):
    """Qwen3-Omni colocated on a crowded GPU ran out of memory in an eager
    warm-up, and every CUDA call after it failed. Leaving the failed steps out
    of the plan would bring the floor to 0."""
    runner = _StubMeasuredRunner(torch.device("cuda", 0), {_DECODE: CaptureCost(_GIB, 2 * _GIB, 0, 2)})

    def dead_context(spec, forwards):
        raise RuntimeError("CUDA error: unknown error")

    runner._warmed = dead_context
    engine = SimpleNamespace(_device=torch.device("cuda", 0), _submodules={"Thinker": None}, _gpu_memory_fraction=None)

    with pytest.raises(RuntimeError, match="slot=0 eagerly to size its capture, with 51200 MiB free before it: CUDA"):
        Engine._plan_captures(engine, {"Thinker": runner}, {"Thinker": {}})


def test_every_bucket_is_sized_whatever_the_pools_size(measured, monkeypatch):
    """Qwen3.5-0.8B with a 64-page KV pool (#728): sizing admitted each
    capture for real pages, the 8,192-token buckets did not fit, and the boot
    failed. A served prompt still pads into them, so each has to be sized for
    the floor and captured. Through the real warm-up, on a real 8-page pool."""
    from mstar.engine.accelerator_graph_config import decode_rows_then_prompt
    from mstar.engine.resources import Segment
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
    from mstar.engine.resources.kv.manager import KVManager

    monkeypatch.setattr(manager_mod, "KVTransferManager", lambda *args, **kwargs: None)
    kv = KVManager(
        cfg=PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=512, max_num_pages=8, page_size=4),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv._cg_max_seq_len = 40
    free = kv._arena.num_free
    # 3 pages, and 13: more than the pool's 7
    fits, wide = _bucket("mixed", 2, 8), _bucket("mixed", 4, 40)

    def declare_step(graph_walk, request_ids, inputs, slot_lease, per_request_info):
        spans = decode_rows_then_prompt(slot_lease.bucket.num_tokens, len(request_ids))
        return SubmoduleStep(
            segments=[], steps={"kv": KVStep(segments=tuple(
                Segment(rid, "main", span) for rid, span in zip(request_ids, spans, strict=True)
            ))},
        )

    def ensure(key, bs):
        rids = list(range(-bs, 0))
        for rid in rids:
            kv.ingest_capture_row(rid)
        return rids

    runner = _StubMeasuredRunner(torch.device("cuda", 0), {wide: _cost(100), fits: _cost(10)})
    runner.NUM_WARMUP = 0
    runner._warmed = partial(AcceleratorGraphRunner._warmed, runner)
    runner._forward_for = lambda spec: lambda **kwargs: _cost(100 if spec.bucket == wide else 10)
    runner._dummy_rows = SimpleNamespace(ensure=ensure, reset=lambda rids, free=False: None)
    runner._dummy_engine_inputs = lambda rids, walk: SimpleNamespace(per_request_info={})
    runner._submodule = SimpleNamespace(declare_step=declare_step, preprocess=lambda **kwargs: {})
    runner._graph_backend = SimpleNamespace(set_device=lambda: None, synchronize=lambda: None)
    runner._step_runner = SimpleNamespace(
        admit=lambda step: kv.admit(step.get("kv"), step.ctx),
        plan=lambda step: kv.plan(step.get("kv"), step.ctx),
    )
    config = SimpleNamespace(
        get_node_inputs=lambda bs, tokens: None, get_config_type=lambda: None,
        input_seq_dims=None, capture_forward_method="forward", compile=False,
    )
    runner.prepare_for_capture = lambda: [
        CGSlotSpec(bucket=b, slot=s, config=config) for b in (wide, fits) for s in (0, 1)
    ]
    engine = SimpleNamespace(_device=torch.device("cuda", 0), _submodules={"LLM": None}, _gpu_memory_fraction=None)

    plan = Engine._plan_captures(engine, {"LLM": runner}, {"LLM": {}})

    assert set(runner._capture_costs) == {wide, fits}, "every bucket must be sized, so the floor covers it"
    assert plan.buckets["LLM"] == {wide, fits}, "a bucket a served step pads into must be captured"
    assert kv._arena.num_free == free, "sizing must leave every page free"


def test_a_cpu_device_never_asks_the_driver(monkeypatch):
    """Asking the CUDA driver for memory fails an XPU or CPU worker at start. A
    CPU device skips the plan even on a host where CUDA exists."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    driver_calls = []
    for name in ("mem_get_info", "empty_cache", "graph_pool_handle", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, name=name, **kwargs: driver_calls.append(name))
    runner = _StubMeasuredRunner(torch.device("cpu"), {_DECODE: CaptureCost(_GIB, 2 * _GIB, 0, 2)})
    engine = SimpleNamespace(_device=torch.device("cpu"), _submodules={"Thinker": None})

    plan = Engine._plan_captures(engine, {"Thinker": runner}, {"Thinker": {}})

    assert plan is None
    assert driver_calls == [], "a CPU worker must not ask the CUDA driver for memory"


def _succeeds(x: torch.Tensor) -> torch.Tensor:
    return x @ x


def _runs_out_of_memory(x: torch.Tensor) -> torch.Tensor:
    x @ x
    return torch.empty(2 * torch.cuda.mem_get_info(x.device)[0], dtype=torch.uint8, device=x.device)


def _fails_in_capture_end(x: torch.Tensor) -> torch.Tensor:
    """A side stream never joined back: the capture runs, and capture_end refuses it."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        return x @ x


@requires_cuda
@pytest.mark.parametrize("forward", [_succeeds, _runs_out_of_memory, _fails_in_capture_end])
def test_a_throwaway_capture_frees_its_memory_pass_or_fail(forward, monkeypatch):
    """torch releases a graph's pool only for a capture that ended. One that
    failed in capture_end kept its pool, and the cuBLAS workspace the capture
    stream took in it, for good: memory every later bucket was sized without."""
    # a captured log record would keep the failed capture's frames, and its memory, alive
    monkeypatch.setattr(accelerator_graph_runner.logger, "disabled", True)
    device = torch.device("cuda")
    x = torch.randn(2048, 2048, device=device)
    # warmed up, as before any capture: cuBLAS can't make its handle inside one
    _succeeds(x)
    # a failed capture drops every stream's workspace, so start with none
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    reserved = torch.cuda.memory_reserved(device)

    taken = accelerator_graph_runner._capture_thrown_away(
        partial(accelerator_graph_runner.capture_into_graph, lambda: forward(x), device=device, autocast_dtype=None),
        forward.__name__, ExecutableMemory(),
    )

    assert (taken is None) == (forward is not _succeeds)
    assert torch.cuda.memory_reserved(device) == reserved, "a throwaway capture must hand back all it took"


@requires_cuda
def test_graphs_take_no_more_executable_memory_than_sizing_counts():
    """A graph's executable memory is outside the allocator, so no pool shows
    it. On Qwen3-Omni, 162 graphs held 724 MiB beyond their pools, and capture
    ended with less free memory than the largest eager step. The plan credits
    a walk's graphs with what its copies held, so once the copies are released
    the graphs must reuse at least that."""
    device = torch.device("cuda")
    x = torch.zeros(8, device=device)

    def many_kernels():
        y = x
        for _ in range(4096):
            y = y + 1
        return y

    many_kernels()
    capture = partial(accelerator_graph_runner.capture_into_graph, many_kernels, device=device, autocast_dtype=None)
    measured = ExecutableMemory()
    memory = accelerator_graph_runner._capture_thrown_away(capture, "4096 adds", measured)
    pool = AcceleratorGraphPool(torch.cuda.graph_pool_handle())
    torch.cuda.empty_cache()
    free, reserved = torch.cuda.mem_get_info()[0], torch.cuda.memory_reserved()

    # captured while the copies are held, so the graphs can't reuse their memory
    graphs = [capture(pool=pool) for _ in range(8)]
    executable = free - torch.cuda.mem_get_info()[0] - (torch.cuda.memory_reserved() - reserved)
    # captured again after the copies are released, as a runner captures
    measured.release()
    free, reserved = torch.cuda.mem_get_info()[0], torch.cuda.memory_reserved()
    graphs += [capture(pool=pool) for _ in range(8)]
    fresh = free - torch.cuda.mem_get_info()[0] - (torch.cuda.memory_reserved() - reserved)
    del graphs

    assert 0.75 * 8 * memory.outside <= executable <= 8 * memory.outside, (
        f"8 graphs took {executable / _MIB:.0f} MiB outside their pool, sizing counted {8 * memory.outside / _MIB:.0f}"
    )
    assert 0 < measured.held and fresh <= 8 * memory.outside - measured.held, (
        f"8 graphs took {fresh / _MIB:.0f} MiB of new memory, the plan expected "
        f"{(8 * memory.outside - measured.held) / _MIB:.0f}"
    )


_PIECE = 2 * _MIB
_COPY = 5300 * 2**10


class _StubDriver:
    """libcuda and free memory for copies of one graph: each copy takes
    ``per_copy`` bytes, first from what is left of the last piece the driver
    mapped, then from free memory in 2 MiB pieces. ``neighbor`` maps a copy
    to what another process takes from free memory while it is made."""

    def __init__(self, neighbor: dict[int, int], fails_at: int | None = None, per_copy: int = _COPY):
        self.per_copy = per_copy
        self.free = 40 * _GIB
        self.left = 3 * _MIB
        self.neighbor = neighbor
        self.fails_at = fails_at
        self.copies = 0
        self.destroyed = 0

    def cuGraphInstantiateWithFlags(self, copy, graph, flags):
        if self.copies + 1 == self.fails_at:
            return 2  # CUDA_ERROR_OUT_OF_MEMORY
        self.copies += 1
        pieces = max(-(-(self.per_copy - self.left) // _PIECE), 0)
        self.left += pieces * _PIECE - self.per_copy
        self.free -= pieces * _PIECE + self.neighbor.get(self.copies, 0)
        return 0

    def cuGraphExecDestroy(self, copy):
        self.destroyed += 1
        return 0


def _measured(monkeypatch, driver: _StubDriver) -> ExecutableMemory:
    """An ``ExecutableMemory`` measured on one-node copies through ``driver``."""
    monkeypatch.setattr(ctypes, "CDLL", lambda name: driver)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (driver.free, 80 * _GIB))
    executable = ExecutableMemory()
    executable.measure(SimpleNamespace(raw_cuda_graph=lambda: 1), nodes=1)
    return executable


@pytest.mark.parametrize(
    "neighbor", [{}, {8: -512 * _MIB}, {8: _GIB}], ids=["alone", "a_neighbor_frees", "a_neighbor_allocates"],
)
def test_a_graphs_executable_memory_is_measured_in_whole_pieces(monkeypatch, neighbor):
    """The first copies of a graph fill a piece the driver mapped before, so
    free memory drops by less than those copies take. And free memory is the
    device's: a worker freeing memory on the same GPU while a copy is made
    would make graphs look cheaper than they are, and a worker allocating
    would make them look costlier."""
    driver = _StubDriver(neighbor)

    executable = _measured(monkeypatch, driver)

    per_copy = executable.per_node
    assert _COPY <= per_copy <= _COPY + _PIECE, f"measured {per_copy / _MIB:.2f} MiB a copy, not {_COPY / _MIB:.2f}"
    assert 0 < executable.held <= driver.copies * _COPY, "the plan must not credit more than the copies hold"


def test_a_graph_too_small_to_need_a_piece_is_counted_at_most_one(monkeypatch):
    """512 copies of a graph of a few nodes fit in the piece the driver mapped
    before. Left unmeasured, the graph would never be captured."""
    driver = _StubDriver({}, per_copy=2 * 2**10)

    executable = _measured(monkeypatch, driver)

    assert executable.per_node >= 2 * 2**10, "a graph that fits in a piece must not be counted short"


def test_a_walks_copies_are_held_until_its_runner_captures(monkeypatch):
    """The driver keeps a destroyed graph's executable memory and maps new
    memory beside it. On an H100, a graph measured after an earlier graph's
    copies were destroyed read a third of its cost."""
    driver = _StubDriver({})
    runner = _StubPlannedRunner([_DECODE], peer=[True])
    runner._executables = {("thinker_decode", None): _measured(monkeypatch, driver)}
    assert driver.copies > 0 and driver.destroyed == 0, "the copies must be held while other walks are measured"

    runner.warmup_and_capture({_DECODE})

    assert driver.destroyed == driver.copies, "a runner that captures must hand its copies' memory to its graphs"


def _measured_walk(held: int) -> ExecutableMemory:
    executable = ExecutableMemory()
    executable.held = held
    return executable


def test_a_walks_graphs_reuse_the_memory_its_copies_held():
    """The driver keeps the memory a walk's measured copies held for graphs,
    and free memory doesn't rise when the copies are released. Counted again
    with each graph, that memory left colocated Qwen3-Omni with 1 GiB of
    ballast no room for any graph. Graphs of another walk reuse only part of
    that memory: on an H100, 4096-add graphs took back 12.6 of the 18 MiB that
    621-add copies held."""
    talker = _measured_walk(16 * _MIB)
    pools = {
        # each graph adds 8 MiB of executable memory, and the first its 4 MiB of scratch
        "Talker": {bs: CaptureCost(0, 4 * _MIB, 0, 1, 8 * _MIB, talker) for bs in range(6)},
        "Thinker": {0: CaptureCost(0, 2 * _MIB, 0, 1, 8 * _MIB)},
    }

    plan = plan_captures(pools, free=20 * _MIB)

    assert plan.buckets == {"Talker": {0, 1, 2, 3}, "Thinker": set()}, (
        "the Talker's copies must pay for two Talker graphs' executable memory, once, and none of the Thinker's"
    )


@pytest.mark.parametrize(
    ("fraction", "reserved", "planned"), [(None, 30, 5), (0.45, 30, 2), (0.45, 26, 5)],
    ids=["uncapped", "the_cap_binds", "free_memory_binds"],
)
def test_a_capped_worker_reuses_the_copies_only_within_its_cap(monkeypatch, fraction, reserved, planned):
    """The cap counts a worker's PyTorch memory and its graphs' executable
    memory, but not the copies. Where the cap binds, the copies' memory would
    be room beyond the cap."""
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (8 * _GIB, 80 * _GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: reserved * _GIB)
    talker = _measured_walk(2 * _GIB)
    # a 2 GiB eager step, and graphs of 1 GiB in their pool and 512 MiB outside it
    costs = {bs: CaptureCost(2 * _GIB, _GIB, _GIB, 1, _GIB // 2, talker) for bs in range(8)}
    runner = SimpleNamespace(size_captures=lambda: costs, size_lent_captures=lambda wanted: costs)
    engine = SimpleNamespace(
        _device=torch.device("cuda", 0), _submodules={"Talker": None}, _gpu_memory_fraction=fraction,
    )

    plan = Engine._plan_captures(engine, {"Talker": runner}, {"Talker": {}})

    assert len(plan.buckets["Talker"]) == planned, "the graphs must fit in free memory and under the cap"


def test_a_measurement_that_runs_out_of_memory_keeps_what_it_counted(monkeypatch):
    """A crowded GPU can fail a copy's instantiation. Sizing continues with
    the copies made before the failure."""
    driver = _StubDriver({}, fails_at=4)

    executable = _measured(monkeypatch, driver)
    executable.release()

    assert executable.per_node >= _COPY, "the copies made before the failure must not be counted short"
    assert driver.destroyed == driver.copies == 3, "every copy the measurement made must be destroyed"


def test_a_graph_whose_executable_memory_cannot_be_measured_is_left_unsized(monkeypatch):
    """Every copy can fail to instantiate on a crowded GPU. Counted as zero,
    the graph's executable memory would go unplanned."""
    monkeypatch.setattr(accelerator_graph_runner, "current_platform", SimpleNamespace(is_cuda=lambda: True))
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (0, 1))
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda *args, **kwargs: [])
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: None, raising=False)
    monkeypatch.setattr(accelerator_graph_runner, "_graph_nodes", lambda graph: 1400)
    monkeypatch.setattr(ExecutableMemory, "measure", lambda self, graph, nodes: None)

    memory = accelerator_graph_runner._capture_thrown_away(
        lambda pool, keep_graph: (object(), {}), "decode", ExecutableMemory(),
    )

    assert memory is None, "a graph whose executable memory went unmeasured must not be planned"


def test_a_throwaway_capture_reads_its_pool_as_the_installed_torch_takes_it(monkeypatch):
    """`memory_snapshot` takes ``include_traces`` only from torch 2.11, and
    pyproject allows 2.9. On 2.9 and 2.10 the call raised TypeError while
    sizing, so no graph was captured and the server ran eager."""
    signature = inspect.signature(torch.cuda.memory_snapshot)
    calls = []
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (0, 1))
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda *args, **kwargs: calls.append((args, kwargs)) or [])
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: None, raising=False)
    monkeypatch.setattr(accelerator_graph_runner, "_graph_nodes", lambda graph: 0)
    measured = ExecutableMemory()
    measured.per_node = 0.0

    accelerator_graph_runner._capture_thrown_away(lambda pool, keep_graph: (object(), {}), "decode", measured)

    (args, kwargs), = calls
    try:
        signature.bind(*args, **kwargs)
    except TypeError as error:
        pytest.fail(f"torch {torch.__version__} refuses the snapshot call, so every capture would fail: {error}")


def _cost(graph: int, kept: int = 0, peak: int = 0, slots: int = 1) -> CaptureCost:
    """A bucket's cost, in MiB."""
    return CaptureCost(peak=peak * _MIB, graph=graph * _MIB, kept=kept * _MIB, slots=slots)


def test_the_plan_takes_small_graphs_from_every_pool_first():
    """700 MiB beside a 1 GiB eager step: largest first, Code2Wav's bs=32 and
    bs=1 would take it all and leave the Talker none."""
    pools = {
        "Code2Wav": {"bs=32": _cost(600), "bs=1": _cost(100)},
        "Talker": {"bs=32": _cost(500, peak=1024), "bs=1": _cost(100)},
    }

    plan = plan_captures(pools, free=(1024 + 700) * _MIB)

    assert plan.floor == 1024 * _MIB, "the floor is the largest eager step in any pool"
    assert plan.buckets == {"Code2Wav": {"bs=1"}, "Talker": {"bs=1", "bs=32"}}, (
        "every pool's cheapest graphs must come before any pool's biggest"
    )


def test_a_graphs_executable_memory_counts_against_the_floor():
    """Planned by their pools alone, colocated Qwen3-Omni's graphs left free
    memory 241 MiB short of its largest eager step, because each graph's
    executable memory is outside the allocator."""
    costs = {bs: _cost(100, kept=100, peak=1024)._replace(outside=50 * _MIB) for bs in (1, 2)}

    plan = plan_captures({"Thinker": costs}, free=(1024 + 250) * _MIB)

    assert len(plan.buckets["Thinker"]) == 1, "both pools fit in 250 MiB, but not with their graphs' 100 MiB outside"
    assert (plan.predicted["Thinker"], plan.outside["Thinker"]) == (100 * _MIB, 50 * _MIB)


def test_a_bucket_goes_in_with_all_its_slots_or_not_at_all():
    """Two slots of a 200 MiB graph that keeps 100 MiB of outputs take 300 MiB:
    the second slot reuses the first's scratch but not its outputs."""
    pools = {"Thinker": {"decode": _cost(200, kept=100, slots=2)}}

    assert plan_captures(pools, free=250 * _MIB).buckets == {"Thinker": set()}, "room for one slot is room for none"
    assert plan_captures(pools, free=300 * _MIB).predicted == {"Thinker": 300 * _MIB}


def test_a_graph_keeps_its_outputs_in_whole_segments():
    """Code2Wav's bs=2 graph keeps 373 KB of outputs. Captured after bs=8 into
    one pool, it took a fresh 2 MiB segment: bs=8's scratch had no small block
    left for it, and the pool came out over the plan."""
    bs2 = CaptureCost(peak=0, graph=390 * _MIB, kept=381780, slots=1)
    pools = {"Code2Wav": {"bs=8": _cost(1552, kept=2), "bs=2": bs2}}

    plan = plan_captures(pools, free=2 * _GIB)

    assert plan.predicted == {"Code2Wav": (1550 + 2 + 2) * _MIB}, "a small tensor must count as the segment it may need"


def test_the_same_costs_give_the_same_plan_in_any_order():
    """Nodes came from a set, so the order a plan sees its pools in changed
    from boot to boot. A tie goes to the node named first."""
    talker, code2wav = {"bs=1": _cost(100)}, {"bs=1": _cost(100)}

    one = plan_captures({"Talker": talker, "Code2Wav": code2wav}, free=150 * _MIB)
    other = plan_captures({"Code2Wav": code2wav, "Talker": talker}, free=150 * _MIB)

    assert one.buckets == other.buckets == {"Code2Wav": {"bs=1"}, "Talker": set()}, (
        "the plan must not depend on the order the pools come in"
    )


class _PeerGroup:
    """One of two ranks, whose peer's flags are ``peer``."""

    world_size = 2

    def __init__(self, peer: list[bool]):
        self._peer = peer

    def barrier(self):
        pass

    def graph_capture(self):
        return GraphCaptureScope()

    def all_gather(self, tensor, dim=0):
        return torch.cat([tensor, torch.tensor(self._peer, dtype=tensor.dtype)])


class _StubPlannedRunner:
    """The real capture loop over stub captures, one slot per bucket, on a rank whose peer planned ``peer``."""

    warmup_and_capture = AcceleratorGraphRunner.warmup_and_capture
    _buckets_captured_everywhere = AcceleratorGraphRunner._buckets_captured_everywhere
    _register_slot = AcceleratorGraphRunner._register_slot
    _report_dropped = AcceleratorGraphRunner._report_dropped
    _refused_buckets = AcceleratorGraphRunner._refused_buckets
    _capture_refusal = staticmethod(lambda spec: None)
    NUM_WARMUP = AcceleratorGraphRunner.NUM_WARMUP

    def __init__(self, buckets: list[BucketKey], peer: list[bool]):
        self._specs = [CGSlotSpec(bucket=bucket, slot=0, config=SimpleNamespace()) for bucket in buckets]
        self._capture_costs = {bucket: _cost(100) for bucket in buckets}
        self._device = torch.device("cpu")
        self._submodule_name = "Thinker"
        self._num_slots = 1
        self._buckets = {}
        self._comm_group = SimpleNamespace(
            tp_group=_PeerGroup(peer), sp_group=SimpleNamespace(world_size=1, barrier=lambda: None),
        )
        self._dummy_rows = SimpleNamespace(release_all=lambda: None)
        self._graph_backend = SimpleNamespace(
            is_available=lambda: True, graph_pool_handle=lambda: AcceleratorGraphPool(object()),
            memory_allocated=lambda: 0,
        )
        self._blocked_buckets = []
        self._executables = {}
        self.tried: list[BucketKey] = []
        self.forwards: list[int] = []

    def prepare_for_capture(self):
        return self._specs

    def _capture_one(self, spec, forwards):
        self.tried.append(spec.bucket)
        self.forwards.append(forwards)
        return spec.bucket

    _get_addtl_slot_specs = staticmethod(lambda spec: [])
    _log_memory = staticmethod(lambda before, after: None)


def test_a_tp_runner_captures_only_what_every_rank_planned():
    """Each rank plans on its own device. A bucket only this rank planned would
    be replayed here while the peer runs it eager, and a collective inside it
    would hang."""
    runner = _StubPlannedRunner([_DECODE, _PREFILL], peer=[True, False])

    runner.warmup_and_capture({_DECODE, _PREFILL})

    assert runner.tried == [_DECODE], "a bucket the peer did not plan must not be captured here"


def test_a_runner_captures_its_costliest_bucket_first():
    """The Codec's buckets come in batch-size order, its smallest window first
    at each size. Captured that way, every larger window outgrew what the last
    had left, and the pool took 11222 MiB against 9068 predicted."""
    runner = _StubPlannedRunner([_DECODE, _PREFILL], peer=[True, True])
    runner._capture_costs = {_DECODE: _cost(100), _PREFILL: _cost(900)}

    runner.warmup_and_capture({_DECODE, _PREFILL})

    assert runner.tried == [_PREFILL, _DECODE], "the costliest graph must go first, so the rest reuse its pool"


@pytest.mark.parametrize(
    ("planned", "forwards"), [({_DECODE, _PREFILL}, 0), (None, AcceleratorGraphRunner.NUM_WARMUP)],
    ids=["sized", "no_plan"],
)
def test_capture_warms_a_bucket_up_only_when_sizing_did_not(planned, forwards):
    """Only CUDA plans, and sizing runs each forward a capture needs first. An
    XPU device has no plan, so it captures every bucket and warms each up itself."""
    runner = _StubPlannedRunner([_DECODE, _PREFILL], peer=[True, True])

    runner.warmup_and_capture(planned)

    assert sorted(runner.tried, key=str) == sorted([_DECODE, _PREFILL], key=str)
    assert runner.forwards == [forwards, forwards], "a capture must run the warm-up sizing didn't"


@pytest.mark.parametrize(
    ("fraction", "reserved", "planned"), [(None, 30, 8), (0.45, 30, 4), (0.45, 40, 0)],
    ids=["uncapped", "capped", "over_the_cap"],
)
def test_the_cap_bounds_the_plan(monkeypatch, fraction, reserved, planned):
    """Two workers share GPU 0 in qwen3tts_1p7b_split. Each saw the other's
    memory as used, never as held back, so the first to capture could take the
    other's room. Under 0.45 of 80 GiB, a worker holding 30 GiB has 6 left: the
    2 GiB eager step and four 1 GiB graphs."""
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (50 * _GIB, 80 * _GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: reserved * _GIB)
    # graphs that share no scratch, so each takes its full GiB
    costs = {bs: CaptureCost(peak=2 * _GIB, graph=_GIB, kept=_GIB, slots=1) for bs in range(8)}
    talker = SimpleNamespace(size_captures=lambda: costs, size_lent_captures=lambda wanted: costs)
    engine = SimpleNamespace(
        _device=torch.device("cuda", 0), _submodules={"Talker": None}, _gpu_memory_fraction=fraction,
    )

    plan = Engine._plan_captures(engine, {"Talker": talker}, {"Talker": {}})

    assert len(plan.buckets["Talker"]) == planned, "the worker's graphs and eager step must fit under its cap"


def test_a_runner_with_no_graphs_keeps_its_largest_bucket_as_its_batch_cap():
    """A runner that captured nothing runs every bucket eager, and the floor
    covers eager steps only up to its largest bucket."""
    code2wav = SimpleNamespace(
        submodule=SimpleNamespace(max_batch_size=lambda walk: None),
        accelerator_graph_runner=None,
        capture_runner=SimpleNamespace(max_batch_size_for=lambda walk: 32),
    )
    engine = SimpleNamespace(_submodules={"Code2Wav": code2wav})

    cap = Engine.get_max_batch_size(engine, "Code2Wav", "code2wav_chunk")

    assert cap == 32, "a batch past the largest bucket is an eager step the floor never measured"

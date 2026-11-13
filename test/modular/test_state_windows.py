"""Windows a read keeps: a kept window on the bidirectional mask, and a window of frames.

LingBot World keeps a sink and the latest frames of its rollout, at sizes that follow each request's resolution, and
Waypoint keeps a ring of frames per layer. A window planned wrong changes what every later frame sees, with no error of
its own, so each declaration today's resources can't plan is refused when the specs are built, naming the read.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from mstar.communication.tensors import LocalTransferEngine
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.resources.attn.config import AttentionConfig, AttentionStep, AttnBackend
from mstar.engine.resources.attn.kept_window import KeptWindowAttention
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.config import (
    KVReqConfig,
    KVSpec,
    KVStep,
    PagedKVConfig,
    RetentionPolicy,
    RingKVConfig,
    RingKVLayerConfig,
    RingKVStep,
)
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import kept_reach
from mstar.engine.resources.kv.ring import RingKVManager
from mstar.engine.resources.kv.rolling import RollingKVConfig, RollingKVReqConfig, RollingKVStep, RollingPosition
from mstar.engine.resources.kv.rolling.manager import RollingKVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.sequence_shard import sequence_parallel_refusals
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Bidirectional,
    Causal,
    Entry,
    Fork,
    FrameWindow,
    KeptWindow,
    KVEntry,
    Layout,
    PagedNHD,
    Positions,
    Read,
    Rope,
    TensorType,
    add_read,
    at_last_token,
    causal_conv,
    host,
    position,
    positions,
    rope,
)
from mstar.engine.state.adapters import RingHandle, build_specs, kv_layouts, node_handles, state_steps
from mstar.engine.state.analysis import state_declaration_problems
from mstar.engine.state.calls import kernel_setting
from mstar.graph.base import GraphNode, Loop
from mstar.model.base import KVLabel, LabelLayout, PrefixKeys
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import ChunkingPolicy

HEADS, DIM = 2, 8


def _array(name="kv", *, reads, max_len=64, **fields) -> Array:
    return Array(name, **{
        "nodes": {"dit"}, "layers": {0, 1}, "entry": KVEntry(heads=HEADS, head_dim=DIM), "max_len": max_len,
        "reads": reads, **fields,
    })


def _read(mask, **fields) -> Read:
    return Read(mask=mask, op=Attend(q_heads=HEADS), **fields)


@pytest.mark.parametrize(("make", "message"), [
    (lambda: Bidirectional(kept_window=1), "Bidirectional.kept_window must be a bool, got 1"),
    (lambda: KeptWindow(-1, window=4), "KeptWindow.sink must be an int >= 0, got -1"),
    (lambda: KeptWindow(2, window=-1), "KeptWindow.window must be an int >= 0, got -1"),
    (lambda: KeptWindow(2.0, window=4), "KeptWindow.sink must be an int >= 0, got 2.0"),
    (lambda: KeptWindow(True, window=4), "KeptWindow.sink must be an int >= 0, got True"),
    (lambda: FrameWindow(0, frames=16), "FrameWindow.tokens must be a positive int, got 0"),
    (lambda: FrameWindow(512, frames=0), "FrameWindow.frames must be a positive int, got 0"),
    (lambda: FrameWindow(512, frames=16, stride=0), "FrameWindow.stride must be a positive int, got 0"),
    (lambda: FrameWindow(512, frames=12, stride=8), "FrameWindow.frames must be a multiple of stride 8, got 12"),
], ids=[
    "kept_window_int", "sink_negative", "window_negative", "sink_float", "sink_bool", "tokens_zero", "frames_zero",
    "stride_zero", "frames_off_stride",
])
def test_a_window_refuses_its_own_bad_fields(make, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        make()


class _Model:
    """Node ``dit`` prefills from ids, then decodes in loop ``loop``, under the policies and prefix keys given."""

    def __init__(self, *arrays, chunked=(), kv_labels=None, rebuilt=False):
        self.arrays = list(arrays)
        self.chunked = chunked
        self.labels = kv_labels or {}
        self.rebuilt = rebuilt

    def declare_state(self):
        return self.arrays

    def get_graph_walk_graphs(self):
        return {
            "prefill": GraphNode(name="dit", input_names=["ids"], outputs=[]),
            "decode": Loop(section=GraphNode(name="dit", input_names=[], outputs=[]), max_iters=8, outputs=[],
                           name="loop"),
        }

    def get_combined_graph_walks(self):
        return {}

    def get_chunking_policies(self):
        return {"dit": (ChunkingPolicy(walks=self.chunked, chunkable=True),)} if self.chunked else {}

    def kv_labels(self):
        return self.labels

    def rebuild_policies(self):
        if not self.rebuilt:
            return {}
        return {"dit": RebuildPolicy(walks=frozenset({"prefill"}), ids_input="ids", sample_output="token", loop="loop")}


def test_a_rebuild_refuses_a_kept_window_on_its_decode_walks():
    """A rebuild recomputes the decode steps in prefill rows, where a kept window cuts other entries than each step
    saw. Only the kept window takes the loop's walks."""
    kept = _array(reads={"attn": _read(Bidirectional(kept_window=True), walks={"decode"})})
    assert state_declaration_problems(_Model(kept, rebuilt=True)) == [
        "Array 'kv'.reads['attn']: Bidirectional on walk 'decode', which dit's rebuild policy replaces with its "
        "prefill rows. A kept window cuts other entries from a prefill row than from the steps it replaces, so drop "
        "the rebuild policy"
    ]
    plain = _array(reads={"attn": _read(Bidirectional(), walks={"decode"})})
    assert state_declaration_problems(_Model(plain, rebuilt=True)) == [], (
        "a plain Bidirectional read on decode walks was refused"
    )


@pytest.mark.parametrize(("kwargs", "cause"), [
    ({"chunked": ("prefill",)}, "which a chunking policy splits across steps"),
    ({"kv_labels": {"kv": {"main": KVLabel(keys=PrefixKeys("ids", ("prefill",)))}}}, "keys for prefix reuse"),
    ({"rebuilt": True}, "which dit's rebuild policy runs it again"),
], ids=["chunked", "keyed", "rebuilt"])
def test_a_frame_window_refuses_a_step_that_splits_a_frame(kwargs, cause):
    """A frame's queries see the whole frame, so a step that ends inside a frame would cut what they see."""
    frames = _array(reads={"local": _read(FrameWindow(4, frames=2), packed=False)})
    (problem,) = state_declaration_problems(_Model(frames, **kwargs))
    assert problem.startswith("Array 'kv'.reads['local']: FrameWindow on walk 'prefill'") and cause in problem
    assert "A step boundary inside a frame cuts what the frame's queries see, so " in problem, (
        "the refusal should give a frame window's own cause"
    )


def test_a_frame_window_beside_other_masks_has_no_layout():
    """No paged kernel runs a frame window, and a read within a step takes only Block."""
    mixed = _array(reads={"local": _read(FrameWindow(4, frames=2), layers={0}), "attn": _read(Causal(), layers={1})})
    with pytest.raises(ValueError, match=r"Array 'kv': no layout yet: reads\['local'\].mask must be Causal, Window"):
        build_specs([mixed])
    within = _array(max_len=0, reads={"local": _read(FrameWindow(4, frames=2))})
    with pytest.raises(ValueError, match=r"Array 'kv': no layout yet: reads\['local'\].mask must be Block"):
        build_specs([within])


# The ring

TPF = 128    # the smallest frame a flex block mask takes


class _StubFlex:
    """Attends every slot of the ring, as eager flex does whatever its mask, and records the layers it was asked for."""

    def __init__(self):
        self.layers = []

    def attend(self, q, k, v, *, enable_gqa, layer_idx):
        if not torch.compiler.is_compiling():
            self.layers.append(layer_idx)
        k, v = k.expand(q.size(0), -1, -1, -1), v.expand(q.size(0), -1, -1, -1)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=enable_gqa)


def _ring_array(*, reads=None, max_len=4 * TPF, **fields) -> Array:
    """Two layers, the second at stride 2, so the ring keeps a different window in each."""
    return _array(reads=reads or {
        "local": _read(FrameWindow(TPF, frames=2), layers={0}, packed=False),
        "global": _read(FrameWindow(TPF, frames=4, stride=2), layers={1}, packed=False),
    }, max_len=max_len, **{"positions": Positions(), **fields})


def _ring(array=None) -> tuple[RingHandle, RingKVManager, _StubFlex]:
    """``array``'s handle through `node_handles`, over its real ring and a flex stub."""
    array = array or _ring_array()
    kv_spec, flex_spec = build_specs([array])
    ring = RingKVManager.build(kv_spec, EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.float32))
    flex = _StubFlex()
    model = SimpleNamespace(declare_state=lambda: [array])
    handles = node_handles(model, "dit", {"kv": ring, flex_spec.resource_key: flex})
    return handles["kv"], ring, flex


def _ring_step(handle, ring, rids=("a",), walk="rollout", capture=False, **declared) -> SubmoduleStep:
    """A step of one frame per row, admitted and planned on the ring, with its calls open."""
    step = state_steps(
        SubmoduleStep(segments=[Segment(rid, "main", TPF) for rid in rids], state={"kv": ArrayStep(**declared)}),
        {"kv": handle}, graph_walk=walk,
    )
    step.set_ctx(StepContext(request_ids=list(rids), graph_walk=walk, slot=0, capture=capture))
    for rid in rids:
        if rid not in ring._known_rids:
            ring.ingest_request(rid)
    assert ring.admit(step.steps["kv"], step.ctx).ok
    ring.plan(step.steps["kv"], step.ctx)
    step.begin_calls()
    return step


def _end(step, ring) -> None:
    step.check_calls()
    ring.commit(step.steps["kv"], step.ctx)
    step.end_calls()


def _pass(handle, keep, rows=1, value=1.0):
    """Each layer's add then read, as a world model's pass over its frame."""
    for layer in (0, 1):
        k = torch.full((rows, TPF, HEADS, DIM), value)
        handle.add(layer, keep=keep, k=k, v=k)
        handle.read(layer, torch.ones(rows, TPF, HEADS, DIM))


def test_the_ring_handle_builds_through_node_handles():
    handle, ring, flex = _ring()
    assert type(handle) is RingHandle
    assert handle.reads == {"local": flex, "global": flex}, "both reads should share the one flex resource"
    assert handle.flex_key == "local" and handle._ring is ring
    assert handle.positions.position == ring.position, "position() should ask the ring for its count"


def test_prime_then_rollout_drives_the_ring_in_order():
    """A prime is one kept pass per layer, and a rollout four scratch passes and one kept pass. The ring counts a frame
    per step and plans each row's next frame from the count."""
    handle, ring, flex = _ring()
    prime = _ring_step(handle, ring, walk="prime")
    assert (ring._static_session_idx[0].item(), ring._static_frame_pos[0].item()) == (0, 0)
    _pass(handle, keep=True)
    _end(prime, ring)
    assert position(handle, "a", "main") == TPF

    rollout = _ring_step(handle, ring)
    assert ring._static_frame_pos[0].item() == 1, "the rollout's frame should follow the prime's"
    for keep in (False, False, False, False, True):
        _pass(handle, keep=keep)
    assert (handle._kept, handle._scratch, handle._read) == ({0, 1}, {0, 1}, {0, 1})
    _end(rollout, ring)
    assert position(handle, "a", "main") == 2 * TPF
    assert flex.layers == [0, 1] * 6, "each read should run its own layer's mask"


def test_a_compiled_region_traces_the_ring_calls_once():
    """Waypoint compiles its two passes with fullgraph=True. The calls in them trace with no break, the checks skip,
    and the step's checks pass with nothing recorded, at capture too."""
    torch._dynamo.reset()
    counter = CompileCounter()
    eager, eager_ring, _ = _ring()
    handle, ring, _ = _ring()

    def denoise(h, x):
        for layer in (0, 1):
            h.add(layer, keep=False, k=x, v=x)
            x = h.read(layer, x)
        return x

    def cache(h, x):
        for layer in (0, 1):
            h.add(layer, keep=True, k=x, v=x)
            x = h.read(layer, x)
        return x

    regions = (torch.compile(denoise, backend=counter, fullgraph=True),
               torch.compile(cache, backend=counter, fullgraph=True))
    capture = _ring_step(handle, ring, rids=("dummy",), capture=True)
    regions[1](handle, torch.ones(1, TPF, HEADS, DIM))
    capture.check_calls()
    capture.end_calls()
    ring.reset_request("dummy")
    for walk, passes in (("prime", (True,)), ("rollout", (False,) * 4 + (True,)), ("rollout", (False,) * 4 + (True,))):
        steps = [_ring_step(h, r, walk=walk) for h, r in ((handle, ring), (eager, eager_ring))]
        for i, keep in enumerate(passes):
            x = torch.full((1, TPF, HEADS, DIM), float(i + 1))
            regions[keep](handle, x)
            (cache if keep else denoise)(eager, x)
        for step, r in zip(steps, (ring, eager_ring), strict=True):
            _end(step, r)
    assert counter.frame_count == 2, "each region should compile once, for prime and rollout alike"
    for traced, run in zip(ring.layers, eager_ring.layers, strict=True):
        assert torch.equal(traced.kv, run.kv), "the traced writes should leave the ring the eager writes leave"


@pytest.mark.parametrize(("change", "message"), [
    ({"reads": {"local": _read(FrameWindow(TPF, frames=2), layers={0}, packed=False),
                "global": _read(FrameWindow(2 * TPF, frames=2), layers={1}, packed=False)}},
     r"reads take frames of \[128, 256\] entries, and a ring holds one size"),
    ({"reads": {"local": _read(FrameWindow(TPF, frames=2), layers={0}, packed=False)}},
     r"layers \[1\] have no read, and a ring sizes each layer by its read"),
    ({"reads": {"local": _read(FrameWindow(TPF, frames=2), layers={0}, packed=False),
                "global": Read(mask=FrameWindow(TPF, frames=2), op=Attend(q_heads=2 * HEADS), layers={1},
                               packed=False)}},
     r"reads have q_heads \[2, 4\], and RingKVConfig has one num_qo_heads"),
    ({"entry": Entry({"k": TensorType((HEADS, DIM), dtype=torch.bfloat16, shard=0),
                      "v": TensorType((HEADS, DIM), dtype=torch.bfloat16, shard=0)})},
     "entry must be a KVEntry"),
    ({"labels": {"main", "cfg"}}, r"labels \['cfg', 'main'\], and a ring holds one session per request"),
    ({"max_len": TPF - 1}, "max_len 127 holds no frame of 128 entries"),
], ids=["frame_sizes", "uncovered_layer", "q_heads", "entry", "labels", "max_len"])
def test_the_ring_refuses_what_it_cannot_size(change, message):
    with pytest.raises(ValueError, match=f"Array 'kv': no layout yet: {message}"):
        build_specs([_ring_array(**change)])


def test_a_ring_takes_no_kernel_settings():
    """A ring's pool is its sessions, which a deployment sizes, and flex attention takes no setting."""
    with pytest.raises(ValueError, match=r"kernel_settings\['kv'\] sets \['max_num_pages'\]; it may set only \[\]"):
        build_specs([_ring_array()], kernel_settings={"kv": {"max_num_pages": 8}})
    with pytest.raises(ValueError, match=r"kernel_settings\['local'\] names no read or update"):
        build_specs([_ring_array()], kernel_settings={"local": {"splits": (1,)}})


def test_the_engine_refuses_a_ring_on_sp_ranks():
    """A ring declares nothing about sequence parallelism and refuses nothing about it. The engine's check refuses its
    node on SP ranks, and where a shard splits the node's steps the engine refuses the ring at build."""
    (refusal,) = sequence_parallel_refusals(build_specs([_ring_array()]), "dit", tp=1, sp=2)
    assert refusal.startswith("node 'dit' runs on 2 SP ranks but neither splits tokens"), refusal
    assert not RingKVManager.supports_split_steps, "a ring that took split steps would pass the engine's build check"


@pytest.mark.parametrize(("read", "message"), [
    (_read(FrameWindow(64, frames=2), packed=False), "mask.tokens is 64, which doesn.t fill whole blocks"),
    (_read(FrameWindow(TPF, frames=2)), "takes packed queries, and flex attends rows of one frame"),
    (_read(FrameWindow(TPF, frames=2), packed=False, walks={"rollout"}), "walks must be None"),
], ids=["partial_block", "packed", "walks"])
def test_flex_refuses_partial_blocks_and_packed_queries(read, message):
    with pytest.raises(ValueError, match=rf"Array 'kv': no layout yet: reads\['local'\].*{message}"):
        build_specs([_ring_array(reads={"local": read}, max_len=4 * 64)])


@pytest.mark.parametrize(("segments", "declared", "message"), [
    ([Segment("a", "main", TPF - 1)], {}, "row a adds 127 entries, and a ring step adds one frame of 128 per row"),
    (None, {"forks": (Fork(src="main", dst="main", at="end"),)}, "a step forks a ring"),
    (None, {"keep": False}, "keep=False on a ring"),
    (None, {"eager_add": True}, "a step sets placeholders or eager_add"),
], ids=["span", "forks", "scratch", "eager_add"])
def test_a_ring_step_adds_one_frame_per_row(segments, declared, message):
    handle, _, _ = _ring()
    step = SubmoduleStep(segments=segments or [Segment("a", "main", TPF)], state={"kv": ArrayStep(**declared)})
    step.set_ctx(StepContext(request_ids=["a"], graph_walk="rollout", slot=0, capture=False))
    with pytest.raises(ValueError, match=f"Array 'kv': .*{message}"):
        handle.begin_step(step)


def test_a_ring_answers_position_and_refuses_ids_and_rope():
    handle, ring, _ = _ring()
    step = _ring_step(handle, ring)
    _pass(handle, keep=True)
    _end(step, ring)
    assert position(handle, "a", "main") == TPF, "a request's position is the entries of the frames it committed"
    _ring_step(handle, ring)
    with pytest.raises(ValueError, match="Array 'kv': no layout yet: positions\\(\\), and its positions are a count"):
        positions(handle)
    q = torch.ones(1, TPF, HEADS, DIM)
    with pytest.raises(ValueError, match="Array 'kv': no layout yet: rope\\(\\), and its positions declare no Rope"):
        rope(handle, q, q)


def test_a_ring_array_takes_no_rope():
    with pytest.raises(ValueError, match="Array 'kv': no layout yet: positions.rope, and its storage keeps no"):
        build_specs([_ring_array(positions=Positions(rope=Rope()))])
    assert build_specs([_ring_array(positions=Positions())]), "a counter alone should build"


def test_a_ring_step_sets_no_advance_or_position_ids():
    handle, _, _ = _ring()
    for declared in ({"advance": (1,)}, {"position_ids": torch.arange(TPF)}):
        step = SubmoduleStep(segments=[Segment("a", "main", TPF)], state={"kv": ArrayStep(**declared)})
        with pytest.raises(ValueError, match="Array 'kv': no layout yet: a step sets advance or position_ids"):
            state_steps(step, {"kv": handle}, graph_walk="rollout")


def test_a_ring_of_waypoint_s_geometry_builds_waypoint_s_specs():
    """The reference is Waypoint's hand-written spec, copied. The differences are the declaration's own: the flex
    resource takes its first read's key, and the ring holds a request to the array's ``max_len``."""
    global_layers = frozenset(range(3, 24, 4))
    array = Array("kv", nodes={"dit"}, layers=frozenset(range(24)), entry=KVEntry(heads=16, head_dim=64),
                  max_len=(1 + 512) * 512, positions=Positions(), reads={
        "local_attn": Read(mask=FrameWindow(512, frames=16), op=Attend(q_heads=32),
                           layers=frozenset(range(24)) - global_layers, packed=False),
        "global_attn": Read(mask=FrameWindow(512, frames=128, stride=8), op=Attend(q_heads=32), layers=global_layers,
                            packed=False),
    })
    hand_written_ring = RingKVConfig(
        num_layers=24, num_kv_heads=16, num_qo_heads=32, head_dim=64, tokens_per_frame=512, num_sessions=1,
        layers=tuple(
            RingKVLayerConfig(ring_frames=16, ring_buckets=16, pinned_dilation=8 if i in global_layers else 1)
            for i in range(24)
        ),
    )
    ring_spec, flex_spec = build_specs([array])
    assert (ring_spec.resource_key, ring_spec.nodes) == ("kv", frozenset({"dit"}))
    assert replace(ring_spec.config, max_frames=None) == hand_written_ring and ring_spec.config.max_frames == 513
    assert (flex_spec.resource_key, flex_spec.nodes) == ("local_attn", frozenset({"dit"}))
    assert flex_spec.config == AttentionConfig(kv_cache="kv", backend=AttnBackend.FLEX)


def test_a_ring_step_is_the_hand_written_one_with_its_rows_segments():
    """Waypoint's hand-written steps named each row's frame. A ring step lists its rows' segments instead, and the ring
    counts the frames itself."""
    handle, _, _ = _ring()
    rows = [Segment(rid, "main", TPF) for rid in ("a", "b")]
    step = state_steps(SubmoduleStep(segments=rows, state={"kv": ArrayStep()}), {"kv": handle}, graph_walk="prime")
    assert replace(step.steps["kv"], segments=None) == RingKVStep()
    assert replace(step.steps["local"], segments=None) == AttentionStep()


@pytest.mark.parametrize(("call", "message"), [
    (lambda h, x: h.write(0, k=x), "write on an append-only array"),
    (lambda h, x: h.set_start({}), "no layout yet: set_start on a RingHandle"),
    (lambda h, x: h.block("k", 0), "no layout yet: block on a RingHandle"),
    (at_last_token, "no layout yet: at_last_token on a RingHandle"),
    (lambda h, x: add_read(h, 0, x, k=x, v=x), "no layout yet: add_read on a RingHandle"),
    (lambda h, x: causal_conv(h, 0, x, weight=x), "no layout yet: an update of type CausalConv on a RingHandle"),
    (lambda h, x: host(h, "a", "main", 0), "no layout yet: host on a RingHandle"),
    (lambda h, x: kernel_setting(h, "local", "splits"), "no splits for read 'local', whose binding FlexAttend takes"),
    (lambda h, x: h.read(0), "no layout yet: a read with no op"),
    (lambda h, x: h.read(0, x, top_k=3), r"no layout yet: mask arguments \['top_k'\]"),
], ids=["write", "set_start", "block", "at_last_token", "add_read", "update", "host", "kernel_setting", "no_q",
        "mask_args"])
def test_a_ring_refuses_the_calls_it_cannot_run(call, message):
    handle, ring, _ = _ring()
    _ring_step(handle, ring)
    with pytest.raises(ValueError, match=f"Array 'kv': {message}"):
        call(handle, torch.ones(1, TPF, HEADS, DIM))


# An exact retention, and the reach of a stream's view


def _pool(page_size=8, pages=64) -> KVManager:
    """A CPU pool of one layer of one head of one value, whose entries can hold their log positions."""
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=1, max_seq_len=pages * page_size, max_num_pages=pages,
        page_size=page_size,
    )
    return KVManager(
        cfg=cfg, name="kv", joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")), device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _open_exact(pool: KVManager, rid: str, sink: int, window: int, exact=True) -> None:
    """Ingest ``rid`` with a layout whose retention keeps ``sink`` and ``window``, through the wire's rows."""
    layout = LabelLayout(growth=(("rollout", 1),), retention=RetentionPolicy(window, sink, exact=exact))
    request = KVReqConfig(needed_labels=["main"])
    request.apply_conductor_config(kv_layout={"main": layout.row()}, loop_caps={"rollout": 10_000})
    pool.ingest_request(rid, overrides=request)


def _write_step(pool: KVManager, rid: str, start: int, span: int, reached=None) -> None:
    """One kept step of ``span`` entries holding their log positions. ``reached`` gets the positions the step's
    kept-window read gathers, after its add."""
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(request_ids=(rid,), graph_walk="rollout", slot=0, capture=False)
    assert pool.admit(step, ctx).ok
    (view,) = pool.plan(step, ctx)["main"].views
    pool.write_kv(*([torch.arange(start, start + span, dtype=torch.float32)[:, None, None]] * 2), layer_idx=0,
                  label="main")
    if reached is not None:
        slots = kept_reach(view, pool.config.tokens_per_page)
        reached.append(pool.layer_view(0)[slots.pages, 0, slots.offsets, 0, 0].long().tolist())
    pool.commit(step, ctx)


def _held(pool: KVManager, rid: str) -> list[int]:
    """The log positions ``rid``'s stream still holds."""
    stream, page, view = pool._streams[rid]["main"], pool.config.tokens_per_page, pool.layer_view(0)
    return [int(view[stream.page_indices[i // page], 0, i % page, 0, 0]) for i in range(stream.stored_len)]


def test_an_exact_retention_keeps_the_whole_window():
    """A sink of 3 ends inside the first page of 8, which the release keeps whole. Counting the page's other entries
    against the budget, as a plain retention does, frees the oldest of the window's 8 at 20 entries."""
    pool = _pool()
    _open_exact(pool, "a", sink=3, window=8)
    for start in range(0, 40, 4):
        _write_step(pool, "a", start, 4)
        held = set(_held(pool, "a"))
        written = start + 4
        assert {0, 1, 2} <= held, "the sink should stay"
        assert set(range(max(0, written - 8), written)) <= held, f"the window should stay whole at {written} entries"


def test_max_pages_counts_what_an_exact_stream_holds():
    """Prefix 1, page 128 and budget 128: the exact stream keeps its first page whole and 128 entries behind it,
    which a commit can leave straddling two more pages."""
    assert RetentionPolicy(128, 1, exact=True).max_pages(128) == 3
    pool = _pool(page_size=128, pages=8)
    _open_exact(pool, "a", sink=1, window=128)
    peak = 0
    for start in range(0, 1024, 64):
        _write_step(pool, "a", start, 64)
        peak = max(peak, len(pool._streams["a"]["main"].page_indices))
    assert peak == 3, "the stream should peak at what max_pages counts"


def test_an_exact_retention_waits_for_its_prefix():
    """LingBot's first chunk at 464x832 commits 6032 entries under a 9048-entry sink. Until the sink is committed
    nothing past it exists, and the policy can't be adopted."""
    pool = _pool(page_size=128, pages=256)
    _open_exact(pool, "a", sink=9048, window=12064)
    for chunk in range(6):
        _write_step(pool, "a", chunk * 6032, 6032)
    assert pool._streams["a"]["main"].protected_prefix == 9048, "the policy should be adopted once the sink is in"


def test_exact_retention_applies_from_the_layout_at_commit():
    """The step declares no retention: the commit takes the request's from its layout."""
    pool = _pool()
    _open_exact(pool, "a", sink=3, window=8)
    for start in range(0, 40, 4):
        _write_step(pool, "a", start, 4)
    assert pool._streams["a"]["main"].released > 0, "the layout's policy should have released the oldest pages"


def test_the_fast_path_takes_every_row_of_a_step_with_no_exact_request():
    """A request with an exact policy costs another request one membership test, and not the fast path."""
    pool = _pool()
    _open_exact(pool, "a", sink=3, window=8)
    pool.ingest_request("b", overrides=KVReqConfig(needed_labels=["main"]))
    arena, fast = pool._arena, []

    class _Recording:
        def __getattr__(self, name):
            return getattr(arena, name)

        def commit_rows(self, *args):
            fast.append(args[1])
            return arena.commit_rows(*args)

    pool._arena = _Recording()
    _write_step(pool, "b", 0, 4)
    _write_step(pool, "a", 0, 4)
    assert [[segment.request_id for segment in rows] for rows in fast] == [["b"]], (
        "only the step with no exact request should take the fast path"
    )


def test_the_exact_flag_survives_the_wire():
    policy = RetentionPolicy(12064, 9048, exact=True)
    request = KVReqConfig(needed_labels=["main"])
    request.apply_conductor_config(kv_layout={"main": LabelLayout(retention=policy).row()})
    assert request.kv_layout["main"].retention == policy


@pytest.mark.parametrize("page_size", [16, 128])
def test_the_reach_maps_through_released_pages(page_size):
    """Against the dense reference: the first ``sink`` log positions, the last ``window`` the stream kept before the
    step, and the step's own, over 40 commits of uneven spans, while the pool releases pages behind the sink."""
    sink, window = 2 * page_size + 5, 3 * page_size + 7
    pool = _pool(page_size=page_size, pages=64)
    _open_exact(pool, "a", sink=sink, window=window)
    reached, start = [], 0
    for step in range(40):
        span = 1 + (7 * step) % (page_size // 2)
        _write_step(pool, "a", start, span, reached)
        expected = [*range(min(sink, start)), *range(max(sink, start - window), start), *range(start, start + span)]
        assert reached[-1] == sorted(set(expected)), f"commit {step} reached other entries"
        start += span


def test_other_streams_carry_no_retention_on_their_views():
    pool = _pool()
    _open_exact(pool, "a", sink=3, window=8)
    _open_exact(pool, "b", sink=3, window=8, exact=False)
    pool.ingest_request("c", overrides=KVReqConfig(needed_labels=["main"]))
    step = KVStep(segments=tuple(Segment(rid, "main", 4) for rid in "abc"))
    ctx = StepContext(request_ids=("a", "b", "c"), graph_walk="rollout", slot=0, capture=False)
    assert pool.admit(step, ctx).ok
    views = pool.plan(step, ctx)["main"].views
    assert [view.retention for view in views] == [RetentionPolicy(8, 3, exact=True), None, None]


def test_the_staged_reach_is_each_row_s_reach_end_to_end():
    """What the kept-window resource and a read with no op both gather from: each row's `kept_reach`, rows in order."""
    from mstar.engine.resources.kv.plan import kept_reach, stage_reach

    pool = _pool()
    _open_exact(pool, "a", sink=3, window=8)
    _open_exact(pool, "b", sink=3, window=8, exact=False)
    step = KVStep(segments=(Segment("a", "main", 4), Segment("b", "main", 6)))
    ctx = StepContext(request_ids=("a", "b"), graph_walk="rollout", slot=0, capture=False)
    assert pool.admit(step, ctx).ok
    views = pool.plan(step, ctx)["main"].views
    staged = stage_reach(views, 8, torch.device("cpu"))
    rows = [kept_reach(view, 8) for view in views]
    assert staged.lengths == (4, 6)
    assert staged.pages.tolist() == [p for row in rows for p in row.pages.tolist()]
    assert staged.offsets.tolist() == [o for row in rows for o in row.offsets.tolist()]


def test_an_exact_window_on_a_pool_with_window_groups_is_refused_at_intake():
    """The release keeps an exact window in the first group alone. A kept-window read is its array's only read, so
    only a layout written by hand reaches this."""
    from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
    from mstar.engine.resources.spec import resolve_spec_dependencies

    cfg = PagedKVConfig(num_layers=3, num_kv_heads=1, head_dim=8, max_seq_len=64, page_size=16, max_num_pages=64)
    spec = KVSpec(resource_key="kv", nodes={"dit"}, config=cfg)
    resolve_spec_dependencies([
        spec, AttentionSpec(resource_key="local", nodes={"dit"}, config=AttentionConfig("kv", sliding_window=4,
                                                                                         layers=(1, 2))),
        AttentionSpec(resource_key="full", nodes={"dit"}, config=AttentionConfig("kv")),
    ])
    assert len(cfg.groups) > 1
    request = KVReqConfig(needed_labels=["main"])
    exact = LabelLayout(growth=(("rollout", 1),), retention=RetentionPolicy(32, 8, exact=True))
    request.apply_conductor_config(kv_layout={"main": exact.row()}, loop_caps={"rollout": 100})
    with pytest.raises(ValueError, match="label 'main' keeps an exact window, and the pool has window groups"):
        request.validate(spec)


def test_a_removed_request_leaves_the_exact_set():
    """A handle comes back for another request, which would inherit the policy."""
    pool = _pool()
    _open_exact(pool, "a", sink=3, window=8)
    pool.reset_request("a")
    assert "a" in pool._exact, "a reset keeps the request's layout, and so its policy"
    pool.remove_request("a")
    assert "a" not in pool._exact


# A kept-window read on the paged pool

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gathers on a CUDA pool")


def _kept_array(**fields) -> Array:
    return _array("dit_kv", reads={"self_attn": _read(Bidirectional(kept_window=True))}, max_len=4096, **fields)


@pytest.mark.parametrize("arrays", [
    [_array("dit_kv", reads={"self_attn": _read(Bidirectional(kept_window=True), layers={0}),
                             "attn": _read(Causal(), layers={1})}, max_len=4096)],
    [_array("dit_kv", reads={"self_attn": _read(Bidirectional(kept_window=True), layers={0}),
                             "other": _read(Bidirectional(kept_window=True), layers={1})}, max_len=4096)],
    [_array("dit_kv", reads={"self_attn": _read(Bidirectional(kept_window=True), layers={0}),
                             "reach": Read(mask=Bidirectional(kept_window=True), layers={1})}, max_len=4096)],
], ids=["beside_causal", "beside_kept_window", "beside_a_read_with_no_op"])
def test_a_kept_window_read_is_its_array_s_only_read(arrays):
    """The pool releases what one read's window no longer reaches, which any other read of the array may reach."""
    with pytest.raises(ValueError, match=r"Array 'dit_kv': no layout yet: reads\['\w+'\] keeps a window beside"):
        build_specs(arrays)


@pytest.mark.parametrize(("arrays", "message"), [
    ([_array("dit_kv", reads={"self_attn": _read(Bidirectional(kept_window=True), walks={"rollout"}),
                              "attn": _read(Causal(), walks={"prefill"})}, max_len=4096)],
     "keeps a window beside reads"),
    ([_array("dit_kv", reads={"self_attn": _read(Bidirectional(kept_window=True), walks={"rollout"})}, max_len=4096)],
     r"reads\['self_attn'\].walks must be None"),
    ([_array("q", reads={"own": _read(Causal())}),
      _array("dit_kv", reads={"self_attn": _read(Bidirectional(kept_window=True), queries="q")}, max_len=4096)],
     r"reads\['self_attn'\].queries must be None"),
], ids=["walks_beside", "walks", "queries"])
def test_a_kept_window_read_takes_no_queries_or_walks(arrays, message):
    """A read with walks lands on the by-walk storage and one with queries on the cross storage, and the kept window
    swaps the binding on each."""
    with pytest.raises(ValueError, match=f"Array 'dit_kv': no layout yet: .*{message}"):
        build_specs(arrays)


def test_a_kept_window_read_takes_packed_queries():
    with pytest.raises(ValueError, match="declares packed=False, and the kept-window kernel takes packed queries"):
        build_specs([_array("dit_kv", reads={"self_attn": _read(Bidirectional(kept_window=True), packed=False)},
                            max_len=4096)])


def test_a_layout_sizes_exactly_its_kept_window_reads():
    """A request's layout gives each kept-window read its sizes, which become the label's exact retention."""
    array = _kept_array()
    (layout,) = kv_layouts([array], {"dit_kv": {"main": Layout(kept_windows={
        "self_attn": KeptWindow(18, window=24),
    })}}, {})["dit_kv"].values()
    assert layout.retention == RetentionPolicy(24, 18, exact=True)
    with pytest.raises(RuntimeError, match=r"kept_windows gives no sizes for the kept-window read 'self_attn'"):
        kv_layouts([array], {"dit_kv": {"main": Layout()}}, {})
    with pytest.raises(RuntimeError, match=r"kept_windows: \['other'\] name no kept-window read"):
        kv_layouts([array], {"dit_kv": {"main": Layout(kept_windows={
            "self_attn": KeptWindow(18, window=24), "other": KeptWindow(1, window=1),
        })}}, {})


class _StubInner:
    """The ragged cross manager under a kept-window read: records the reach each call attends, returns q."""

    force_double_buffer = True

    def __init__(self):
        self.keys = []
        self.plans = []

    def plan(self, step, ctx):
        self.plans.append(step)

    def run(self, q, k, v, label=None):
        self.keys.append(k.clone())
        return q


def test_a_kept_window_read_is_never_captured():
    """Its reach changes length every step, so the capture leaves it out, and a plan under a lease is refused."""
    attention = KeptWindowAttention(_StubInner(), "dit_kv")
    assert not attention.can_capture()
    leased = StepContext(request_ids=["a"], graph_walk="rollout", slot=0, capture=False,
                         slot_lease=SimpleNamespace(captured=True))
    with pytest.raises(RuntimeError, match="Array 'dit_kv': a kept-window read is eager-only"):
        attention.plan(AttentionStep(segments=[Segment("a", "main", 4)]), leased)


def test_the_kept_window_resource_s_refusals_name_the_array():
    attention = KeptWindowAttention(_StubInner(), "dit_kv")
    ctx = StepContext(request_ids=["a"], graph_walk="rollout", slot=0, capture=False)
    with pytest.raises(RuntimeError, match=r"Array 'dit_kv': no layout yet: a kept-window read over labels"):
        attention.plan(AttentionStep(segments=[Segment("a", "main", 4), Segment("a", "uncond", 4)]), ctx)
    with pytest.raises(ValueError, match="Array 'dit_kv': no layout yet: at_last_token"):
        attention.select_last_hidden(torch.zeros(1))


def _kept_and_rolling(device: torch.device, chunks=6, frame=3):
    """LingBot's window at a frame of ``frame`` entries, a 6-frame sink and 18 frames attended, chunks of 4: the
    keys a kept-window read on the paged pool gathers at every pass, and the ones `main`'s rolling store gives for
    the same writes, through `RollingKVManager.window`."""
    sink, window, chunk = 6 * frame, 8 * frame, 4 * frame
    array = Array("dit_kv", nodes={"dit"}, layers={0}, entry=KVEntry(heads=1, head_dim=1), max_len=4096,
                  reads={"self_attn": Read(mask=Bidirectional(kept_window=True), op=Attend(q_heads=1))})
    kv_spec, attention_spec = build_specs([array], kernel_settings={"dit_kv": {"max_num_pages": 64}})
    pool = KVManager(
        cfg=replace(kv_spec.config, page_size=8), name="dit_kv", joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")), device=device,
        dtype=torch.float32,
    )
    inner = _StubInner()
    attention = KeptWindowAttention(inner, "dit_kv")
    attention.bind_dependencies({"dit_kv": pool})
    handle = node_handles(SimpleNamespace(declare_state=lambda: [array]), "dit",
                          {"dit_kv": pool, attention_spec.resource_key: attention})["dit_kv"]
    (row,) = kv_layouts([array], {"dit_kv": {"main": Layout(
        growth=(("rollout", chunk),), kept_windows={"self_attn": KeptWindow(sink, window=window)},
    )}}, {})["dit_kv"].values()
    request = KVReqConfig(needed_labels=["main"])
    request.apply_conductor_config(kv_layout={"main": row.row()}, loop_caps={"rollout": chunks})
    pool.ingest_request("a", overrides=request)
    rolling = RollingKVManager(RollingKVConfig(num_layers=1, num_heads=1, head_dim=1, pool_tokens=sink + window + chunk,
                                               dtype=torch.float32), device)
    rolling.ingest_request("a", RollingKVReqConfig(tokens=sink + window + chunk))
    assert rolling.reserve("a", "dit", "rollout", None) is None
    gathered, expected = [], []
    for index in range(chunks):
        ctx = StepContext(request_ids=["a"], graph_walk="rollout", slot=0, capture=False)
        step = state_steps(SubmoduleStep(segments=[Segment("a", "main", chunk)], state={"dit_kv": ArrayStep()}),
                           {"dit_kv": handle}, graph_walk="rollout")
        step.set_ctx(ctx)
        assert pool.admit(step.steps["dit_kv"], ctx).ok
        ctx.plan_results = {"dit_kv": pool.plan(step.steps["dit_kv"], ctx)}
        attention.plan(step.steps["self_attn"], ctx)
        rolled = RollingKVStep(segments=(Segment("a", "chunk", chunk),),
                               positions={"a": RollingPosition(index * chunk, sink, window + chunk)})
        assert rolling.admit(rolled, ctx).ok
        rolling.plan(rolled, ctx)
        step.begin_calls()
        for keep in (False, False, False, False, True):
            k = (index * chunk + torch.arange(chunk, dtype=torch.float32, device=device)) + (0.5 if not keep else 0)
            k = k[:, None, None]
            handle.add(0, keep=keep, k=k, v=-k)
            handle.read(0, torch.zeros(chunk, 1, 1, device=device))
            gathered.append(inner.keys[-1].flatten())
            rolling.write(0, k[None], -k[None])
            expected.append(rolling.window(0)[0].flatten().clone())
        step.check_calls()
        pool.commit(step.steps["dit_kv"], ctx)
        rolling.commit(rolled, ctx)
        step.end_calls()
    assert pool._streams["a"]["main"].released, "the pool should have released pages behind the sink"
    return gathered, expected


def test_a_kept_window_read_sees_main_s_rolling_window():
    """The one test that ties the port to `main`'s store: at every pass of 6 chunks, past the window's fill at chunk
    4, the kept window gathers exactly the keys, in the order, that the rolling store's view holds."""
    gathered, expected = _kept_and_rolling(torch.device("cpu"))
    for i, (got, want) in enumerate(zip(gathered, expected, strict=True)):
        assert torch.equal(got, want), f"pass {i % 5} of chunk {i // 5} gathered other keys than the rolling view"


@requires_cuda
def test_the_reach_equals_the_store_s_view_on_cuda():
    gathered, expected = _kept_and_rolling(torch.device("cuda"))
    for i, (got, want) in enumerate(zip(gathered, expected, strict=True)):
        assert torch.equal(got, want), f"pass {i % 5} of chunk {i // 5} gathered other keys than the rolling view"


# A read with no op


def _reach_handle(sink=18, window=24, chunk=12):
    """A no-op kept-window read through `node_handles` over a CPU pool of pages of 8, with request ``a`` laid out."""
    array = Array("dit_kv", nodes={"dit"}, layers={0}, entry=KVEntry(heads=1, head_dim=1), max_len=4096,
                  reads={"self_attn": Read(mask=Bidirectional(kept_window=True))})
    (kv_spec,) = build_specs([array], kernel_settings={"dit_kv": {"max_num_pages": 64}})
    pool = KVManager(
        cfg=replace(kv_spec.config, page_size=8), name="dit_kv", joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")), device=torch.device("cpu"),
        dtype=torch.float32,
    )
    handle = node_handles(SimpleNamespace(declare_state=lambda: [array]), "dit", {"dit_kv": pool})["dit_kv"]
    (row,) = kv_layouts([array], {"dit_kv": {"main": Layout(
        growth=(("rollout", chunk),), kept_windows={"self_attn": KeptWindow(sink, window=window)},
    )}}, {})["dit_kv"].values()
    request = KVReqConfig(needed_labels=["main"])
    request.apply_conductor_config(kv_layout={"main": row.row()}, loop_caps={"rollout": 100})
    pool.ingest_request("a", overrides=request)
    return handle, pool


def _reach_step(handle, pool, span, capture=False) -> SubmoduleStep:
    step = state_steps(SubmoduleStep(segments=[Segment("a", "main", span)], state={"dit_kv": ArrayStep()}),
                       {"dit_kv": handle}, graph_walk="rollout")
    step.set_ctx(StepContext(request_ids=["a"], graph_walk="rollout", slot=0, capture=capture))
    assert pool.admit(step.steps["dit_kv"], step.ctx).ok
    pool.plan(step.steps["dit_kv"], step.ctx)
    return step


def test_a_read_with_no_op_returns_the_reach():
    """Against the dense reference: the reach of each step, the step's own adds included, packed by row."""
    sink, window, chunk = 18, 24, 12
    handle, pool = _reach_handle(sink, window, chunk)
    for index in range(6):
        start = index * chunk
        step = _reach_step(handle, pool, chunk)
        step.begin_calls()
        k = (start + torch.arange(chunk, dtype=torch.float32))[:, None, None]
        handle.add(0, k=k, v=-k)
        reached = handle.read(0)
        expected = [*range(min(sink, start)), *range(max(sink, start - window), start), *range(start, start + chunk)]
        assert reached.fields["k"].flatten().long().tolist() == sorted(set(expected)), f"chunk {index}"
        assert torch.equal(reached.fields["v"], -reached.fields["k"])
        assert reached.row_offsets.tolist() == [0, len(set(expected))]
        step.check_calls()
        pool.commit(step.steps["dit_kv"], step.ctx)
        step.end_calls()


def test_add_read_through_a_read_with_no_op_returns_the_reach():
    handle, pool = _reach_handle()
    step = _reach_step(handle, pool, 12)
    step.begin_calls()
    k = torch.arange(12, dtype=torch.float32)[:, None, None]
    reached = add_read(handle, 0, k=k, v=-k)
    assert torch.equal(reached.fields["k"], k) and reached.row_offsets.tolist() == [0, 12]
    step.check_calls()
    step.end_calls()


def test_a_fused_add_writes_the_pages_a_reach_gathers():
    """A state op's fused kernel may write a reach-only array's pages: the reach reads the pages, as a kept-window
    kernel does, so nothing waits for k and v at the read."""
    handle, _ = _reach_handle()
    op = SimpleNamespace(adds=True, formats=(PagedNHD(),))
    assert handle.takes_fused(op)


def test_a_reach_only_array_keeps_the_fused_ops_it_was_given():
    """The handle drops its fused ops on a step whose reads attend the step's own k and v, and it asks each read. A
    reach reads the pages alone."""
    handle, pool = _reach_handle()
    handle.fused_ops = frozenset({"gdn"})
    _reach_step(handle, pool, 12).begin_calls()
    assert handle.fused_ops == {"gdn"}, "a reach reads the pages, where a fused op's write lands"


def test_reference_mode_builds_its_pool():
    """With no read that runs a kernel, the pool takes its KV heads as its query heads, and no attention is built."""
    array = Array("dit_kv", nodes={"dit"}, layers={0, 1}, entry=KVEntry(heads=HEADS, head_dim=DIM), max_len=64,
                  reads={"self_attn": Read(mask=Bidirectional(kept_window=True))})
    (kv_spec,) = build_specs([array])
    assert kv_spec.config.num_qo_heads == HEADS


def test_a_reach_read_s_add_is_on_the_pages_before_its_read():
    """Every add writes its pages, which the reach reads back. The handle holds the add for its read, as it does for
    every read, and the read takes it."""
    handle, pool = _reach_handle()
    _reach_step(handle, pool, 12).begin_calls()
    k = torch.arange(12, dtype=torch.float32)[:, None, None]
    handle.add(0, k=k, v=-k)
    page = pool._streams["a"]["main"].page_indices[0]
    assert pool.layer_view(0)[page, 0, :8].flatten().tolist() == list(range(8)), "the add should be on the pages"
    handle.read(0)
    assert handle._held is None, "the read should take the add the handle held for it"


@pytest.mark.parametrize("mask", [Bidirectional(), Causal()], ids=["bidirectional", "causal"])
def test_a_read_with_no_op_on_kept_entries_needs_a_kept_window(mask):
    with pytest.raises(ValueError, match=r"reads\['attn'\].op must be Attend, the reads today's resources run"):
        build_specs([_array(reads={"attn": Read(mask=mask)})])


def test_a_reach_is_refused_under_capture():
    """A reach changes length every step, so a captured graph has nothing fixed to replay. Validation sees no
    capture, so the step that would capture refuses."""
    handle, pool = _reach_handle()
    step = _reach_step(handle, pool, 12, capture=True)
    with pytest.raises(ValueError, match="Array 'dit_kv': no layout yet: a read with no op in a captured step"):
        step.begin_calls()


def test_a_reach_is_refused_on_a_pool_placed_over_ranks():
    """A placed pool keeps part of each page on every rank, and a reach is gathered by slot with no kernel that
    knows the placement, so each rank would gather other entries. The handles are built with the engine, so the
    deployment doesn't load."""
    array = Array("dit_kv", nodes={"dit"}, layers={0}, entry=KVEntry(heads=1, head_dim=1), max_len=4096,
                  reads={"self_attn": Read(mask=Bidirectional(kept_window=True))})
    (kv_spec,) = build_specs([array], kernel_settings={"dit_kv": {"max_num_pages": 64}})
    ranks = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1])
    pool = KVManager(
        cfg=replace(kv_spec.config, page_size=8, context_parallel=True), name="dit_kv",
        joint_comm_group=JointGroups(tp_group=CommGroup.trivial(), sp_group=ranks, instance=ranks),
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")), device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert pool.placement is not None
    with pytest.raises(ValueError, match="Array 'dit_kv': no layout yet: a read with no op gathers its reach from "
                                         "this rank's pages, and resources.dit_kv sets context_parallel"):
        node_handles(SimpleNamespace(declare_state=lambda: [array]), "dit", {"dit_kv": pool})


def test_a_reach_only_array_has_no_last_token_yet():
    handle, pool = _reach_handle()
    _reach_step(handle, pool, 12).begin_calls()
    with pytest.raises(ValueError, match="Array 'dit_kv': no layout yet: at_last_token on an array whose read has"):
        at_last_token(handle, torch.zeros(12, 1))


def test_the_views_are_the_live_step_s_through_a_pre_plan():
    """The forward reads the views of the step it runs: its own plan's, its promoted pre-plan's, or the read-only plan
    it reuses. A pre-plan of the next step, which waits for this step's commit, stages its views without making them
    live, and a commit drops the live ones, so a plan goes with its last reference."""
    pool = _pool()
    pool.ingest_request("a", overrides=KVReqConfig(needed_labels=["main"]))
    first, second = (KVStep(segments=(Segment("a", "main", span),)) for span in (4, 2))
    ctx = StepContext(request_ids=("a",), graph_walk="rollout", slot=0, capture=False)
    assert pool.admit(first, ctx).ok
    planned = pool.plan(first, ctx)
    assert pool.views("main") is planned["main"].views
    pool.commit(first, ctx)
    ahead = StepContext(request_ids=("a",), graph_walk="rollout", slot=1, capture=False, is_preplan=True)
    assert pool.admit(second, ahead).ok
    staged = pool.plan(second, ahead)
    with pytest.raises(RuntimeError, match="views of 'main' outside a step's forward, where no plan is live"):
        pool.views("main")
    assert pool.admit(second, ctx).ok
    assert pool.plan(second, ctx) is staged, "the step should promote its pre-plan"
    assert pool.views("main") is staged["main"].views, "the promoted step's views should be live"
    pool.commit(second, ctx)
    readonly = KVStep(segments=(Segment("a", "main", 0),), commit=False)
    for _ in range(2):
        assert pool.admit(readonly, ctx).ok
        planned = pool.plan(readonly, ctx)
        assert pool.views("main") is planned["main"].views, "a reused read-only plan's views should be live"
        pool.commit(readonly, ctx)


def test_a_kept_window_picks_its_binding_only_through_for_mask():
    """The mask's field picks the binding once, in `OpBinding.for_mask`, for a read with an op and for one without; a
    read with no op through any other mask has no binding."""
    from mstar.engine.state.adapters import BINDINGS, PAGED, PAGED_CROSS, KeptWindowAttend, PagedNoOp, PagedReach

    kept, plain = Bidirectional(kept_window=True), Bidirectional()
    assert isinstance(BINDINGS[PAGED, Attend].for_mask(kept), KeptWindowAttend)
    assert isinstance(BINDINGS[PAGED_CROSS, Attend].for_mask(kept), KeptWindowAttend)
    assert type(BINDINGS[PAGED, Attend].for_mask(plain)) is type(BINDINGS[PAGED, Attend])
    assert isinstance(BINDINGS[PAGED, type(None)].for_mask(kept), PagedReach)
    assert isinstance(BINDINGS[PAGED, type(None)].for_mask(plain), PagedNoOp)


@pytest.mark.parametrize(("tokens", "fits"), [(128, True), (512, True), (64, False), (200, False)])
def test_a_frame_fits_the_block_mask_by_one_rule(tokens, fits):
    """The read's check at validation and the ring at build ask the same predicate."""
    from mstar.engine.resources.kv.ring.cache import LayerRingCache, fits_block_mask

    assert fits_block_mask(tokens) is fits
    build = lambda: LayerRingCache(num_sessions=1, n_kv_heads=1, ring_frames=2, ring_buckets=2,  # noqa: E731
                                   d_head=8, tokens_per_frame=tokens, pinned_dilation=1, dtype=torch.float32,
                                   device="cpu")
    if fits:
        build()
    else:
        with pytest.raises(ValueError, match="must be a multiple of the sparse block size"):
            build()


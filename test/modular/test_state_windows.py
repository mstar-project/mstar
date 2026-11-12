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

from mstar.engine.resources.attn.config import AttentionConfig, AttentionStep, AttnBackend
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.config import RingKVConfig, RingKVLayerConfig, RingKVStep
from mstar.engine.resources.kv.ring import RingKVManager
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
from mstar.model.base import KVLabel, PrefixKeys
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


def test_a_kept_window_has_no_layout_until_it_is_bound():
    kept = Bidirectional(kept_window=True)
    for arrays in (
        [_array(reads={"attn": _read(kept)})],
        [_array(reads={"attn": _read(kept, walks={"a"}), "other": _read(Causal(), walks={"b"})})],
        [_array("q", reads={"own": _read(Causal())}), _array(reads={"attn": _read(kept, queries="q")})],
    ):
        with pytest.raises(ValueError, match=r"Array 'kv': no layout yet: reads\['attn'\].mask keeps a window"):
            build_specs(arrays)
    plain = _array(reads={"attn": _read(Causal())})
    with pytest.raises(ValueError, match="no layout yet: label 'main' gives kept_windows"):
        kv_layouts([plain], {"kv": {"main": Layout(kept_windows={"attn": KeptWindow(4, window=8)})}}, {})


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
        "prefill rows. A Block or Bidirectional read sees each step's adds as a whole, so drop the rebuild policy"
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

    def attend(self, q, k, v, visible, *, enable_gqa, layer_idx=None):
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
    per step, plans each row's next frame from the count, and its admit checks the frame the step declares, which is
    the count Waypoint declares on each walk."""
    handle, ring, flex = _ring()
    prime = _ring_step(handle, ring, walk="prime")
    assert prime.steps["kv"].frames == (("a", 0),)
    assert (ring._static_session_idx[0].item(), ring._static_frame_pos[0].item()) == (0, 0)
    _pass(handle, keep=True)
    _end(prime, ring)
    assert position(handle, "a", "main") == TPF

    rollout = _ring_step(handle, ring)
    assert rollout.steps["kv"].frames == (("a", 1),)
    assert ring._static_frame_pos[0].item() == 1
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
    ({"seq_sharded": True}, "seq_sharded takes max_len 0"),
], ids=["frame_sizes", "uncovered_layer", "q_heads", "entry", "labels", "max_len", "seq_sharded"])
def test_the_ring_refuses_what_it_cannot_size(change, message):
    with pytest.raises(ValueError, match=f"Array 'kv': no layout yet: {message}"):
        build_specs([_ring_array(**change)])


def test_a_ring_takes_no_kernel_settings():
    """A ring's pool is its sessions, which a deployment sizes, and flex attention takes no setting."""
    with pytest.raises(ValueError, match=r"kernel_settings\['kv'\] sets \['max_num_pages'\]; it may set only \[\]"):
        build_specs([_ring_array()], kernel_settings={"kv": {"max_num_pages": 8}})
    with pytest.raises(ValueError, match=r"kernel_settings\['local'\] names no read or update"):
        build_specs([_ring_array()], kernel_settings={"local": {"splits": (1,)}})


@pytest.mark.parametrize(("read", "message"), [
    (_read(FrameWindow(64, frames=2), packed=False), "mask.tokens is 64, and a flex block mask holds whole blocks"),
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


def test_a_ring_of_waypoint_s_geometry_builds_waypoint_s_specs_and_steps():
    """The reference is Waypoint's hand-written spec and step, copied. Two differences are the declaration's own: the
    flex resource takes its first read's key, and the ring holds a request to the array's ``max_len``. A step lists its
    rows' segments, which Waypoint's didn't."""
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

    handle, _, _ = _ring()
    rows = [Segment(rid, "main", TPF) for rid in ("a", "b")]
    step = state_steps(SubmoduleStep(segments=rows, state={"kv": ArrayStep()}), {"kv": handle}, graph_walk="prime")
    assert replace(step.steps["kv"], segments=None) == RingKVStep(frames=(("a", 0), ("b", 0)))
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

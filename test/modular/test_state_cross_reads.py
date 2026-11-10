"""A read of another array's queries, and a read under sequence parallelism, plan and exchange what a model did by hand.

LTX-2.5 wrote each attention's plan and exchange in model code, one case per attention, so a port that gets a case wrong
attends the wrong tokens with no error. The three cases follow from two flags: both sides sharded runs Ulysses over
whole segments, only the queries sharded runs on the rank's share, and only the keys sharded merges the ranks' partial
attentions. A DiT compares its output bytes exactly, so a break or a plan that moves is a regression.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionSpec,
    RaggedCrossAttentionSpec,
    RaggedCrossAttentionStep,
)
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep, Attend, Block, Causal, KVEntry, Read, TensorType, Value, adapters
from mstar.engine.state.adapters import (
    WITHIN_STEP,
    WithinStepHandle,
    _bindings,
    build_specs,
    node_handles,
    sdpa_resources,
    state_steps,
)
from mstar.engine.state.analysis import deployment_problems
from mstar.engine.state.calls import kernel_setting
from mstar.engine.state.contract import Entry
from mstar.model.components.diffusion.attention import sdpa_attention

HEADS, DIM = 4, 8


def _entry(heads=HEADS, dtype=None) -> Entry:
    t = TensorType((heads, DIM), dtype=dtype, shard=0)
    return Entry({"k": t, "v": t})


def _array(
    name, read=None, *, queries=None, sharded=False, label=None, heads=HEADS, nodes=("dit",), dtype=None, packed=False,
) -> Array:
    return Array(
        name, nodes=set(nodes), layers={0, 1}, entry=_entry(heads, dtype), max_len=0, seq_sharded=sharded,
        labels={label or name},
        reads={read or f"{name}_read": Read(mask=Block(), op=Attend(q_heads=heads), queries=queries, packed=packed)},
    )


def _av() -> list[Array]:
    """The shape of LTX-2.5's declaration: two self reads and three cross reads, one stream sharded."""
    return [
        _array("video", "video_self", sharded=True),
        _array("audio", "audio_self", heads=2),
        _array("text_v", "video_text", queries="video", label="text"),
        _array("text_a", "audio_text", queries="audio", label="text", heads=2),
        _array("audio_v", "audio_to_video", queries="video", label="audio", heads=2),
        _array("video_a", "video_to_audio", queries="audio", label="video", sharded=True, heads=2),
    ]


class _Model(SimpleNamespace):
    def __init__(self, *declarations):
        super().__init__(declare_state=lambda: list(declarations))


class _StubRagged:
    """Records the label each read asked for and the tokens it was handed, and returns q."""

    def __init__(self):
        self.runs = []

    def run(self, q, k=None, v=None, label=None, return_lse=False):
        self.runs.append((label, q.shape, k.shape))
        if return_lse:
            return q, torch.zeros(q.shape[:2])
        return q


class _StubGroup:
    """A sequence-parallel group's size and rank, which pick the exchange and the shares. Nothing here exchanges."""

    def __init__(self, world_size=2, rank=0):
        self.world_size, self.rank = world_size, rank


def _within(array, ragged, **within) -> WithinStepHandle:
    return WithinStepHandle(array, {next(iter(array.reads)): ragged}, None, _bindings(WITHIN_STEP, array), **within)


def _handles(arrays, *, group=None, sdpa=frozenset()) -> dict[str, WithinStepHandle]:
    specs = build_specs(arrays)
    return node_handles(
        _Model(*arrays), "dit", {spec.resource_key: _StubRagged() for spec in specs}, sp_group=group, sdpa=sdpa,
    )


def _av_step(rows=(1,), video=9, audio=3, text=5) -> SubmoduleStep:
    def segments(label, span):
        return tuple(Segment(rid, label, span) for rid in rows)

    return SubmoduleStep(state={
        "video": ArrayStep(segments=segments("video", video)),
        "audio": ArrayStep(segments=segments("audio", audio)),
        "text_v": ArrayStep(segments=segments("text", text)),
        "text_a": ArrayStep(segments=segments("text", text)),
        "audio_v": ArrayStep(segments=segments("audio", audio)),
        "video_a": ArrayStep(segments=segments("video", video)),
    })


def _by_label(step) -> dict[str, tuple[Segment, ...]]:
    labels: dict[str, list[Segment]] = {}
    for segment in step.segments:
        labels.setdefault(segment.label, []).append(segment)
    return {label: tuple(segments) for label, segments in labels.items()}


def _ctx(rows=(1,), capture=False, replay=False) -> StepContext:
    lease = SimpleNamespace(captured=True) if replay else None
    return StepContext(request_ids=list(rows), graph_walk="gen", slot=0, capture=capture, slot_lease=lease)


# The specs


def test_reads_of_one_kind_and_config_share_the_first_ones_resource():
    specs = build_specs(_av())
    assert [(type(s), s.resource_key) for s in specs] == [
        (RaggedAttentionSpec, "video_self"), (RaggedAttentionSpec, "audio_self"),
        (RaggedCrossAttentionSpec, "video_text"), (RaggedCrossAttentionSpec, "audio_text"),
    ], "two heads' audio cross reads and the video one should make two cross resources"


def test_a_self_read_and_a_cross_read_of_one_geometry_stay_apart():
    specs = build_specs([_array("a", "a_self"), _array("b", "b_cross", queries="a")])
    assert [type(s) for s in specs] == [RaggedAttentionSpec, RaggedCrossAttentionSpec]


def test_reads_on_different_nodes_never_share_a_resource():
    specs = build_specs([_array("a", nodes=("x",)), _array("b", nodes=("y",))])
    assert [s.resource_key for s in specs] == ["a_read", "b_read"]


@pytest.mark.parametrize("key, narrows", [
    ("video_self", True), ("audio_self", False), ("video_text", False), ("audio_text", False),
])
def test_heads_narrow_by_sp_only_where_both_sides_are_sharded(key, narrows):
    specs = {spec.resource_key: spec for spec in build_specs(_av())}
    assert specs[key].config.shard_over_sp is narrows


def test_a_node_with_nothing_sharded_keeps_the_default_narrowing():
    (spec,) = build_specs([_array("a")])
    assert spec.config.shard_over_sp, "a model with no sequence parallelism leaves the default"


def test_a_declared_dtype_reaches_the_spec():
    (spec,) = build_specs([_array("a", dtype=torch.bfloat16)])
    assert spec.config.dtype is torch.bfloat16


def test_the_new_settings_reach_the_shared_spec_from_any_of_its_reads():
    specs = build_specs(_av(), kernel_settings={
        "video_to_audio": {"workspace_mb": 128}, "audio_self": {"max_segments_per_request": 3},
        "video_self": {"compile_op": True},
    })
    by_key = {spec.resource_key: spec for spec in specs}
    assert by_key["audio_text"].config.workspace_mb == 128
    assert by_key["audio_self"].config.max_segments_per_request == 3
    assert by_key["video_self"].config.compile_op


@pytest.mark.parametrize("array, cause", [
    (Array("a", nodes={"dit"}, layers={0}, entry=KVEntry(heads=HEADS, head_dim=DIM), max_len=8, seq_sharded=True,
           reads={"r": Read(mask=Causal(), op=Attend(q_heads=HEADS))}), "seq_sharded and queries take max_len 0"),
    (Array("a", nodes={"dit"}, layers={0}, entry=_entry(), max_len=0, labels={"x", "y"},
           reads={"r": Read(mask=Block(), op=Attend(q_heads=HEADS))}), "max_len 0 takes one label"),
    (Array("a", nodes={"dit"}, layers={0}, entry=_entry(dtype=torch.bfloat16), max_len=8,
           reads={"r": Read(mask=Causal(), op=Attend(q_heads=HEADS))}), "with no dtype"),
])
def test_what_todays_specs_cant_hold_is_refused_by_name(array, cause):
    with pytest.raises(ValueError, match=f"Array 'a': no layout yet: .*{cause}"):
        build_specs([array])


def test_a_seq_sharded_value_is_refused_by_name():
    array = Value("v", nodes={"dit"}, layers={0}, entry=Entry({"x": TensorType((1,), dtype=None)}))
    sharded = Array("v", nodes=array.nodes, layers=array.layers, entry=array.entry, max_len=1, rewritable=True,
                    seq_sharded=True)
    with pytest.raises(ValueError, match="Array 'v': no layout yet"):
        build_specs([sharded])


# The steps, per rank


@pytest.mark.parametrize("world_size, rank, share", [(1, 0, 9), (2, 0, 5), (2, 1, 4)])
def test_each_read_plans_over_the_segments_its_kernel_sees(world_size, rank, share):
    group = _StubGroup(world_size, rank)
    expanded = state_steps(_av_step(rows=(1, 2)), _handles(_av(), group=group))
    def segments(label, span):
        return tuple(Segment(rid, label, span) for rid in (1, 2))

    video_self = expanded.steps["video_self"]
    assert isinstance(video_self, AttentionStep) and not video_self.causal
    assert tuple(video_self.segments) == segments("video", 9), "Ulysses runs over the whole sequence after the exchange"
    assert tuple(expanded.steps["audio_self"].segments) == segments("audio", 3)
    video_cross = expanded.steps["video_text"]
    assert isinstance(video_cross, RaggedCrossAttentionStep) and video_cross.pairs == (("video", "text"),)
    assert _by_label(video_cross) == {"video": segments("video", share), "text": segments("text", 5)}
    audio_cross = expanded.steps["audio_text"]
    assert audio_cross.pairs == (("audio", "text"), ("video", "audio"), ("audio", "video"))
    assert _by_label(audio_cross) == {
        "video": segments("video", share), "audio": segments("audio", 3), "text": segments("text", 5),
    }


def test_a_cross_read_with_both_sides_sharded_plans_over_whole_segments():
    arrays = [_array("a", "a_self", sharded=True), _array("b", "b_cross", queries="a", sharded=True)]
    step = SubmoduleStep(state={
        "a": ArrayStep(segments=(Segment(1, "a", 9),)), "b": ArrayStep(segments=(Segment(1, "b", 7),)),
    })
    expanded = state_steps(step, _handles(arrays, group=_StubGroup()))
    assert _by_label(expanded.steps["b_cross"]) == {"a": (Segment(1, "a", 9),), "b": (Segment(1, "b", 7),)}


def test_arrays_sharing_a_resource_plan_each_label_once():
    arrays = [_array("image"), _array("caption"), _array("joint")]
    step = SubmoduleStep(state={
        "image": ArrayStep(segments=(Segment(1, "image", 4),)),
        "caption": ArrayStep(segments=(Segment(1, "caption", 2),)),
        "joint": ArrayStep(segments=(Segment(1, "joint", 6),)),
    })
    expanded = state_steps(step, _handles(arrays))
    assert expanded.steps.keys() == {"image_read"}, "three arrays of one geometry are one resource"
    assert _by_label(expanded.steps["image_read"]) == {
        "image": (Segment(1, "image", 4),), "caption": (Segment(1, "caption", 2),), "joint": (Segment(1, "joint", 6),),
    }


def test_two_arrays_with_one_label_and_other_segments_on_one_resource_are_refused():
    arrays = [_array("a", label="x"), _array("b", label="x")]
    step = SubmoduleStep(state={
        "a": ArrayStep(segments=(Segment(1, "x", 4),)), "b": ArrayStep(segments=(Segment(1, "x", 2),)),
    })
    with pytest.raises(ValueError, match="Array 'b': label 'x' on its resource already holds other segments"):
        state_steps(step, _handles(arrays))


def test_a_cross_read_whose_query_array_isnt_in_the_step_is_refused():
    arrays = [_array("a"), _array("b", queries="a")]
    step = SubmoduleStep(state={"b": ArrayStep(segments=(Segment(1, "b", 2),))})
    with pytest.raises(ValueError, match="Array 'b': a step lists it without 'a', whose segments lay out its queries"):
        state_steps(step, _handles(arrays))


def test_a_cross_read_over_other_requests_is_refused_when_it_opens():
    arrays = [_array("a"), _array("b", queries="a")]
    step = SubmoduleStep(state={
        "a": ArrayStep(segments=(Segment(1, "a", 4), Segment(2, "a", 4))),
        "b": ArrayStep(segments=(Segment(2, "b", 2), Segment(1, "b", 2))),
    })
    expanded = state_steps(step, _handles(arrays))
    expanded.set_ctx(_ctx(rows=(1, 2)))
    with pytest.raises(ValueError, match="Array 'b': .*same requests in the same order"):
        expanded.begin_calls()


def test_a_ulysses_exchange_over_mixed_lengths_is_refused_when_it_opens():
    arrays = [_array("a", sharded=True)]
    step = SubmoduleStep(state={"a": ArrayStep(segments=(Segment(1, "a", 4), Segment(2, "a", 6)))})
    expanded = state_steps(step, _handles(arrays, group=_StubGroup()))
    expanded.set_ctx(_ctx(rows=(1, 2)))
    with pytest.raises(ValueError, match="Array 'a': no layout yet: a Ulysses exchange over spans"):
        expanded.begin_calls()


def test_a_replay_skips_the_cross_check():
    arrays = [_array("a"), _array("b", queries="a")]
    step = SubmoduleStep(state={
        "a": ArrayStep(segments=(Segment(1, "a", 4), Segment(2, "a", 4))),
        "b": ArrayStep(segments=(Segment(2, "b", 2), Segment(1, "b", 2))),
    })
    expanded = state_steps(step, _handles(arrays))
    expanded.set_ctx(_ctx(rows=(1, 2), replay=True))
    expanded.begin_calls()
    expanded.end_calls()


# The exchange


def _q(tokens=4, heads=HEADS):
    return torch.ones(1, tokens, heads, DIM)


def _opened(handle, segments):
    step = SubmoduleStep(segments=segments, state={handle.name: ArrayStep()})
    step.set_ctx(_ctx())
    step.set_state_handles((handle,))
    step.begin_calls()
    return step


def test_a_read_with_no_exchange_packs_rows_and_unpacks_them_as_main_does():
    ragged = _StubRagged()
    array = _array("a", label="main")
    h = _within(array, ragged)
    _opened(h, [Segment(1, "main", 4)])
    h.add(0, k=_q(), v=_q())
    assert h.read(0, _q()).shape == (1, 4, HEADS, DIM)
    assert ragged.runs == [("main", (4, HEADS, DIM), (4, HEADS, DIM))], "the kernel takes packed tokens"


def test_a_read_of_packed_tokens_goes_straight_to_the_kernel():
    ragged = _StubRagged()
    h = _within(_array("a", label="main", packed=True), ragged)
    _opened(h, [Segment(1, "main", 4)])
    h.add(0, k=_q()[0], v=_q()[0])
    assert h.read(0, _q()[0]).shape == (4, HEADS, DIM)


def test_a_cross_read_runs_under_the_label_of_both_sides():
    ragged = _StubRagged()
    array = _array("b", queries="a", label="text")
    h = _within(array, ragged, queried=_array("a", label="video"))
    step = SubmoduleStep(state={
        "a": ArrayStep(segments=(Segment(1, "video", 4),)), "b": ArrayStep(segments=(Segment(1, "text", 2),)),
    })
    step.set_ctx(_ctx())
    h.begin_step(step)
    h.add(0, k=_q(tokens=2), v=_q(tokens=2))
    h.read(0, _q())
    assert ragged.runs[0][0] == "video<-text"


@pytest.mark.parametrize("sharded, queried_sharded, exchange", [
    (True, True, "ulysses"), (True, False, "merge"), (False, True, "plain"), (False, False, "plain"),
])
def test_the_two_flags_pick_the_exchange(monkeypatch, sharded, queried_sharded, exchange):
    calls = []
    monkeypatch.setattr(adapters, "ulysses_attention", lambda *a, **k: calls.append(("ulysses", a[1], k)))
    monkeypatch.setattr(adapters, "attention_over_sharded_keys", lambda *a, **k: calls.append(("merge", a[1], k)))
    queried = _array("a", sharded=queried_sharded)
    array = _array("b", queries="a", sharded=sharded)
    h = _within(array, _StubRagged(), queried=queried, sp_group=_StubGroup())
    step = SubmoduleStep(state={
        "a": ArrayStep(segments=(Segment(1, "a", 9),)), "b": ArrayStep(segments=(Segment(1, "b", 9),)),
    })
    step.set_ctx(_ctx())
    h.begin_step(step)
    h.add(0, k=_q(), v=_q())
    h.read(0, _q())
    assert (calls[0][0] if calls else "plain") == exchange
    if exchange == "ulysses":
        assert h._seq_sizes == [5, 4], "rank 0 takes the odd token, as sp_seq_split does"
        assert calls[0][2] == {"seq_dim": 1, "head_dim": 2}


def test_no_flag_exchanges_at_sp1(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("attention_over_sharded_keys breaks the graph, so it must not run at SP1")

    monkeypatch.setattr(adapters, "ulysses_attention", fail)
    monkeypatch.setattr(adapters, "attention_over_sharded_keys", fail)
    h = _within(_array("a", sharded=True), _StubRagged(), sp_group=_StubGroup(world_size=1))
    _opened(h, [Segment(1, "a", 4)])
    h.add(0, k=_q(), v=_q())
    h.read(0, _q())


def test_ulysses_gets_one_callable_per_array_never_one_per_layer(monkeypatch):
    callables = []
    monkeypatch.setattr(adapters, "ulysses_attention", lambda group, q, k, v, run, *a, **kw: callables.append(run))
    h = _within(_array("a", sharded=True), _StubRagged(), sp_group=_StubGroup())
    _opened(h, [Segment(1, "a", 8)])
    for layer in (0, 1):
        h.add(layer, k=_q(), v=_q())
        h.read(layer, _q())
    assert callables[0] is callables[1], "Dynamo would retrace ulysses_attention's frame once per layer"


# backend: sdpa


def test_a_resource_on_sdpa_builds_no_spec():
    assert build_specs([_array("a")], sdpa={"a_read"}) == []


def test_kernel_settings_on_a_resource_on_sdpa_set_nothing_and_still_refuse_a_bad_field():
    assert build_specs([_array("a")], sdpa={"a_read"}, kernel_settings={"a_read": {"workspace_mb": 1}}) == []
    with pytest.raises(ValueError, match="it may set only"):
        build_specs([_array("a")], sdpa={"a_read"}, kernel_settings={"a_read": {"backend": "dense"}})


def test_sdpa_on_a_resource_no_read_runs_on_is_refused():
    with pytest.raises(ValueError, match=r"resources \['vae'\]: backend sdpa, and no read runs on them"):
        build_specs([_array("a")], sdpa={"a_read", "vae"})


def test_a_kernel_setting_of_a_read_on_sdpa_is_refused_by_name():
    handle = node_handles(_Model(_array("a")), "dit", {}, sdpa={"a_read"})["a"]
    with pytest.raises(ValueError, match="Array 'a': no max_segments_per_request for read 'a_read', whose binding "
                                         "SdpaAttend takes no kernel settings"):
        kernel_setting(handle, "a_read", "max_segments_per_request")


def test_a_step_of_reads_on_sdpa_is_no_step_at_all():
    handles = node_handles(_Model(_array("a")), "dit", {}, sdpa={"a_read"})
    step = SubmoduleStep(state={"a": ArrayStep(segments=(Segment(1, "a", 4),))})
    assert state_steps(step, handles) is None, "a step on sdpa declares no step, and its forward still runs"


def test_a_read_on_sdpa_equals_sdpa_with_no_step_open():
    q, k, v = (torch.randn(2, 4, HEADS, DIM) for _ in range(3))
    h = node_handles(_Model(_array("a")), "dit", {}, sdpa={"a_read"})["a"]
    h.add(0, k=k, v=v)
    assert torch.equal(h.read(0, q), sdpa_attention(q, k, v))


def test_a_read_on_sdpa_refuses_packed_tokens():
    with pytest.raises(ValueError, match="Array 'a': no layout yet: reads\\['a_read'\\] runs SDPA, which can't tell"):
        node_handles(_Model(_array("a", packed=True)), "dit", {}, sdpa={"a_read"})


def test_the_setting_is_read_per_resource_from_its_block():
    config = {"resources": {"a_read": {"backend": "sdpa"}, "b_read": {"backend": "dense"}, "kv": {"page_size": 8}}}
    assert sdpa_resources(config) == {"a_read"}


@pytest.mark.parametrize("config, cause", [
    ({"node_groups": [{"node_names": ["dit"], "attention_backend": "sdpa"}]},
     "node_groups\\[0\\].attention_backend: the backend is a resource's setting"),
    ({"resources": {"a_read": {"backend": "sdpa", "max_segments_per_step": 2}}},
     "resources.a_read: backend sdpa builds no resource, so it takes no other setting"),
])
def test_a_bad_setting_is_refused_naming_the_cause(config, cause):
    with pytest.raises(ValueError, match=cause):
        sdpa_resources(config)


def test_sdpa_on_a_read_that_keeps_entries_is_refused_at_load():
    array = Array("kv", nodes={"llm"}, layers={0}, entry=KVEntry(heads=HEADS, head_dim=DIM), max_len=8,
                  reads={"r": Read(mask=Causal(), op=Attend(q_heads=HEADS))})
    with pytest.raises(ValueError, match="Array 'kv': no layout yet: reads\\['r'\\].mask must be Block, the only mask"):
        build_specs([array], sdpa={"r"})


def test_a_read_sharded_on_both_sides_needs_heads_that_divide_by_tp_times_sp():
    (problem,) = deployment_problems(_Model(_array("a", sharded=True, heads=HEADS)), {}, head_shards={"dit": 3})
    assert problem.startswith("Array 'a'.reads['a_read']: q_heads 4 doesn't divide by 3")
    assert deployment_problems(_Model(_array("a", sharded=True)), {}, head_shards={"dit": 4}) == []
    assert deployment_problems(_Model(_array("a")), {}, head_shards={"dit": 3}) == [], "only both sides narrow"

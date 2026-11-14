"""An array whose read is `Retained`: a sink and a window of a stream that starts with a shared source, on a bounded KV
resource.

Token2wav's caches keep, per request, a sink and a window of a stream that starts with its voice's own caches, which
every request of that voice reads where they live. The read runs today's `BoundedKVSpec`; a step names each row's source
and the resource fixes where its stream stands from what it committed, so the model passes no positions. The numerics
are token2wav's own tests' (``test_token2wav.py``), which serve windows through these handles against the one-request
reference.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import StepRunner
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.bounded import BoundedKVSpec, BoundedKVStep, SinkWindow
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Causal,
    Entry,
    KVEntry,
    Read,
    Retained,
    Source,
    TensorType,
    retained,
)
from mstar.engine.state.adapters import build_specs, node_handles, state_steps
from mstar.engine.state.analysis import state_declaration_problems

HEADS, DIM, LAYERS = 2, 16, 3


def _array(mask=None, *, source=Source(min_len=2, max_len=6), heads=HEADS, bias=False, **read) -> Array:
    kv = TensorType((HEADS, DIM), dtype=torch.float32)
    return Array(
        "cache", nodes={"vocoder"}, layers=frozenset(range(LAYERS)), entry=Entry({"k": kv, "v": kv}), max_len=64,
        source=source,
        reads={"cache_attn": Read(
            mask=mask or Retained(sink=1, window=4, source_in_sink=True), op=Attend(q_heads=heads, distance_bias=bias),
            packed=read.pop("packed", False), **read,
        )},
    )


def _handles(array: Array):
    (spec,) = build_specs([array], kernel_settings={"cache": {"max_slots": 2}})
    resources = {"cache": build_resource(spec, EngineResourceInfo(device=torch.device("cpu"), kv_dtype=None))}
    return resources, node_handles(SimpleNamespace(declare_state=lambda: [array]), "vocoder", resources)


def _sources(n=2, length=6):
    torch.manual_seed(0)
    return {field: torch.randn(n, LAYERS, length, HEADS, DIM) for field in "kv"}, [6, 4][:n]


def _step(rows, span, sources=None, keep=True) -> SubmoduleStep:
    return SubmoduleStep(state={"cache": ArrayStep(
        segments=tuple(Segment(rid, "main", span) for rid in rows), keep=keep,
        sources=sources if sources is not None else dict.fromkeys(rows, 0),
    )})


def test_a_retained_read_builds_a_bounded_kv_with_the_masks_retention():
    (spec,) = build_specs([_array(Retained(sink=1, window=4, source_in_sink=True, reverse_steps=True))])
    assert isinstance(spec, BoundedKVSpec) and spec.resource_key == "cache"
    config = spec.config
    assert (config.num_layers, config.num_heads, config.head_dim, config.rows_per_request) == (LAYERS, HEADS, DIM, 1)
    assert (config.min_source_len, config.max_source_len, config.dtype) == (2, 6, torch.float32)
    assert config.reverse_step_order
    assert [config.retention_for(n) for n in (2, 6)] == [SinkWindow(3, 4), SinkWindow(7, 4)]


def test_an_entry_with_a_rows_axis_holds_a_requests_rows_apart():
    """A guidance pair: each of an entry's rows attends its own keys under its own query row, a request's rows
    consecutive rows of the read's queries; the resource holds them as rows of a request."""
    kv = TensorType((2, HEADS, DIM), dtype=torch.float32)
    array = Array(
        "cache", nodes={"vocoder"}, layers=frozenset(range(LAYERS)), entry=Entry({"k": kv, "v": kv}), max_len=64,
        source=Source(min_len=2, max_len=6),
        reads={"cache_attn": Read(mask=Retained(sink=1, window=4, source_in_sink=True), op=Attend(q_heads=HEADS),
                                  packed=False)},
    )
    (spec,) = build_specs([array])
    assert (spec.config.rows_per_request, spec.config.num_heads) == (2, HEADS)
    resources, handles = _handles(array)
    torch.manual_seed(0)
    fields = {field: torch.randn(1, LAYERS, 6, 2, HEADS, DIM) for field in "kv"}
    handles["cache"].set_sources(fields, [6])
    runner = StepRunner(resources)
    step = state_steps(_step([7, 8], 2, {7: 0, 8: 0}), handles)
    step.set_ctx(StepContext(request_ids=[7, 8], graph_walk="w", slot=0, capture=False))
    assert runner.admit(step).ok
    runner.plan(step)
    step.begin_calls()
    h = handles["cache"]
    # two requests of two rows each: four query rows, request-major
    x = torch.randn(4, 3, HEADS, DIM)
    for layer in range(LAYERS):
        h.add(layer, k=x, v=x)
        out = h.read(layer, x)
    assert out.shape == (4, 3, HEADS, DIM)
    # with the same source rows and the same step, a request's two rows differ only by their own entries
    assert not torch.allclose(out[0], out[1])
    step.check_calls()
    step.end_calls()
    runner.commit(step)


def test_a_step_names_each_rows_source_and_no_position():
    resources, handles = _handles(_array())
    handles["cache"].set_sources(*_sources())
    expanded = state_steps(_step([7, 8], 3, {7: 0, 8: 1}), handles)
    assert expanded.steps == {"cache": BoundedKVStep(
        segments=(Segment(7, "main", 3), Segment(8, "main", 3)), sources={7: (6, 0), 8: (4, 1)},
    )}


def _run(resources, handles, rows, span, sources, t=4, bias=None):
    """One step through the runner and the handle: admit, plan, add and read every layer, commit. The read's output
    and each row's retained count."""
    runner = StepRunner(resources)
    step = state_steps(_step(rows, span, sources), handles)
    step.set_ctx(StepContext(request_ids=list(rows), graph_walk="w", slot=0, capture=False))
    assert runner.admit(step).ok
    runner.plan(step)
    step.begin_calls()
    h = handles["cache"]
    x = torch.randn(len(rows), t, HEADS, DIM)
    for layer in range(LAYERS):
        h.add(layer, k=x, v=x)
        out = h.read(layer, x, **({} if bias is None else {"distance_bias": bias}))
    counts = retained(h).clone()
    step.check_calls()
    step.end_calls()
    runner.commit(step)
    return out, counts


def test_the_resource_fixes_where_each_rows_stream_stands_from_what_it_committed():
    """A row's retained count grows by its span each step, until its sink and window are full: the model passes no
    position, and a refused step leaves nothing to undo."""
    resources, handles = _handles(_array())
    handles["cache"].set_sources(*_sources())
    _, first = _run(resources, handles, [7, 8], 3, {7: 0, 8: 1})
    # the source alone at the first step: 6 and 4 entries
    assert first.tolist() == [6, 4]
    _, second = _run(resources, handles, [7, 8], 3, {7: 0, 8: 1})
    # each kept its first written entry in the sink (source + 1) and its window of 4 holds what came after
    assert second.tolist() == [6 + 3, 4 + 3]
    _, third = _run(resources, handles, [7, 8], 0, {7: 0, 8: 1})
    assert third.tolist() == [6 + 1 + 4, 4 + 1 + 4]


def test_a_read_returns_rows_of_one_length_and_matches_its_own_attention():
    """With nothing kept yet and no source rows read past their length, a row attends its source and its own step:
    the handle's rows of one length are the kernel's, transposed."""
    resources, handles = _handles(_array())
    fields, lengths = _sources(1)
    handles["cache"].set_sources(fields, lengths)
    torch.manual_seed(1)
    out, _ = _run(resources, handles, [7], 2, {7: 0}, t=2)
    assert out.shape == (1, 2, HEADS, DIM)


@pytest.mark.parametrize("change, cause", [
    (dict(sources={}), r"rows \[7\] name no source, and the array declares one"),
    (dict(keep=False), r"keeps nothing on a Retained array"),
])
def test_a_step_a_retained_array_cant_run_is_refused_when_it_opens(change, cause):
    _, handles = _handles(_array())
    handles["cache"].set_sources(*_sources())
    step = state_steps(_step([7], 2, **change), handles)
    step.set_ctx(StepContext(request_ids=[7], graph_walk="w", slot=0, capture=False))
    with pytest.raises(ValueError, match=r"Array 'cache': .*" + cause):
        step.begin_calls()


def test_sources_are_refused_after_the_first_step_and_in_the_wrong_shape_or_dtype():
    resources, handles = _handles(_array())
    h = handles["cache"]
    fields, lengths = _sources()
    with pytest.raises(ValueError, match="takes fields k and v"):
        h.set_sources({"k": fields["k"]}, lengths)
    with pytest.raises(ValueError, match="a length per source"):
        h.set_sources(fields, lengths[:1])
    with pytest.raises(ValueError, match=r"outside 2\.\.6"):
        h.set_sources(fields, [6, 1])
    with pytest.raises(ValueError, match="the array holds torch.float32"):
        h.set_sources({f: t.half() for f, t in fields.items()}, lengths)
    h.set_sources(fields, lengths)
    _run(resources, handles, [7], 1, {7: 0})
    with pytest.raises(ValueError, match="after the handle's first step"):
        h.set_sources(fields, lengths)


def test_a_read_takes_a_distance_bias_exactly_where_its_op_declares_one():
    resources, handles = _handles(_array(bias=True))
    handles["cache"].set_sources(*_sources())
    with pytest.raises(ValueError, match="takes distance_bias exactly where its op declares one"):
        _run(resources, handles, [7], 1, {7: 0})


@pytest.mark.parametrize("array, cause", [
    (lambda: _array(heads=2 * HEADS), "the bounded kernel takes the entry's 2 heads"),
    (lambda: _array(packed=True), "takes rows of one length"),
    (lambda: _array(walks={"w"}), "takes rows of one length"),
])
def test_what_a_bounded_resource_cant_hold_is_refused_by_name(array, cause):
    with pytest.raises(ValueError, match=r"Array 'cache': no layout yet: .*" + cause):
        build_specs([array()])


def test_a_distance_bias_on_another_read_is_refused_by_name():
    paged = Array("kv", nodes={"llm"}, layers=frozenset({0}), entry=KVEntry(heads=2, head_dim=16), max_len=8,
                  reads={"attn": Read(mask=Causal(), op=Attend(q_heads=2, distance_bias=True))})
    with pytest.raises(ValueError, match="takes a distance_bias, which only a Retained read runs"):
        build_specs([paged])


def test_a_retained_mask_that_keeps_nothing_or_counts_a_missing_source_is_refused():
    with pytest.raises(ValueError, match="keep something"):
        _array(Retained())
    with pytest.raises(ValueError, match="counts the source's length, and the array declares no source"):
        _array(Retained(window=2, source_in_window=True), source=None)


def test_a_source_on_an_array_with_no_retained_read_is_a_problem():
    kv = KVEntry(heads=2, head_dim=16)
    array = Array("kv", nodes={"llm"}, layers=frozenset({0}), entry=kv, max_len=8, source=Source(max_len=4),
                  reads={"attn": Read(mask=Causal(), op=Attend(q_heads=2))})
    model = SimpleNamespace(
        declare_state=lambda: [array], kv_labels=dict, get_graph_walk_graphs=lambda: {},
        get_combined_graph_walks=dict,
    )
    assert any("only a Retained read reads a source" in problem for problem in state_declaration_problems(model))

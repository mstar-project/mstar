"""The engine builds today's steps, specs, layouts and request configs from declarations, so nothing downstream changes.

The resources, the scheduler and the wire read those objects as they did, and a field the expansion drops or a default
it changes would make a step plan something else silently. A request's settings and layouts come from model code on
another process than the workers', so a name the model doesn't declare fails that request and not a worker later.
"""

from __future__ import annotations

import sys
from dataclasses import fields, replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine import accelerator_graph_runner as agr
from mstar.engine import engine as engine_module
from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig
from mstar.engine.engine import Engine
from mstar.engine.resources import AttentionStep, Segment, StepContext, SubmoduleStep
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig, RaggedAttentionSpec
from mstar.engine.resources.kv import manager as kv_manager
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.spec import CombinedSplit
from mstar.engine.resources.step import BucketKey, SlotLease
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Block,
    Causal,
    CausalConv,
    Entry,
    Fork,
    GatedDeltaNet,
    KVEntry,
    Layout,
    Positions,
    Read,
    Rope,
    Sampler,
    SampleSettings,
    SampleStep,
    Span,
    TensorType,
    Value,
    Window,
    sample,
)
from mstar.engine.state.adapters import (
    PagedArrayHandle,
    WithinStepHandle,
    build_specs,
    kv_layouts,
    node_handles,
    sampling_configs,
    state_steps,
)
from mstar.engine.state.analysis import deployment_problems
from mstar.engine.state.calls import SamplerHandle
from mstar.model.base import LabelLayout, Model
from mstar.model.submodule_base import ARNodeSubmodule, DeviceLoopback, NodeInputs, NodeSubmodule, engine_step
from mstar.testing_utils.model import make_model

SEGMENTS = [Segment(1, "main", 4), Segment(2, "main", 1)]
_READ = Read(mask=Causal(), op=Attend(q_heads=4))


def _kv(**overrides) -> Array:
    return Array("kv", **{
        "nodes": {"LLM"}, "layers": {0, 1}, "entry": KVEntry(heads=4, head_dim=8), "max_len": 16,
        "reads": {"attn": _READ}, **overrides,
    })


def _value() -> Array:
    entry = Entry({
        "state": TensorType((4, 8, 8), dtype=torch.float32, shard=0),
        "conv": TensorType((32, 3), dtype=None, shard=0),
    })
    updates = {"conv": CausalConv(), "linear": GatedDeltaNet()}
    return Value("gdn", nodes={"LLM"}, layers={2}, entry=entry, updates=updates)


def _within() -> Array:
    return Array("vision", nodes={"vit"}, layers={0}, entry=KVEntry(heads=4, head_dim=8), max_len=0,
                 reads={"vision_attn": Read(mask=Block(), op=Attend(q_heads=4))})


class _StubModel:
    """Declares what it is given, and nothing else a handle reads."""

    def __init__(self, *declarations):
        self.declarations = list(declarations)

    def declare_state(self):
        return self.declarations


class _StubKV:
    """Hands back a page tensor per layer."""

    def layer_view(self, i):
        return torch.zeros(1)


class _StubPool:
    """Hands back a block tensor per field and layer."""

    def block(self, name, i):
        return torch.zeros(1)


def _handles(*declarations, node="LLM") -> dict:
    resources = {
        "kv": _StubKV(), "attn": SimpleNamespace(), "local": SimpleNamespace(), "global": SimpleNamespace(),
        "rope": SimpleNamespace(), "kv_pos": SimpleNamespace(), "other_pos": SimpleNamespace(),
        "other_kv": _StubKV(), "gdn": _StubPool(), "linear": SimpleNamespace(), "conv": None,
        "sampler": SimpleNamespace(), "vision_attn": SimpleNamespace(),
    }
    return node_handles(_StubModel(*declarations), node, resources)


def _expanded(declared, *declarations, node="LLM", segments=SEGMENTS) -> dict:
    handles = _handles(*declarations, node=node)
    return state_steps(SubmoduleStep(segments=segments, state=declared), handles).steps


# Steps


def test_an_append_only_array_with_a_read_and_positions_expands_to_todays_three_steps():
    array = _kv(positions=Positions())
    assert _expanded({"kv": ArrayStep()}, array) == {
        "kv": KVStep(segments=SEGMENTS),
        "attn": AttentionStep(segments=SEGMENTS, causal=True),
        "rope": PositionStep(segments=SEGMENTS),
    }


def test_every_read_of_an_array_gets_a_step_of_its_own():
    array = _kv(reads={"local": Read(mask=Window(4), op=Attend(q_heads=4), layers={0}),
                       "global": Read(mask=Causal(), op=Attend(q_heads=4), layers={1})})
    steps = _expanded({"kv": ArrayStep()}, array)
    assert steps["local"] == AttentionStep(segments=SEGMENTS, causal=True)
    assert steps["global"] == AttentionStep(segments=SEGMENTS, causal=True)


def test_a_steps_array_fields_reach_todays_kv_and_position_steps():
    ids = torch.arange(5)
    placeholders = frozenset({(2, "main")})
    steps = _expanded(
        {"kv": ArrayStep(keep=True, placeholders=placeholders, eager_add=True, advance=(9, 1), position_ids=ids)},
        _kv(positions=Positions()),
    )
    assert steps["kv"] == KVStep(segments=SEGMENTS, commit=True, sink=placeholders, eager_write=True)
    assert steps["rope"].advance == (9, 1) and steps["rope"].pos_ids is ids


def test_a_step_that_keeps_nothing_advances_no_position():
    steps = _expanded({"kv": ArrayStep(keep=False, advance=(9, 1))}, _kv(positions=Positions()))
    assert steps["kv"].commit is False
    assert steps["rope"].advance == (0, 0), "a row keeps its next position when its entries weren't kept"


def test_no_position_ids_stays_none_which_a_resident_capture_needs():
    steps = _expanded({"kv": ArrayStep()}, _kv(positions=Positions()))
    assert steps["rope"].pos_ids is None, "a dict of None would read as explicit ids, and resident planning refuses it"


def test_forks_reach_kv_at_their_time_with_their_rows():
    forks = (Fork(src="main", dst="cfg", at="start", rows=frozenset({1})),
             Fork(src="cfg", dst="main", at="end", rows=frozenset({1})))
    steps = _expanded({"kv": ArrayStep(forks=forks)}, _kv(labels={"main", "cfg"}))
    assert steps["kv"].pre_forks == (("main", "cfg"),)
    assert steps["kv"].post_forks == (("cfg", "main"),)
    assert steps["kv"].fork_rids == frozenset({1})


def test_forks_naming_different_rows_have_no_layout_yet():
    forks = (Fork(src="main", dst="a", at="start", rows=frozenset({1})), Fork(src="main", dst="b", at="start"))
    with pytest.raises(ValueError, match=r"Array 'kv': no layout yet: a step's forks name different rows"):
        _expanded({"kv": ArrayStep(forks=forks)}, _kv(labels={"main", "a", "b"}))


def test_a_value_expands_to_its_pool_and_its_gated_delta_net():
    assert _expanded({"gdn": ArrayStep()}, _kv(), _value()) == {
        "gdn": RecurrentStep(segments=SEGMENTS), "linear": LinearAttnStep(segments=SEGMENTS),
    }


def test_a_value_fork_copies_its_entry():
    fork = Fork(src="main", dst="cfg", at="end")
    steps = _expanded({"gdn": ArrayStep(forks=(fork,))}, _kv(), _value())
    assert steps["gdn"].post_forks == (("main", "cfg"),) and steps["gdn"].fork_rids is None


def test_an_array_with_max_len_zero_has_no_pool_and_a_non_causal_read():
    assert _expanded({"vision": ArrayStep()}, _within(), node="vit") == {
        "vision_attn": AttentionStep(segments=SEGMENTS, causal=False),
    }


def test_a_sampler_step_reaches_todays_fields():
    tokens = {1: torch.arange(3)}
    kept = frozenset({2})
    steps = _expanded(
        {"sampler": SampleStep(apply_penalty=False, apply_filters=False, prompt_tokens=tokens, kept_rows=kept)},
        Sampler("sampler", nodes={"LLM"}, vocab_size=8),
    )
    assert steps == {"sampler": SamplerStep(
        segments=SEGMENTS, apply_penalty=False, apply_filters=False, prefill_tracked_tokens=tokens, kept_rids=kept,
    )}


def test_an_array_steps_own_segments_reach_its_steps_and_the_others_keep_the_steps():
    own = (Segment(1, "main", 2),)
    steps = _expanded(
        {"kv": ArrayStep(segments=own), "sampler": SampleStep()},
        _kv(positions=Positions()), Sampler("sampler", nodes={"LLM"}, vocab_size=8),
    )
    assert {key: step.segments for key, step in steps.items()} == {
        "kv": own, "attn": own, "rope": own, "sampler": SEGMENTS,
    }


def test_expanding_keeps_what_the_step_declares_itself():
    extra = PositionStep()
    step = SubmoduleStep(segments=SEGMENTS, steps={"other": extra}, state={"kv": ArrayStep()}, cg_key_info="bucket")
    expanded = state_steps(step, _handles(_kv()))
    assert expanded.steps["other"] is extra and expanded.cg_key_info == "bucket"
    assert step.steps == {"other": extra}, "a declaration reused across calls should keep its own steps"


def test_each_step_gets_fresh_step_objects():
    handles = _handles(_kv())
    first = state_steps(SubmoduleStep(segments=[Segment(1, "main", 1)], state={"kv": ArrayStep()}), handles)
    second = state_steps(SubmoduleStep(segments=[Segment(2, "main", 2)], state={"kv": ArrayStep()}), handles)
    assert first.steps["kv"].segments == [Segment(1, "main", 1)], "SubmoduleStep writes segments onto its step objects"
    assert second.steps["kv"].segments == [Segment(2, "main", 2)]


def test_a_step_with_no_state_comes_back_as_it_was():
    step = SubmoduleStep(segments=SEGMENTS, steps={"kv": KVStep()})
    assert state_steps(step, {}) is step
    assert state_steps(None, {}) is None, "a leased vision step declares none"


def test_a_step_may_declare_nothing_in_steps():
    assert SubmoduleStep(segments=SEGMENTS, state={"kv": ArrayStep()}).steps == {}


def test_a_name_the_node_does_not_declare_is_refused():
    with pytest.raises(ValueError, match=r"state\['kv'\] is a ArrayStep, and on this node 'kv' is not declared"):
        state_steps(SubmoduleStep(segments=SEGMENTS, state={"kv": ArrayStep()}), {})


def test_an_array_needs_an_array_step_and_a_sampler_a_sample_step():
    handles = _handles(_kv(), Sampler("sampler", nodes={"LLM"}, vocab_size=8))
    with pytest.raises(ValueError, match=r"state\['kv'\] is a SampleStep, and on this node 'kv' is PagedArrayHandle"):
        state_steps(SubmoduleStep(segments=SEGMENTS, state={"kv": SampleStep()}), handles)
    with pytest.raises(ValueError, match=r"state\['sampler'\] is a ArrayStep"):
        state_steps(SubmoduleStep(segments=SEGMENTS, state={"sampler": ArrayStep()}), handles)


def test_an_array_with_no_segments_anywhere_is_refused():
    with pytest.raises(ValueError, match=r"Array 'kv': a step lists it with no segments"):
        state_steps(SubmoduleStep(state={"kv": ArrayStep()}), _handles(_kv()))


class _StubSubmodule:
    """What the engine reads off a submodule for a step."""

    segment_keys = ()

    def __init__(self, handles):
        self.state_handles = handles

    def declare_step(self, **kwargs):
        return SubmoduleStep(segments=SEGMENTS, state={"kv": ArrayStep()})


def test_engine_step_expands_what_declare_step_returns():
    step = engine_step(_StubSubmodule(_handles(_kv())), graph_walk="decode")
    assert set(step.steps) == {"kv", "attn"}
    assert [h.name for h in step.state_handles] == ["kv"]


# The step's lifecycle


def _driven_step(handle, ctx=None) -> SubmoduleStep:
    state = {handle.name: SampleStep() if isinstance(handle, SamplerHandle) else ArrayStep()}
    step = SubmoduleStep(steps={}, segments=[Segment(1, "main", 1)], state=state)
    step.set_state_handles((handle,))
    step.set_ctx(ctx)
    return step


def _drive(step, forward, committed: list) -> None:
    """The engine's real `_drive_step`, with the resources and the forward cut down to what a test counts."""
    engine = object.__new__(Engine)
    engine._enable_nvtx = engine._enable_profile = False
    engine._order_after_pre_plans = lambda: None
    engine._runner = SimpleNamespace(plan=lambda step, phase: None, commit=committed.append)
    engine._forward = lambda *args: forward()
    engine._foreign_planners = {}
    mgmt = SimpleNamespace(
        accelerator_graph_runner=None, resident=None, resources={}, piecewise_runners={}, fp32_precision=None,
        submodule=SimpleNamespace(preprocess=lambda *args, **kwargs: {}),
    )
    batch = SimpleNamespace(
        node_name="LLM", preplan_event=None, per_request_input_metadata=None, launch_started_event=None, committed=None,
    )
    engine._drive_step(batch, mgmt, [1], [], {}, step.ctx, step.ctx.slot_lease, False, step, False)


def _sampler() -> SamplerHandle:
    resource = SimpleNamespace(sample=lambda rows, logits: logits.argmax(-1))
    return SamplerHandle(Sampler("sampler", nodes={"LLM"}, vocab_size=8), resource)


def _eager_ctx() -> StepContext:
    return StepContext(request_ids=(1,), graph_walk="decode", slot=0, capture=False)


def test_a_forward_that_raises_leaves_no_handle_open_and_commits_nothing():
    handle, committed = _sampler(), []
    step = _driven_step(handle, _eager_ctx())

    def forward():
        assert handle._open, "a step's calls run with its handles open"
        raise RuntimeError("the forward failed")

    with pytest.raises(RuntimeError, match="the forward failed"):
        _drive(step, forward, committed)
    assert not handle._open and committed == []


def test_a_wrong_draw_count_commits_nothing_and_leaves_no_handle_open():
    handle, committed = _sampler(), []
    step = _driven_step(handle, _eager_ctx())
    with pytest.raises(ValueError, match=r"Sampler 'sampler': a step listing it sampled 0 times, and draws is 1"):
        _drive(step, dict, committed)
    assert not handle._open and committed == []


def test_a_right_draw_count_commits_and_closes_the_handle():
    handle, committed = _sampler(), []
    step = _driven_step(handle, _eager_ctx())
    _drive(step, lambda: sample(handle, torch.eye(1, 8)), committed)
    assert not handle._open and committed == [step]


def test_a_replay_commits_with_no_draws_run_in_python():
    handle, committed = _sampler(), []
    lease = SlotLease(slot=0, bucket=BucketKey("decode", 1, 1))
    step = _driven_step(handle, StepContext(
        request_ids=(1,), graph_walk="decode", slot=0, capture=False, slot_lease=lease,
    ))
    _drive(step, dict, committed)
    assert not handle._open and committed == [step]


def _captured_forward(handle, draws: int):
    def forward(**kwargs):
        for _ in range(draws):
            sample(handle, torch.eye(1, 8))
        return {}
    return forward


@pytest.fixture
def _capture_runner(monkeypatch):
    """The real `AcceleratorGraphRunner._warmed`, over one sampler handle and no resources."""
    handle = _sampler()
    step = _driven_step(handle)
    monkeypatch.setattr(agr, "engine_step", lambda *args, **kwargs: step)
    runner = object.__new__(agr.AcceleratorGraphRunner)
    runner._dummy_rows = SimpleNamespace(ensure=lambda key, n: [1], reset=lambda rids, free=False: None)
    runner._dummy_engine_inputs = lambda rids, walk: SimpleNamespace(per_request_info={}, step=None)
    runner._step_runner = SimpleNamespace(
        admit=lambda step: SimpleNamespace(ok=True), plan=lambda step: None,
    )
    runner._resident = None
    runner._graph_backend = SimpleNamespace(set_device=lambda: None, synchronize=lambda: None)
    spec = SimpleNamespace(
        bucket=BucketKey("decode", 1, 1), slot=0, bs=1, num_tokens=1, config_idx=0,
        config=BatchedAcceleratorGraphConfig(
            capture_graph_walk="decode", capture_batch_sizes=[1],
            single_request_inputs=NodeInputs(tensor_inputs={}, input_seq_len=1),
        ),
    )
    return runner, spec, handle


def test_a_capture_opens_its_handles_for_preprocess_and_for_each_forward(_capture_runner):
    runner, spec, handle = _capture_runner
    opened = []
    runner._submodule = SimpleNamespace(preprocess=lambda **kwargs: opened.append(handle._open) or {})
    runner._forward_for = lambda spec: _captured_forward(handle, 1)
    with runner._warmed(spec, forwards=0) as warmed:
        assert opened == [True] and not handle._open, "preprocess reads the step's label, and then the handle closes"
        warmed.run()
        warmed.run()
    assert not handle._open


def test_a_capture_with_a_wrong_draw_count_is_refused_and_leaves_no_handle_open(_capture_runner):
    runner, spec, handle = _capture_runner
    runner._submodule = SimpleNamespace(preprocess=lambda **kwargs: {})
    runner._forward_for = lambda spec: _captured_forward(handle, 0)
    with runner._warmed(spec, forwards=0) as warmed:
        with pytest.raises(ValueError, match=r"Sampler 'sampler': a step listing it sampled 0 times, and draws is 1"):
            warmed.run()
        assert not handle._open


def test_a_capture_whose_begin_raises_leaves_no_handle_open(_capture_runner):
    runner, spec, handle = _capture_runner
    runner._submodule = SimpleNamespace(preprocess=lambda **kwargs: {})
    begin = handle.begin_step

    def refuse(step):
        begin(step)
        raise ValueError("a step touches a label")

    handle.begin_step = refuse
    with pytest.raises(ValueError, match="a step touches a label"):
        with runner._warmed(spec, forwards=0):
            pass
    assert not handle._open, "an open handle would drop every later bucket on the next begin"


def test_a_piecewise_capture_opens_its_handles_for_each_forward():
    handle = _sampler()
    step = _driven_step(handle, StepContext(request_ids=(1,), graph_walk="decode", slot=0, capture=True))
    runner = object.__new__(agr.PiecewiseAcceleratorGraphRunner)
    runner._dummy_rows = SimpleNamespace(ensure=lambda key, n: [1], reset=lambda rids: None)
    runner._dummy_key = lambda shape, slot: "key"
    runner._config = SimpleNamespace(
        make_static_inputs=lambda shape: {}, compile=False,
        capture_fn=lambda call: _captured_forward(handle, call)(),
    )
    runner._bucket = lambda shape: None
    runner._graph_backend = SimpleNamespace(synchronize=lambda: None)
    runner._plan = lambda step, shape: None
    runner._normalize_output = lambda out: out
    shape = SimpleNamespace(bs=1, seq_lens=None)

    for draws, refused in ((1, False), (0, True)):
        runner._declare = lambda *args, **kwargs: step
        runner._call_inputs = lambda static, rids, step, draws=draws: draws
        with runner._warmed(shape, slot=0, forwards=0) as warmed:
            if refused:
                with pytest.raises(ValueError, match=r"sampled 0 times, and draws is 1"):
                    warmed.run()
            else:
                warmed.run()
            assert not handle._open


# Specs


def test_positions_become_a_position_spec_under_rope_with_todays_defaults():
    specs = build_specs([_kv(positions=Positions())])
    assert specs[-1] == PositionSpec(resource_key="rope", nodes={"LLM"}, config=PositionConfig(kv_cache="kv"))


@pytest.mark.parametrize("name, key", [("kv", "rope"), ("kv_cache", "rope"), ("thinker_kv", "thinker_pos")])
def test_a_positions_key_comes_from_the_arrays_name_alone(name, key):
    array = replace(_kv(positions=Positions()), name=name)
    assert build_specs([array])[-1].resource_key == key
    other = Array("other_kv", nodes={"LLM"}, layers={0}, entry=KVEntry(heads=4, head_dim=8), max_len=8,
                  reads={"other_attn": _READ}, positions=Positions())
    keys = {spec.resource_key for spec in build_specs([array, other]) if isinstance(spec, PositionSpec)}
    assert keys == {key, "other_pos"}, "another array's declarations don't change the key"


def test_a_rope_reaches_every_position_config_field():
    rope = Rope(theta=5.0, rotary_dim=4, interleave=True, scale=2.0, low_freq_factor=1.0, high_freq_factor=4.0,
                old_context_len=8192, dtype=torch.bfloat16, table_dtype=torch.float32)
    spec = build_specs([_kv(positions=Positions(rope=rope))])[-1]
    assert spec.config == PositionConfig(
        kv_cache="kv", rotary_dim=4, interleave=True, rope_scale=2.0, rope_theta=5.0, rope_dtype=torch.bfloat16,
        rope_table_dtype=torch.float32, low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192,
    )


def test_two_arrays_with_positions_each_get_their_own_key():
    specs = build_specs([
        Array("thinker_kv", nodes={"a"}, layers={0}, entry=KVEntry(heads=4, head_dim=8), max_len=8,
              reads={"t": _READ}, positions=Positions(rope=Rope())),
        Array("talker_kv", nodes={"b"}, layers={0}, entry=KVEntry(heads=4, head_dim=8), max_len=8,
              reads={"u": _READ}, positions=Positions(rope=Rope())),
    ])
    keys = [spec.resource_key for spec in specs if isinstance(spec, PositionSpec)]
    assert keys == ["thinker_pos", "talker_pos"]


def test_a_sampler_becomes_a_sampler_spec():
    sampler = Sampler("s", nodes={"LLM"}, vocab_size=8, penalty=False, min_p=True, eos_ids=(1, 2), draws=3)
    assert build_specs([sampler]) == [SamplerSpec(
        resource_key="s", nodes={"LLM"}, vocab_size=8, enable_repetition_penalty=False, enable_min_p=True,
        min_tokens_stop_ids=(1, 2),
    )]


def test_a_block_read_becomes_a_ragged_spec_with_the_default_compile_op():
    assert build_specs([_within()]) == [RaggedAttentionSpec(
        resource_key="vision_attn", nodes={"vit"},
        config=RaggedAttentionConfig(num_qo_heads=4, num_kv_heads=4, head_dim=8),
    )]


def test_kernel_settings_set_a_within_step_reads_segment_ceiling_and_compile_op():
    settings = {"max_segments_per_step": 64, "compile_op": True}
    config = build_specs([_within()], kernel_settings={"vision_attn": settings})[0].config
    assert (config.max_segments_per_step, config.compile_op) == (64, True)


@pytest.mark.parametrize("field, value", [("splits", ()), ("decode_fmha", True)])
def test_a_within_step_read_refuses_the_paged_kernel_settings(field, value):
    with pytest.raises(ValueError, match=rf"Array 'vision': kernel_settings\['vision_attn'\] sets \['{field}'\]"):
        build_specs([_within()], kernel_settings={"vision_attn": {field: value}})


@pytest.mark.parametrize("field, value", [("max_segments_per_step", 1), ("compile_op", True)])
def test_a_paged_read_refuses_the_within_step_kernel_settings(field, value):
    with pytest.raises(ValueError, match=rf"Array 'kv': kernel_settings\['attn'\] sets \['{field}'\]"):
        build_specs([_kv()], kernel_settings={"attn": {field: value}})


def test_kernel_settings_still_reach_a_paged_read_and_update():
    split = CombinedSplit("mixed", prefill_walks=("prefill",))
    specs = build_specs(
        [_kv(), _value()],
        kernel_settings={"attn": {"splits": (split,), "decode_fmha": True}, "linear": {"splits": (split,)}},
    )
    by_key = {spec.resource_key: spec for spec in specs}
    assert by_key["attn"].config.splits == (split,) and by_key["attn"].config.decode_fmha
    assert by_key["linear"].config.splits == (split,)


def test_a_value_links_to_the_array_that_keeps_entries_not_a_within_step_one():
    within = Array("tokens", nodes={"LLM"}, layers={0}, entry=KVEntry(heads=4, head_dim=8), max_len=0,
                   reads={"t": Read(mask=Block(), op=Attend(q_heads=4))})
    by_key = {spec.resource_key: spec for spec in build_specs([_kv(), within, _value()])}
    assert by_key["gdn"].config.kv_cache == "kv"


@pytest.mark.parametrize("array, cause", [
    (Array("v", nodes={"a"}, layers={0}, entry=KVEntry(heads=4, head_dim=8), max_len=0), "needs one read"),
    (Array("v", nodes={"a"}, layers={0}, entry=KVEntry(heads=4, head_dim=8), max_len=0,
           reads={"r": Read(mask=Block())}), r"reads\['r'\].op must be Attend"),
    (Array("v", nodes={"a"}, layers={0}, entry=Entry({"x": TensorType((1,), dtype=None)}), max_len=0,
           reads={"r": Read(mask=Block())}), "entry must be a KVEntry"),
    (Array("v", nodes={"a"}, layers={0}, entry=KVEntry(heads=4, head_dim=8), max_len=0,
           reads={"r": Read(mask=Block(), op=Attend(q_heads=4), walks={"prefill"})}),
     r"reads\['r'\].walks must be None"),
])
def test_a_within_step_array_today_has_no_spec_for_is_refused(array, cause):
    with pytest.raises(ValueError, match=rf"Array 'v': no layout yet: .*{cause}"):
        build_specs([array])


def test_a_ported_model_need_not_write_get_node_resources():
    assert Model.get_node_resources(object()) == []
    assert "get_node_resources" not in Model.__abstractmethods__


# Layouts


_SPAN = Span("ids", "prefill", 4, 4, "text_inputs")


def test_a_layout_becomes_todays_label_layout_and_a_values_is_dropped():
    layouts = kv_layouts(
        [_kv(), _value()],
        {"kv": {"main": Layout(spans=(_SPAN,), growth=(("decode_loop", 1),), scratch=3)}, "gdn": {"main": Layout()}},
        {},
    )
    assert layouts == {"kv": {"main": LabelLayout(spans=(_SPAN,), growth=(("decode_loop", 1),), scratch=3)}}


def test_keyed_spans_reach_their_label_and_only_that_one():
    layouts = kv_layouts(
        [_kv(labels={"main", "cfg"})],
        {"kv": {"main": Layout(spans=(_SPAN,)), "cfg": Layout(spans=(_SPAN,))}},
        {"kv": {"main": 0}},
    )
    assert layouts["kv"]["main"].keyed_spans == 0
    assert layouts["kv"]["cfg"].keyed_spans is None, "the default keys every span before the first rows span"


def test_a_converted_layout_serializes_like_a_hand_written_one():
    layout = Layout(spans=(_SPAN,), growth=(("decode_loop", 1),))
    converted = kv_layouts([_kv()], {"kv": {"main": layout}}, {})["kv"]["main"]
    assert converted.row() == LabelLayout(spans=(_SPAN,), growth=(("decode_loop", 1),)).row()


@pytest.mark.parametrize("state_layout, keyed, cause", [
    ({"other": {"main": Layout()}}, {}, r"state_layout\['other'\]: it names no declared array"),
    ({"kv": {"cfg": Layout()}}, {}, r"state_layout\['kv'\]: labels \['cfg'\] aren't in Array 'kv'.labels"),
    ({"kv": {"main": Layout()}}, {"other": {"main": 0}}, r"keyed_spans\['other'\]: it names no declared array"),
    ({"kv": {"main": Layout()}}, {"kv": {"cfg": 0}}, r"keyed_spans\['kv'\]: labels \['cfg'\] aren't"),
])
def test_a_layout_naming_what_the_model_does_not_declare_fails_the_request(state_layout, keyed, cause):
    with pytest.raises(RuntimeError, match=cause):
        kv_layouts([_kv()], state_layout, keyed)


def test_keyed_spans_on_a_label_with_no_layout_fail_the_request():
    with pytest.raises(RuntimeError, match=r"keyed_spans\['kv'\]: labels \['cfg'\] have no paged layout"):
        kv_layouts([_kv(labels={"main", "cfg"})], {"kv": {"main": Layout()}}, {"kv": {"cfg": 0}})


def test_keyed_spans_on_a_value_fail_the_request():
    value = Value("gdn", nodes={"LLM"}, layers={0}, entry=Entry({"x": TensorType((1,), dtype=None)}))
    with pytest.raises(RuntimeError, match=r"keyed_spans\['gdn'\]: labels \['main'\] have no paged layout"):
        kv_layouts([value], {"gdn": {"main": Layout()}}, {"gdn": {"main": 3}})


# Request settings


def test_sample_settings_have_todays_request_fields_less_the_seed_and_grammar():
    # the conductor sets both: the seed, and the grammar from the request (SamplerSpec.request_config)
    today = {f.name: f.default for f in fields(SamplingReqConfig) if f.name not in ("_seed", "grammar")}
    settings = {f.name: f.default for f in fields(SampleSettings)}
    assert settings == today


def test_settings_become_a_sampling_config_that_takes_the_conductors_seed():
    configs = sampling_configs({"s": SampleSettings(temperature=0.0, top_k=5)},
                               [Sampler("s", nodes={"LLM"}, vocab_size=8)])
    config = configs["s"]
    assert isinstance(config, SamplingReqConfig) and (config.temperature, config.top_k) == (0.0, 5)
    config.apply_conductor_config(seed=3)
    assert config.seed == 3


def test_a_config_that_is_not_settings_passes_through():
    other = SamplingReqConfig()
    assert sampling_configs({"s": other}, [])["s"] is other


def test_settings_for_no_declared_sampler_fail_the_request():
    with pytest.raises(ValueError, match=r"SampleSettings for 's', which names no declared sampler"):
        sampling_configs({"s": SampleSettings()}, [_kv()])


def test_log_probs_on_several_draws_fail_the_request():
    sampler = Sampler("s", nodes={"LLM"}, vocab_size=8, draws=2)
    with pytest.raises(ValueError, match=r"Sampler 's': return_logprobs, and draws is 2"):
        sampling_configs({"s": SampleSettings(return_logprobs=True)}, [sampler])


def test_log_probs_on_several_draws_are_refused_at_load():
    model = _StubModel(Sampler("s", nodes={"LLM"}, vocab_size=8, draws=2))
    problems = deployment_problems(model, {"s": {"enable_logprobs": True}})
    assert len(problems) == 1 and problems[0].startswith("Sampler 's'.draws") and "log-probs" in problems[0]


def test_log_probs_on_one_draw_are_allowed_at_load():
    model = _StubModel(Sampler("s", nodes={"LLM"}, vocab_size=8))
    assert deployment_problems(model, {"s": {"enable_logprobs": True}}) == []


class _LoopbackNode(ARNodeSubmodule):
    """Reads its last sampled token back on the device."""

    device_loopback = DeviceLoopback(walk="decode", signal="text_inputs", sampler="s", buffer="new_token")
    forward = prepare_inputs = preprocess = None


def test_a_device_loop_back_on_several_draws_is_refused_at_load():
    handles = {"s": SimpleNamespace(sampler=Sampler("s", nodes={"LLM"}, vocab_size=8, draws=2))}
    with pytest.raises(ValueError, match=r"Sampler 's': a device loop-back reads one token per step, and draws is 2"):
        _LoopbackNode().bind_node_state(handles)


# Binding


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bound = None

    def bind_state(self, handles):
        self.bound = handles


class _Node(NodeSubmodule):
    forward = prepare_inputs = None

    def __init__(self):
        super().__init__()
        self.layer = _Layer()
        self.bound = None

    def bind_state(self, handles):
        self.bound = handles


def test_handles_reach_the_layers_and_the_submodule_itself():
    node, handles = _Node(), {"kv": object()}
    node.bind_node_state(handles)
    assert node.layer.bound is handles and node.bound is handles and node.state_handles is handles


def test_a_node_gets_only_its_own_handles():
    declarations = (_kv(), _within(), Sampler("sampler", nodes={"LLM"}, vocab_size=8))
    assert set(_handles(*declarations)) == {"kv", "sampler"}
    assert set(_handles(*declarations, node="vit")) == {"vision"}


def test_an_array_that_keeps_nothing_gets_the_handle_that_stays_in_its_graph():
    assert type(_handles(_kv())["kv"]) is PagedArrayHandle
    assert type(_handles(_within(), node="vit")["vision"]) is WithinStepHandle


class _EngineNode(NodeSubmodule):
    declared_handles = None

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        raise NotImplementedError

    def forward(self, graph_walk, engine_inputs, **kwargs):
        raise NotImplementedError

    def bind_node_state(self, handles):
        self.declared_handles = handles


def _load_engine(monkeypatch, node, model):
    monkeypatch.setattr(engine_module, "node_handles", lambda model, name, resources: {"kv": name})
    joint = SimpleNamespace(world_size=1, tp_group=SimpleNamespace(barrier=lambda: None))
    groups = SimpleNamespace(
        get_joint_group_for_node=lambda name: joint, get_instance_world_size_for_node=lambda name: 1,
        get_instance_rank_for_node=lambda name: 0,
    )
    engine = Engine(None, autocast_dtype=None)
    engine.load_model(
        submodules={"LLM": node}, specs=[], parallel_groups=groups, device=torch.device("cpu"),
        transfer_engine_info=None, model=model,
    )


def test_the_engine_binds_a_nodes_handles_when_it_loads_a_model(monkeypatch):
    node = _EngineNode()
    _load_engine(monkeypatch, node, make_model(checkpoint_path=lambda: None))
    assert node.declared_handles == {"kv": "LLM"}


def test_the_engine_binds_nothing_when_it_loads_no_model(monkeypatch):
    node = _EngineNode()
    _load_engine(monkeypatch, node, model=None)
    assert node.declared_handles is None


def test_a_model_that_declares_nothing_builds_no_handles():
    assert node_handles(make_model(), "LLM", {}) == {}


# The label check over a real pool


def test_a_within_step_arrays_layout_is_dropped_since_it_has_no_stream():
    assert kv_layouts([_within()], {"vision": {"main": Layout()}}, {}) == {}


def test_a_kv_manager_says_which_labels_a_request_opened(monkeypatch):
    monkeypatch.setattr(kv_manager, "KVTransferManager", lambda *args, **kwargs: SimpleNamespace())
    kv = kv_manager.KVManager(
        cfg=PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, max_num_pages=64, page_size=16),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.ingest_request(1, KVReqConfig(needed_labels=["main"]))
    assert kv.opens_label(1, "main")
    assert not kv.opens_label(1, "cfg"), "a step on a label its request didn't open has no stream to read"
    assert kv.opens_label(2, "cfg"), "a padding row has no config"
    kv.ingest_request(3, KVReqConfig(needed_labels_per_node={"LLM": ["cfg"]}))
    assert kv.opens_label(3, "cfg"), "a label opened per node isn't in needed_labels"

"""A layout a model declares goes through the data worker and the conductor as the hand-written one did.

The data worker keys a prompt's pages from its layouts and serializes them into the rows the conductor reads, so a
converted layout must give the same keys and rows, or a request would hit another request's pages or none. These names
are the model's to set: a client that planted them could name a span's keys, so they join the worker's scrub list.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from unittest.mock import patch

sys.path.insert(0, ".")

import pytest
import torch
from test_kv_prefix_request_keys import PROMPT, _deployment, _Model, _run, _worker

from mstar.conductor.conductor import (
    Conductor,
    _check_mask_layouts,
    _required_mask_layouts,
    _with_chunk_causal_layout,
    _with_expert_layout,
)
from mstar.conductor.request_info import WalkStep
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.state import (
    Array,
    Attend,
    Causal,
    ChunkCausal,
    ChunkCausalLayout,
    Entry,
    ExpertSelf,
    ExpertSelfLayout,
    KVEntry,
    Layout,
    Read,
    Sampler,
    SampleSettings,
    Size,
    Span,
    TensorType,
    Value,
)
from mstar.engine.state.adapters import CHUNK_CAUSAL_LAYOUT_KWARG, EXPERT_LAYOUT_KWARG
from mstar.graph.base import GraphNode
from mstar.model.base import KVLabel, LabelLayout, ProcessPromptOutput

_SPAN = Span("ids", "prefill", len(PROMPT), len(PROMPT), "text_inputs")
_ARRAY = Array("kv", nodes={"LLM"}, layers={0}, entry=KVEntry(heads=1, head_dim=8), max_len=4096,
               reads={"attn": Read(mask=Causal(), op=Attend(q_heads=1))})


class _Declaring(_Model):
    """The same model, returning a contract ``Layout``."""

    def __init__(self, extra=None, **kwargs):
        super().__init__(**kwargs)
        self.extra = extra or {}
        self.handed = None

    nodes = ("LLM",)

    def declare_state(self):
        return [_ARRAY]

    def get_autocast_dtype(self):
        return torch.bfloat16

    def get_node_resources(self):
        return []

    def get_graph_walk_graphs(self):
        return {"prefill": GraphNode(name="LLM", input_names=[], outputs=[])}

    def get_combined_graph_walks(self):
        return {}

    def get_chunking_policies(self):
        return {}

    def process_prompt(self, *args, **kwargs):
        self.handed = kwargs
        tensors = {"text_inputs": [torch.tensor(PROMPT, dtype=torch.long)]}
        return ProcessPromptOutput(tensors, {"state_layout": {"kv": {"main": Layout(spans=(_SPAN,))}}, **self.extra})


class _HandWritten(_Model):
    """A model that wrote the layout by hand, with ``keyed_spans``."""

    def __init__(self, keyed_spans):
        super().__init__()
        self.keyed_spans = keyed_spans

    def process_prompt(self, *args, **kwargs):
        tensors = {"text_inputs": [torch.tensor(PROMPT, dtype=torch.long)]}
        layout = LabelLayout(spans=(_SPAN,), keyed_spans=self.keyed_spans)
        return ProcessPromptOutput(tensors, {"kv_layout": {"kv": {"main": layout}}})


class _ChunkDeclaring(_Declaring):
    def declare_state(self):
        return [*_Declaring.declare_state(self), Array(
            "chunk", nodes={"LLM"}, layers={0}, entry=KVEntry(heads=1, head_dim=8), max_len=0,
            reads={"chunk_attn": Read(mask=ChunkCausal(8), op=Attend(q_heads=1))},
        )]

    def process_prompt(self, *args, **kwargs):
        output = super().process_prompt(*args, **kwargs)
        output.metadata["state_layout"]["chunk"] = {
            "main": Layout(chunk_causal={"chunk_attn": ChunkCausalLayout(global_len=4, noised_len=16, clean_len=8)})
        }
        return output


class _CombinedDeclaring(_ChunkDeclaring):
    def kv_labels(self):
        # A step-written Value cannot be restored by a KV prefix hit.
        return {"kv": {"main": KVLabel()}}

    def declare_state(self):
        return [*super().declare_state(),
                Array(
                    "expert", nodes={"LLM"}, layers={0}, entry=KVEntry(heads=1, head_dim=8), max_len=0,
                    reads={"expert_attn": Read(mask=ExpertSelf(2, future_per_chunk=1), op=Attend(q_heads=1))},
                ),
                Value(
                    "sized", nodes={"LLM"}, layers={0},
                    entry=Entry({"buf": TensorType((Size("n"),), dtype=torch.float32)}), largest={"n": 4},
                )]

    def process_prompt(self, *args, **kwargs):
        output = super().process_prompt(*args, **kwargs)
        output.metadata["state_layout"]["expert"] = {
            "main": Layout(expert_self={"expert_attn": ExpertSelfLayout(num_chunks=2)})
        }
        output.metadata["state_layout"]["sized"] = {"main": Layout(sizes={"n": 2})}
        return output


@pytest.mark.parametrize("keyed, extra", [(None, {}), (0, {"keyed_spans": {"kv": {"main": 0}}})],
                         ids=["every_span_keyed", "no_keyed_spans"])
def test_a_declared_layout_gives_the_keys_and_rows_of_a_hand_written_one(keyed, extra):
    converted = _run(_worker(_Declaring(extra), _deployment()))
    written = _run(_worker(_HandWritten(keyed), _deployment()))
    assert converted["kv_layout"] == written["kv_layout"], "the conductor reads these rows"
    for name in ("prefix_keys", "prefix_tail", "prefix_decode"):
        assert converted.get(name) == written.get(name), f"{name} differs from the hand-written layout's"
    assert not {"state_layout", "keyed_spans"} & converted.keys(), "a Layout on the wire would break msgpack"


def test_a_data_worker_builds_the_declarations_once():
    model = _Declaring()
    worker = _worker(model, _deployment())
    with patch.object(model, "declare_state", wraps=model.declare_state) as declare_state:
        _run(worker)
        _run(worker)
    declare_state.assert_called_once()


def test_chunk_lengths_cross_the_data_worker_as_wire_safe_rows():
    model = _ChunkDeclaring()
    kwargs = _run(_worker(model, _deployment()), {CHUNK_CAUSAL_LAYOUT_KWARG: {"chunk": {"main": "client"}}})
    assert CHUNK_CAUSAL_LAYOUT_KWARG not in model.handed
    assert kwargs[CHUNK_CAUSAL_LAYOUT_KWARG] == {"chunk": {"main": {"chunk_attn": [4, 16, 8]}}}
    assert "state_layout" not in kwargs


def test_expert_chunk_and_sized_layouts_cross_the_data_worker_together():
    model = _CombinedDeclaring()
    planted = {
        CHUNK_CAUSAL_LAYOUT_KWARG: {"chunk": {"main": "client"}},
        EXPERT_LAYOUT_KWARG: {"expert": {"main": "client"}},
        "state_sizes": {"sized": {"main": "client"}},
    }
    kwargs = _run(_worker(model, _deployment()), planted)
    assert not planted.keys() & model.handed.keys()
    assert kwargs[CHUNK_CAUSAL_LAYOUT_KWARG] == {"chunk": {"main": {"chunk_attn": [4, 16, 8]}}}
    assert kwargs[EXPERT_LAYOUT_KWARG] == {"expert": {"main": {"expert_attn": [2]}}}
    assert kwargs["state_sizes"] == {"sized": {"main": {"n": 2}}}


def test_missing_expert_layout_fails_at_data_worker_intake():
    class MissingExpert(_CombinedDeclaring):
        def process_prompt(self, *args, **kwargs):
            output = super().process_prompt(*args, **kwargs)
            output.metadata["state_layout"]["expert"]["main"] = Layout()
            return output

    with pytest.raises(RuntimeError, match="expert_self is missing reads"):
        _run(_worker(MissingExpert(), _deployment()))


def test_expert_array_can_be_planned_after_data_worker_intake():
    class MissingExpert(_CombinedDeclaring):
        def process_prompt(self, *args, **kwargs):
            output = super().process_prompt(*args, **kwargs)
            del output.metadata["state_layout"]["expert"]
            return output

    kwargs = _run(_worker(MissingExpert(), _deployment()))
    assert kwargs["state_sizes"] == {"sized": {"main": {"n": 2}}}
    assert EXPERT_LAYOUT_KWARG not in kwargs


def test_chunk_and_expert_masks_can_follow_kv_and_sizes_at_walk_plan():
    class Hybrid(_CombinedDeclaring):
        def process_prompt(self, *args, **kwargs):
            output = super().process_prompt(*args, **kwargs)
            del output.metadata["state_layout"]["chunk"]
            del output.metadata["state_layout"]["expert"]
            return output

    model = Hybrid()
    kwargs = _run(_worker(model, _deployment()))
    assert "kv" in kwargs["kv_layout"]
    assert kwargs["state_sizes"] == {"sized": {"main": {"n": 2}}}
    assert CHUNK_CAUSAL_LAYOUT_KWARG not in kwargs
    assert EXPERT_LAYOUT_KWARG not in kwargs

    required = _required_mask_layouts(model, model.declare_state())
    with pytest.raises(RuntimeError, match="has no chunk-causal layout"):
        _check_mask_layouts([WalkStep("prefill")], required)
    chunk = {"chunk": {"main": {"chunk_attn": [4, 16, 8]}}}
    with pytest.raises(RuntimeError, match="has no expert layout"):
        _check_mask_layouts([WalkStep("prefill", step_metadata={CHUNK_CAUSAL_LAYOUT_KWARG: chunk})], required)

    expert = {"expert": {"main": {"expert_attn": [2]}}}
    steps = [WalkStep("prefill", step_metadata={
        CHUNK_CAUSAL_LAYOUT_KWARG: chunk, EXPERT_LAYOUT_KWARG: expert,
    })]
    steps = _with_chunk_causal_layout(steps, kwargs.get(CHUNK_CAUSAL_LAYOUT_KWARG))
    steps = _with_expert_layout(steps, kwargs.get(EXPERT_LAYOUT_KWARG))
    _check_mask_layouts(steps, required)
    assert steps[0].step_metadata[CHUNK_CAUSAL_LAYOUT_KWARG] == chunk
    assert steps[0].step_metadata[EXPERT_LAYOUT_KWARG] == expert

    prompt_chunk = {"chunk": {"main": {"chunk_attn": [12, 8, 8]}}}
    prompt_expert = {"expert": {"main": {"expert_attn": [1]}}}
    overlaid = _with_expert_layout(_with_chunk_causal_layout(steps, prompt_chunk), prompt_expert)
    assert overlaid[0].step_metadata[CHUNK_CAUSAL_LAYOUT_KWARG] == chunk
    assert overlaid[0].step_metadata[EXPERT_LAYOUT_KWARG] == expert


def test_missing_chunk_lengths_fail_at_data_worker_intake():
    class MissingChunk(_ChunkDeclaring):
        def process_prompt(self, *args, **kwargs):
            output = super().process_prompt(*args, **kwargs)
            output.metadata["state_layout"]["chunk"]["main"] = Layout()
            return output

    with pytest.raises(RuntimeError, match="chunk_causal is missing reads \\['chunk_attn'\\]"):
        _run(_worker(MissingChunk(), _deployment()))


def test_missing_chunk_array_can_be_planned_after_data_worker_intake():
    class MissingChunk(_ChunkDeclaring):
        def process_prompt(self, *args, **kwargs):
            output = super().process_prompt(*args, **kwargs)
            del output.metadata["state_layout"]["chunk"]
            return output

    kwargs = _run(_worker(MissingChunk(), _deployment()))
    assert "kv" in kwargs["kv_layout"]
    assert CHUNK_CAUSAL_LAYOUT_KWARG not in kwargs


def test_chunk_layout_can_be_supplied_by_walk_plan_after_intake():
    class PlannedChunk(_ChunkDeclaring):
        def process_prompt(self, *args, **kwargs):
            output = super().process_prompt(*args, **kwargs)
            del output.metadata["state_layout"]
            return output

        def walk_plan(self, *args, **kwargs):
            return [WalkStep("prefill", step_metadata={
                CHUNK_CAUSAL_LAYOUT_KWARG: {"chunk": {"main": {"chunk_attn": [12, 8, 8]}}},
            })]

    kwargs = _run(_worker(PlannedChunk(), _deployment()))
    assert CHUNK_CAUSAL_LAYOUT_KWARG not in kwargs

    model = PlannedChunk()
    required = _required_mask_layouts(model, model.declare_state())
    with pytest.raises(RuntimeError, match="has no chunk-causal layout"):
        _check_mask_layouts([WalkStep("prefill")], required)

    request_layout = {"chunk": {"main": {"chunk_attn": [4, 16, 8]}}}
    planned_layout = model.walk_plan("main", [], [], {}, kwargs)[0].step_metadata[CHUNK_CAUSAL_LAYOUT_KWARG]
    plan = _with_chunk_causal_layout(
        model.walk_plan("main", [], [], {}, kwargs), request_layout,
    )
    _check_mask_layouts(plan, required)
    assert plan[0].step_metadata[CHUNK_CAUSAL_LAYOUT_KWARG] == planned_layout


def test_a_prompt_with_no_keyed_spans_keys_no_page():
    kwargs = _run(_worker(_Declaring({"keyed_spans": {"kv": {"main": 0}}}), _deployment()))
    assert kwargs["kv_layout"]["kv"]["main"][4] == 0
    assert not (kwargs.get("prefix_keys") or {}).get("kv", {}).get("main"), "a vision prompt keys nothing"


@pytest.mark.parametrize("name", [
    "state_layout", "keyed_spans", "state_sizes", EXPERT_LAYOUT_KWARG, CHUNK_CAUSAL_LAYOUT_KWARG,
])
def test_a_client_cannot_plant_a_layout_key(name):
    model = _Declaring()
    kwargs = _run(_worker(model, _deployment()), {name: {"kv": {"main": 99}}})
    assert name not in model.handed, f"process_prompt was handed a {name} the client made up"
    assert kwargs["kv_layout"]["kv"]["main"][4] == 1, "the client's value decided which spans are keyed"


def test_a_model_with_no_sized_value_computes_no_sizes():
    with patch("mstar.api_server.data_worker.state_sizes") as state_sizes:
        kwargs = _run(_worker(_Declaring(), _deployment()))
    state_sizes.assert_not_called()
    assert "state_sizes" not in kwargs


def test_a_layout_naming_an_undeclared_array_fails_the_request():
    extra = {"state_layout": {"other": {"main": Layout()}}}
    with pytest.raises(RuntimeError, match=r"state_layout\['other'\]: it names no declared array"):
        _run(_worker(_Declaring(extra), _deployment()))


class _Settings:
    """Returns the request settings it was given, and declares one sampler."""

    def __init__(self, configs):
        self.configs = configs

    def get_request_resource_configs(self, input_modalities, output_modalities, model_kwargs):
        return self.configs

    def declare_state(self):
        return [Sampler("sampler", nodes={"LLM"}, vocab_size=8, draws=2)]


def _configs(*args):
    conductor = object.__new__(Conductor)
    conductor.model = _Settings(*args)
    conductor._kv_labels = {}
    conductor._node_resources_cache = []
    return conductor._get_resource_configs([], [], {})


def test_the_conductor_hands_on_settings_as_todays_config():
    assert _configs({"sampler": SampleSettings(temperature=0.0)}) == {"sampler": SamplingReqConfig(temperature=0.0)}


def test_settings_for_an_undeclared_sampler_fail_the_request_at_the_conductor():
    with pytest.raises(ValueError, match=r"SampleSettings for 'other', which names no declared sampler"):
        _configs({"other": SampleSettings()})


def test_log_probs_on_several_draws_fail_the_request_at_the_conductor():
    with pytest.raises(ValueError, match=r"Sampler 'sampler': return_logprobs, and draws is 2"):
        _configs({"sampler": replace(SampleSettings(), return_logprobs=True)})

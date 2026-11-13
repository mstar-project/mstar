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

from mstar.conductor.conductor import Conductor
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.state import Array, Attend, Causal, KVEntry, Layout, Read, Sampler, SampleSettings, Span
from mstar.graph.base import GraphNode
from mstar.model.base import LabelLayout, ProcessPromptOutput

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


def test_a_prompt_with_no_keyed_spans_keys_no_page():
    kwargs = _run(_worker(_Declaring({"keyed_spans": {"kv": {"main": 0}}}), _deployment()))
    assert kwargs["kv_layout"]["kv"]["main"][4] == 0
    assert not (kwargs.get("prefix_keys") or {}).get("kv", {}).get("main"), "a vision prompt keys nothing"


@pytest.mark.parametrize("name", ["state_layout", "keyed_spans"])
def test_a_client_cannot_plant_a_layout_key(name):
    model = _Declaring()
    kwargs = _run(_worker(model, _deployment()), {name: {"kv": {"main": 99}}})
    assert name not in model.handed, f"process_prompt was handed a {name} the client made up"
    assert kwargs["kv_layout"]["kv"]["main"][4] == 1, "the client's value decided which spans are keyed"


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

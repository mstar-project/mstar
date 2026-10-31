"""A paged label the model leaves undeclared is refused, at load or at intake, and so is a
request bigger than the pool that would hold it. A request that asks for no length has its
default cut to the pool instead."""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
from test_kv_layout_intake import LAYOUT, _conduct, _Model, _preprocess

from mstar.conductor.conductor import Conductor
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, PagedKVConfig
from mstar.graph.base import GraphNode, Loop
from mstar.model.base import KVLabel, LabelLayout, Span
from mstar.validation.resources import _undeclared_pools, _unknown_prefix_walks


def _raise_first(problems):
    """The checks report every problem; these tests look at one."""
    for problem in problems:
        raise ValueError(problem)


def _refuse_undeclared_pools(specs, model):
    _raise_first(_undeclared_pools(specs, model))


def _refuse_unknown_walks(model):
    _raise_first(_unknown_prefix_walks(model))


LLM = GraphNode(name="llm", input_names=["text_inputs"], outputs=[])


class _Declaring:

    def __init__(self, labels: dict, decode: Loop | None = None):
        self._labels = labels
        self._decode = decode or Loop(name="decode_loop", max_iters=8, outputs=[], section=LLM)

    def kv_labels(self):
        return {"kv": self._labels} if self._labels else {}

    def get_graph_walk_graphs(self):
        return {"prefill": LLM, "decode": self._decode}


def _paged(key: str = "kv") -> KVSpec:
    return KVSpec(resource_key=key, nodes={"llm"}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, max_num_pages=8, page_size=16,
    ))


def test_a_paged_pool_with_no_labels_is_refused_at_load():
    with pytest.raises(ValueError, match="'kv'") as refusal:
        _refuse_undeclared_pools([_paged()], _Declaring({}))

    assert "kv_labels" in str(refusal.value), "the refusal doesn't say what to add"


def test_growth_on_an_unnamed_loop_is_refused_at_load():
    unnamed = Loop(max_iters=8, outputs=[], section=LLM)
    label = KVLabel(growth=(("decode", "decode_loop"),))

    with pytest.raises(ValueError, match="'decode_loop', which 'decode' doesn't run") as refusal:
        _refuse_unknown_walks(_Declaring({"main": label}, decode=unnamed))

    assert unnamed.name in str(refusal.value), "the refusal doesn't name the loops the walk runs"


@pytest.mark.parametrize(("layout", "message"), [
    (None, "returned no kv_layout for 'kv'/'main', which the request opens"),
    ({"kv": {"main": LAYOUT["kv"]["main"]}}, "returned no kv_layout for 'kv'/'uncond', which the request opens"),
    ({"kv": {**LAYOUT["kv"], "x": LabelLayout()}}, "returned a kv_layout for 'kv'/'x', which it does not declare"),
], ids=["no layout", "a label left out", "an undeclared label"])
def test_a_request_laid_out_wrong_fails_with_a_500(tmp_path, monkeypatch, caplog, layout, message):
    model = _Model(layout=layout)
    kwargs = _preprocess(model, {"max_output_tokens": 30})

    sent = _conduct(model, kwargs, tmp_path, monkeypatch).sent

    [failure] = [msg.body for msg in sent if getattr(msg, "message_type", None) == "request_failed"]
    # a model bug, not the client's: the client gets an error id, the log the reason
    match = re.fullmatch(r"Ingest failed: internal error \(error id ([0-9a-f]{12})\)", failure.error_message)
    assert failure.status == 500 and match, f"the request failed with {failure.status}: {failure.error_message}"
    [logged] = [r.getMessage() for r in caplog.records if f"[error id {match.group(1)}]" in r.getMessage()]
    assert message in logged
    assert not [msg for msg in sent if getattr(msg.message_type, "name", "") == "NEW_REQUEST"], (
        "a worker heard of a request the conductor refused"
    )


def _one_loop() -> _Model:
    """main takes 100 slots plus one per decode_loop iteration, and uncond takes 2 pages and grows
    on no loop here. So only decode_loop decides whether a footprint fits the pool."""
    uncond = LabelLayout(spans=(Span("rows", "prefill", 10, 10, "cond"),), scratch=7)
    return _Model(layout={"kv": {"main": LAYOUT["kv"]["main"], "uncond": uncond}})


def test_a_length_past_its_pool_is_a_400_before_dispatch(tmp_path, monkeypatch):
    model = _one_loop()
    kwargs = _preprocess(model, {"max_output_tokens": 30})

    sent = _conduct(model, kwargs, tmp_path, monkeypatch, resources={"kv": {"max_num_pages": 11}}).sent

    failures = [msg.body for msg in sent if getattr(msg, "message_type", None) == "request_failed"]
    assert [failure.status for failure in failures] == [400], "an asked length past the pool was not refused"
    assert "11 pages of 'kv' for labels ['main', 'uncond'], and the pool has 10" in failures[0].error_message
    assert not [msg for msg in sent if getattr(msg.message_type, "name", "") == "NEW_REQUEST"]


def test_a_request_that_asks_for_no_length_is_cut_to_its_pool(tmp_path, monkeypatch):
    model = _one_loop()
    kwargs = _preprocess(model, {})

    sent = _conduct(model, kwargs, tmp_path, monkeypatch, resources={"kv": {"max_num_pages": 11}}).sent

    [new_request] = [msg.body for msg in sent if getattr(msg.message_type, "name", "") == "NEW_REQUEST"]
    assert new_request.request_info.max_tokens == new_request.loop_caps["decode_loop"] == 28, (
        "the default was not cut to the longest output the pool holds"
    )


def test_each_worker_counts_only_the_labels_its_nodes_open():
    config = KVReqConfig(needed_labels_per_node_walk={("llm", "prefill"): ["cfg"], ("talker", "decode"): ["uncond"]})
    config.apply_conductor_config(kv_layout={
        label: [[[n, n, "prefill", 0]], [], 0, None, 0] for label, n in (("main", 144), ("cfg", 80), ("uncond", 112))
    }, loop_caps={})
    pool = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, max_num_pages=17, page_size=16)
    conductor = Conductor.__new__(Conductor)
    conductor.model = SimpleNamespace(get_node_resources=lambda: [
        KVSpec(resource_key="kv", nodes={"llm", "talker"}, config=pool),
    ])
    conductor.model_config = {}
    conductor._per_worker_graphs = {
        node: [SimpleNamespace(section=GraphNode(name=node, input_names=[], outputs=[]))] for node in ("llm", "talker")
    }

    for specs in conductor._resource_specs():
        config.validate(specs["kv"])
    with pytest.raises(ValueError, match="21 pages"):
        config.validate(KVSpec(resource_key="kv", nodes={"llm", "talker"}, config=pool))

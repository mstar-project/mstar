"""A paged label the model leaves undeclared is refused, at load or at intake."""

from __future__ import annotations

import re
import sys

sys.path.insert(0, ".")

import pytest
from test_kv_layout_intake import LAYOUT, _conduct, _Model, _preprocess

from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.graph.base import GraphNode, Loop
from mstar.model.base import KVLabel, LabelLayout
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

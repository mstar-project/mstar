"""A walk the cache can serve whole feeds no other node, checked at load.

A walk the cache serves whole runs no forward and completes with no outputs,
so a node that reads its per-token outputs would read nothing, and nothing
would say why. Only a stream that names layout walks has walks the cache can
serve whole; a single span always runs at least its last token, which is why
Orpheus, whose prefill streams to its codec, is not checked.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.graph.base import GraphEdge, GraphNode, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import PrefixStream
from mstar.worker.engine_manager import _refuse_unservable_walks


class _StubModel:
    """Declares a stream an image walk writes too, over whichever graphs it is given."""

    def __init__(
        self, text_llm_feeds=EMIT_TO_CLIENT, image_llm_feeds=EMIT_TO_CLIENT,
        image_walk_runs_llm=True,
    ):
        self._text_feeds = text_llm_feeds
        self._image_feeds = image_llm_feeds
        self._image_walk_runs_llm = image_walk_runs_llm

    def prefix_key_streams(self):
        return {"kv": {"main": PrefixStream(
            "text_inputs", "ids", "prefill", "decode", ("prefill_image",),
        )}}

    def get_graph_walk_graphs(self):
        vit = GraphNode(
            name="vit", input_names=["image_inputs"],
            outputs=[GraphEdge(
                next_node="LLM" if self._image_walk_runs_llm else EMIT_TO_CLIENT,
                name="img_emb",
            )],
        )
        return {
            "prefill": GraphNode(
                name="LLM", input_names=["text_inputs"],
                outputs=[GraphEdge(next_node=self._text_feeds, name="new_token")],
            ),
            "prefill_image": Sequential([
                vit,
                GraphNode(
                    name="LLM", input_names=["img_emb"],
                    outputs=[GraphEdge(next_node=self._image_feeds, name="hidden")],
                ),
            ]) if self._image_walk_runs_llm else vit,
            "decode": GraphNode(name="LLM", input_names=["text_inputs"], outputs=[]),
        }


def _specs(nodes=frozenset({"LLM"})) -> list[KVSpec]:
    return [KVSpec(
        resource_key="kv", nodes=set(nodes),
        config=PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64),
    )]


# the resource's specs and the model, and what the refusal names
_REFUSED = {
    "a layout walk feeds a node": (_specs(), _StubModel(image_llm_feeds="talker"), r"'prefill_image', where LLM"),
    "its own walk feeds a node": (_specs(), _StubModel(text_llm_feeds="talker"), r"'prefill', where LLM"),
    "no keyed node in a layout walk": (_specs(), _StubModel(image_walk_runs_llm=False), r"'prefill_image', where none"),
    "no keyed node in its own walk": (_specs(nodes={"Talker"}), _StubModel(), r"'prefill', where none"),
}


@pytest.mark.parametrize(("specs", "model", "message"), _REFUSED.values(), ids=_REFUSED.keys())
def test_a_walk_the_cache_cannot_serve_whole_is_refused_at_load(specs, model, message):
    with pytest.raises(ValueError, match=message):
        _refuse_unservable_walks(specs, model)


"""A combined split's prefill rows follow the scheduler's caps.

The prefill half of a captured mixed step is sized to the rows its prefill
walks may hold in one step (`max_rows_in_combined`), an uncapped walk counting
the bucket's rows; the engine hands each resource those caps for the walks of
the nodes it serves.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mstar.engine.engine import Engine
from mstar.engine.resources.spec import CombinedSplit


def test_the_prefill_rows_are_the_prefill_walks_caps():
    split = CombinedSplit("mixed", prefill_walks=("prefill_text",))

    assert split.max_prefill_rows({"prefill_text": 4, "decode": None}, 32) == 4


def test_an_uncapped_prefill_walk_counts_the_bucket_rows():
    split = CombinedSplit("mixed", prefill_walks=("prefill_text",))

    assert split.max_prefill_rows({"prefill_text": None}, 32) == 32
    assert split.max_prefill_rows({}, 16) == 16


def test_several_prefill_walks_add_up_to_at_most_the_bucket():
    split = CombinedSplit("mixed", prefill_walks=("prefill_text", "prefill_audio"))

    assert split.max_prefill_rows({"prefill_text": 4, "prefill_audio": 2}, 32) == 6
    assert split.max_prefill_rows({"prefill_text": 4, "prefill_audio": 2}, 4) == 4


def _model(walk_nodes: dict[str, list[str]]):
    return SimpleNamespace(get_graph_walk_graphs=lambda: {
        walk: SimpleNamespace(get_nodes=lambda nodes=nodes: nodes)
        for walk, nodes in walk_nodes.items()
    })


def _submodule(caps: dict[str, int | None]):
    return SimpleNamespace(max_rows_in_combined=caps.get)


def test_a_resource_gets_the_caps_of_its_nodes_walks():
    model = _model({"prefill_text": ["LLM"], "decode": ["LLM"], "encode": ["ViT"]})
    submodules = {"LLM": _submodule({"prefill_text": 4}), "ViT": _submodule({"encode": 1})}

    caps = Engine._max_rows_in_combined("attn", {"LLM"}, submodules, model)

    assert caps == {"prefill_text": 4, "decode": None}


def test_nodes_that_cap_one_walk_differently_are_refused():
    model = _model({"prefill_text": ["A", "B"]})
    submodules = {"A": _submodule({"prefill_text": 4}), "B": _submodule({"prefill_text": 2})}

    with pytest.raises(ValueError, match="differently"):
        Engine._max_rows_in_combined("attn", {"A", "B"}, submodules, model)

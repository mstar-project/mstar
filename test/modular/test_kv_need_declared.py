"""A model with a paged KV pool must say what a request will take from it.

The pool admits a request only once what it may take fits, and that is only as
good as the count. A pool no model counts admits every request, and is the one
that can still run out of pages halfway through one; so load refuses it rather
than serve with it.
"""

import sys

sys.path.insert(0, ".")

import pytest

from mstar.conductor.conductor import Conductor
from mstar.engine.resources import KVSpec, PagedKVConfig
from mstar.model.base import Model, PrefixStream


class _Model:
    """A paged pool, and whatever the test declares for it."""

    counted = False
    keyed = False

    def get_node_resources(self):
        return [KVSpec(
            resource_key="kv", nodes={"LLM"},
            config=PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096),
        )]

    def prefix_key_streams(self):
        return {"kv": {"main": PrefixStream("text_inputs", "ids", "prefill")}} if self.keyed else {}

    def kv_counted(self):
        return frozenset({"kv"}) if self.counted else Model.kv_counted(self)

    paged_kv_keys = Model.paged_kv_keys


def test_a_paged_pool_nothing_counts_is_refused_at_load():
    with pytest.raises(ValueError, match=r"declares no need for its paged KV pools \['kv'\]"):
        Conductor._counted_kv_keys(_Model())


def test_a_pool_counted_by_its_prompt_or_its_keys_loads():
    counted = _Model()
    counted.counted = True
    keyed = _Model()
    keyed.keyed = True

    assert Conductor._counted_kv_keys(counted) == Conductor._counted_kv_keys(keyed) == {"kv"}

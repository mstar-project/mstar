"""A step of a walk run on split kernels holds its decode rows first.

`GDNMixedWrapper` and `FlashInferMixedWrapper` read a mixed step's leading
one-token rows as the step's first tokens, in place, and size their prefill
half to the prompt rows the scheduler lets in; a decode row behind a prompt
row would take a prefill row's place. The engine records the order from the
resources' `CombinedSplit`s, and every batch the worker makes follows it.
"""

from types import SimpleNamespace

import pytest

from mstar.engine.engine import Engine
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.spec import CombinedSplit
from mstar.worker.worker import Worker

SPLIT = CombinedSplit("mixed", ("prefill_text",))


def _engine(*specs):
    engine = Engine.__new__(Engine)
    engine._split_prompt_walks = {}
    for spec in specs:
        engine._add_split_row_order(spec, spec.nodes)
    return engine


def _specs(split=SPLIT):
    return (
        AttentionSpec("attn", {"LLM"}, AttentionConfig(kv_cache="kv", splits=(split,))),
        LinearAttnSpec("linear_attn", {"LLM"}, LinearAttnConfig(
            recurrent_state="gdn", variant=LinearAttnVariant.GDN, splits=(split,),
        )),
    )


WALKS = {0: "prefill_text", 1: "decode", 2: "prefill_text", 3: "decode", 4: "decode"}


def test_a_split_walks_prompt_rows_go_last_in_their_order():
    engine = _engine(*_specs())

    assert engine.order_split_rows("LLM", "mixed", [0, 1, 2, 3, 4], WALKS) == [1, 3, 4, 0, 2]


def test_other_walks_and_nodes_keep_their_order():
    engine = _engine(*_specs())

    assert engine.order_split_rows("LLM", "decode", [0, 1, 2], WALKS) == [0, 1, 2]
    assert engine.order_split_rows("vision", "mixed", [0, 1, 2], WALKS) == [0, 1, 2]


def test_resources_that_disagree_on_a_splits_prompt_walks_are_refused():
    attn, _ = _specs()
    _, linear = _specs(CombinedSplit("mixed", ("prefill_text", "prefill_vision")))

    with pytest.raises(ValueError, match="prefill walks"):
        _engine(attn, linear)


def test_the_worker_makes_a_mixed_step_decode_first():
    worker = Worker.__new__(Worker)
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: _engine(*_specs()))

    batch = worker._make_executing_batch(
        node_name="LLM", graph_walk="mixed", request_ids=[0, 1, 2, 3, 4],
        per_request_input_tensors={}, per_request_info={}, request_walks=WALKS,
    )

    assert batch.request_ids == (1, 3, 4, 0, 2)

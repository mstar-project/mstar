"""A decode loop's counter is i on the loop's iteration i, so a sampler's step index can be read off it.

CosyVoice3's LM draws each token from a hash of the request's seed and its step index, so a step that read the wrong
index would draw a token another step drew. Its port reads the index off the loop counter in place of a count the node
kept itself. The worker refreshes the counter after a step's routing, and a step it speculates is prepared before that,
so the step's own copy of the counters counts one iteration on, and that copy is what `prepare_inputs` reads.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from test_graph_runtime import _build_on_real_manager, _complete, _fwd_info, _ingest, _ready

from mstar.engine.engine import ForwardPassInfoWrapper
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.special_destinations import EMPTY_DESTINATION
from mstar.worker.executing_batch import input_metadata_and_fwd_info

DECODE_LOOP = "decode_loop"


def _decode() -> Loop:
    """CosyVoice3's decode walk: the LM feeds each token back to itself."""
    return Loop(
        name=DECODE_LOOP,
        section=GraphNode(
            name="llm", input_names={"prev_token"},
            outputs=[
                GraphEdge(name="prev_token", next_node="llm"),
                GraphEdge(name="speech_tokens", next_node=EMPTY_DESTINATION),
            ],
        ),
        max_iters=16,
        outputs=[],
    )


def test_the_decode_loop_counts_i_on_its_iteration_i(tmp_path):
    tm, runtime, rid = _build_on_real_manager(_decode(), {"llm"}, tmp_path, loops={DECODE_LOOP})
    _ingest(runtime, rid, [GraphEdge(name="prev_token", next_node="llm")])
    info = _fwd_info("w")
    for i in range(4):
        assert _ready(runtime) == [("llm", [rid])], f"iteration {i} should be ready"
        runtime.update_dynamic_loop_iters({rid: info}, "default")
        assert info.dynamic_loop_iter_counts[DECODE_LOOP] == i, f"iteration {i} would sample at another step's index"
        token = torch.ones(1, dtype=torch.long)
        _complete(tm, runtime, rid, "llm", {"prev_token": [token], "speech_tokens": [token]},
                  ["prev_token", "speech_tokens"])


def test_a_speculated_step_reads_the_next_iteration(tmp_path):
    info = _fwd_info("decode")
    info.dynamic_loop_iter_counts[DECODE_LOOP] = 2
    request_state = SimpleNamespace(get_fwd_info=lambda rid, partition: info)
    streams = SimpleNamespace(possible=lambda: False)
    metadata, _ = input_metadata_and_fwd_info(
        request_state, streams, "llm", "default", {7: {}}, increment_loop_rids=[7], increment_loop_name=DECODE_LOOP,
    )
    seen = ForwardPassInfoWrapper(info, metadata[7]).dynamic_loop_iter_counts[DECODE_LOOP]
    assert info.dynamic_loop_iter_counts[DECODE_LOOP] == 2, "the step in flight's counter should be left alone"
    assert seen == 3, "a speculated step would sample at the index of the step in flight"

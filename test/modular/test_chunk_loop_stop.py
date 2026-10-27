"""Ending a chunkable walk's Loop.

A chunk is one iteration of the node's Loop, so something has to end it on the
last chunk -- nothing inside the node knows which one that is. Keyed on the
walk, not on the row having a chunk range: a walk with no token budget is
never measured or cut, so its prompts run whole and carry no range, and a Loop
left unended runs to ``max_iters`` while the request holds its KV. That showed
up as pool exhaustion rather than as a scheduling bug.
"""

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.model.submodule_base import chunk_loop_name
from mstar.worker.worker import Worker

NODE = "LLM"


def _batch(rows, chunk_ranges=None):
    """``rows`` is rid -> walk."""
    return SimpleNamespace(
        node_name=NODE,
        batch=SimpleNamespace(
            request_to_worker_graph=dict.fromkeys(rows, 0),
            walk_of=rows.get,
            chunk_ranges=chunk_ranges or {},
        ),
    )


def _run(rows, chunk_ranges=None, stops=None, chunkable=("prefill",)):
    worker = Worker.__new__(Worker)
    # what the worker precomputes at load from the model's chunking policies
    worker._chunk_loop_walks = frozenset((NODE, w) for w in chunkable)
    stops = stops if stops is not None else {}
    mid = Worker._end_finished_chunk_loops(
        worker, _batch(rows, chunk_ranges), stops,
    )
    return stops, mid


def test_a_row_that_ran_whole_ends_its_loop():
    """No range at all: its walk has no budget, so it was never cut."""
    stops, mid = _run({1: "prefill"})
    assert stops == {1: {chunk_loop_name(NODE, "prefill")}}
    assert not mid


def test_a_final_chunk_ends_its_loop():
    stops, mid = _run({1: "prefill"}, {1: (4, 10, 10)})
    assert stops == {1: {chunk_loop_name(NODE, "prefill")}}
    assert not mid


def test_a_non_final_chunk_leaves_the_loop_running():
    stops, mid = _run({1: "prefill"}, {1: (0, 4, 10)})
    assert stops == {}
    assert mid == {1}


def test_a_non_final_chunks_own_stop_is_dropped():
    """Its sampled token is not a real sample, so a stop drawn from it would
    cut the prompt short mid-prefill."""
    stops, _ = _run({1: "prefill"}, {1: (0, 4, 10)}, stops={1: {"decode_loop"}})
    assert stops == {}


def test_a_walk_without_a_loop_is_left_alone():
    """A decode row sharing a mixed step: no chunk Loop, so nothing to end."""
    stops, mid = _run({1: "prefill", 2: "decode"}, {1: (0, 10, 10)})
    assert stops == {1: {chunk_loop_name(NODE, "prefill")}}
    assert not mid


def test_a_node_that_chunks_nowhere_does_no_work_at_all():
    """Most nodes never chunk; postprocess must not pay per row for them."""
    rows = {rid: "decode" for rid in range(1, 21)}
    stops, mid = _run(rows, chunkable=())
    assert stops == {} and not mid

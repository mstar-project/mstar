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

from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import WorkerGraph
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


def test_a_mid_prefill_row_keys_no_prefix():
    """Postprocess must hand the rows it names mid-prefill to the prefix
    chain as skips: a non-final chunk's sampled token is not a real sample,
    so keying the chain on it keys a token the request never generated.

    A structural guard, not a behavioural one -- driving the real postprocess
    needs most of a worker. It is here because the call was dropped once and
    nothing else noticed: the chain simply stopped being extended.
    """
    import inspect

    src = inspect.getsource(Worker._postprocess_batch)
    assert "extend_prefix_chains(" in src, "the prefix chain is never extended"
    call = src[src.index("extend_prefix_chains("):]
    call = call[: call.index(")")]
    assert "skip_rids=mid_prefill_rids" in call, (
        "mid-prefill rows must be skipped, or a non-final chunk's token keys "
        "the prefix cache"
    )


def test_a_loop_is_ended_even_where_the_deployment_chunks_nothing():
    """``apply_chunking`` wraps the node from the MODEL's policy, so the Loop
    is in the walk's graph whether or not this deployment chunks it. A null
    budget or a node under TP/SP runs the walk inside the Loop, once, and it
    still has to be ended -- otherwise it spins to max_iters holding the
    request's KV, which reads as pool exhaustion rather than a hung loop."""
    # The worker reads its set off the graphs it was given, so it holds a
    # walk whose chunking the config or TP turned off -- the Loop is still
    # there, built from the model's policy, and runs once.
    stops, mid = _run({1: "prefill"}, chunkable=("prefill",))
    assert stops == {1: {chunk_loop_name(NODE, "prefill")}} and not mid


def _wrapped_walk(chunkable: bool, policy=None):
    """A walk's graph as ``apply_chunking`` leaves it, wrapped or not."""
    from mstar.model.base import _wrap_node_in_loop
    from mstar.model.submodule_base import ChunkingPolicy

    node = GraphNode(
        name=NODE, input_names={"text_inputs"},
        outputs=[GraphEdge(name="new_token", next_node=EMIT_TO_CLIENT)],
    )
    if not chunkable:
        return node
    return _wrap_node_in_loop(
        node, NODE, "prefill",
        policy or ChunkingPolicy(walks=("prefill",), chunkable=True, max_chunks=8),
    )


def test_a_default_chunk_loop_runs_a_long_prompt_to_its_end():
    """A policy that leaves ``max_chunks`` unset must not cap its Loop: a
    capped Loop reports the prompt done after its last allowed chunk, and the
    next walk starts with the rest of the prompt still to run."""
    from mstar.model.submodule_base import ChunkingPolicy

    loop = _wrapped_walk(True, ChunkingPolicy(walks=("prefill",), chunkable=True))
    assert loop.max_iters >= 1 << 20


def _wg(section):
    return WorkerGraph(
        section=section, graph_walks={"prefill"}, ranks=[0], worker_graph_id=0,
    )


def test_the_worker_finds_its_chunk_loops_in_its_own_graphs():
    assert Worker._chunk_loops_in([_wg(_wrapped_walk(True))]) == {(NODE, "prefill")}


def test_an_unwrapped_walk_contributes_no_chunk_loop():
    """A deployment that excludes a node from chunking builds no Loop there,
    and the worker must not expect one -- it would end a loop that is not."""
    assert Worker._chunk_loops_in([_wg(_wrapped_walk(False))]) == frozenset()


def test_another_loop_is_not_mistaken_for_a_chunk_loop():
    """A decode Loop is not a chunk Loop; only the derived name counts."""
    decode = Loop(
        name="decode_loop",
        section=GraphNode(
            name=NODE, input_names={"text_inputs"},
            outputs=[GraphEdge(name="text_inputs", next_node=NODE)],
        ),
        max_iters=8, outputs=[],
    )
    assert Worker._chunk_loops_in([_wg(decode)]) == frozenset()


def test_a_row_the_cache_served_whole_still_ends_its_loop():
    """The prefix cache can serve a chunkable walk entirely, and then the node
    never runs: no forward, no chunk, no output. Its Loop is still in the
    graph and still has to be ended, or the request holds its state until
    max_iters.

    Postprocess reconciles the scheduled batch down to ``request_ids |
    cached_rids`` before the stop, so a served row is still there -- and it
    carries no chunk range, nothing having been cut for it, which is the same
    shape as a row that ran whole.
    """
    stops, mid = _run({1: "prefill"})
    assert stops == {1: {chunk_loop_name(NODE, "prefill")}}
    assert not mid, "nothing is mid-prefill: there was nothing left to prefill"


def test_a_cache_served_row_is_ended_even_beside_a_chunked_one():
    """A served row shares its step with rows that did run; the served one
    ends, the one mid-prefill does not."""
    stops, mid = _run({1: "prefill", 2: "prefill"}, {2: (0, 4, 10)})
    assert stops == {1: {chunk_loop_name(NODE, "prefill")}}
    assert mid == {2}

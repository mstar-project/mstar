"""Ending a chunkable walk's Loop.

A chunk is one iteration of the node's Loop, which counts the tokens its
chunks ran and ends at the prompt's length. The worker hands each row's count
to the runtime as its loop stride (``ChunkLoops.loop_strides``). Keyed on the
walk, not on the row having a chunk range: a walk with no token budget is
never measured or cut, so its prompts run whole and carry no range, and a Loop
left unended runs to ``max_iters`` while the request holds its KV. That showed
up as pool exhaustion rather than as a scheduling bug.
"""

import sys

sys.path.insert(0, ".")

from mstar.engine.chunking import ChunkRange
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import WorkerGraph
from mstar.model.submodule_base import CHUNK_LOOP_END
from mstar.worker.chunk_loops import ChunkLoops
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Worker

NODE = "LLM"


def _batch(rows, chunk_ranges=None):
    """``rows`` is rid -> walk: a combined step if they differ."""
    walks = set(rows.values())
    return ScheduledBatch(
        node_name=NODE,
        graph_walk=next(iter(walks)) if len(walks) == 1 else "mixed",
        request_to_worker_graph=dict.fromkeys(rows, 0),
        request_walks=dict(rows) if len(walks) > 1 else {},
        chunk_ranges=chunk_ranges or {},
    )


def _strides(rows, chunk_ranges=None, chunkable=("prefill",)):
    # what the worker reads off its graphs at load
    loops = ChunkLoops((NODE, w) for w in chunkable)
    return loops.loop_strides(_batch(rows, chunk_ranges), list(rows))


def test_a_row_that_ran_whole_ends_its_loop():
    """No range at all: its walk has no budget, so it was never cut."""
    assert _strides({1: "prefill"}) == [CHUNK_LOOP_END]


def test_a_chunk_moves_its_loop_by_its_stride():
    """A first chunk counts from 0, past any cached prefix; a later one by
    its own tokens. The Loop ends itself once the count reaches its cap."""
    assert _strides({1: "prefill"}, {1: ChunkRange(6, 8, 10, 8)}) == [8]
    assert _strides({1: "prefill"}, {1: ChunkRange(8, 10, 10, 2)}) == [2]


def test_a_walk_without_a_loop_counts_one():
    """A decode row sharing a mixed step: its loop is the decode loop."""
    assert _strides(
        {1: "prefill", 2: "decode"}, {1: ChunkRange(0, 10, 10, 10)},
    ) == [10, 1]


def test_a_node_that_chunks_nowhere_sends_no_strides():
    """Most nodes never chunk; their routes carry no stride column."""
    rows = {rid: "decode" for rid in range(1, 21)}
    assert _strides(rows, chunkable=()) is None


def test_a_decode_step_beside_a_chunk_loop_does_no_work_per_row():
    """A node with a chunk Loop on its prefill walk still runs decode steps
    every pass; those must cost one lookup, not one per row."""

    class _CountingBatch(ScheduledBatch):
        looked_up = 0

        def walk_of(self, rid):
            type(self).looked_up += 1
            return super().walk_of(rid)

    batch = _CountingBatch(
        node_name=NODE, graph_walk="decode",
        request_to_worker_graph=dict.fromkeys(range(1, 21), 0),
    )
    assert ChunkLoops({(NODE, "prefill")}).loop_strides(batch, list(range(1, 21))) is None
    assert _CountingBatch.looked_up == 0


def test_a_mid_prefill_row_keys_no_prefix():
    """Postprocess must hand the rows mid-prefill to the prefix chain as
    skips: a non-final chunk's sampled token is not a real sample, so keying
    the chain on it keys a token the request never generated.

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
    """The Loop is in the walk's graph whether or not this deployment chunks
    it. A null budget or a node under TP/SP runs the walk inside the Loop,
    once, and it still has to be ended -- otherwise it spins to max_iters
    holding the request's KV, which reads as pool exhaustion rather than a
    hung loop."""
    assert _strides({1: "prefill"}, chunkable=("prefill",)) == [CHUNK_LOOP_END]


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
        policy or ChunkingPolicy(walks=("prefill",), chunkable=True),
    )


def test_a_chunk_loop_is_uncapped_until_its_first_chunk():
    """The Loop's own cap only covers a row nothing has measured; a prompt's
    first chunk caps it at the prompt's length (``set_loop_caps``)."""
    from mstar.model.submodule_base import ChunkingPolicy

    loop = _wrapped_walk(True, ChunkingPolicy(walks=("prefill",), chunkable=True))
    assert loop.max_iters == CHUNK_LOOP_END


def _wg(section):
    return WorkerGraph(
        section=section, graph_walks={"prefill"}, ranks=[0], worker_graph_id=0,
    )


def test_the_worker_finds_its_chunk_loops_in_its_own_graphs():
    assert ChunkLoops.walks_in([_wg(_wrapped_walk(True))]) == {(NODE, "prefill")}


def test_an_unwrapped_walk_contributes_no_chunk_loop():
    """A deployment that excludes a node from chunking builds no Loop there,
    and the worker must not expect one -- it would end a loop that is not."""
    assert ChunkLoops.walks_in([_wg(_wrapped_walk(False))]) == frozenset()


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
    assert ChunkLoops.walks_in([_wg(decode)]) == frozenset()


def test_a_row_the_cache_served_whole_still_ends_its_loop():
    """The prefix cache can serve a chunkable walk entirely, and then the node
    never runs: no forward, no chunk, no output. Its Loop is still in the
    graph and still has to be ended, or the request holds its state until
    max_iters.

    Postprocess reconciles the scheduled batch down to ``request_ids |
    cached_rids`` before routing, so a served row is still there -- and it
    carries no chunk range, nothing having been cut for it, which is the same
    shape as a row that ran whole.
    """
    assert _strides({1: "prefill"}) == [CHUNK_LOOP_END]


def test_a_cache_served_row_is_ended_even_beside_a_chunked_one():
    """A served row shares its step with rows that did run; the served one
    ends, the one mid-prefill moves on by its chunk."""
    assert _strides(
        {1: "prefill", 2: "prefill"}, {2: ChunkRange(0, 4, 10, 4)},
    ) == [CHUNK_LOOP_END, 4]

"""A stream-terminated partition ends on the worker's final-chunk signal.

The worker sets ``partition_done`` on the worker-graphs-done report of the pass
that consumed a stream's final chunk. The conductor must treat that as the
partition's ``request_done`` (so ``producer_done`` reaches its downstream
connections) even when the model reports the partition as not done: the model
only sees connection counters, and ``consumed_count`` counts chunks popped for
execution as reported by every colocated worker-graphs-done message, so a
counter-based end can fire while the partition's last step is still running.
"""
from dataclasses import dataclass, field

from mstar.conductor.conductor import Conductor, RequestData
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    PartitionState,
    StreamingConnectionState,
)
from mstar.model.base import ForwardPassArgs


class _StubModel:
    """Never ends a partition on its own; records what it was asked."""

    def __init__(self):
        self.calls = []

    def get_partition_forward_pass_args(self, partition_name, partition_metadata, persist_signals,
                                        incoming_connections=None):
        self.calls.append((partition_name, [c.edge_name for c in incoming_connections or []]))
        return ForwardPassArgs(full_metadata=partition_metadata, inputs=[], unpersist_tensors=[],
                               request_done=False)


@dataclass
class _Recorder:
    producer_done: list = field(default_factory=list)
    partition_inputs: list = field(default_factory=list)


def _conductor(rid="r"):
    c = object.__new__(Conductor)
    c.model = _StubModel()
    rec = _Recorder()
    c._send_producer_done = lambda request_id, frm, to: rec.producer_done.append((request_id, frm, to))
    c._send_partition_inputs = lambda request_id, pname, fwd_args: rec.partition_inputs.append(pname)
    c._un_persist_tensors = lambda request_id, tensors: None
    c._set_partition_worker_graph_ids = lambda request_id, pname, walk: None
    meta = lambda walk: CurrentForwardConductorMetadata(graph_walk=walk, is_prefill=False)  # noqa: E731
    conns = {
        "A->B": StreamingConnectionState(from_partition="A", to_partition="B", edge_name="tok"),
        "B->C": StreamingConnectionState(from_partition="B", to_partition="C", edge_name="codes"),
    }
    req = RequestData(
        persist_signals={}, persist_signal_ref_cnt={}, worker_graph_to_workers={}, all_worker_graph_ids=set(),
        max_output_tokens=10_000, random_seed=0, resource_configs={},
        partition_states={p: PartitionState(partition_name=p, metadata=meta(p.lower())) for p in "ABC"},
        partition_definitions={p: PartitionDefinition(name=p, graph_walks={p.lower()}) for p in "ABC"},
        streaming_connections=conns,
    )
    c.requests = {rid: req}
    return c, req, rec


def test_worker_final_chunk_signal_ends_the_partition_and_propagates_producer_done():
    c, req, rec = _conductor()
    # B consumed the final chunk of A->B: B is done, and C must learn its producer is done
    all_done = c._process_done_forward("r", "B", partition_done_from_worker=True)
    assert req.partition_states["B"].is_done is True
    assert req.streaming_connections["B->C"].producer_done is True
    assert rec.producer_done == [("r", "B", "C")]
    assert all_done is False          # A and C are still running
    assert c.model.calls == [("B", ["tok"])]


def test_counter_heuristics_are_not_needed_and_an_ordinary_pass_does_not_end_it():
    c, req, rec = _conductor()
    # B finished a pass but did not consume the final chunk: it waits for more stream data
    all_done = c._process_done_forward("r", "B", partition_done_from_worker=False)
    assert req.partition_states["B"].is_done is False
    assert req.streaming_connections["B->C"].producer_done is False
    assert rec.producer_done == [] and rec.partition_inputs == []
    assert all_done is False


def test_partition_without_incoming_connections_ignores_the_flag():
    """A producer-only partition (no stream in) ends only through the model's answer."""
    c, req, rec = _conductor()
    c._process_done_forward("r", "A", partition_done_from_worker=True)
    assert req.partition_states["A"].is_done is False
    assert rec.producer_done == []


def test_chain_closes_partition_by_partition():
    c, req, rec = _conductor()
    c.model.get_partition_forward_pass_args = lambda partition_name, partition_metadata, persist_signals, \
        incoming_connections=None: ForwardPassArgs(full_metadata=partition_metadata, inputs=[], unpersist_tensors=[],
                                                   request_done=partition_name == "A")
    assert c._process_done_forward("r", "A") is False           # A ends itself (its own stream input is exhausted)
    assert req.streaming_connections["A->B"].producer_done is True
    assert c._process_done_forward("r", "B", partition_done_from_worker=True) is False
    assert c._process_done_forward("r", "C", partition_done_from_worker=True) is True   # every partition done
    assert rec.producer_done == [("r", "A", "B"), ("r", "B", "C")]

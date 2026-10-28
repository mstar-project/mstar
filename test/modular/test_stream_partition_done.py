"""A partition ends on the worker's ``partition_done`` signal.

The worker sets ``partition_done`` on the worker-graphs-done report of the pass
that consumed a stream's final chunk, or whose stop check returned
``END_PARTITION`` (a prefill that sampled a stop token). The conductor must
treat that as the partition's ``request_done`` (so ``producer_done`` reaches
its downstream connections) even when the model reports the partition as not
done: the model only sees connection counters, and ``consumed_count`` counts
chunks popped for execution as reported by every colocated worker-graphs-done
message, so a counter-based end can fire while the partition's last step is
still running.
"""
from dataclasses import dataclass, field

import pytest

from mstar.conductor.conductor import Conductor, RequestData
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    PartitionState,
    StreamingConnectionState,
)
from mstar.model.base import ForwardPassArgs
from mstar.utils.ipc_format import WorkerGraphsDone


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


def _done_forward(c, req, partition, partition_done=False):
    """``_process_done_forward`` after a pass whose report carried ``partition_done``."""
    req.partition_states[partition].worker_reported_done = partition_done
    return c._process_done_forward("r", partition)


def test_worker_final_chunk_signal_ends_the_partition_and_propagates_producer_done():
    c, req, rec = _conductor()
    # B consumed the final chunk of A->B: B is done, and C must learn its producer is done
    all_done = _done_forward(c, req, "B", partition_done=True)
    assert req.partition_states["B"].is_done is True
    assert req.streaming_connections["B->C"].producer_done is True
    assert rec.producer_done == [("r", "B", "C")]
    assert all_done is False          # A and C are still running
    assert c.model.calls == [("B", ["tok"])]


def test_counter_heuristics_are_not_needed_and_an_ordinary_pass_does_not_end_it():
    c, req, rec = _conductor()
    # B finished a pass but did not consume the final chunk: it waits for more stream data
    all_done = _done_forward(c, req, "B")
    assert req.partition_states["B"].is_done is False
    assert req.streaming_connections["B->C"].producer_done is False
    assert rec.producer_done == [] and rec.partition_inputs == []
    assert all_done is False


@pytest.mark.parametrize("flag_on", ["first", "last"])
def test_the_flag_on_any_report_ends_a_partition_without_a_stream_in(flag_on):
    """A producer-only partition whose prefill sampled a stop token: the flag
    comes on the LLM's report, and the pass's other worker graph (the
    encoder's) can report before or after it."""
    c, req, rec = _conductor()
    c.enable_prof = False
    pstate = req.partition_states["A"]
    pstate.current_worker_graph_ids = {0, 1}
    req.worker_graph_to_workers = {0: ["enc"], 1: ["llm"]}
    reports = [
        WorkerGraphsDone(request_id="r", partition_name="A", worker_graph_ids=[wg], is_first_tp_rank=True,
                         partition_done=wg == 1)
        for wg in ((1, 0) if flag_on == "first" else (0, 1))
    ]
    assert c._process_worker_graphs_done(reports[0]) == []
    assert c._process_worker_graphs_done(reports[1]) == ["A"]
    c._process_done_forward("r", "A")
    assert pstate.is_done is True
    assert rec.partition_inputs == [], "no further walk starts"
    assert rec.producer_done == [("r", "A", "B")]
    assert pstate.worker_reported_done is False, "reset for the next pass"


def test_chain_closes_partition_by_partition():
    c, req, rec = _conductor()
    c.model.get_partition_forward_pass_args = lambda partition_name, partition_metadata, persist_signals, \
        incoming_connections=None: ForwardPassArgs(full_metadata=partition_metadata, inputs=[], unpersist_tensors=[],
                                                   request_done=partition_name == "A")
    assert c._process_done_forward("r", "A") is False           # A ends itself (its own stream input is exhausted)
    assert req.streaming_connections["A->B"].producer_done is True
    assert _done_forward(c, req, "B", partition_done=True) is False
    assert _done_forward(c, req, "C", partition_done=True) is True   # every partition done
    assert rec.producer_done == [("r", "A", "B"), ("r", "B", "C")]

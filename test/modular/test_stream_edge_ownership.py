"""A popped stream chunk is either held by the graph or handed back to its
buffer, never both.

The chunk's edge holds a reference to each of its items, and a sliding
window's overlap items are shared with the next chunk and the buffer. A chunk
handed back while the graph still holds it would be released twice once both
are consumed, freeing an item the next chunk still needs. This drives the
paths that pop and then ingest or hand back -- the poll, a speculation's prep,
and its settle -- over real stream buffers, with a runtime that records what
it holds.
"""
from types import SimpleNamespace

import pytest
import torch

from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore
from mstar.communication.tensors import TensorCommunicationManager
from mstar.graph.base import TensorPointerInfo
from mstar.graph.runtime.base import ColumnarEdgeSpecs, SpeculationPrepOutput
from mstar.streaming.chunk_policy import SlidingWindowChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.utils.profiler import PhaseLaps
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.speculation import Speculation, SpeculationContext, SpeculationPreparer
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import Worker

EDGE, NODE, SPEC_ID = "new_token", "snac_decoder", 7


class _TM:
    get_tensor = TensorCommunicationManager.get_tensor
    get_tensors = TensorCommunicationManager.get_tensors
    dereference_batch_uniform = TensorCommunicationManager.dereference_batch_uniform
    cleanup_collectable = TensorCommunicationManager.cleanup_collectable
    increment_ref_batch_uniform = TensorCommunicationManager.increment_ref_batch_uniform

    def __init__(self):
        self.tensor_store = TensorStore(bookkeeping=PythonTensorBookkeeping())
        self.uuid_to_shard_dim = {}

    def _cleanup_by_uuid(self, uuid, registered=None):
        if self.tensor_store.check_uuid_presence(uuid):
            self.tensor_store.remove_tensor(uuid)


class _Ledger(StreamBufferManager):
    """The real manager, recording what it pops and hands back, and failing a
    hand-back of a chunk the runtime holds."""

    def __init__(self, *args):
        super().__init__(*args)
        self.held: set[int] = set()
        self.popped: dict[int, object] = {}
        self.returned: set[int] = set()
        self.offered: list = []

    def pop_edge(self, sbuf, edge_name, request_id, **kwargs):
        se = super().pop_edge(sbuf, edge_name, request_id, **kwargs)
        if se is not None:
            self.popped[id(se)] = se
            self.returned.discard(id(se))  # a re-pop of a returned chunk
        return se

    def edge_block(self, polled):
        self.offered = [se for _rid, se in polled]
        return super().edge_block(polled)

    def edge_specs(self, polled):
        self.offered = [se for _rid, se in polled]
        return super().edge_specs(polled)

    def return_edge(self, request_id, streaming_edge):
        assert id(streaming_edge) not in self.held, "handed back a chunk the graph holds"
        assert id(streaming_edge) not in self.returned, "handed back twice"
        self.returned.add(id(streaming_edge))
        super().return_edge(request_id, streaming_edge)

    def unsettled(self) -> list:
        """Popped chunks neither held nor handed back."""
        return [se for k, se in self.popped.items() if k not in self.held and k not in self.returned]


class _Runtime:
    """Holds whatever it accepts, as the graph does until the node consumes it
    or a speculation's settle undoes the ingest."""

    def __init__(self, streams: _Ledger, refuse: set[int]):
        self.streams, self.refuse = streams, refuse
        self.spec_held: dict[int, list] = {}  # rid -> chunks a speculation ingested

    def ingest_inputs_batch(self, block, can_buffer, is_streaming):
        for i, se in enumerate(self.streams.offered):
            if i not in self.refuse:
                self.streams.held.add(id(se))
        return sorted(self.refuse)

    def prep_spec_rids(self, inp):
        consumed, ready = [], []
        rid_of = [rid for rid, n in zip(inp.rids, inp.streaming_edges_per_rid, strict=True) for _ in range(n)]
        for i, se in enumerate(self.streams.offered):
            if i in self.refuse:
                continue
            consumed.append(i)
            self.streams.held.add(id(se))
            self.spec_held.setdefault(rid_of[i], []).append(se)
            ready.append(rid_of[i])
        ready = list(dict.fromkeys(ready))
        return SpeculationPrepOutput(
            consumed_streaming_edge_idxs=consumed, ready_rids=ready, wg_ids=[0] * len(ready),
            input_edges=ColumnarEdgeSpecs.empty(), spec_id=SPEC_ID,
        )

    def commit_speculation(self, spec_id, success, dropped_rids, node=None, wg_ids=(), scheduled_rids=()):
        undone = self.spec_held if not success else {r: self.spec_held.get(r, []) for r in dropped_rids}
        for edges in undone.values():
            for se in edges:
                self.streams.held.discard(id(se))


def _streams(rids, refuse=frozenset(), window=3, stride=2, items=5):
    """Each rid with one buffer holding enough items for two chunks."""
    tm = _TM()
    infos = {}
    for k, rid in enumerate(rids):
        buf = StreamBuffer(rid, EDGE, "LLM", SlidingWindowChunkPolicy(window, stride))
        for j in range(items):
            uuid = 1000 * (k + 1) + j
            tm.tensor_store.put_tensor(rid, uuid, torch.tensor([j]), TensorPointerInfo(
                dims=[1], dtype=torch.int64, nbytes=8, address=0, stride=[1], uuid=uuid,
                source_session_id="", source_entity="w0",
            ))
            tm.tensor_store.increment_ref(uuid, 1)
            buf.pre_read_register(uuid)
            buf.put(uuid, uuid)
        infos[rid] = SimpleNamespace(
            stream_buffers={EDGE: buf}, stream_buffers_by_consumer={NODE: {EDGE: buf}},
            ended_streams=set(),
        )
    streams = _Ledger(SimpleNamespace(per_request_info=infos), tm, [])
    streams.consumer_node = {EDGE: NODE}
    streams._rids = set(rids)
    runtime = _Runtime(streams, set(refuse))
    return streams, runtime


@pytest.mark.parametrize("refuse", [set(), {0}, {1}, {0, 1}])
def test_the_poll_holds_or_hands_back_every_chunk(refuse):
    streams, runtime = _streams([1, 2], refuse)
    w = Worker.__new__(Worker)
    w.streams, w._graph_runtime = streams, runtime
    Worker._poll_stream_buffers(w)
    assert len(streams.popped) == 2
    assert not streams.unsettled()


@pytest.mark.parametrize("success, dropped", [(False, set()), (True, set()), (True, {2})])
@pytest.mark.parametrize("refuse", [set(), {1}])
def test_a_speculation_hands_back_only_what_its_settle_undid(success, dropped, refuse):
    streams, runtime = _streams([1, 2], refuse)
    ctx = SpeculationContext(
        graph_runtime=runtime, tensor_manager=streams._tensor_manager,
        request_state=streams._request_state, scheduler=None, streams=streams,
    )
    prep = SpeculationPreparer(ctx)._prepare_dicts(
        ScheduledBatch(node_name=NODE, graph_walk="w"), NODE, "w", [1, 2], None, PhaseLaps("t."),
    )
    spec = Speculation(
        scheduled_batch=ScheduledBatch(
            node_name=NODE, graph_walk="w",
            request_to_worker_graph={r: 0 for r in prep.ready_rids if r not in dropped},
        ),
        node_batch=SimpleNamespace(), consumed_edges=set(), continuing_rids=set(prep.ready_rids),
        partition="p", is_new_iter=False, is_same_node=True,
        consumed_streaming_edges=prep.consumed_streaming_edges, spec_id=prep.spec_id,
    )
    w = Worker.__new__(Worker)
    w.streams, w._graph_runtime = streams, runtime
    w.engine_manager = SimpleNamespace(get_engine=lambda node: SimpleNamespace(release_cached_prefix=lambda b: None))
    Worker._settle_speculation(w, spec, success, dropped)
    assert not streams.unsettled()
    kept = {id(se) for r, edges in runtime.spec_held.items() for se in edges if success and r not in dropped}
    assert streams.held == kept


def test_the_ledger_catches_a_chunk_handed_back_while_held():
    streams, runtime = _streams([1])
    w = Worker.__new__(Worker)
    w.streams, w._graph_runtime = streams, runtime
    Worker._poll_stream_buffers(w)
    (se,) = streams.popped.values()
    with pytest.raises(AssertionError, match="graph holds"):
        streams.return_edge(1, se)

"""A STREAM_ITEMS frame has the effect of the INPUT_SIGNALS it replaces: one
per row, each a single streamed edge. Same reads started, same items
registered with each stream buffer in the same order, the same rows parked
(request not here yet) or dropped and ACKed (request removed), and nothing for
a request being drained -- only the reads start in one call."""
import logging
from types import SimpleNamespace

import pytest
import torch

from mstar.communication.stream_items import StreamItemsBuilder, row_input_signals, rows
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.utils.containers import RecentSet
from mstar.utils.ipc_format import WorkerMessage, WorkerMessageType
from mstar.worker.worker import Worker

EDGE, NODE = "new_token", "snac_decoder"
HANDLES = {"r1": 1, "r2": 2, "draining": 3}


def _info(uuid):
    return TensorPointerInfo(
        dims=[1], dtype=torch.int64, nbytes=8, address=8 * uuid, stride=[1], uuid=uuid,
        source_session_id="h:1", source_entity="w0", shm_segment="seg0", shm_offset=64 * uuid,
        _source_node_name="LLM", _source_graph_walk="decode",
    )


def _frame():
    """Two known requests (one with two items), one not here yet, one
    removed, one being drained."""
    b = StreamItemsBuilder("snac")
    for wire_rid, uuids in [("r1", [1, 2]), ("later", [3]), ("removed", [4]), ("draining", [5]), ("r2", [6])]:
        assert b.try_add(wire_rid, GraphEdge(
            next_node=NODE, name=EDGE, tensor_info=[_info(u) for u in uuids], is_streaming=True,
        ))
    return b.build()


def _worker():
    w = Worker.__new__(Worker)
    w.worker_id = "w1"
    w.enable_nvtx = False
    w._draining_rids = {"draining"}
    w._removed_rids = RecentSet(16)
    w._removed_rids.add("removed")
    w._unprocessed_messages = {}
    w._rid = HANDLES.get
    w.request_state = SimpleNamespace(per_request_info={
        h: SimpleNamespace(stream_buffers={EDGE: StreamBuffer(h, EDGE, "LLM", FixedChunkPolicy(1))})
        for h in HANDLES.values()
    })
    w.scheduler = SimpleNamespace(failed_rids=set())
    w.reads, w.acked = [], []

    def start_read_tensors(rid, edges, graph_walk=None):
        w.reads.extend((rid, e.name, [i.uuid for i in e.tensor_info]) for e in edges if e.tensor_info)
        return []

    def start_read_tensors_multi(reads):
        for rid, edges in reads:
            start_read_tensors(rid, edges)
        return []

    w.tensor_manager = SimpleNamespace(
        start_read_tensors=start_read_tensors,
        start_read_tensors_multi=start_read_tensors_multi,
        ack_unread_tensors=lambda edges: w.acked.append(
            [i.uuid for e in edges for i in e.tensor_info],
        ),
    )
    w.wakeup_event = SimpleNamespace(register_futures=lambda futures: None)
    return w


def _state(w):
    return (
        w.reads, w.acked,
        {h: list(i.stream_buffers[EDGE]._tensor_ids_in_order) for h, i in w.request_state.per_request_info.items()
         if EDGE in i.stream_buffers},
        {r: [m.body for m in ms] for r, ms in w._unprocessed_messages.items()},
    )


def test_a_frame_does_what_its_rows_as_input_signals_did():
    frame = _frame()
    batched = _worker()
    batched._process_stream_items(frame)
    one_by_one = _worker()
    for row in rows(frame):
        one_by_one._process_message_list([WorkerMessage(
            message_type=WorkerMessageType.INPUT_SIGNALS, body=row_input_signals(frame, row),
        )])
    assert _state(batched) == _state(one_by_one)
    reads, acked, registered, parked = _state(batched)
    assert reads == [(1, EDGE, [1, 2]), (2, EDGE, [6])]
    assert acked == [[4]]
    assert registered == {1: [1, 2], 2: [6], 3: []}
    assert list(parked) == ["later"]


def test_a_frame_starts_its_reads_in_one_call():
    w = _worker()
    calls = []
    w.tensor_manager.start_read_tensors_multi = lambda reads: calls.append(reads) or []
    w._process_stream_items(_frame())
    assert [[(rid, [i.uuid for e in edges for i in e.tensor_info]) for rid, edges in c] for c in calls] == [
        [(1, [1, 2]), (2, [6])],
    ]


@pytest.mark.parametrize("failed", [True, False])
def test_a_known_request_with_no_stream_buffer_is_acked_unread(failed, caplog):
    """A request whose ingest failed here is known but has no stream buffer:
    its rows are acked unread (warning unless the ingest failure is known),
    as INPUT_SIGNALS does, and the frame's other rows still read."""
    def worker():
        w = _worker()
        w.request_state.per_request_info[1].stream_buffers = {}
        if failed:
            w.scheduler.failed_rids.add(1)
        return w

    frame = _frame()
    batched = worker()
    with caplog.at_level(logging.WARNING, logger="mstar.worker.worker"):
        batched._process_stream_items(frame)
    warned = [r for r in caplog.records if "no stream buffer" in r.getMessage()]
    caplog.clear()
    one_by_one = worker()
    for row in rows(frame):
        one_by_one._process_message_list([WorkerMessage(
            message_type=WorkerMessageType.INPUT_SIGNALS, body=row_input_signals(frame, row),
        )])
    assert _state(batched) == _state(one_by_one)
    reads, acked, registered, _ = _state(batched)
    assert reads == [(2, EDGE, [6])]
    assert acked == [[1, 2], [4]]
    assert registered == {2: [6], 3: []}
    assert len(warned) == (0 if failed else 1)

"""Internal request IDs remain integers through the typed Python wire."""

import mstar.communication.wire_types  # noqa: F401 - registers stable wire tags
from mstar.communication import wire
from mstar.engine.resources.base import MoveKind
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    EdgeItems,
    FailRequests,
    NewRequestConductor,
    OffloadDelta,
    ScheduleTPNode,
    WorkerMessage,
    WorkerMessageType,
)

RID = (1 << 62) | 7


def _round_trip(message):
    return wire.decode(wire.encode(message))


def test_request_entry_and_failure_keep_a_63_bit_integer():
    incoming = ConductorMessage(
        ConductorMessageType.NEW_REQUEST,
        NewRequestConductor(
            request_id=RID,
            initial_signals={},
            initial_input_modalities=["text"],
            initial_output_modalities=["text"],
            input_metadata={},
            model_kwargs={},
            fallback_seed=19,
        ),
    )
    received = _round_trip(incoming)
    assert type(received.body.request_id) is int
    assert received.body.request_id == RID
    assert received.body.fallback_seed == 19
    failure = ConductorMessage(ConductorMessageType.FAIL_REQUESTS, FailRequests({RID: "failed"}))
    assert _round_trip(failure).body.errors == {RID: "failed"}


def test_tp_and_stream_columns_keep_integer_request_keys():
    delta = OffloadDelta.new()
    delta.add(MoveKind.RESERVE, RID, pool="kv", victims=(RID + 1,))
    schedule = WorkerMessage(
        WorkerMessageType.SCHEDULE_TP,
        ScheduleTPNode(
            node_name="llm", graph_walk="decode", request_ids=[RID],
            resident_delta=delta,
            prefix_hits={RID: {}}, page_trails={"kv": (1, 2)},
        ),
    )
    body = _round_trip(schedule).body
    assert body.request_ids == [RID]
    assert body.resident_delta.rids == [RID]
    assert body.resident_delta.victims == [[RID + 1]]
    assert body.prefix_hits == {RID: {}}
    assert body.page_trails == {"kv": (1, 2)}

    items = WorkerMessage(
        WorkerMessageType.EDGE_ITEMS,
        EdgeItems(partition_name="p", source_session_id="session", source_entity="worker",
                  tp_size=1, tp_rank=0, request_ids=[RID]),
    )
    assert _round_trip(items).body.request_ids == [RID]

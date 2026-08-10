"""Wire-format contract for TP async scheduling (implementation map row 3).

Pins the two properties the rest of the protocol is built on:

1. Tagging ``ScheduleTPNode`` with speculation does not disturb today's serial
   broadcast — existing three-arg construction still works and is
   non-speculative by omission.
2. ``CancelSpec`` is its own message type, dispatchable without touching the
   schedule FIFO, and names exactly one speculation by seq.

The *semantics* of Cancel (void-only, never retract) are enforced by the CPU
model checker in ``tp_async_sim.py``; these tests only pin the carrier.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

# Import mstar from THIS worktree, not the venv's editable install. Without
# this, the PEP-660 finder falls back to the primary checkout and the test
# silently validates code that isn't the code under change (the 2026-08-09
# lesson that made 2/4 of the tp-follow fence tests "fail").
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.utils.ipc_format import (  # noqa: E402
    CancelSpec,
    ScheduleTPNode,
    WorkerMessage,
    WorkerMessageType,
)


def test_existing_construction_is_unchanged_and_non_speculative():
    """Today's leader broadcast site passes three positional args. It must keep
    compiling AND must not accidentally become speculative."""
    msg = ScheduleTPNode("node_a", "decode", ["r1", "r2"])
    assert msg.node_name == "node_a"
    assert msg.graph_walk == "decode"
    assert msg.request_ids == ["r1", "r2"]
    assert msg.speculative is False
    assert msg.spec_seq == -1


def test_speculative_head_carries_its_seq():
    msg = ScheduleTPNode("node_a", "decode", ["r1"], speculative=True, spec_seq=7)
    assert msg.speculative is True
    assert msg.spec_seq == 7


def test_cancel_names_exactly_one_speculation():
    c = CancelSpec(spec_seq=7)
    assert c.spec_seq == 7
    # One seq per message: a Cancel that could name a range would invite
    # "cancel everything pending", which is the retract semantics the model
    # checker refutes.
    fields = [f.name for f in dataclasses.fields(CancelSpec)]
    assert fields == ["spec_seq"]


def test_cancel_is_a_distinct_message_type():
    """Cancel must be dispatchable without going through the schedule path —
    a void has to reach a rank whose FIFO must not be disturbed."""
    assert WorkerMessageType.CANCEL_SPEC != WorkerMessageType.SCHEDULE_TP
    assert WorkerMessageType.CANCEL_SPEC.value == "cancel_spec"
    wrapped = WorkerMessage(
        message_type=WorkerMessageType.CANCEL_SPEC, body=CancelSpec(spec_seq=3)
    )
    assert wrapped.body.spec_seq == 3


def test_spec_seq_round_trips_through_pickle():
    """TP messages travel as pickled dataclasses over ZMQ, so the tag has to
    survive that specific path — not a dict round trip."""
    import pickle

    msg = ScheduleTPNode("n", "decode", ["r1"], speculative=True, spec_seq=11)
    back = pickle.loads(pickle.dumps(msg))
    assert back == msg
    assert back.speculative is True and back.spec_seq == 11

    c = pickle.loads(pickle.dumps(CancelSpec(spec_seq=11)))
    assert c.spec_seq == 11

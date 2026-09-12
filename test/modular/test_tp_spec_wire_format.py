"""Wire-format contract for TP async scheduling (implementation map row 3).

Pins the properties the rest of the protocol is built on:

1. Tagging ``ScheduleTPNode`` with speculation does not disturb today's serial
   broadcast — existing three-arg construction still works and is
   non-speculative by omission.
2. A speculative head names itself (``spec_seq``) and the batch it was built
   from (``spec_from_seq``); a follower matches ``spec_from_seq`` against its
   own in-flight batch to decide whether it may build the head early.
3. There is NO cancel/commit message type: voids are derived from state that
   is identical on every rank, never signalled (see the ``ScheduleTPNode``
   docstring and ``tp_async_sim.py`` for why a signalled retract is unsafe).
   The only other message is ``TPNoSpeculation`` — "no head from step s" —
   which carries no verdict, only the absence of a head.

The *semantics* (void-only, symmetric derivation) are enforced by the CPU
model checker in ``tp_async_sim.py``; these tests only pin the carrier.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Import mstar from THIS worktree, not the venv's editable install. Without
# this, the PEP-660 finder falls back to the primary checkout and the test
# silently validates code that isn't the code under change (the 2026-08-09
# lesson that made 2/4 of the tp-follow fence tests "fail").
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.utils.ipc_format import (  # noqa: E402
    ScheduleTPNode,
    TPNoSpeculation,
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
    assert msg.spec_from_seq == -1


def test_speculative_head_names_itself_and_its_parent():
    msg = ScheduleTPNode(
        "node_a", "decode", ["r1"], speculative=True, spec_seq=7, spec_from_seq=6
    )
    assert msg.speculative is True
    assert msg.spec_seq == 7
    assert msg.spec_from_seq == 6


def test_no_cancel_or_commit_message_type():
    """Voids are derived on every rank from replicated state, never signalled.
    A cancel type reappearing here means someone re-introduced the retract
    race the model checker refutes (``B2_RETRACT``)."""
    names = {m.name for m in WorkerMessageType}
    assert "CANCEL_SPEC" not in names
    assert "COMMIT_SPEC" not in names
    assert "SCHEDULE_TP" in names
    assert "TP_NO_SPEC" in names


def test_no_spec_marker_names_the_step_and_round_trips():
    import pickle

    msg = TPNoSpeculation("n", "decode", spec_from_seq=6)
    wrapped = WorkerMessage(message_type=WorkerMessageType.TP_NO_SPEC, body=msg)
    back = pickle.loads(pickle.dumps(wrapped))
    assert back.message_type is WorkerMessageType.TP_NO_SPEC
    assert back.body == msg and back.body.spec_from_seq == 6


def test_wrapped_head_dispatches_on_schedule_tp():
    wrapped = WorkerMessage(
        message_type=WorkerMessageType.SCHEDULE_TP,
        body=ScheduleTPNode("n", "decode", ["r1"], speculative=True, spec_seq=3, spec_from_seq=2),
    )
    assert wrapped.message_type is WorkerMessageType.SCHEDULE_TP
    assert wrapped.body.spec_seq == 3


def test_seqs_round_trip_through_pickle():
    """TP messages travel as pickled dataclasses over ZMQ, so the tags have to
    survive that specific path — not a dict round trip."""
    import pickle

    msg = ScheduleTPNode(
        "n", "decode", ["r1"], speculative=True, spec_seq=11, spec_from_seq=10
    )
    back = pickle.loads(pickle.dumps(msg))
    assert back == msg
    assert back.speculative is True
    assert back.spec_seq == 11 and back.spec_from_seq == 10

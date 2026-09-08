"""A failed request is reported to the conductor exactly once.

Failures arrive in two waves: ``prepare_inputs`` / ``postprocess`` raise inside
``exec_and_postprocess`` and are handled by the main loop, then
``check_stop_for_batch`` runs later, inside ``_postprocess_batch``, and can add
more. Both waves have to be reported, and neither twice — which means the first
wave has to be cleared off the batch once it has been handled.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.worker.worker import Worker


def _pending(rids, failed):
    return SimpleNamespace(
        batch=SimpleNamespace(
            node_objects={rid: object() for rid in rids},
            request_to_worker_graph=dict.fromkeys(rids, "wg0"),
        ),
        node_batch=SimpleNamespace(
            request_ids=list(rids),
            per_request_info={rid: object() for rid in rids},
            failed_requests=dict(failed),
        ),
    )


def _drop(pending, outputs, failed):
    # unbound: _drop_failed_rids touches nothing on self
    Worker._drop_failed_rids(None, pending, outputs, failed)


def test_a_handled_failure_leaves_the_batch():
    pending = _pending(["r0", "r1"], {"r0": "boom"})
    outputs = {"r0": {}, "r1": {}}

    _drop(pending, outputs, dict(pending.node_batch.failed_requests))

    assert pending.node_batch.request_ids == ["r1"]
    assert "r0" not in outputs
    assert "r0" not in pending.batch.node_objects
    assert "r0" not in pending.node_batch.per_request_info


def test_a_handled_failure_is_cleared_so_it_is_not_reported_again():
    """The regression: ``failed_requests`` kept the rid, so the later
    ``_postprocess_batch`` block re-reported it to the conductor."""
    pending = _pending(["r0", "r1"], {"r0": "boom"})

    _drop(pending, {}, dict(pending.node_batch.failed_requests))

    assert pending.node_batch.failed_requests == {}


def test_a_later_check_stop_failure_is_still_distinguishable():
    """What the clearing buys: after the first wave is handled, whatever
    ``check_stop`` adds is exactly the set left to report."""
    pending = _pending(["r0", "r1"], {"r0": "prepare_inputs blew up"})

    _drop(pending, {}, dict(pending.node_batch.failed_requests))
    # check_stop_for_batch raises for r1 and registers it
    pending.node_batch.failed_requests["r1"] = "check_stop blew up"

    assert pending.node_batch.failed_requests == {"r1": "check_stop blew up"}
    _drop(pending, {}, dict(pending.node_batch.failed_requests))
    assert pending.node_batch.request_ids == []

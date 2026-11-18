"""Where each request is in its lifecycle on this worker.

Bookkeeping only: this records what arrived and answers which requests can be
torn down or ACKed now. It asks the graph runtime, which owns the handle <-> wire
id table, to translate, and calls no other component. The worker runs the
teardowns.

Removes and drains are recorded when their message arrives, and applied at the
worker's teardown points, where nothing that reads the state they move is
outstanding.
"""
from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Set

from mstar.utils.containers import RecentSet
from mstar.utils.ipc_format import (
    DrainRequest,
    MessageSource,
    RemoveRequest,
    WorkerMessage,
)
from mstar.worker.services import WorkerLayout

logger = logging.getLogger(__name__)

# Removed wire ids to remember, to drop a message that arrives after its
# request's teardown instead of parking it for good. Such a message trails its
# teardown by about a loop iteration and a socket hop, far fewer teardowns than
# this; a count, not a time, so a stalled worker that reads both late still
# matches them. Every submission has its own key, so a remembered key never gets
# another NEW_REQUEST.
_REMOVED_RIDS_KEEP = 4096

# `no_new_work` with nothing to exclude, so a scan with no teardown allocates nothing
_NONE: frozenset[int] = frozenset()


class RequestLifecycle:
    def __init__(self, layout: WorkerLayout, graph_runtime):
        self._is_tp_follower = layout.is_tp_follower
        self._runtime = graph_runtime

        # rid -> the order requests arrived at this worker in. Also which rids
        # are admitted here and not yet torn down: the runtime keeps a handle
        # past `removed`, until the teardown's last step frees it.
        self.arrivals: dict[int, int] = {}
        self._arrived = 0

        # wire id -> messages for a request that has not arrived. Keyed by the
        # string: these come before the NEW_REQUEST that mints the handle.
        self._parked: dict[str, list[WorkerMessage]] = {}
        # Wire ids of requests torn down here, or that the conductor removed
        # without this worker admitting them. Keyed by the string: the handle is gone.
        self._removed: RecentSet[str] = RecentSet(_REMOVED_RIDS_KEEP)

        # Reported, or about to be: no new work, until the removal comes back.
        self.failed_rids: set[int] = set()
        self._unreported: dict[int, str] = {}
        # the HTTP status of an unreported failure that isn't a 500
        self._statuses: dict[int, int] = {}

        # rid -> the step its remove follows (-1 unstamped), waiting for no step
        # to hold the rid. No step that still needs an input is ahead of these,
        # so inputs for them are dropped.
        self.pending_removes: dict[int, int] = {}
        # step seq -> rids rank 0 released after that step, held until this rank
        # has consumed it, since rank 0 released their pages between two steps.
        self._removes_awaiting_step: dict[int, list[int]] = {}

        # Drains (abort/fail). Waiting for no step to hold the rid; taking no new
        # work until the remove; and having sent READS_DONE, after which no read
        # starts. Keyed by the wire string: a drain can precede the NEW_REQUEST.
        self.pending_drains: set[str] = set()
        self.draining: set[str] = set()
        self._reads_done_sent: set[str] = set()

    # -- arrival ------------------------------------------------------------

    def add(self, rid: int, wire_rid: str) -> list[WorkerMessage]:
        """A NEW_REQUEST, of which a request with several partitions here sends
        one per partition: it keeps its first arrival. Its parked messages
        stay parked until ``take_parked``."""
        if rid not in self.arrivals:
            self._arrived += 1
            self.arrivals[rid] = self._arrived
        if wire_rid in self.draining or wire_rid in self.pending_drains:
            # Drained before admission: no new work, while committed TP steps still run.
            self.failed_rids.add(rid)

    def take_parked(self, wire_rid: str) -> list[WorkerMessage]:
        """The messages parked for a request, for the caller to replay now. Taken
        only at the replay, so a setup that raises leaves them for the request's
        teardown to drop."""
        return self._parked.pop(wire_rid, [])

    def park(self, message: WorkerMessage) -> bool:
        """Hold a message for a request that has not arrived. False if it never
        will, so the caller drops the message."""
        wire_rid = message.body.request_id
        if wire_rid in self._removed:
            return False
        self._parked.setdefault(wire_rid, []).append(message)
        return True

    def takes_input(self, wire_rid: str) -> bool:
        """Whether this rank still reads the inputs of NEW_REQUEST and INPUT_SIGNALS
        for a request: until it sends READS_DONE, draining or not, since a step
        rank 0 committed may wait on one that arrives behind the forwarded drain.
        After READS_DONE the producer may free what it sent."""
        return wire_rid not in self._reads_done_sent

    # -- failure -------------------------------------------------------------

    def fail(self, errors: dict[int, str], status: int = 500) -> None:
        """Requests this worker can no longer serve, and the HTTP status they fail
        with. Rids already torn down are dropped: the conductor answers a failure with
        the REMOVE that clears the entry, and it sends none for a request it no longer knows."""
        for rid, msg in errors.items():
            if rid in self.arrivals:
                self.failed_rids.add(rid)
                # the first error is the specific one; a step's generic one can follow it
                if rid not in self._unreported:
                    self._unreported[rid] = msg
                    if status != 500:
                        self._statuses[rid] = status

    def take_statuses(self, rids) -> dict[int, int]:
        """The status of each of ``rids`` (just taken from `take_failures`) that isn't a 500."""
        statuses = self._statuses
        return {rid: statuses.pop(rid) for rid in rids if rid in statuses} if statuses else {}

    def take_failures(self) -> dict[int, str]:
        """The failures not yet reported to the conductor."""
        if not self._unreported:
            return self._unreported
        failures, self._unreported = self._unreported, {}
        return failures

    # -- removes -------------------------------------------------------------

    def remove(self, rid: int, body: RemoveRequest, last_consumed_seq: int) -> None:
        """A REMOVE for a request this worker holds. A follower takes only rank
        0's, to tear down in rank 0's order; it can't release pages rank 0
        released between two steps until it has consumed the earlier one."""
        if not self._is_tp_follower:
            self.pending_removes.setdefault(rid, -1)
        elif body.source == MessageSource.TP_RANK_0:
            self._removes_awaiting_step.setdefault(body.after_tp_seq, []).append(rid)
            self._land_removes(last_consumed_seq)

    def _land_removes(self, last_consumed_seq: int) -> None:
        if self._removes_awaiting_step:
            for seq in sorted(s for s in self._removes_awaiting_step if s <= last_consumed_seq):
                self.pending_removes.update(dict.fromkeys(self._removes_awaiting_step.pop(seq), seq))

    def removes_ready(self, in_flight: Collection[int], last_consumed_seq: int) -> list[int]:
        """The requests to tear down now, in order; each stays pending until
        ``removed``, so a teardown that raises doesn't lose the ones after it. A
        step holding a rid defers its remove; on a follower it also defers the
        ones after it, which rank 0 tore down later. The free list hands pages
        back out in the order they were freed."""
        self._land_removes(last_consumed_seq)
        if not self.pending_removes:
            return []
        ready = []
        # A remove replayed from the parking lot can arrive after a later stamp.
        pending = self.pending_removes
        for rid in sorted(pending, key=pending.__getitem__) if self._is_tp_follower else pending:
            if rid not in in_flight:
                ready.append(rid)
            elif self._is_tp_follower:
                break
        return ready

    def removed(self, rid: int | None, wire_rid: str) -> Iterable[WorkerMessage]:
        """Forget a request that was torn down, or never admitted (``rid`` None).
        Returns the messages that were parked for it, for the caller to drop."""
        if rid is not None:
            self.arrivals.pop(rid, None)
            self.failed_rids.discard(rid)
            self.pending_removes.pop(rid, None)
            self._unreported.pop(rid, None)
            self._statuses.pop(rid, None)
            for seq, rids in list(self._removes_awaiting_step.items()):
                if rid in rids:
                    rids[:] = [r for r in rids if r != rid]
                    if not rids:
                        del self._removes_awaiting_step[seq]
        self.pending_drains.discard(wire_rid)
        self.draining.discard(wire_rid)
        self._reads_done_sent.discard(wire_rid)
        self._removed.add(wire_rid)
        return self._parked.pop(wire_rid, ())

    def teardown_failed(self, rid: int) -> None:
        """A teardown that raised after ``removed``: the request's state is
        partly gone, so it takes no more work. It is not retried, since it would
        likely raise again every pass, and its handle is never freed (that is
        the teardown's last step), so no later request inherits the mark."""
        self.failed_rids.add(rid)

    # -- drains --------------------------------------------------------------

    def drain(self, body: DrainRequest) -> None:
        """A DRAIN. A follower honors only the leader's forward, like a remove,
        but for a request never admitted here: the conductor sends a NEW_REQUEST
        ahead of its drain on one FIFO socket, so the leader has no handle either
        and forwards nothing."""
        if (
            not self._is_tp_follower or body.source == MessageSource.TP_RANK_0
            or self._handle(body.request_id) is None
        ):
            self.pending_drains.add(body.request_id)

    def drains_ready(self, in_flight: Collection[int]) -> list[str]:
        """The drains to begin now; each stays pending until ``begin_drain``. A
        step holding the rid defers its drain, since clearing the scheduler
        under it would race the GPU thread."""
        if not self.pending_drains:
            return []
        handle = self._handle
        return [wire for wire in self.pending_drains if handle(wire) not in in_flight]

    def begin_drain(self, wire_rid: str) -> int | None:
        """Take no new work for the request. Its handle, if it has been admitted."""
        self.pending_drains.discard(wire_rid)
        rid = self._handle(wire_rid)
        if rid is not None:
            self.failed_rids.add(rid)
        self.draining.add(wire_rid)
        return rid

    def open_drains(self) -> list[tuple[str, int | None]]:
        """The drains still owing READS_DONE, each with its handle (None if the
        request was never admitted, so it holds no handle-keyed state)."""
        if not self.draining:
            return []
        handle = self._handle
        return [(w, handle(w)) for w in self.draining if w not in self._reads_done_sent]

    def reads_done(self, wire_rid: str) -> None:
        """READS_DONE went out. The rid stays draining until its remove."""
        self._reads_done_sent.add(wire_rid)

    def _handle(self, wire_rid: str) -> int | None:
        """``wire_rid``'s handle while it is admitted here, else None."""
        rid = self._runtime.get_rid_handle(wire_rid)
        return rid if rid in self.arrivals else None

    # -- queries --------------------------------------------------------------

    def teardowns_pending(self) -> bool:
        """Whether any rid might be tearing down, so a speculation has to filter
        its candidates one by one."""
        return bool(
            self.pending_removes or self.failed_rids
            or self.pending_drains or self.draining
        )

    def no_new_work(self, *also: Iterable[int]) -> Set[int]:
        """The rids a scan gives no new step: failed or with a remove pending,
        plus ``also``'s. Unlike `closing`, a drain counts only once begun, when
        it marks its rid failed."""
        if not also and not self.failed_rids and not self.pending_removes:
            return _NONE
        return self.failed_rids.union(self.pending_removes, *also)

    def closing(self, rid: int, wire_rid: str | None) -> bool:
        """Whether ``rid`` (``wire_rid`` on the wire) is on its way out:
        `no_new_work`'s, or with a drain pending or begun."""
        return (
            rid in self.pending_removes or rid in self.failed_rids
            or wire_rid in self.pending_drains or wire_rid in self.draining
        )

    def is_tearing_down(self, rid: int) -> bool:
        """`closing`, for a rid without its wire id: no further speculative
        work for ``rid``.

        A deferred drain only fires once no step holds the rid, and a same-node
        speculation chain keeps holding it every step, so a drain the chain does
        not see would never fire and the rollout would run to ``max_iters`` for
        a client that has gone."""
        wire = None
        if (self.pending_drains or self.draining) and rid in self.arrivals:
            wire = self._runtime.get_rid_string(rid)
        return self.closing(rid, wire)

    def publishable(self, batch) -> list[int]:
        """The rows of a landed step that may publish: no late KV snapshot for a
        request on its way out, whose remove waits behind this very step, and
        none for a rebuild row, whose request keeps the publication from before
        its drop. Called on the GPU thread, so it reads these live."""
        closing, rebuild_rids = self.closing, batch.step_context.rebuild_rids
        return [
            rid for rid in batch.request_ids
            if (info := batch.per_request_info.get(rid)) is not None
            if rid not in rebuild_rids and not closing(rid, info.request_id)
        ]

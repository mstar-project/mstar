"""Worker-side session bookkeeping.

The state a worker keeps for the persistent sessions it holds, and every
decision that state drives: which requests belong to a session, whether a
resumed request may be ingested yet, which deferred removal also ends its
session, and which held teardown is free to run.

Requests are the worker's rid *handles*, like the rest of its per-request state;
a held NEW_REQUEST is the one exception, since its handle is minted only once the
worker admits it.

Nothing here talks to the engine or the communicator — the worker does that. The
one thing it cannot answer for itself is whether a request is still on its way
out of the worker, so that predicate is injected.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from mstar.utils.ipc_format import NewRequest

logger = logging.getLogger(__name__)


def _session_id_of(body: NewRequest) -> str | None:
    """The session a held ingest belongs to. Only ever held with one."""
    session = body.request_info.session
    return session.session_id if session is not None else None


@dataclass
class WorkerSessionManager:
    """The persistent sessions one worker holds state for."""

    # rid handle -> is it still on its way out of this worker (held by an
    # in-flight step, a deferred removal, or still registered)? Only the worker
    # knows.
    is_leaving: Callable[[int], bool]

    _session_rids: dict[str, set[int]] = field(default_factory=dict)
    _rid_session: dict[int, str] = field(default_factory=dict)
    # TEARDOWN_SESSIONs waiting on their requests to finish leaving
    _pending_session_teardowns: set[str] = field(default_factory=set)
    # session id -> the TP followers this worker leads for it, recorded as its
    # requests are bound; a teardown is forwarded to them
    _session_followers: dict[str, set[str]] = field(default_factory=dict)
    # rid handle -> nodes whose loop ran a speculative step past its stop for
    # it; handed to the engine at removal. Session requests only.
    _overshot: dict[int, set[str]] = field(default_factory=dict)
    # NEW_REQUESTs held until their session's previous request has finished
    # leaving, so the resumed request is handed the state that request
    # built rather than an empty stream.
    _pending_session_ingests: list[NewRequest] = field(default_factory=list)

    # ------------------------------------------------------------------
    # What the worker holds
    # ------------------------------------------------------------------

    def get_rids(self, session_id: str) -> set[int]:
        return self._session_rids.get(session_id, set())

    def session_of(self, request_id: int) -> str | None:
        return self._rid_session.get(request_id)

    def requests_still_leaving(
        self, session_id: str, except_rid: int | None = None,
    ) -> bool:
        """Whether any of the session's requests is still on its way out.

        ``except_rid`` skips one: the conductor sends a NEW_REQUEST per
        partition, so a worker serving two of them sees the same rid twice and
        must not read its own first ingest as a request still leaving.
        """
        return any(
            rid != except_rid and self.is_leaving(rid)
            for rid in self.get_rids(session_id)
        )

    @property
    def has_pending(self) -> bool:
        """Whether anything is held waiting for a session to free up."""
        return bool(
            self._pending_session_ingests or self._pending_session_teardowns
        )

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def hold_if_not_ready(
        self, body: NewRequest, own_handle: int | None = None,
    ) -> bool:
        """Hold a resumed request whose session has not handed its state over.

        ``own_handle`` is the handle this request already has here, if the
        worker has seen another of its partitions — it is not a request leaving.
        True when the request was held; the worker must then leave it alone
        until ``take_held_ingests`` gives it back.
        """
        session = body.request_info.session
        if session is None or not self.requests_still_leaving(
            session.session_id, except_rid=own_handle,
        ):
            return False
        # ingesting now would adopt nothing
        if not any(
            held.request_id == body.request_id
            for held in self._pending_session_ingests
        ):
            logger.info(
                "Holding request %s: session %s is still releasing its "
                "previous request", body.request_id, session.session_id,
            )
        self._pending_session_ingests.append(body)
        return True

    def drop_held(self, request_id: str) -> bool:
        """Forget a held request (by its wire id) that was removed before it
        could be admitted. True when one was held."""
        kept = [
            held for held in self._pending_session_ingests
            if held.request_id != request_id
        ]
        dropped = len(kept) != len(self._pending_session_ingests)
        self._pending_session_ingests = kept
        return dropped

    def take_held_ingests(self) -> tuple[list[NewRequest], list[NewRequest]]:
        """``(ready, dropped)``: the requests waiting on their session, split by
        whether that session is still there.

        ``dropped`` is for a session whose teardown is itself waiting. Admitting
        one would give that session a live request again, so its teardown would
        never come ready and every later teardown of the same id here would
        queue behind it. The worker fails those rather than admitting them into
        a session that is going.

        A ready one that turns out still not to be ready holds itself again, so
        the caller re-checks rather than trusting this.
        """
        held, self._pending_session_ingests = self._pending_session_ingests, []
        ready, dropped = [], []
        for body in held:
            going = _session_id_of(body) in self._pending_session_teardowns
            (dropped if going else ready).append(body)
        if dropped:
            logger.info(
                "Dropped %d request(s) waiting on a session that is being torn "
                "down: %s", len(dropped), [b.request_id for b in dropped],
            )
        return ready, dropped

    def bind(
        self, request_id: int, session_id: str | None,
        tp_followers: set[str] | None = None,
    ) -> None:
        """Record which session a request belongs to; a no-op without one.

        ``tp_followers`` are the ranks this worker leads for the request; the
        session's teardown goes to them too.
        """
        if session_id is None:
            return
        self._session_rids.setdefault(session_id, set()).add(request_id)
        self._rid_session[request_id] = session_id
        if tp_followers:
            self._session_followers.setdefault(session_id, set()).update(
                tp_followers
            )

    def tp_followers(self, session_id: str) -> set[str]:
        """The TP followers this worker leads for the session."""
        return set(self._session_followers.get(session_id, ()))

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def note_overshoot(self, request_id: int, node_name: str) -> None:
        """The request's loop on ``node_name`` ran one speculative step past
        its stop. Kept for a session request only: no one else continues
        from the KV that step wrote into."""
        if request_id in self._rid_session:
            self._overshot.setdefault(request_id, set()).add(node_name)

    def take_overshot(self, request_id: int) -> frozenset[str]:
        return frozenset(self._overshot.pop(request_id, ()))

    def release(self, request_id: int) -> str | None:
        """Drop the request, returning the session it belonged to.

        The session itself stays — it keeps what the request built — unless the
        worker follows up with ``forget_session``.
        """
        self._overshot.pop(request_id, None)
        session_id = self._rid_session.pop(request_id, None)
        if session_id is not None:
            self.get_rids(session_id).discard(request_id)
        return session_id

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def hold_teardown(self, session_id: str) -> None:
        """Wait for the session's requests to finish leaving: one of their
        removals would otherwise hand state back to a session that is gone."""
        self._pending_session_teardowns.add(session_id)

    def ready_teardowns(self) -> list[str]:
        """The held teardowns whose requests have all left."""
        return [
            session_id for session_id in sorted(self._pending_session_teardowns)
            if not self.requests_still_leaving(session_id)
        ]

    def forget_session(self, session_id: str) -> None:
        """Drop every trace of the session; its state is being freed.

        Including any NEW_REQUEST still held for it: the state it is waiting to
        adopt is going, and ``_apply_pending_sessions`` admits held ingests
        first, so leaving it behind would admit a request nobody will schedule
        and block every later teardown of the same id on this worker.
        """
        self._pending_session_teardowns.discard(session_id)
        self._session_followers.pop(session_id, None)
        for rid in self._session_rids.pop(session_id, set()):
            self._rid_session.pop(rid, None)
            self._overshot.pop(rid, None)
        dropped = [
            held for held in self._pending_session_ingests
            if _session_id_of(held) == session_id
        ]
        if dropped:
            self._pending_session_ingests = [
                held for held in self._pending_session_ingests
                if _session_id_of(held) != session_id
            ]
            logger.info(
                "Dropped %d request(s) held for session %s; it is being torn "
                "down: %s", len(dropped), session_id,
                [held.request_id for held in dropped],
            )

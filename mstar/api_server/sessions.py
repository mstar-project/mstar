"""API-server-side session tracking.

Owns what the HTTP layer needs to answer for a session before anything reaches
the conductor: which sessions exist, which one a request may name, when a
session has gone idle long enough to be collected, and which ids are held by a
tombstone while their state is being freed.

The registry is the only place that decides a session's fate; the conductor and
the workers just carry it out.
"""

import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from mstar.model.sessions import (
    SessionCapacityPolicy,
    SessionsConfig,
    SessionTTLMode,
)

logger = logging.getLogger(__name__)

# What a client may name its session: it goes in a URL path
# (``DELETE /sessions/{id}``), so nothing that needs escaping or reads as a
# path segment. A minted id is a uuid4, which fits.
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _mint_key(session_id: str) -> str:
    """The key a session goes by below the API server; unique per session, so
    a stale teardown for an earlier session under the same id cannot free this
    one's state. The '.' keeps it apart from any id a client may choose."""
    return f"{session_id}.{uuid.uuid4().hex[:12]}"


class SessionError(Exception):
    """A session request the server refuses, with the status to answer with.

    ``status_code`` / ``detail`` are HTTPException's names on purpose: a route
    that already reports ``getattr(exc, "status_code", 500)`` -- the OpenAI
    router does -- then answers a refused session correctly with no new case.
    """

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class SessionRecord:
    session_id: str
    # What the conductor and workers know this session by; see _mint_key.
    key: str
    timeout_s: float
    created_at: float
    last_activity: float
    # One in flight at most, for now; a set so the cap can be lifted later.
    active_request_ids: set[str] = field(default_factory=set)
    # Closing: its state is being freed and every request naming it is refused
    # until the conductor's ACK arrives.
    closing: bool = False
    # Why it is closing, reported to a client that races the teardown.
    closing_reason: str = ""
    # When it started closing, and whether the conductor has been asked to
    # free its state: the tombstone sweep chases both.
    closing_since: float | None = None
    teardown_asked: bool = False

    def deadline(self, ttl_mode: SessionTTLMode) -> float:
        start = (
            self.created_at if ttl_mode is SessionTTLMode.ABSOLUTE
            else self.last_activity
        )
        return start + self.timeout_s


@dataclass
class SessionRequest:
    """What ``/generate``'s session flags resolved to."""

    session_id: str | None = None
    # The internal key sent downstream in place of the client's id.
    key: str | None = None
    # Set when the server minted the id, so it is reported back to the client.
    created: bool = False
    end_session: bool = False
    # Continuing a session that already exists, rather than opening one.
    resumed: bool = False


class SessionRegistry:
    """Tracks live sessions and the tombstones of the ones being freed.

    Thread-safe: the HTTP handlers and the result-draining thread both reach
    it. ``teardown`` is the injected side effect that asks the conductor to
    free a session's state; the registry holds the tombstone until
    ``torn_down`` says it is gone.
    """

    def __init__(
        self,
        config: SessionsConfig | None,
        # called with a session's internal key, never its client id
        teardown: Callable[[str], None],
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self._teardown = teardown
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionRecord] = {}
        # internal key -> client id, for the conductor's teardown ACKs
        self._by_key: dict[str, str] = {}
        self._request_to_session: dict[str, str] = {}
        # How long a tombstone may stand before the sweep chases it; above the
        # conductor's own barrier TTL, which force-finalizes at 120s.
        self._tombstone_grace_s = float(
            os.environ.get("MSTAR_SESSION_TOMBSTONE_GRACE_S", "180")
        )

    @property
    def enabled(self) -> bool:
        return self.config is not None

    # ------------------------------------------------------------------
    # Intake
    # ------------------------------------------------------------------

    def resolve(
        self,
        *,
        start_session: bool,
        resume_session: bool,
        end_session: bool,
        session_id: str | None,
        session_timeout_s: float | None,
        request_id: str,
    ) -> SessionRequest:
        """Validate a request's session flags and bind it to its session.

        Raises :class:`SessionError` for anything the deployment refuses; the
        caller turns that into the HTTP response. On success the request is
        already registered as the session's in-flight one, so a concurrent
        resume is refused.
        """
        if not (start_session or resume_session or end_session or session_id):
            return SessionRequest()
        config = self.config
        if config is None:
            raise SessionError(
                400,
                "this deployment does not support sessions; drop the "
                "session_* fields",
            )
        if start_session and resume_session:
            raise SessionError(
                400, "start_session and resume_session are mutually exclusive"
            )
        if not (start_session or resume_session):
            raise SessionError(
                400,
                "a request naming a session must set start_session or "
                "resume_session; use DELETE /sessions/{id} to end one without "
                "a request",
            )
        if resume_session and session_id is None:
            raise SessionError(400, "resume_session requires a session_id")
        if (
            start_session and session_id is not None
            and not SESSION_ID_PATTERN.fullmatch(session_id)
        ):
            raise SessionError(
                400,
                f"session_id {session_id!r} must be 1-128 characters of "
                "letters, digits, '_' and '-'",
            )
        try:
            timeout_s = config.resolve_timeout_s(session_timeout_s)
        except ValueError as e:
            raise SessionError(400, str(e)) from e

        now = self._clock()
        evicted = None
        with self._lock:
            if start_session:
                resolved, evicted = self._start_locked(session_id, timeout_s, now)
                created = session_id is None
            else:
                resolved = self._resume_locked(session_id)
                created = False
            record = self._sessions[resolved]
            record.active_request_ids.add(request_id)
            record.last_activity = now
            self._request_to_session[request_id] = resolved
        if evicted is not None:
            logger.info(
                "Evicted idle session %s to make room for %s", evicted, resolved,
            )
            self._teardown(evicted)
        logger.info(
            "Request %s %s session %s", request_id,
            "started" if start_session else "resumed", resolved,
        )
        return SessionRequest(
            session_id=resolved, key=record.key, created=created,
            end_session=end_session, resumed=resume_session,
        )

    def _start_locked(
        self, session_id: str | None, timeout_s: float, now: float,
    ) -> tuple[str, str | None]:
        """Open a session, returning its id and the key of whatever was
        evicted for it."""
        config = self.config
        if session_id is not None:
            existing = self._sessions.get(session_id)
            if existing is not None:
                raise SessionError(
                    409,
                    f"session {session_id!r} already exists"
                    + (" and is being torn down" if existing.closing else ""),
                )
        evicted = None
        live = sum(1 for r in self._sessions.values() if not r.closing)
        if live >= config.max_concurrent_sessions:
            victim = (
                self._lru_idle_locked()
                if config.capacity_policy is SessionCapacityPolicy.EVICT
                else None
            )
            if victim is None:
                raise SessionError(
                    429,
                    f"this deployment holds {config.max_concurrent_sessions} "
                    "concurrent sessions at most; end one first",
                )
            self._mark_closing_locked(victim, "evicted to make room")
            victim.teardown_asked = True
            evicted = victim.key
        resolved = session_id or str(uuid.uuid4())
        key = _mint_key(resolved)
        self._sessions[resolved] = SessionRecord(
            session_id=resolved,
            key=key,
            timeout_s=timeout_s,
            created_at=now,
            last_activity=now,
        )
        self._by_key[key] = resolved
        return resolved, evicted

    def _lru_idle_locked(self) -> SessionRecord | None:
        """The least recently used session that may be evicted: idle, and not
        already closing. A session with a request in flight is writing its state
        right now, so it is never a candidate however old it is."""
        idle = [
            record for record in self._sessions.values()
            if not record.closing and not record.active_request_ids
        ]
        return min(idle, key=lambda r: r.last_activity, default=None)

    def _resume_locked(self, session_id: str) -> str:
        record = self._sessions.get(session_id)
        if record is None:
            raise SessionError(404, f"unknown session {session_id!r}")
        if record.closing:
            raise SessionError(
                409,
                f"session {session_id!r} is being torn down"
                + (f": {record.closing_reason}" if record.closing_reason else ""),
            )
        if record.active_request_ids:
            # One request at a time; see SessionsConfig.max_requests_in_flight.
            raise SessionError(
                409,
                f"session {session_id!r} already has a request in flight "
                f"({sorted(record.active_request_ids)}); wait for it to "
                "finish before resuming",
            )
        return session_id

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def finish_request(
        self, request_id: str, failed: bool = False, error: str = "",
    ) -> None:
        """Release a request from its session.

        A request that failed takes its session with it: v1 keeps no rollback,
        so the state it half-wrote is not something a later request should
        continue from.
        """
        with self._lock:
            session_id = self._request_to_session.pop(request_id, None)
            if session_id is None:
                return
            record = self._sessions.get(session_id)
            if record is None:
                return
            record.active_request_ids.discard(request_id)
            record.last_activity = self._clock()
            if not failed:
                return
            reason = error or f"request {request_id} failed"
            self._mark_closing_locked(record, reason)
            # A session already closing because its last request carried
            # end_session normally goes with that request's teardown — but if
            # the request never reached the conductor, nobody would ask.
            if record.teardown_asked:
                return
            record.teardown_asked = True
            key = record.key
        logger.warning("Tearing session %s down: %s", key, reason)
        self._teardown(key)

    def rebind_request(self, request_id: str, key: str) -> None:
        """Re-key a request bound by ``resolve`` under the submission key it
        was given, which is what every later message about it names."""
        with self._lock:
            session_id = self._request_to_session.pop(request_id, None)
            if session_id is None:
                return
            self._request_to_session[key] = session_id
            record = self._sessions.get(session_id)
            if record is not None:
                record.active_request_ids.discard(request_id)
                record.active_request_ids.add(key)

    def session_of(self, request_id: str) -> str | None:
        with self._lock:
            return self._request_to_session.get(request_id)

    def note_ending(self, session_id: str) -> None:
        """A request carrying ``end_session`` was accepted: hold the tombstone
        now, so nothing else can name the session while it winds down."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is not None:
                self._mark_closing_locked(
                    record, "ended by its last request"
                )

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def delete(self, session_id: str) -> None:
        """``DELETE /sessions/{id}``: free the session without a request."""
        if self.config is None:
            raise SessionError(
                400, "this deployment does not support sessions"
            )
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                raise SessionError(404, f"unknown session {session_id!r}")
            if record.active_request_ids:
                raise SessionError(
                    409,
                    f"session {session_id!r} has a request in flight "
                    f"({sorted(record.active_request_ids)}); wait for it to "
                    "finish or abort it first",
                )
            if record.closing:
                raise SessionError(
                    409,
                    f"session {session_id!r} is already being torn down"
                    + (f": {record.closing_reason}" if record.closing_reason else ""),
                )
            self._mark_closing_locked(record, "deleted by the client")
            record.teardown_asked = True
        self._teardown(record.key)

    def sweep(self) -> list[str]:
        """Close every session whose TTL has passed. Returns what was closed.

        A session with a request in flight is never collected: its state is
        being written to right now.

        Also chases a tombstone that has stood past ``_tombstone_grace_s`` —
        asking for the teardown nobody asked for, or releasing an id whose ACK
        is never coming — so a stuck session cannot hold its id forever.
        """
        if self.config is None:
            return []
        now = self._clock()
        ttl_mode = self.config.ttl_mode
        expired: list[SessionRecord] = []
        stalled: list[str] = []
        abandoned: list[str] = []
        with self._lock:
            for record in list(self._sessions.values()):
                if record.closing:
                    since = record.closing_since
                    if since is None or now - since <= self._tombstone_grace_s:
                        continue
                    if not record.teardown_asked:
                        record.teardown_asked = True
                        record.closing_since = now
                        stalled.append(record.key)
                    else:
                        # the conductor force-finalizes its own barrier well
                        # inside this window, so nothing is coming
                        abandoned.append(record.key)
                    continue
                if record.active_request_ids:
                    continue
                if record.deadline(ttl_mode) <= now:
                    self._mark_closing_locked(
                        record, f"idle past its {record.timeout_s:.0f}s timeout"
                    )
                    record.teardown_asked = True
                    expired.append(record)
        for record in expired:
            logger.info("Session %s expired; freeing its state", record.key)
            self._teardown(record.key)
        for key in stalled:
            logger.warning(
                "Session %s has been closing for %.0fs with no teardown under "
                "way; asking for one now", key, self._tombstone_grace_s,
            )
            self._teardown(key)
        for key in abandoned:
            logger.error(
                "Session %s was never confirmed torn down within %.0fs; "
                "releasing its id. A worker may still hold its state.",
                key, self._tombstone_grace_s,
            )
            self.torn_down(key)
        return [record.session_id for record in expired]

    def torn_down(self, key: str) -> None:
        """The conductor confirmed the state under ``key`` is gone; lift the
        tombstone. A key that is not a live record's (a late ACK for an earlier
        session under the same client id) changes nothing."""
        with self._lock:
            session_id = self._by_key.pop(key, None)
            if session_id is None:
                return
            record = self._sessions.pop(session_id)
            for request_id in record.active_request_ids:
                self._request_to_session.pop(request_id, None)
        logger.info("Session %s released", key)

    def _mark_closing_locked(self, record: SessionRecord, reason: str) -> None:
        if not record.closing:
            record.closing = True
            record.closing_reason = reason
            record.closing_since = self._clock()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def snapshot(self) -> list[dict]:
        """Every session, ids included. For tests and in-process callers
        only: an id is all it takes to resume or delete a session, so the HTTP
        surface reports ``counts`` and per-id ``describe`` instead."""
        with self._lock:
            return [self._describe_locked(r) for r in self._sessions.values()]

    def describe(self, session_id: str) -> dict | None:
        """One session, for a caller that already holds its id."""
        with self._lock:
            record = self._sessions.get(session_id)
            return None if record is None else self._describe_locked(record)

    def counts(self) -> dict:
        """How many sessions are live and closing, against the cap."""
        with self._lock:
            closing = sum(1 for r in self._sessions.values() if r.closing)
            return {
                "live": len(self._sessions) - closing,
                "closing": closing,
                "max_concurrent_sessions": (
                    None if self.config is None
                    else self.config.max_concurrent_sessions
                ),
            }

    def _describe_locked(self, r: SessionRecord) -> dict:
        now = self._clock()
        ttl_mode = (
            self.config.ttl_mode if self.config is not None
            else SessionTTLMode.IDLE
        )
        return {
            "session_id": r.session_id,
            "timeout_s": r.timeout_s,
            "age_s": round(now - r.created_at, 3),
            "idle_s": round(now - r.last_activity, 3),
            "expires_in_s": round(r.deadline(ttl_mode) - now, 3),
            "active_request_ids": sorted(r.active_request_ids),
            "closing": r.closing,
        }

    def fail_all(self, reason: str) -> None:
        """Forget every session, for a deployment that is going down."""
        with self._lock:
            self._sessions.clear()
            self._by_key.clear()
            self._request_to_session.clear()
        logger.info("Dropped all session state: %s", reason)

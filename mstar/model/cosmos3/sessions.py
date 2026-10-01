"""Per-session world state for windowed rollouts.

A ``session_id`` on a windowed request names a world the client may come back
to (``resume_session``). The DiT node keeps the rollout's last window under
it and the streaming decoder its decode context, each in a ``SessionStore``.
The store is bounded and LRU-ordered, and it knows which sessions have a
request in flight: those are never evicted and refuse a second request, so
two rollouts cannot write the same world at once. Idle sessions expire after
their timeout, so a client that crashes mid-rollout does not leave a state
behind forever.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class _Entry:
    value: Any                 # what the node keeps for the session; None until stored
    expires_at: float          # clock time an idle entry is dropped at
    live: str | None = None    # request id generating under this session


class SessionStore:
    """Bounded, LRU-ordered session state with in-flight pins and a TTL.

    A request that names a session ``begin``s it, which pins it for the
    request's duration (a second request on a live session is refused),
    reads the stored state with ``get`` (which promotes it, so every node's
    store ages the same way), and either ``put``s the new state when it is
    done or ``end``s the session. ``release`` unpins whatever a removed
    request held. Only idle sessions count against ``capacity`` and can be
    evicted (expired ones first, then the oldest); live ones never are, so
    the store holds at most ``capacity`` idle plus the in-flight sessions.
    """

    def __init__(
        self, capacity: int, default_ttl_s: float, max_ttl_s: float,
        clock=time.monotonic,
    ):
        self.capacity = max(1, int(capacity))
        self.default_ttl_s = float(default_ttl_s)
        self.max_ttl_s = float(max_ttl_s)
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def ttl(self, requested: float | None) -> float:
        """A request's session timeout: the default when unset, else clamped
        to the served maximum."""
        if requested is None:
            return self.default_ttl_s
        ttl = float(requested)
        if not ttl > 0:
            raise ValueError(f"session_timeout_s must be > 0, got {requested!r}")
        return min(ttl, self.max_ttl_s)

    def begin(self, session_id: str, request_id: str) -> None:
        """Pin ``session_id`` for ``request_id`` until it stores or ends the
        session (or is removed). Refused while another request holds it."""
        self._sweep()
        entry = self._entries.get(session_id)
        if entry is None:
            entry = self._entries[session_id] = _Entry(value=None, expires_at=math.inf)
        elif entry.live not in (None, request_id):
            raise ValueError(
                f"session {session_id!r} is in use by request {entry.live!r}; "
                "wait for it to finish or pick another session_id"
            )
        entry.live = request_id
        self._entries.move_to_end(session_id)

    def get(self, session_id: str) -> Any | None:
        """The stored state, promoting the session; None when the session is
        unknown, expired or has not stored anything yet."""
        self._sweep()
        entry = self._entries.get(session_id)
        if entry is None or entry.value is None:
            return None
        self._entries.move_to_end(session_id)
        return entry.value

    def put(self, session_id: str, value: Any, ttl_s: float | None = None) -> None:
        """Store the session's state and unpin it: the request under it is
        done with the session. Idle sessions over capacity go, oldest first."""
        entry = self._entries.get(session_id)
        if entry is None:
            entry = self._entries[session_id] = _Entry(value=None, expires_at=0.0)
        entry.value = value
        entry.expires_at = self._clock() + self.ttl(ttl_s)
        entry.live = None
        self._entries.move_to_end(session_id)
        self._evict()

    def end(self, session_id: str) -> None:
        """Forget the session, stored state and pin alike."""
        self._entries.pop(session_id, None)

    def release(self, request_id: str) -> None:
        """Unpin whatever ``request_id`` holds (the request was removed before
        it stored or ended its session). A session it never stored goes."""
        for session_id, entry in list(self._entries.items()):
            if entry.live == request_id:
                entry.live = None
                if entry.value is None:
                    del self._entries[session_id]
        self._evict()

    def live(self, session_id: str) -> str | None:
        """The request generating under the session, if any."""
        entry = self._entries.get(session_id)
        return None if entry is None else entry.live

    def ids(self) -> list[str]:
        """Sessions in LRU order, oldest first."""
        return list(self._entries)

    def __contains__(self, session_id: str) -> bool:
        entry = self._entries.get(session_id)
        return entry is not None and entry.value is not None

    def __len__(self) -> int:
        return len(self._entries)

    def _sweep(self) -> None:
        now = self._clock()
        for session_id, entry in list(self._entries.items()):
            if entry.live is None and entry.expires_at <= now:
                del self._entries[session_id]

    def _evict(self) -> None:
        idle = [sid for sid, entry in self._entries.items() if entry.live is None]
        while len(idle) > self.capacity:
            del self._entries[idle.pop(0)]

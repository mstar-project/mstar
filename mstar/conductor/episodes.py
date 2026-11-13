"""Conductor-owned episode records for request planning and replica placement.

The admission layer assigns a logical slot. This table records that assignment
but allocates no storage and sends no worker messages. The normal request
lifecycle still adds and removes each call from the replica router.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from mstar.conductor.routing import Placement
from mstar.model.base import EpisodeView


@dataclass(slots=True)
class EpisodeRecord:
    calls: int
    placement: Placement
    slot: int
    held: dict[str, int] = field(default_factory=dict)
    keys: dict[str, str] = field(default_factory=dict)
    in_flight: str | None = None
    last_end: float = 0.0
    cold: str | None = None


def _snapshot(record: EpisodeRecord | None, episode_id: str | None) -> EpisodeView:
    if record is None:
        return EpisodeView(episode_id, 0, MappingProxyType({}))
    return EpisodeView(episode_id, record.calls, MappingProxyType(record.held.copy()), record.cold)


class EpisodeRecords:
    """Pin a named episode's placement while its rows survive requests.

    Only the conductor may call these methods. A storage receives the assigned
    slot through NEW_REQUEST and binds the worker-local request handle to it.
    """

    def __init__(self) -> None:
        self._records: dict[str, EpisodeRecord] = {}

    def view(self, episode_id: str | None) -> EpisodeView:
        """Snapshot state before a call is admitted; idless calls are cold."""
        return _snapshot(self._records.get(episode_id) if episode_id is not None else None, episode_id)

    def placement_for(self, episode_id: str | None, pick: Callable[[], Placement]) -> Placement:
        """Re-use a named episode's replica without consulting the router again."""
        record = self._records.get(episode_id) if episode_id is not None else None
        if record is None:
            return pick()
        # Routing facts are predictions for a new pick, not facts about this
        # call. The regular router.add still counts its live load separately.
        return Placement(record.placement.replicas.copy(), {})

    def admit(
        self, episode_id: str | None, request_id: str, slot: int,
        placement: Placement,
    ) -> EpisodeView:
        """Record an admission decision made elsewhere, before worker dispatch.

        This is not a slot allocator. A repeated call must use the same slot
        and replica; a later admission stage will queue concurrent calls.
        """
        if type(slot) is not int or slot < 0:
            raise ValueError(f"episode slot must be a non-negative int, got {slot!r}")
        if not request_id:
            raise ValueError("request id must be non-empty")
        if episode_id is None:
            return _snapshot(None, None)
        if type(episode_id) is not str or not episode_id:
            raise ValueError(f"episode id must be a non-empty str or None, got {episode_id!r}")

        record = self._records.get(episode_id)
        before = _snapshot(record, episode_id)
        if record is None:
            record = EpisodeRecord(0, Placement(placement.replicas.copy(), {}), slot)
            self._records[episode_id] = record
        elif record.in_flight is not None:
            raise ValueError(f"episode {episode_id!r} already runs request {record.in_flight!r}")
        elif record.slot != slot or record.placement.replicas != placement.replicas:
            raise ValueError(f"episode {episode_id!r} must keep its slot and replica")
        record.in_flight = request_id
        record.calls += 1
        return before

    def finish(self, episode_id: str, request_id: str, held: Mapping[str, int]) -> None:
        """Commit the worker-reported resident counts after the call tears down."""
        record = self._records[episode_id]
        if record.in_flight != request_id:
            raise ValueError(f"episode {episode_id!r} does not run request {request_id!r}")
        if any(type(count) is not int or count < 0 for count in held.values()):
            raise ValueError("held counts must be non-negative ints")
        record.held = dict(held)
        record.in_flight = None
        record.last_end = time.monotonic()
        record.cold = None

    def reset(self, episode_id: str, reason: str = "reset") -> None:
        """Empty the planning view without changing the seed/budget call count."""
        record = self._records[episode_id]
        if record.in_flight is not None:
            raise ValueError(f"episode {episode_id!r} is in flight")
        record.held.clear()
        record.cold = reason

    def end(self, episode_id: str) -> int:
        """Forget an idle episode after worker storage has released its rows."""
        record = self._records[episode_id]
        if record.in_flight is not None:
            raise ValueError(f"episode {episode_id!r} is in flight")
        del self._records[episode_id]
        return record.slot

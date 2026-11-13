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


class EpisodeCapacityError(Exception):
    """All candidate rows are occupied by calls that cannot be evicted yet."""


class EpisodeCallLimitError(ValueError):
    """A connection used its configured number of admitted calls."""


@dataclass(slots=True)
class EpisodeRecord:
    calls: int
    placement: Placement | None
    slot: int | None
    held: dict[str, int] = field(default_factory=dict)
    keys: dict[str, str] = field(default_factory=dict)
    in_flight: str | None = None
    last_end: float = 0.0
    cold: str | None = None
    workers: set[str] = field(default_factory=set)
    ending: bool = False


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
        self._closed: dict[str, float] = {}

    def view(
        self, episode_id: str | None, keys: Mapping[str, str] | None = None,
        reset: bool = False,
    ) -> EpisodeView:
        """Snapshot state before a call; changed keys read as empty."""
        record = self._records.get(episode_id) if episode_id is not None else None
        view = _snapshot(record, episode_id)
        if record is None or (not reset and not keys):
            return view
        held = {} if reset else dict(view.held)
        if not reset:
            for name, key in keys.items():
                if record.keys.get(name) != key:
                    held.pop(name, None)
        return EpisodeView(episode_id, view.calls, MappingProxyType(held), "reset" if reset else view.cold)

    def clear_arrays(self, episode_id: str | None, keys: Mapping[str, str] | None) -> list[str]:
        """Arrays whose content key changed since the previous call."""
        record = self._records.get(episode_id) if episode_id is not None else None
        if record is None or keys is None:
            return []
        return sorted(name for name, key in keys.items() if record.keys.get(name) != key)

    def can_admit(self, episode_id: str | None, max_calls: int | None = None) -> None:
        if episode_id is None:
            return
        if type(episode_id) is not str or not episode_id:
            raise ValueError(f"episode id must be a non-empty str or None, got {episode_id!r}")
        if episode_id in self._closed:
            raise ValueError(f"episode {episode_id!r} has ended")
        record = self._records.get(episode_id)
        if record is not None and record.ending:
            raise ValueError(f"episode {episode_id!r} is ending")
        if record is not None and record.in_flight is not None:
            raise ValueError(f"episode {episode_id!r} already runs request {record.in_flight!r}")
        if max_calls is not None and record is not None and record.calls >= max_calls:
            raise EpisodeCallLimitError(f"episode reached max_calls={max_calls}")

    def busy(self, episode_id: str | None) -> bool:
        record = self._records.get(episode_id) if episode_id is not None else None
        return record is not None and record.in_flight is not None

    def slot_for(self, episode_id: str | None) -> int | None:
        record = self._records.get(episode_id) if episode_id is not None else None
        return record.slot if record is not None else None

    def mark_ending(self, episode_id: str) -> bool:
        """Mark a close while a call drains; return whether it is in flight."""
        record = self._records.get(episode_id)
        if record is None:
            self._closed[episode_id] = time.monotonic()
            return False
        record.ending = True
        return record.in_flight is not None

    def has(self, episode_id: str) -> bool:
        return episode_id in self._records

    def ending(self, episode_id: str) -> bool:
        record = self._records.get(episode_id)
        return record is not None and record.ending

    def closed_or_ending(self, episode_id: str) -> bool:
        return episode_id in self._closed or self.ending(episode_id)

    def placement_for(self, episode_id: str | None, pick: Callable[[], Placement]) -> Placement:
        """Re-use a named episode's replica without consulting the router again."""
        record = self._records.get(episode_id) if episode_id is not None else None
        if record is None or record.slot is None:
            return pick()
        # Routing facts are predictions for a new pick, not facts about this
        # call. The regular router.add still counts its live load separately.
        return Placement(record.placement.replicas.copy(), {})

    def admit(
        self, episode_id: str | None, request_id: str, slot: int,
        placement: Placement, *, keys: Mapping[str, str] | None = None,
        reset: bool = False, max_calls: int | None = None,
        workers: set[str] | None = None,
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

        self.can_admit(episode_id, max_calls)
        record = self._records.get(episode_id)
        before = self.view(episode_id, keys, reset)
        if record is None:
            record = EpisodeRecord(0, Placement(placement.replicas.copy(), {}), slot)
            self._records[episode_id] = record
        elif record.slot is None:
            record.placement = Placement(placement.replicas.copy(), {})
            record.slot = slot
        elif record.slot != slot or record.placement.replicas != placement.replicas:
            raise ValueError(f"episode {episode_id!r} must keep its slot and replica")
        if reset:
            record.held.clear()
            record.keys.clear()
            record.cold = "reset"
        else:
            for name in self.clear_arrays(episode_id, keys):
                record.held.pop(name, None)
        if keys:
            record.keys.update(keys)
        if workers:
            record.workers.update(workers)
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
        record.held.update(held)
        record.in_flight = None
        record.last_end = time.monotonic()
        record.cold = None

    def reset(self, episode_id: str, reason: str = "reset") -> None:
        """Empty the planning view without changing the seed/budget call count."""
        record = self._records[episode_id]
        if record.in_flight is not None:
            raise ValueError(f"episode {episode_id!r} is in flight")
        record.held.clear()
        record.keys.clear()
        record.cold = reason

    def end(self, episode_id: str) -> tuple[int | None, set[str]]:
        """Forget an idle episode after worker storage has released its rows."""
        record = self._records[episode_id]
        if record.in_flight is not None:
            raise ValueError(f"episode {episode_id!r} is in flight")
        del self._records[episode_id]
        self._closed[episode_id] = time.monotonic()
        return record.slot, record.workers

    def evict(self, episode_id: str, reason: str = "evicted") -> tuple[int, set[str]]:
        """Release rows while retaining the call count and cold reason."""
        record = self._records[episode_id]
        if record.in_flight is not None or record.slot is None:
            raise ValueError(f"episode {episode_id!r} has no idle slot to evict")
        slot, workers = record.slot, record.workers
        record.slot = None
        record.placement = None
        record.workers = set()
        record.held.clear()
        record.keys.clear()
        record.cold = reason
        return slot, workers

    def fail(self, episode_id: str, request_id: str) -> tuple[int, set[str]]:
        """Discard a call that may have changed rows before it failed."""
        record = self._records[episode_id]
        if record.in_flight != request_id:
            raise ValueError(f"episode {episode_id!r} does not run request {request_id!r}")
        del self._records[episode_id]
        self._closed[episode_id] = time.monotonic()
        return record.slot, record.workers

    def idle(self) -> list[tuple[str, EpisodeRecord]]:
        return sorted(
            ((name, record) for name, record in self._records.items()
             if record.in_flight is None and record.slot is not None),
            key=lambda item: item[1].last_end,
        )

    def state_counts(self) -> dict[str, int]:
        counts = {"running": 0, "idle": 0, "cold": 0}
        for record in self._records.values():
            state = "running" if record.in_flight is not None else (
                "cold" if record.slot is None else "idle"
            )
            counts[state] += 1
        return counts

    def forget_expired_cold(self, now: float, ttl: float) -> None:
        """Bound records after their slot and frontend connection are gone."""
        for name, record in list(self._records.items()):
            if record.slot is None and record.in_flight is None and now - record.last_end >= 2 * ttl:
                del self._records[name]
        for name, ended_at in list(self._closed.items()):
            if now - ended_at >= 2 * ttl:
                del self._closed[name]


class EpisodeSlots:
    """One logical row index across the workers of a pinned placement."""

    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("episodes.max_episodes must be a positive int")
        self.capacity = capacity
        self._owners: dict[tuple[str, int], str] = {}

    def reserve(self, owner: str, workers: set[str]) -> int | None:
        if not workers:
            raise ValueError("an episode needs at least one worker")
        for slot in range(self.capacity):
            if all((worker, slot) not in self._owners for worker in workers):
                for worker in workers:
                    self._owners[worker, slot] = owner
                return slot
        return None

    def has_free_on_any(self, workers: list[str]) -> bool:
        return any(
            (worker, slot) not in self._owners
            for worker in workers for slot in range(self.capacity)
        )

    def used_by_worker(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for worker, _ in self._owners:
            counts[worker] = counts.get(worker, 0) + 1
        return counts

    def release(self, owner: str, workers: set[str], slot: int) -> None:
        if any(self._owners.get((worker, slot)) != owner for worker in workers):
            raise ValueError(f"episode slot {slot} is not held by {owner!r} on all workers")
        for worker in workers:
            del self._owners[worker, slot]

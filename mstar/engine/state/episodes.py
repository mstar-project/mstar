"""Bind a request handle to the episode slot the conductor assigned.

This table owns no tensors and chooses no slots. A storage uses the returned
slot while a request runs, then keeps or frees its rows according to the
binding's lifetime. The conductor remains the authority for placement and
admission; all ranks receive the same slot assignment.
"""

from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True)
class EpisodeBinding:
    slot: int
    cold: bool
    temporary: bool


class EpisodeRowBindings:
    """One logical owner per slot, with at most one active request per owner."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._by_request: dict[int, EpisodeBinding] = {}
        self._owner_by_slot: dict[int, str | None] = {}
        self._active_by_slot: dict[int, int] = {}
        self._slot_by_episode: dict[str, int] = {}

    def bind(self, request_handle: int, slot: int, episode_id: str | None) -> EpisodeBinding:
        """Attach the request to the assigned slot; ``None`` borrows it until unbind.

        ``cold`` means the slot has no prior rows for this episode. It tells
        storage whether to clear a reused physical allocation before the first
        step. Explicit reset is a separate lifecycle action.
        """
        if type(request_handle) is not int or request_handle < 0:
            raise ValueError(f"request handle must be a non-negative int, got {request_handle!r}")
        if type(slot) is not int or slot < 0:
            raise ValueError(f"episode slot must be a non-negative int, got {slot!r}")
        if episode_id is not None and (type(episode_id) is not str or not episode_id):
            raise ValueError(f"episode id must be a non-empty str or None, got {episode_id!r}")

        with self._lock:
            if request_handle in self._by_request:
                raise ValueError(f"request handle {request_handle} is already bound")
            if slot in self._active_by_slot:
                raise ValueError(f"episode slot {slot} already runs request {self._active_by_slot[slot]}")

            known = slot in self._owner_by_slot
            if episode_id is None:
                if known:
                    raise ValueError(f"temporary request cannot borrow occupied episode slot {slot}")
            else:
                other_slot = self._slot_by_episode.get(episode_id)
                if other_slot is not None and other_slot != slot:
                    raise ValueError(f"episode {episode_id!r} owns slot {other_slot}, not {slot}")
                if known and self._owner_by_slot[slot] != episode_id:
                    raise ValueError(f"episode slot {slot} belongs to {self._owner_by_slot[slot]!r}")

            if not known:
                self._owner_by_slot[slot] = episode_id
                if episode_id is not None:
                    self._slot_by_episode[episode_id] = slot
            binding = EpisodeBinding(slot, cold=not known, temporary=episode_id is None)
            self._by_request[request_handle] = binding
            self._active_by_slot[slot] = request_handle
            return binding

    def owner_of(self, request_handle: int) -> int:
        """Return the slot to bind into a resource before its request starts."""
        with self._lock:
            return self._by_request[request_handle].slot

    def slot_busy(self, slot: int) -> bool:
        """Whether a request still owns the slot on this worker."""
        with self._lock:
            return slot in self._active_by_slot

    def unbind(self, request_handle: int) -> EpisodeBinding:
        """Detach a request, releasing its slot only if it was temporary."""
        with self._lock:
            binding = self._by_request.pop(request_handle)
            del self._active_by_slot[binding.slot]
            if binding.temporary:
                del self._owner_by_slot[binding.slot]
            return binding

    def end(self, episode_id: str) -> int:
        """Release an idle named episode; callers free its storage rows first."""
        with self._lock:
            slot = self._slot_by_episode[episode_id]
            if slot in self._active_by_slot:
                raise ValueError(f"episode {episode_id!r} still runs request {self._active_by_slot[slot]}")
            del self._slot_by_episode[episode_id]
            del self._owner_by_slot[slot]
            return slot

    def idle_slot(self, episode_id: str) -> int | None:
        """Return an idle owner's slot, or None when this replica never held it."""
        with self._lock:
            slot = self._slot_by_episode.get(episode_id)
            if slot is not None and slot in self._active_by_slot:
                raise ValueError(f"episode {episode_id!r} still runs request {self._active_by_slot[slot]}")
            return slot

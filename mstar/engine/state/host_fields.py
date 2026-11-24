"""Host fields: small values a Value keeps per request, per label and per layer, read on the host.

A step declares its writes, and they apply when it commits. A pre-plan's declaration can run a second time for one step,
and a step that never runs commits nothing, so applying them any earlier would count steps that never happened.
"""

from collections.abc import Collection, Mapping
from typing import NamedTuple, get_args, get_origin

from mstar.engine.state.contract import Array, ArrayStep, FrozenMap


def is_host_value(value: object, kind: type) -> bool:
    """Whether ``value`` is exactly of the declared ``kind``, so a ``bool`` never passes as an ``int``."""
    if get_origin(kind) is not tuple:
        return type(value) is kind
    if type(value) is not tuple:
        return False
    args = get_args(kind)
    if len(args) == 2 and args[1] is Ellipsis:
        return all(is_host_value(item, args[0]) for item in value)
    return len(value) == len(args) and all(map(is_host_value, value, args))


def _zero(kind: type) -> object:
    return () if get_origin(kind) is tuple else kind()


# The stored key. Lookups pass a plain tuple in this field order, which hashes and compares equal and skips building
# the NamedTuple on every decode step, so this has to stay a tuple.
class EntryKey(NamedTuple):
    array: str
    label: str
    layer: int


class HostFields:
    """Every entry's host fields on one engine, by request handle, then array, label and layer.

    Handles are recycled, so a request's fields go on the `remove_request` that releases its handle. A row is nested
    under its handle because removal runs on the worker's main thread and commits on its GPU thread: popping one row
    never resizes a mapping the other thread is iterating.
    """

    def __init__(self):
        self._arrays: dict[str, Array] = {}
        self._episode_arrays: dict[str, Array] = {}
        self._fields: dict[int, dict[EntryKey, FrozenMap]] = {}
        self._episode_fields: dict[int, dict[EntryKey, FrozenMap]] = {}
        self._episode_written: dict[int, set[str]] = {}
        self._temp_written: dict[int, set[str]] = {}
        self._episode_slot_by_request: dict[int, int] = {}

    @property
    def episode_arrays(self) -> Mapping[str, Array]:
        return self._episode_arrays

    @property
    def has_episode_arrays(self) -> bool:
        return bool(self._episode_arrays)

    def declare(self, array: Array) -> None:
        self._arrays[array.name] = array
        if array.lifetime == "episode":
            self._episode_arrays[array.name] = array

    @staticmethod
    def _zero_entries(array: Array) -> dict[EntryKey, FrozenMap]:
        return {
            EntryKey(array.name, label, layer): FrozenMap(
                {name: _zero(kind) for name, kind in array.entry.host.items()},
            )
            for label in array.labels for layer in array.layers
        }

    def open(self, request: int) -> None:
        """Fill ``request``'s entries with their zero values, once its handle is minted."""
        if self._arrays:
            entries: dict[EntryKey, FrozenMap] = {}
            for array in self._arrays.values():
                entries.update(self._zero_entries(array))
            self._fields[request] = entries

    def bind_episode(
        self, request: int, slot: int, episode_id: str | None, *, cold: bool, reset_arrays: Collection[str] = (),
    ) -> None:
        """Bind ``request``'s episode-lifetime host arrays to ``slot`` for a named episode."""
        if episode_id is None or not self._episode_arrays:
            return
        if cold:
            if slot in self._episode_fields:
                raise ValueError(f"episode slot {slot} already has host fields")
            entries: dict[EntryKey, FrozenMap] = {}
            for array in self._episode_arrays.values():
                entries.update(self._zero_entries(array))
            self._episode_fields[slot] = entries
            self._episode_written[slot] = set()
        elif slot not in self._episode_fields:
            raise ValueError(f"episode slot {slot} has no host fields to resume")
        for name in reset_arrays:
            array = self._episode_arrays.get(name)
            if array is not None:
                self._episode_fields[slot].update(self._zero_entries(array))
                self._episode_written.get(slot, set()).discard(name)
        req_entries = self._fields.get(request)
        if req_entries is not None:
            for key in [k for k in req_entries if k[0] in self._episode_arrays]:
                del req_entries[key]
        self._episode_slot_by_request[request] = slot

    def end_episode(self, slot: int) -> None:
        """Release an idle named episode's host fields."""
        if slot in self._episode_slot_by_request.values():
            raise ValueError(f"episode slot {slot} still has an active request")
        self._episode_fields.pop(slot, None)
        self._episode_written.pop(slot, None)

    def remove_request(self, request: int) -> None:
        self._episode_slot_by_request.pop(request, None)
        self._temp_written.pop(request, None)
        self._fields.pop(request, None)

    def is_open(self, request: int) -> bool:
        return request in self._fields or request in self._episode_slot_by_request

    def _row_entries(self, request: int, array: str) -> dict[EntryKey, FrozenMap] | None:
        if array in self._episode_arrays:
            slot = self._episode_slot_by_request.get(request)
            if slot is not None:
                return self._episode_fields.get(slot)
        return self._fields.get(request)

    def fields(self, request: int, array: str, label: str, layer: int) -> Mapping[str, object] | None:
        """One entry's fields, or None for an entry that doesn't exist."""
        entries = self._row_entries(request, array)
        return None if entries is None else entries.get((array, label, layer))

    def held(self, request: int, array: str) -> int:
        if array not in self._episode_arrays:
            return 0
        slot = self._episode_slot_by_request.get(request)
        if slot is not None:
            return 1 if array in self._episode_written.get(slot, ()) else 0
        return 1 if array in self._temp_written.get(request, ()) else 0

    def _mark_written(self, request: int, array: str) -> None:
        if array not in self._episode_arrays:
            return
        slot = self._episode_slot_by_request.get(request)
        if slot is not None:
            self._episode_written.setdefault(slot, set()).add(array)
        else:
            self._temp_written.setdefault(request, set()).add(array)

    def commit(self, array: Array, declared: ArrayStep, rows: Collection[int]) -> None:
        """Apply what a kept step declares of ``array``: a fork at its start copies before the step's writes, and a fork
        at its end after them."""
        self._fork(array, declared, "start", rows)
        for (row, label, layer), written in declared.host.items():
            entries, key = self._row_entries(row, array.name), (array.name, label, layer)
            entries[key] = FrozenMap({**entries[key], **written})
            if written:
                self._mark_written(row, array.name)
        self._fork(array, declared, "end", rows)

    def _fork(self, array: Array, declared: ArrayStep, at: str, rows: Collection[int]) -> None:
        for fork in declared.forks:
            if fork.at == at:
                for row in rows if fork.rows is None else fork.rows:
                    entries = self._row_entries(row, array.name)
                    for layer in array.layers:
                        src, dst = (array.name, fork.src, layer), (array.name, fork.dst, layer)
                        entries[dst] = entries[src]

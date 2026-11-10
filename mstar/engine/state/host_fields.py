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
        self._fields: dict[int, dict[EntryKey, FrozenMap]] = {}

    def declare(self, array: Array) -> None:
        self._arrays[array.name] = array

    def open(self, request: int) -> None:
        """Fill ``request``'s entries with their zero values, once its handle is minted."""
        if self._arrays:
            self._fields[request] = {
                EntryKey(array.name, label, layer): FrozenMap(
                    {name: _zero(kind) for name, kind in array.entry.host.items()},
                )
                for array in self._arrays.values() for label in array.labels for layer in array.layers
            }

    def remove_request(self, request: int) -> None:
        self._fields.pop(request, None)

    def is_open(self, request: int) -> bool:
        return request in self._fields

    def fields(self, request: int, array: str, label: str, layer: int) -> Mapping[str, object] | None:
        """One entry's fields, or None for an entry that doesn't exist."""
        return self._fields.get(request, {}).get(EntryKey(array, label, layer))

    def commit(self, array: Array, declared: ArrayStep, rows: Collection[int]) -> None:
        """Apply what a kept step declares of ``array``: a fork at its start copies before the step's writes, and a fork
        at its end after them."""
        self._fork(array, declared, "start", rows)
        for (row, label, layer), written in declared.host.items():
            entries, key = self._fields[row], EntryKey(array.name, label, layer)
            entries[key] = FrozenMap({**entries[key], **written})
        self._fork(array, declared, "end", rows)

    def _fork(self, array: Array, declared: ArrayStep, at: str, rows: Collection[int]) -> None:
        for fork in declared.forks:
            if fork.at == at:
                for row in rows if fork.rows is None else fork.rows:
                    entries = self._fields[row]
                    for layer in array.layers:
                        src, dst = EntryKey(array.name, fork.src, layer), EntryKey(array.name, fork.dst, layer)
                        entries[dst] = entries[src]

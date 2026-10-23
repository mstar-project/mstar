"""A speculative step's per-row objects, built as columns.

The common speculation is a loop's next iteration whose inputs all come back
on the device (a decode step's sampled token). Its rows need no tensors and no
per-row decisions, so everything the model and the engine read per row is
built here with C-level maps over the runtime's columns, rather than by a
Python loop per row.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableMapping
from itertools import repeat
from operator import attrgetter
from types import MappingProxyType

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.submodule_base import InputMetadata

_LOOP_COUNTS = attrgetter("dynamic_loop_iter_counts")
_NO_STREAM_CHUNKS = MappingProxyType({})
# loop_counter_snapshots builds InputMetadata positionally, from C
assert InputMetadata._fields == ("stream_chunks", "dynamic_loop_iter_counts")


class LoopbackRowInputs(MutableMapping):
    """``per_request_input_tensors`` for rows whose inputs come from the step
    in flight and stay on the device: each row maps every signal in
    ``signals`` to an empty list, built the first time the row is read.

    A step the engine prepares through ``uniform_row_inputs`` reads none of
    them. Rows can also be given concrete inputs (``rows[rid] = inputs``),
    which ``set_signals`` leaves alone.
    """

    __slots__ = ("_rows", "_built", "_signals")

    def __init__(self, request_ids: Iterable[int], signals: Iterable[str] = ()):
        # rid -> its concrete inputs, or None while it takes the template
        self._rows: dict[int, NameToTensorList | None] = dict.fromkeys(request_ids)
        # the template rows read so far
        self._built: dict[int, NameToTensorList] = {}
        self._signals = tuple(signals)

    def set_signals(self, signals: Iterable[str]) -> None:
        """Every template row now carries ``signals``, each with no tensors,
        including a row already read."""
        self._signals = tuple(signals)
        for inputs in self._built.values():
            for name in self._signals:
                inputs.setdefault(name, [])

    def __getitem__(self, rid: int) -> NameToTensorList:
        inputs = self._rows[rid]
        if inputs is not None:
            return inputs
        inputs = self._built.get(rid)
        if inputs is None:
            inputs = self._built[rid] = {name: [] for name in self._signals}
        return inputs

    def __setitem__(self, rid: int, inputs: NameToTensorList) -> None:
        self._rows[rid] = inputs
        self._built.pop(rid, None)

    def __delitem__(self, rid: int) -> None:
        del self._rows[rid]
        self._built.pop(rid, None)

    def __contains__(self, rid) -> bool:
        return rid in self._rows

    def __iter__(self) -> Iterator[int]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def materialize(self) -> dict[int, NameToTensorList]:
        """A plain dict of every row, for a path that edits rows one by one."""
        return {rid: self[rid] for rid in self._rows}


def loop_counter_snapshots(
    infos: list[CurrentForwardPassInfo], loop_name: str | None, n_next_iter: int,
) -> list[InputMetadata]:
    """Per row, ``InputMetadata`` carrying its own copy of the loop counters.

    The first ``n_next_iter`` rows run one iteration past the shared
    counters in ``loop_name``; the rest run at them. A copy, not a view: the
    shared counters move when the in-flight step is routed, which can happen
    while this step is being prepared.
    """
    counts = list(map(dict.copy, map(_LOOP_COUNTS, infos)))
    if loop_name is not None:
        for c in counts[:n_next_iter]:
            c[loop_name] = c.get(loop_name, 0) + 1
    return list(map(
        tuple.__new__, repeat(InputMetadata), zip(repeat(_NO_STREAM_CHUNKS), counts),
    ))

"""A node's persistent rows: one ``RowSpace`` per opted-in walk, and each
resource's ``RowState`` in it.

The engine drives the step lifecycle; a resource only says what its rows hold
(``RowState``) and writes its per-step changes through ``ctx.rows``.
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from mstar.engine.rows.space import RowSpace, StepRows

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


class RowState:
    """A resource's share of one (node, walk)'s persistent rows, made by
    ``Resource.row_state``, which registers its buffers on the space.

    While a request holds a row, the row is its state for this resource: the
    resource answers host queries from the mirror (``StepRows.mirror``) and
    leaves its own per-request copy alone. A request can be removed while
    holding a row (``remove_request`` runs then); anything the row's
    check-in would write to (a store row) stays reserved until ``released``
    names its slot. The base class is the state of a resource that keeps
    nothing per row.
    """

    def check_out(self, rows: StepRows) -> None:
        """Write every joining row (``rows.diff.join_rids`` into
        ``join_slots``) from the request's state, on every buffer registered."""

    def check_in(self, rows: StepRows) -> None:
        """Read every leaving row back (``rows.diff.leave_rids`` from
        ``leave_slots``, skipping ``rows.diff.dropped``): device state with
        ``copy_out``, host state from the committed mirror in an
        ``on_commit`` hook, so a step that never commits changes nothing."""

    def released(self, slots: np.ndarray) -> None:
        """Rows of removed requests left ``slots``: free what was kept for them."""


class _Walk(NamedTuple):
    space: RowSpace
    states: tuple[RowState, ...]


class NodeRows:
    """The persistent walks of one node."""

    def __init__(self, node_name: str, walks: Mapping[str, _Walk]):
        self.node_name = node_name
        self._walks = dict(walks)
        self._spaces = [w.space for w in self._walks.values()]

    @classmethod
    def build(
        cls, node_name: str, walks: Collection[str],
        max_rows: Callable[[str], int | None],
        resources: Mapping[str, Resource], device: torch.device,
    ) -> NodeRows:
        """Each walk's space, sized to the most rows a step of it may hold,
        with every resource of the node taking part. Raises if a walk has no
        batch cap or a resource can't keep its state in rows."""
        built = {}
        for walk in sorted(walks):
            rows = max_rows(walk)
            if rows is None:
                raise ValueError(
                    f"{node_name}: walk {walk!r} keeps persistent rows but has no "
                    "batch cap (a capture config or max_batch_size) to size them"
                )
            space = RowSpace(rows, device)
            states = []
            for key, resource in resources.items():
                state = resource.row_state(space, node_name, walk)
                if state is None:
                    raise ValueError(
                        f"{node_name}: walk {walk!r} keeps persistent rows, but "
                        f"resource {key!r} ({type(resource).__name__}) can't keep "
                        "its per-row state in them"
                    )
                states.append(state)
            built[walk] = _Walk(space, tuple(states))
        return cls(node_name, built)

    @property
    def walks(self) -> tuple[str, ...]:
        return tuple(self._walks)

    def space(self, walk: str) -> RowSpace | None:
        w = self._walks.get(walk)
        return None if w is None else w.space

    def persistent(self, walk: str) -> bool:
        return walk in self._walks

    def holds_any(self, rids: Sequence[int]) -> bool:
        return any(
            space.holds(rid) for space in self._spaces if space.batch.n_active
            for rid in rids
        )

    def begin(self, walk: str, rids: Sequence[int]) -> StepRows:
        """Propose a step of ``walk`` over ``rids``, with every resource's
        joining and leaving rows written. Commits nothing."""
        w = self._walks[walk]
        rows = w.space.propose(rids)
        diff = rows.diff
        if diff.leave_rids:
            for state in w.states:
                state.check_in(rows)
        if diff.join_rids:
            for state in w.states:
                state.check_out(rows)
        return rows

    def commit(self, walk: str, rows: StepRows) -> None:
        """Commit a step of ``walk`` that is launching; ``rows.apply`` puts it
        on the device, on the step's stream."""
        released = rows.commit()
        if len(released):
            for state in self._walks[walk].states:
                state.released(released)

    def check_in(self, rids: Sequence[int], skip_walk: str | None = None) -> None:
        """Check ``rids`` in from every walk but ``skip_walk``, on the current
        stream, for a step that reads their state elsewhere."""
        for walk, w in self._walks.items():
            if walk == skip_walk or not w.space.batch.n_active:
                continue
            held = [rid for rid in rids if w.space.holds(rid)]
            if not held:
                continue
            rows = w.space.without(held)
            for state in w.states:
                state.check_in(rows)
            rows.stage()
            self.commit(walk, rows)
            rows.apply()

    def forget(self, rid: int) -> None:
        """``rid`` was removed: its rows leave at their walk's next step."""
        for space in self._spaces:
            space.forget(rid)

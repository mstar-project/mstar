"""A node's persistent rows: one ``RowSpace`` per opted-in walk, and each
resource's ``RowState`` in it.

The engine drives the step lifecycle; a resource only says what its rows hold
(``RowState``) and writes its per-step changes through ``ctx.rows``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from mstar.engine.rows.space import RowSpace, StepRows

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource
    from mstar.engine.resources.step import SlotLease


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

    def before_forward(self, lease: SlotLease) -> None:
        """Run before the forward of every graph captured for the walk, the
        one ``lease`` names, so the graph builds from the rows what its
        forward reads (see ``RowFills``)."""


class RowFills(RowState):
    """A ``RowState`` that builds per-step device inputs from the rows inside
    the step's graph: the capture's plan arms a fill (``arm``) aimed at the
    buffers that graph reads, and the capture records it ahead of the
    forward, so a replay rebuilds them from the rows as they are then. A step
    that isn't captured plans as it would without rows.

    A spec is planned several times before its capture (once per warm-up
    forward) and other specs are planned in between, so fills are kept per
    lease and per ``key``: a re-plan replaces its fill, and a capture records
    only its own lease's."""

    def __init__(self):
        self._fills: dict[SlotLease, dict[object, Callable[[], None]]] = {}

    def arm(self, lease: SlotLease, key: object, fill: Callable[[], None]) -> None:
        self._fills.setdefault(lease, {})[key] = fill

    def before_forward(self, lease: SlotLease) -> None:
        for fill in self._fills.pop(lease, {}).values():
            fill()


class RowWalk(NamedTuple):
    """How a walk keeps persistent rows (``NodeSubmodule.persistent_row_walks``).

    ``labels`` names, by KV cache resource key, the stream labels each step
    of the walk extends by its one token a row. The rows are allocated before
    any step runs, so a resource keeping per-label state in them (positions,
    KV lengths and pages) learns its labels here.
    """
    labels: Mapping[str, tuple[str, ...]] = MappingProxyType({})


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
        cls, node_name: str, walks: Mapping[str, RowWalk],
        max_rows: Callable[[str], int | None],
        resources: Mapping[str, Resource], device: torch.device,
        max_live: int | None = None,
    ) -> NodeRows:
        """Each walk's space, a step of it holding at most ``max_rows(walk)``
        rows, with a resident slot for each of ``max_live`` live requests
        (at least a step's rows; without a limit, just those, so rows that
        rotate out of the steps are checked in), and every resource of the
        node taking part. Raises if a walk has no batch cap or a resource
        can't keep its state in rows."""
        built = {}
        for walk in sorted(walks):
            rows = max_rows(walk)
            if rows is None:
                raise ValueError(
                    f"{node_name}: walk {walk!r} keeps persistent rows but has no "
                    "batch cap (a capture config or max_batch_size) to size them"
                )
            space = RowSpace(rows, device, labels=walks[walk].labels, capacity=max(rows, max_live or 0))
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
            # before capture records the rows' addresses
            space.seal()
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

    def holds_any(self, rids: Sequence[int], skip_walk: str | None = None) -> bool:
        """Whether a walk but ``skip_walk`` holds a row of any of ``rids``."""
        return any(
            w.space.holds(rid) for walk, w in self._walks.items()
            if walk != skip_walk and w.space.batch.n_held
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

    def commit(self, walk: str, rows: StepRows, forward: bool = True) -> None:
        """Commit a step of ``walk`` that is launching, running the walk's
        forward unless ``forward`` is False (a check-in); ``rows.apply`` puts
        it on the device, on the step's stream."""
        released = rows.commit(forward)
        if len(released):
            for state in self._walks[walk].states:
                state.released(released)

    def check_in(self, rids: Sequence[int], skip_walk: str | None = None) -> None:
        """Check ``rids`` in from every walk but ``skip_walk``, on the current
        stream, for a step that reads their state elsewhere."""
        for walk, w in self._walks.items():
            if walk == skip_walk or not w.space.batch.n_held:
                continue
            held = [rid for rid in rids if w.space.holds(rid)]
            if not held:
                continue
            rows = w.space.without(held)
            for state in w.states:
                state.check_in(rows)
            rows.stage()
            self.commit(walk, rows, forward=False)
            rows.apply()

    def before_forward(self, walk: str, lease: SlotLease) -> None:
        """Before the forward of the graph captured for ``walk`` on ``lease``:
        every state's ``before_forward``, which builds from the step's
        gathered rows."""
        for state in self._walks[walk].states:
            state.before_forward(lease)

    def gather(self, walk: str, n: int) -> None:
        """A step of ``walk`` over ``n`` rows, once its rows are applied: see ``RowSpace.gather``."""
        self._walks[walk].space.gather(n)

    def after_forward(self, walk: str) -> None:
        """After a forward of ``walk``: see ``RowSpace.after_forward``."""
        self._walks[walk].space.after_forward()

    def forget(self, rid: int) -> None:
        """``rid`` was removed: its rows leave at their walk's next step."""
        for space in self._spaces:
            space.forget(rid)

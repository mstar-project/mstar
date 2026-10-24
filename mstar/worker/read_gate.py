"""Rank 0's record of what each follower of its lockstep instance has landed.

Every rank of a TP/SP instance reads its own copy of a request's inputs, and
rank 0 schedules from its own copy alone. A follower whose read fails after
rank 0 broadcast a step that needs it can never run that step, and the
instance's collectives hang. So rank 0 runs a lockstep row only once every
follower has the row's inputs:

- Rank 0 marks its external inputs by uuid: those that land as reads (a walk
  input, a persisted signal, another worker's output, after any shard
  consolidation) and those a node outside the instance stores on rank 0 (an
  encoder colocated with rank 0). The instance's own outputs, such as a decode
  step's token, are produced on every rank and read by none, so they are never
  marked and never waited on.
- The first time a row consumes an external edge, rank 0 numbers it per
  request and edge name: the k-th such edge rank 0 consumes is the k-th edge of
  that name each follower receives, one per delivery whatever tensors each rank
  holds for it (its own shard, a consolidated copy, a loop's accumulated
  outputs). Followers count the external edges that land on them and report
  the counts; the row waits until every follower's count reaches its number.
  An input delivered again is a new delivery and gets a new number.
- A streamed input is consumed in chunks rebuilt per rank, so it is matched by
  position instead: the row's chunk ends at item ``n`` of the stream, and every
  follower has to hold its first ``n`` items in order.

A follower whose read fails never counts it, so the row stays out of every
step; the failure reaches the conductor, whose drain fails that request alone.
A request whose followers host a node with an edge into the lockstep node is
not gated: that node feeds the follower without a read, which the follower has
no way to count. Other nodes beside a follower do not matter.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence

from mstar.graph.base import GraphSection
from mstar.graph.runtime.base import ColumnarEdgeSpecs


def lockstep_feeders(section: GraphSection, lockstep: frozenset[str]) -> set[str]:
    """The nodes outside ``lockstep`` that can hand an edge to it: a node's
    own outputs, or a loop's terminal and accumulated outputs, which then
    count for every node in that loop."""
    edges = list(getattr(section, "outputs", None) or ())
    edges += getattr(section, "accumulated_outputs", None) or ()
    found = set()
    if any(edge.next_node in lockstep for edge in edges):
        found |= {name for name in section.get_nodes() if name not in lockstep}
    children = getattr(section, "sections", None)
    if children is None and isinstance(getattr(section, "section", None), GraphSection):
        children = [section.section]
    for child in children or ():
        found |= lockstep_feeders(child, lockstep)
    return found


def signals_by_rid(edges: ColumnarEdgeSpecs) -> dict[int, list[str]]:
    """The signal names each rid's edges carry, in edge order."""
    out: dict[int, list[str]] = {}
    for rid, idx in zip(edges.rids, edges.signal_name_idxs, strict=True):
        out.setdefault(rid, []).append(edges.signal_names[idx])
    return out


def edges_by_rid(edges: ColumnarEdgeSpecs) -> dict[int, list[tuple[str, tuple[int, ...]]]]:
    """Each rid's edges as (signal name, the uuids the edge carries)."""
    out: dict[int, list[tuple[str, tuple[int, ...]]]] = {}
    pos = 0
    for rid, idx, n in zip(
        edges.rids, edges.signal_name_idxs, edges.tensors_per_edge, strict=True,
    ):
        out.setdefault(rid, []).append(
            (edges.signal_names[idx], tuple(edges.uuids[pos:pos + n])),
        )
        pos += n
    return out


class FollowerReadGate:
    """Keyed by rank 0's request handle; ``forget`` on removal."""

    def __init__(self) -> None:
        # rid -> rank 0's uuids of external inputs not yet consumed by a row
        self._external: dict[int, set[int]] = defaultdict(set)
        # rid -> an external edge's uuids -> (edge name, its number)
        self._numbered: dict[int, dict[tuple[int, ...], tuple[str, int]]] = defaultdict(dict)
        # rid -> edge name -> external edges of it numbered so far
        self._consumed: dict[int, dict[str, int]] = defaultdict(dict)
        # rid -> follower worker -> edge -> external edges of it it reported
        self._counts: dict[int, dict[str, dict[str, int]]] = defaultdict(dict)
        # rid -> follower worker -> stream edge -> items it holds in order
        self._items: dict[int, dict[str, dict[str, int]]] = defaultdict(dict)
        # rids with a numbered edge some follower was behind on when last checked
        self._unsettled: set[int] = set()

    def may_wait(self, rid: int) -> bool:
        """Cheap first check: False means no row of ``rid`` can wait on an
        external input (streams are checked by the caller)."""
        return bool(self._external.get(rid)) or rid in self._unsettled

    def external(self, rid: int, uuids: Iterable[int]) -> None:
        """``uuids`` hold an external input on rank 0 (landed as a read, or
        stored by a node outside the instance). Landing again, a uuid is a new
        delivery, and its edge is numbered afresh when next consumed."""
        uuids = set(uuids)
        if not uuids:
            return
        self._external[rid] |= uuids
        numbered = self._numbered.get(rid)
        if numbered:
            for key in [k for k in numbered if uuids.intersection(k)]:
                del numbered[key]

    def record(
        self, rid: int, follower: str,
        counts: Mapping[str, int], stream_items: Mapping[str, int],
    ) -> None:
        mine = self._counts[rid].setdefault(follower, {})
        for edge, n in counts.items():
            mine[edge] = max(mine.get(edge, 0), n)
        items = self._items[rid].setdefault(follower, {})
        for edge, n in stream_items.items():
            items[edge] = max(items.get(edge, 0), n)

    def _behind(self, rid: int, followers: Sequence[str], edge: str, n: int) -> bool:
        counts = self._counts.get(rid, {})
        return any(counts.get(f, {}).get(edge, 0) < n for f in followers)

    def missing(
        self, rid: int, followers: Sequence[str],
        row_edges: Callable[[], Iterable[tuple[str, tuple[int, ...]]]],
        stream_ends: Mapping[str, int],
    ) -> bool:
        """Whether some follower lacks what a row of ``rid`` needs: its
        external edges among ``row_edges()`` and stream chunks ending at
        ``stream_ends``. An external edge is numbered the first time a row
        consumes it, so the k-th one of a name rank 0 consumes matches the
        k-th one a follower counts. ``row_edges`` is only called while the
        request has an external input not yet numbered, or a follower is
        behind on one that is."""
        external = self._external.get(rid)
        consumed = self._consumed.get(rid, {})
        if any(self._behind(rid, followers, e, n) for e, n in consumed.items()):
            self._unsettled.add(rid)
        else:
            self._unsettled.discard(rid)
        if external or rid in self._unsettled:
            numbered = self._numbered[rid]
            for edge, uuids in row_edges():
                if not uuids:
                    continue
                key = numbered.get(uuids)
                if key is None:
                    if not external or external.isdisjoint(uuids):
                        continue  # the instance's own output: never waited on
                    n = self._consumed[rid].get(edge, 0) + 1
                    self._consumed[rid][edge] = n
                    key = numbered[uuids] = (edge, n)
                    external.difference_update(uuids)
                if self._behind(rid, followers, *key):
                    self._unsettled.add(rid)
                    return True
        if stream_ends:
            items = self._items.get(rid, {})
            for follower in followers:
                have = items.get(follower, {})
                if any(have.get(edge, 0) < end for edge, end in stream_ends.items()):
                    return True
        return False

    def forget(self, rid: int) -> None:
        self._external.pop(rid, None)
        self._numbered.pop(rid, None)
        self._consumed.pop(rid, None)
        self._counts.pop(rid, None)
        self._items.pop(rid, None)
        self._unsettled.discard(rid)

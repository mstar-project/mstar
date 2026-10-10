"""Host-side layout of one row of a bounded KV step.

A row's retained keys are at most five contiguous ranges, in stream order:
the sink's source part, the sink's written part, the window's source part and
the window's ring (two ranges when it wraps). Its writes are at most three:
the sink's unfilled part and the ring (again two when it wraps). Each is a
``(start, length)`` pair in the source or the slot, so a step's layout is a
small integer table, and the kernels need nothing else.
"""

from typing import NamedTuple

from mstar.engine.resources.kv.bounded.config import SinkWindow

# reads: (start, len) pairs in this order; READ_FROM_SOURCE marks which index
# the source rather than the slot
NUM_READS = 5
READ_FROM_SOURCE = (True, False, True, False, False)
# writes: (fresh offset of the range's first entry, slot start, len) triples
NUM_WRITES = 3
# slot, reads, writes
ROW_INTS = 1 + 2 * NUM_READS + 3 * NUM_WRITES


class RowLayout(NamedTuple):
    reads: tuple[tuple[int, int], ...]
    writes: tuple[tuple[int, int, int], ...]


def _ring_ranges(first: int, count: int, window: int) -> list[tuple[int, int]]:
    """``count`` ring positions from ``first`` (mod ``window``) as at most two ranges."""
    if count <= 0:
        return []
    start = first % window
    head = min(count, window - start)
    return [(start, head)] + ([(0, count - head)] if count > head else [])


def row_layout(
    source_len: int, written: int, span: int, policy: SinkWindow, sink_capacity: int,
    reverse: bool = False,
) -> RowLayout:
    """The reads and writes of a row whose stream holds ``source_len + written``
    entries and gains ``span`` this step. Slot offsets put the sink's written
    part at ``[0, sink_capacity)`` and the ring after it. With ``reverse`` a
    step's tokens join the stream last-first (see ``BoundedKVConfig``)."""
    sink, window, includes_step = policy
    n = source_len + written
    ring = sink_capacity

    reads: list[tuple[int, int]] = [(0, min(sink, source_len))]
    reads.append((0, max(0, min(sink, n) - source_len)))
    lo = max(sink, n - (window - span if includes_step else window))
    reads.append((lo, max(0, source_len - lo)))
    first = max(lo, source_len)
    wrapped = _ring_ranges(first - sink, n - first, window)
    reads += [(ring + s, c) for s, c in wrapped] + [(0, 0)] * (2 - len(wrapped))

    def fresh(i: int) -> int:
        """Which of this step's tokens is stream entry ``i``."""
        return n + span - 1 - i if reverse else i - n

    writes: list[tuple[int, int, int]] = []
    sink_end = min(sink, n + span)
    if sink_end > n:
        writes.append((fresh(n), n - source_len, sink_end - n))
    if window > 0:
        first = max(sink, n, n + span - window)
        for s, c in _ring_ranges(first - sink, n + span - first, window):
            writes.append((fresh(first), ring + s, c))
            first += c
    writes += [(0, 0, 0)] * (NUM_WRITES - len(writes))
    return RowLayout(tuple(reads), tuple(writes))


EMPTY_ROW = RowLayout(((0, 0),) * NUM_READS, ((0, 0, 0),) * NUM_WRITES)


def flatten_row(slot: int, layout: RowLayout) -> list[int]:
    out = [slot]
    for start, count in layout.reads:
        out += (start, count)
    for offset, start, count in layout.writes:
        out += (offset, start, count)
    return out

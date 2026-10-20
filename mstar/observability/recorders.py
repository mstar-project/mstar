"""Dependency-free accumulators for the families in :mod:`.catalog`.

A process other than the API server keeps one recorder per family and sends
``snapshot()`` on a message the API server already receives; the API server
exports the latest snapshot of each family. Snapshots are cumulative, never
deltas, so a lost or reordered message cannot corrupt a counter.

Not thread-safe: update and snapshot a recorder from one thread (an
``EdgeTransfers`` hands ``latest`` to another thread whole).
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import accumulate

from mstar.observability.catalog import (
    HISTOGRAM,
    TRANSFER_BYTES,
    TRANSFER_SECONDS,
    TRANSFERS,
    Family,
)

Labels = tuple[str, ...]


@dataclass
class MetricSnapshot:
    """One family's series, columnar so the wire codec encodes it typed."""
    name: str
    #: One row per series, in the family's label order.
    label_values: list[list[str]]
    #: Counter or gauge: one value per series.
    values: list[float] = field(default_factory=list)
    #: Histogram: per series, cumulative counts per bucket, ``+Inf`` last
    #: (so the last is the observation count), and the sum of observations.
    bucket_counts: list[list[int]] = field(default_factory=list)
    sums: list[float] = field(default_factory=list)


class Gauge:
    def __init__(self, family: Family):
        self.family = family
        self._values: dict[Labels, float] = {}

    def set(self, labels: Labels, value: float) -> None:
        self._values[labels] = value

    def snapshot(self) -> MetricSnapshot:
        return MetricSnapshot(
            self.family.name, [list(k) for k in self._values], list(self._values.values()),
        )


class Counter(Gauge):
    def inc(self, labels: Labels, amount: float = 1.0) -> None:
        self._values[labels] = self._values.get(labels, 0.0) + amount


class Histogram:
    """Fixed buckets from the family. ``observe`` bumps one bucket; the
    cumulative counts Prometheus wants are built at ``snapshot``."""

    def __init__(self, family: Family):
        self.family = family
        self._bounds = family.buckets
        self._counts: dict[Labels, list[int]] = {}
        self._sums: dict[Labels, float] = {}

    def observe(self, labels: Labels, value: float) -> None:
        counts = self._counts.get(labels)
        if counts is None:
            counts = self._counts[labels] = [0] * (len(self._bounds) + 1)
            self._sums[labels] = 0.0
        # le is inclusive: a value equal to a bound falls in that bucket.
        counts[bisect_left(self._bounds, value)] += 1
        self._sums[labels] += value

    def snapshot(self) -> MetricSnapshot:
        return MetricSnapshot(
            self.family.name,
            [list(k) for k in self._counts],
            bucket_counts=[list(accumulate(c)) for c in self._counts.values()],
            sums=list(self._sums.values()),
        )


class EdgeTransfers:
    """One tensor manager's transfer families (``TRANSFER_BYTES``,
    ``TRANSFERS``, ``TRANSFER_SECONDS``). ``transport`` is the manager's
    protocol.

    ``record`` only appends to five columns; ``snapshot`` folds them into one
    row per (src, dst) once per status interval, so an edge read creates no
    object and touches no per-edge state.

    ``latest`` is the last ``publish()``: the data worker publishes from its
    own thread and the API server reads it from another, swapping whole lists.
    """

    # row layout: bytes, tensors, seconds sum, then one count per bucket
    _BUCKETS0 = 3

    def __init__(self, transport: str):
        self.transport = transport
        self._bounds = TRANSFER_SECONDS.buckets
        self._width = self._BUCKETS0 + len(self._bounds) + 1
        self._rows: dict[tuple[str, str], list] = {}
        self._srcs: list[str] = []
        self._dsts: list[str] = []
        self._bytes: list[int] = []
        self._tensors: list[int] = []
        self._seconds: list[float] = []
        self.latest: list[MetricSnapshot] = self.snapshot()

    def record(self, src: str, dst: str, nbytes: int, tensors: int, seconds: float) -> None:
        self._srcs.append(src)
        self._dsts.append(dst)
        self._bytes.append(nbytes)
        self._tensors.append(tensors)
        self._seconds.append(seconds)

    def _fold(self) -> None:
        rows, bounds, b0 = self._rows, self._bounds, self._BUCKETS0
        for src, dst, nbytes, tensors, seconds in zip(
                self._srcs, self._dsts, self._bytes, self._tensors, self._seconds, strict=True):
            row = rows.get((src, dst))
            if row is None:
                row = rows[(src, dst)] = [0, 0, 0.0] + [0] * (self._width - b0)
            row[0] += nbytes
            row[1] += tensors
            row[2] += seconds
            # le is inclusive: a value equal to a bound falls in that bucket.
            row[b0 + bisect_left(bounds, seconds)] += 1
        for column in (self._srcs, self._dsts, self._bytes, self._tensors, self._seconds):
            column.clear()

    def snapshot(self) -> list[MetricSnapshot]:
        self._fold()

        def labels():
            return [[src, dst, self.transport] for src, dst in self._rows]

        rows = list(self._rows.values())
        b0 = self._BUCKETS0
        return [
            MetricSnapshot(TRANSFER_BYTES.name, labels(), [float(r[0]) for r in rows]),
            MetricSnapshot(TRANSFERS.name, labels(), [float(r[1]) for r in rows]),
            MetricSnapshot(
                TRANSFER_SECONDS.name, labels(),
                bucket_counts=[list(accumulate(r[b0:])) for r in rows],
                sums=[r[2] for r in rows],
            ),
        ]

    def publish(self) -> None:
        self.latest = self.snapshot()


def merge(family: Family, snapshots: Iterable[MetricSnapshot]) -> MetricSnapshot:
    """One snapshot of ``family`` summing the series of ``snapshots`` that
    share label values: several processes' views of one deployment (the
    replicas of a graph node on different workers, say)."""
    values: dict[Labels, float] = {}
    counts: dict[Labels, list[int]] = {}
    sums: dict[Labels, float] = {}
    for snap in snapshots:
        if family.kind != HISTOGRAM:
            for row, value in zip(snap.label_values, snap.values, strict=True):
                key = tuple(row)
                values[key] = values.get(key, 0.0) + value
            continue
        for row, row_counts, total in zip(
                snap.label_values, snap.bucket_counts, snap.sums, strict=True):
            key = tuple(row)
            have = counts.get(key)
            counts[key] = list(row_counts) if have is None else [
                a + b for a, b in zip(have, row_counts, strict=True)]
            sums[key] = sums.get(key, 0.0) + total
    if family.kind != HISTOGRAM:
        return MetricSnapshot(family.name, [list(k) for k in values], list(values.values()))
    return MetricSnapshot(
        family.name, [list(k) for k in counts],
        bucket_counts=list(counts.values()), sums=list(sums.values()),
    )

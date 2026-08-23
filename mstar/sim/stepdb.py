"""Measured per-step GPU cost table ("stepdb").

The simulator prices one engine step — one ``NodeBatch`` for one
``(node, graph_walk)`` — by looking it up here. Costs are *measured*, never
derived from a roofline or composed from individual kernel curves: what the
GPU actually does for a step includes CUDA-graph padding waste, the CFG label
multiplication, in-graph collectives, sampling, and whatever kernel the engine
really launched. Composing those from separate measurements is where
simulators go wrong; measuring the whole step sidesteps it.

Why a whole-step table is tractable here: mstar replays CUDA graphs keyed
``(graph_walk, requires_cfg, bs, num_tokens)`` and pads every batch up to a
captured bucket, so GPU time is a function of the *padded* shape. The key
domain is therefore small and discrete — a few hundred rows per node — rather
than a continuous space needing interpolation in every dimension.

Key design points:

* Rows are keyed by the padded shape actually executed, plus the KV context
  size, TP/SP degrees, and whether the step ran as a captured graph or eager.
  A bucket miss in the real engine is a different execution regime, so
  ``mode`` is part of the key rather than something to interpolate across.
* Only ``kv_len_total`` is continuous. Everything else is exact-match. A query
  between two measured KV points is linearly interpolated and flagged;
  outside the measured range it is clamped-extrapolated and flagged more
  loudly. Nothing is silently invented.
* Every lookup returns a :class:`StepCost` carrying ``coverage`` flags. A
  consumer that reports numbers without checking them is the failure mode this
  is designed to make visible, so the flags propagate through aggregation.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from bisect import bisect_left
from dataclasses import dataclass, field, replace
from enum import IntFlag
from typing import Iterable

SCHEMA_VERSION = 1


class Coverage(IntFlag):
    """Quality of a cost estimate. ORs upward through aggregation."""

    EXACT = 0
    #: KV length fell between measured points; linearly interpolated.
    INTERPOLATED = 1
    #: KV length fell outside the measured range; extrapolated from the edge.
    EXTRAPOLATED = 2
    #: No row for this key at all; the cost is a fallback guess.
    MISSING = 4
    #: Row exists but came from a different model revision than the caller's.
    STALE = 8

    def describe(self) -> str:
        if self == Coverage.EXACT:
            return "exact"
        return "|".join(f.name.lower() for f in Coverage if f and f & self)


@dataclass(frozen=True)
class StepKey:
    """Identity of one measured engine step.

    ``padded_bs``/``padded_num_tokens`` are the shape the GPU actually ran,
    after the CUDA-graph runner rounded the real batch up to a captured
    bucket. ``mode`` distinguishes captured-graph replay from the eager
    fallback, which is a genuinely different cost regime.
    """

    model: str
    node: str
    graph_walk: str
    padded_bs: int
    padded_num_tokens: int
    tp_size: int = 1
    sp_size: int = 1
    requires_cfg: bool = False
    mode: str = "graph"  # "graph" | "eager"

    def as_row(self) -> tuple:
        return (
            self.model, self.node, self.graph_walk,
            self.padded_bs, self.padded_num_tokens,
            self.tp_size, self.sp_size,
            int(self.requires_cfg), self.mode,
        )


@dataclass
class StepCost:
    """A step's modeled cost, in seconds."""

    gpu_s: float
    #: Engine-internal CPU phases, measured alongside the GPU time.
    prepare_s: float = 0.0
    plan_s: float = 0.0
    launch_s: float = 0.0
    sample_s: float = 0.0
    coverage: Coverage = Coverage.EXACT
    #: Free-form note explaining a non-exact coverage, for reports.
    note: str = ""

    @property
    def cpu_s(self) -> float:
        """Total engine-side CPU work for the step."""
        return self.prepare_s + self.plan_s + self.launch_s + self.sample_s

    def scaled(self, factor: float) -> "StepCost":
        return replace(
            self,
            gpu_s=self.gpu_s * factor,
            prepare_s=self.prepare_s * factor,
            plan_s=self.plan_s * factor,
            launch_s=self.launch_s * factor,
            sample_s=self.sample_s * factor,
        )


@dataclass
class StepSample:
    """One measurement to be written to the table."""

    key: StepKey
    kv_len_total: int
    gpu_s: float
    prepare_s: float = 0.0
    plan_s: float = 0.0
    launch_s: float = 0.0
    sample_s: float = 0.0
    #: Dispersion of the repeats behind ``gpu_s`` (seconds), for diagnostics.
    gpu_s_stdev: float = 0.0
    repeats: int = 1
    provenance: dict = field(default_factory=dict)


_CREATE = """
CREATE TABLE IF NOT EXISTS steps (
    model            TEXT NOT NULL,
    node             TEXT NOT NULL,
    graph_walk       TEXT NOT NULL,
    padded_bs        INTEGER NOT NULL,
    padded_num_tokens INTEGER NOT NULL,
    tp_size          INTEGER NOT NULL,
    sp_size          INTEGER NOT NULL,
    requires_cfg     INTEGER NOT NULL,
    mode             TEXT NOT NULL,
    kv_len_total     INTEGER NOT NULL,
    gpu_s            REAL NOT NULL,
    prepare_s        REAL NOT NULL DEFAULT 0,
    plan_s           REAL NOT NULL DEFAULT 0,
    launch_s         REAL NOT NULL DEFAULT 0,
    sample_s         REAL NOT NULL DEFAULT 0,
    gpu_s_stdev      REAL NOT NULL DEFAULT 0,
    repeats          INTEGER NOT NULL DEFAULT 1,
    gpu_name         TEXT,
    provenance       TEXT,
    created_at       REAL,
    PRIMARY KEY (model, node, graph_walk, padded_bs, padded_num_tokens,
                 tp_size, sp_size, requires_cfg, mode, kv_len_total)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT
);
"""


class StepDB:
    """SQLite-backed store of measured step costs.

    Re-measuring a key overwrites its metrics and provenance but never its
    identity columns, so a table can be topped up incrementally as new models
    or bucket shapes appear.
    """

    def __init__(self, path: str, gpu_name: str | None = None):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_CREATE)
        self.conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()
        self.gpu_name = gpu_name or _current_gpu_name()
        # (key, gpu_name) -> sorted [(kv_len, StepCost)], built lazily.
        self._cache: dict[tuple, list[tuple[int, StepCost]]] = {}

    # ── writing ──────────────────────────────────────────────────────────

    def add(self, sample: StepSample) -> None:
        self.add_many([sample])

    def add_many(self, samples: Iterable[StepSample]) -> None:
        rows = []
        now = time.time()
        for s in samples:
            rows.append(
                s.key.as_row()
                + (
                    s.kv_len_total, s.gpu_s, s.prepare_s, s.plan_s,
                    s.launch_s, s.sample_s, s.gpu_s_stdev, s.repeats,
                    self.gpu_name, json.dumps(s.provenance), now,
                )
            )
        self.conn.executemany(
            "INSERT OR REPLACE INTO steps VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        self._cache.clear()

    # ── reading ──────────────────────────────────────────────────────────

    def _curve(self, key: StepKey) -> list[tuple[int, StepCost]]:
        """The measured KV curve for ``key``, ascending by kv_len."""
        cache_key = key.as_row() + (self.gpu_name,)
        hit = self._cache.get(cache_key)
        if hit is not None:
            return hit
        cur = self.conn.execute(
            "SELECT kv_len_total, gpu_s, prepare_s, plan_s, launch_s, sample_s "
            "FROM steps WHERE model=? AND node=? AND graph_walk=? AND padded_bs=? "
            "AND padded_num_tokens=? AND tp_size=? AND sp_size=? AND requires_cfg=? "
            "AND mode=? AND gpu_name=? ORDER BY kv_len_total",
            key.as_row() + (self.gpu_name,),
        )
        curve = [
            (row[0], StepCost(gpu_s=row[1], prepare_s=row[2], plan_s=row[3],
                              launch_s=row[4], sample_s=row[5]))
            for row in cur
        ]
        self._cache[cache_key] = curve
        return curve

    def lookup(self, key: StepKey, kv_len_total: int = 0) -> StepCost:
        """Cost of one step, interpolating along KV length only.

        Never raises on a miss: returns a zero cost flagged ``MISSING`` so a
        simulation can run to completion and report exactly which steps it
        could not price, rather than dying halfway through a sweep.
        """
        curve = self._curve(key)
        if not curve:
            return StepCost(
                gpu_s=0.0,
                coverage=Coverage.MISSING,
                note=f"no rows for {key.node}/{key.graph_walk} "
                     f"bs={key.padded_bs} tok={key.padded_num_tokens} "
                     f"mode={key.mode} on {self.gpu_name}",
            )

        lens = [c[0] for c in curve]
        if len(curve) == 1:
            only_len, only_cost = curve[0]
            if only_len == kv_len_total:
                return only_cost
            return replace(
                only_cost,
                coverage=Coverage.EXTRAPOLATED,
                note=f"single kv point {only_len}, asked {kv_len_total}",
            )

        i = bisect_left(lens, kv_len_total)
        if i < len(lens) and lens[i] == kv_len_total:
            return curve[i][1]

        if i == 0:
            lo_len, lo = curve[0]
            hi_len, hi = curve[1]
            cov, note = Coverage.EXTRAPOLATED, f"kv {kv_len_total} below {lo_len}"
        elif i >= len(lens):
            lo_len, lo = curve[-2]
            hi_len, hi = curve[-1]
            cov, note = Coverage.EXTRAPOLATED, f"kv {kv_len_total} above {hi_len}"
        else:
            lo_len, lo = curve[i - 1]
            hi_len, hi = curve[i]
            cov, note = Coverage.INTERPOLATED, ""

        span = hi_len - lo_len
        w = 0.0 if span == 0 else (kv_len_total - lo_len) / span

        def lerp(a: float, b: float) -> float:
            # Clamp at zero: extrapolating a steep negative slope off the low
            # end could otherwise produce a negative duration.
            return max(0.0, a + (b - a) * w)

        return StepCost(
            gpu_s=lerp(lo.gpu_s, hi.gpu_s),
            prepare_s=lerp(lo.prepare_s, hi.prepare_s),
            plan_s=lerp(lo.plan_s, hi.plan_s),
            launch_s=lerp(lo.launch_s, hi.launch_s),
            sample_s=lerp(lo.sample_s, hi.sample_s),
            coverage=cov,
            note=note,
        )

    # ── introspection ────────────────────────────────────────────────────

    def keys(self, model: str | None = None) -> list[StepKey]:
        sql = ("SELECT DISTINCT model, node, graph_walk, padded_bs, "
               "padded_num_tokens, tp_size, sp_size, requires_cfg, mode FROM steps")
        args: tuple = ()
        if model:
            sql += " WHERE model=?"
            args = (model,)
        return [
            StepKey(r[0], r[1], r[2], r[3], r[4], r[5], r[6], bool(r[7]), r[8])
            for r in self.conn.execute(sql + " ORDER BY node, graph_walk, padded_bs", args)
        ]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]

    def gpu_names(self) -> list[str]:
        """Device names that have rows, for callers that must pick one."""
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT gpu_name FROM steps ORDER BY gpu_name")]

    def models(self) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT model FROM steps ORDER BY model")]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "StepDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _current_gpu_name() -> str:
    """Device name used as a row key, or ``cpu`` when there is no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def pad_to_bucket(value: int, buckets: Iterable[int]) -> int | None:
    """Round ``value`` up to the smallest bucket that fits it.

    Returns None when no bucket is large enough — in the real engine that is
    a capture miss that falls back to eager, which the caller must model as a
    different ``mode`` rather than as a bigger bucket.
    """
    best: int | None = None
    for b in buckets:
        if b >= value and (best is None or b < best):
            best = b
    return best

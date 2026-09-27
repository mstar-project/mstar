"""Windowed (sliding-window) autoregressive generation support.

``WindowSchedule`` is pure window arithmetic over abstract sequence units
(latent frames for video models), and turns its context horizon into the KV
cache's ``RetentionPolicy`` for a stream (``WindowSchedule.retention``): the
immutable prefix protected, everything older than the horizon released as
each window commits. The model puts that policy on the ``KVStep`` of each
window commit and the pool applies it inside the commit, which is what makes
the release safe under the engine's step pre-planning. Models own their walk,
conditioning math and step declarations; this module owns the schedule and
the retention arithmetic.

Ported from #198 (merceod) onto the resource-pool engine.
"""
from dataclasses import dataclass

from mstar.engine.resources.kv.config import RetentionPolicy


@dataclass(frozen=True)
class WindowPlan:
    """One window's slice of a windowed generation.

    Unit indices are absolute within the full sequence. The leading
    ``cond_units`` of the window re-pin the tail of the previous window as
    clean conditioning (the chained-mode overlap; 0 when there is no
    overlap). ``commit_start:commit_end`` is the span this window newly
    generates — the span a kv-mode commit pass appends to the cache
    (overlap units were already committed by the previous window).
    """
    index: int
    start: int
    end: int
    cond_units: int

    @property
    def units(self) -> int:
        return self.end - self.start

    @property
    def commit_start(self) -> int:
        return self.start + self.cond_units

    @property
    def commit_end(self) -> int:
        return self.end


class WindowSchedule:
    """Window arithmetic for one request.

    ``total_units`` are generated in windows of ``window_units`` advancing by
    ``window_units - overlap_units``; the final window may be short, and every
    unit is generated exactly once (commit spans partition ``[0, total)``).
    ``context_units`` bounds the committed history retained in the cache
    after each commit; 0 means retain everything (no release).
    """

    def __init__(
        self,
        total_units: int,
        window_units: int,
        context_units: int = 0,
        overlap_units: int = 0,
    ):
        if total_units < 1:
            raise ValueError(f"total_units must be >= 1, got {total_units}")
        if window_units < 1:
            raise ValueError(f"window_units must be >= 1, got {window_units}")
        if not 0 <= overlap_units < window_units:
            raise ValueError(
                f"overlap_units must be in [0, window_units), got "
                f"{overlap_units} with window_units={window_units}"
            )
        if context_units < 0:
            raise ValueError(f"context_units must be >= 0, got {context_units}")
        self.total_units = total_units
        self.window_units = window_units
        self.context_units = context_units
        self.overlap_units = overlap_units
        self.stride = window_units - overlap_units
        if total_units <= window_units:
            self.num_windows = 1
        else:
            self.num_windows = 1 + -(-(total_units - window_units) // self.stride)

    def window(self, index: int) -> WindowPlan:
        if not 0 <= index < self.num_windows:
            raise IndexError(
                f"window {index} out of range [0, {self.num_windows})"
            )
        start = index * self.stride
        end = min(start + self.window_units, self.total_units)
        cond = self.overlap_units if index > 0 else 0
        return WindowPlan(index=index, start=start, end=end, cond_units=cond)

    def windows(self):
        return (self.window(k) for k in range(self.num_windows))

    def released_end(self, index: int) -> int:
        """Units released from the front of the committed stream once window
        ``index`` has committed: everything older than ``context_units``
        behind the commit frontier. 0 when context is unbounded."""
        if self.context_units == 0:
            return 0
        return max(0, self.window(index).commit_end - self.context_units)

    def retention(self, tokens_per_unit: int, prefix_tokens: int) -> RetentionPolicy | None:
        """The KV retention a window commit declares for its stream: the
        context horizon in cache tokens behind ``prefix_tokens`` of protected
        head (the text prefix, say). ``None`` when the schedule retains
        everything, since there is never a release. Releases are page-floored
        by the pool and the shortfall re-offered at the next commit, so the
        realized context tracks the nominal one within a page."""
        if tokens_per_unit < 1:
            raise ValueError(f"tokens_per_unit must be >= 1, got {tokens_per_unit}")
        if self.context_units == 0:
            return None
        return RetentionPolicy(
            context_budget=self.context_units * tokens_per_unit,
            protected_prefix=prefix_tokens,
        )

"""Windowed (sliding-window) autoregressive generation support.

``WindowSchedule`` is pure window arithmetic over abstract sequence units
(latent frames for video models). ``WindowedKVSession`` turns a schedule into
the KV cache's retention policy — the immutable prefix protected, everything
older than the context horizon released as each window commits — so a model
drives windowed generation without hand-rolling page/token bookkeeping.
Models own their walk, conditioning math, and step declarations; this module
owns the schedule and the retention arithmetic. The handle is the KV resource
(``KVManager.set_retention``); the pool applies the policy inside each commit,
which is what makes the release safe under the engine's step pre-planning.

Ported from #198 (merceod) onto the resource-pool engine.
"""
from dataclasses import dataclass

from mstar.engine.resources.kv.manager import RetentionPolicy


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


class WindowedKVSession:
    """KV retention for one (request, label) under a ``WindowSchedule``.

    ``handle`` provides ``set_retention(request_id, policy, label=...)`` (the
    ``KVManager`` surface). Units convert to cache tokens via
    ``tokens_per_unit``. Releases are page-floored by the pool and the
    shortfall re-offered at the next commit, so the realized context tracks
    the nominal one within a page.
    """

    def __init__(
        self,
        handle,
        request_id: str,
        label: str,
        schedule: WindowSchedule,
        tokens_per_unit: int,
    ):
        if tokens_per_unit < 1:
            raise ValueError(
                f"tokens_per_unit must be >= 1, got {tokens_per_unit}"
            )
        self._handle = handle
        self._request_id = request_id
        self._label = label
        self._schedule = schedule
        self._tokens_per_unit = tokens_per_unit
        self._bound = False

    @property
    def context_tokens(self) -> int | None:
        """Committed generation tokens the cache keeps behind the prefix;
        ``None`` when the schedule retains everything."""
        if self._schedule.context_units == 0:
            return None
        return self._schedule.context_units * self._tokens_per_unit

    def bind(self, prefix_tokens: int) -> RetentionPolicy | None:
        """Install the retention once the immutable stream head (e.g. the
        text prefix) has committed and before the first window commits. A
        schedule with unbounded context installs nothing (there is never a
        release, so nothing to protect). Returns the installed policy."""
        if self._bound:
            raise RuntimeError(
                f"retention already bound for request "
                f"{self._request_id!r} label {self._label!r}"
            )
        self._bound = True
        budget = self.context_tokens
        if budget is None:
            return None
        policy = RetentionPolicy(
            context_budget=budget, protected_prefix=prefix_tokens,
        )
        self._handle.set_retention(self._request_id, policy, label=self._label)
        return policy

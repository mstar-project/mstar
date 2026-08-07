"""Windowed (sliding-window) autoregressive generation support.

``WindowSchedule`` is pure window arithmetic over abstract sequence units
(latent frames for video models). ``WindowedKVSession`` turns schedule events
into KV-cache lifecycle calls — protect the immutable prefix once, release
aged-out context after each window commit — so a model drives windowed
generation without hand-rolling page/token bookkeeping. Models own their
walk, conditioning math, and attention plans; this module owns the schedule
and the lifecycle sequencing.
"""
from dataclasses import dataclass


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
    """KV lifecycle driver for one (request, label) under a ``WindowSchedule``.

    ``handle`` provides ``protect_prefix(request_id, num_tokens, label=...)``
    and ``release_oldest(request_id, num_tokens, label=...) -> int`` (the
    ``BatchedCacheManager`` passthrough surface). Units convert to cache
    tokens via ``tokens_per_unit``. Releases are page-floored by the
    allocator; the session re-offers the shortfall on the next call, so the
    realized context tracks the nominal one within a page.
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
        self._prefix_protected = False
        self._released_tokens = 0

    @property
    def released_tokens(self) -> int:
        return self._released_tokens

    def protect_prefix(self, prefix_tokens: int) -> None:
        """Protect the immutable stream head (e.g. the text prefix). Call
        once, after the prefix has committed and before any release."""
        if self._prefix_protected:
            raise RuntimeError(
                f"prefix already protected for request "
                f"{self._request_id!r} label {self._label!r}"
            )
        self._handle.protect_prefix(
            self._request_id, prefix_tokens, label=self._label
        )
        self._prefix_protected = True

    def after_commit(self, window_index: int) -> int:
        """Release context that aged out once ``window_index`` committed.
        Returns tokens actually freed (0 when nothing aged out yet, or the
        accumulated shortfall is still under a page)."""
        target = (
            self._schedule.released_end(window_index) * self._tokens_per_unit
        )
        ask = target - self._released_tokens
        if ask <= 0:
            return 0
        freed = self._handle.release_oldest(
            self._request_id, ask, label=self._label
        )
        self._released_tokens += freed
        return freed

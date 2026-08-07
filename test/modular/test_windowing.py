"""Tests for ``mstar.engine.windowing``: window arithmetic and the KV
lifecycle session.

The schedule invariant that everything downstream leans on: commit spans
partition ``[0, total_units)`` exactly — every unit is generated once, no
gaps, no double-commits — for any (total, window, overlap) combination,
including a short final window.
"""

from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")

import pytest

from mstar.engine.windowing import WindowedKVSession, WindowSchedule


class TestWindowSchedule:
    def test_single_window_when_total_fits(self):
        s = WindowSchedule(total_units=5, window_units=8)
        assert s.num_windows == 1
        w = s.window(0)
        assert (w.start, w.end, w.cond_units) == (0, 5, 0)
        assert (w.commit_start, w.commit_end) == (0, 5)

    def test_exact_tiling_no_overlap(self):
        s = WindowSchedule(total_units=48, window_units=8)
        assert s.num_windows == 6
        assert [w.start for w in s.windows()] == [0, 8, 16, 24, 32, 40]
        assert all(w.units == 8 for w in s.windows())

    def test_overlap_and_short_final_window(self):
        s = WindowSchedule(total_units=48, window_units=8, overlap_units=1)
        assert s.stride == 7
        assert s.num_windows == 7
        last = s.window(6)
        assert (last.start, last.end, last.cond_units) == (42, 48, 1)
        assert last.units == 6

    def test_commit_spans_partition_total(self):
        rng = random.Random(1)
        for _ in range(200):
            window = rng.randrange(1, 12)
            overlap = rng.randrange(0, window)
            total = rng.randrange(1, 80)
            s = WindowSchedule(total, window, overlap_units=overlap)
            covered = 0
            for w in s.windows():
                assert w.commit_start == covered, (total, window, overlap)
                assert w.commit_end > w.commit_start
                covered = w.commit_end
            assert covered == total, (total, window, overlap)

    def test_released_end_tracks_context_bound(self):
        s = WindowSchedule(48, 8, context_units=16)
        ends = [s.released_end(k) for k in range(s.num_windows)]
        assert ends == [0, 0, 8, 16, 24, 32]
        # Retained span after each commit never exceeds the context bound.
        for k, w in enumerate(s.windows()):
            assert w.commit_end - ends[k] <= 16

    def test_unbounded_context_never_releases(self):
        s = WindowSchedule(48, 8, context_units=0)
        assert all(s.released_end(k) == 0 for k in range(s.num_windows))

    def test_validation(self):
        with pytest.raises(ValueError):
            WindowSchedule(0, 8)
        with pytest.raises(ValueError):
            WindowSchedule(8, 0)
        with pytest.raises(ValueError):
            WindowSchedule(8, 4, overlap_units=4)
        with pytest.raises(ValueError):
            WindowSchedule(8, 4, context_units=-1)
        with pytest.raises(IndexError):
            WindowSchedule(8, 4).window(2)


class _StubHandle:
    """Cache-handle stub that page-floors releases like the real allocator."""

    def __init__(self, page_size: int):
        self.page_size = page_size
        self.protected = {}
        self.freed_total = 0

    def protect_prefix(self, request_id, num_tokens, label=None):
        self.protected[(request_id, label)] = num_tokens

    def release_oldest(self, request_id, num_tokens, label=None):
        freed = (num_tokens // self.page_size) * self.page_size
        self.freed_total += freed
        return freed


class TestWindowedKVSession:
    def test_release_targets_with_page_floor_shortfall(self):
        # 60 tokens/unit against 128-token pages: per-window release asks are
        # never page-aligned, so the session must re-offer the shortfall.
        s = WindowSchedule(48, 8, context_units=16)
        h = _StubHandle(page_size=128)
        sess = WindowedKVSession(h, "r", "main", s, tokens_per_unit=60)

        sess.protect_prefix(300)
        assert h.protected[("r", "main")] == 300

        for k in range(s.num_windows):
            sess.after_commit(k)
            target = s.released_end(k) * 60
            # Realized release tracks the nominal target within one page.
            assert 0 <= target - sess.released_tokens < 128

        assert sess.released_tokens == h.freed_total

    def test_no_release_before_context_fills(self):
        s = WindowSchedule(48, 8, context_units=16)
        h = _StubHandle(page_size=8)
        sess = WindowedKVSession(h, "r", "main", s, tokens_per_unit=8)
        assert sess.after_commit(0) == 0
        assert sess.after_commit(1) == 0
        assert sess.after_commit(2) > 0

    def test_protect_once(self):
        s = WindowSchedule(8, 8)
        sess = WindowedKVSession(_StubHandle(8), "r", "main", s, 8)
        sess.protect_prefix(16)
        with pytest.raises(RuntimeError, match="already protected"):
            sess.protect_prefix(16)

    def test_tokens_per_unit_validation(self):
        with pytest.raises(ValueError):
            WindowedKVSession(_StubHandle(8), "r", "main", WindowSchedule(8, 8), 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

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
    """Records the retention the session installs, like the pool would."""

    def __init__(self):
        self.policies = {}

    def set_retention(self, request_id, policy, label=None):
        self.policies[(request_id, label)] = policy


class TestWindowedKVSession:
    def test_bind_installs_the_schedule_budget(self):
        # 60 tokens/unit, 16 units of context behind a 300-token prefix.
        s = WindowSchedule(48, 8, context_units=16)
        h = _StubHandle()
        sess = WindowedKVSession(h, "r", "main", s, tokens_per_unit=60)
        assert sess.context_tokens == 16 * 60
        policy = sess.bind(300)
        assert h.policies[("r", "main")] is policy
        assert policy.context_budget == 960 and policy.protected_prefix == 300

    def test_unbounded_context_installs_nothing(self):
        s = WindowSchedule(48, 8)
        h = _StubHandle()
        sess = WindowedKVSession(h, "r", "main", s, tokens_per_unit=8)
        assert sess.context_tokens is None
        assert sess.bind(16) is None
        assert h.policies == {}

    def test_bind_once(self):
        s = WindowSchedule(8, 8, context_units=4)
        sess = WindowedKVSession(_StubHandle(), "r", "main", s, 8)
        sess.bind(16)
        with pytest.raises(RuntimeError, match="already bound"):
            sess.bind(16)

    def test_tokens_per_unit_validation(self):
        with pytest.raises(ValueError):
            WindowedKVSession(_StubHandle(), "r", "main", WindowSchedule(8, 8), 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

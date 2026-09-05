import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.utils.containers import RecentSet  # noqa: E402


def test_membership_and_len():
    s = RecentSet(3)
    assert len(s) == 0 and 1 not in s
    s.add(1)
    s.add(2)
    assert 1 in s and 2 in s and 3 not in s
    assert len(s) == 2


def test_evicts_oldest_first_once_full():
    s = RecentSet(3)
    for i in range(5):
        s.add(i)
    assert list(s) == [2, 3, 4]
    assert 0 not in s and 1 not in s
    assert len(s) == 3


def test_readding_is_a_noop_and_keeps_original_age():
    s = RecentSet(2)
    s.add("a")
    s.add("b")
    s.add("a")           # already present: no eviction, no age refresh
    assert list(s) == ["a", "b"]
    s.add("c")           # evicts "a" — it is still the oldest
    assert list(s) == ["b", "c"]
    assert "a" not in s


def test_maxlen_must_be_positive():
    with pytest.raises(ValueError):
        RecentSet(0)

"""ParallelList's iteration contract.

It is a NamedTuple whose __iter__ is overridden to yield (key, value) pairs.
That combination is a trap: ``k, v = pl`` reads like field unpacking but goes
through __iter__, and at exactly two entries it SUCCEEDS while binding two
pairs instead of the two lists. Pinning the behaviour here so the docstring
warning has something enforcing it.
"""
import sys

sys.path.insert(0, ".")

import pytest

from mstar.utils.containers import ParallelList


def test_iteration_yields_pairs():
    pl = ParallelList([1, 2, 3], ["a", "b", "c"])
    assert list(pl) == [(1, "a"), (2, "b"), (3, "c")]
    assert len(pl) == 3


def test_tuple_unpacking_does_not_give_the_two_lists():
    """The trap. At two entries this binds pairs and raises nothing."""
    pl = ParallelList([1, 2], ["a", "b"])
    first, second = pl
    assert first == (1, "a") and second == (2, "b"), \
        "unpacking goes through __iter__, not the NamedTuple fields"
    # The lists are reached by name.
    assert pl.keys == [1, 2] and pl.values == ["a", "b"]


def test_mismatched_lengths_raise_on_iteration():
    """Same length is the invariant of the type; a silent zip truncation here
    would drop requests."""
    with pytest.raises(ValueError, match="shorter than"):
        list(ParallelList([1, 2, 3], ["a"]))


def test_dict_of_a_parallel_list_needs_an_explicit_iter():
    """``dict`` probes for a ``.keys()`` METHOD to decide it was handed a
    mapping. ParallelList.keys is a list attribute, so the probe calls it and
    raises. The failure is loud but the cause is not obvious."""
    pl = ParallelList([1, 2], ["a", "b"])
    with pytest.raises(TypeError, match="not callable"):
        dict(pl)
    assert dict(iter(pl)) == {1: "a", 2: "b"}


def test_from_dict_round_trips():
    d = {1: "a", 2: "b"}
    pl = ParallelList.from_dict(d)
    assert dict(iter(pl)) == d
    # from_dict is a classmethod; without the decorator cls would bind to the
    # dict and the keys/values would come out swapped.
    assert isinstance(pl, ParallelList)

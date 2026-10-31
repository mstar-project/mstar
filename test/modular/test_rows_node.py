"""``NodeRows.commit``: a removed request's store row is released only after
the step's row update, which copies its row out into that store row, is on
the stream."""
import numpy as np

from mstar.engine.rows.node import NodeRows, RowState, _Walk


class _Rows:
    def __init__(self, calls, released):
        self.calls = calls
        self._released = released

    def commit(self):
        self.calls.append("commit")
        return self._released

    def apply(self):
        self.calls.append("apply")


class _State(RowState):
    def __init__(self, calls):
        self.calls = calls

    def released(self, slots):
        self.calls.append(("released", slots.tolist()))


def test_released_follows_the_row_update():
    calls = []
    node = NodeRows("n", {"decode": _Walk(space=None, states=(_State(calls),))})
    node.commit("decode", _Rows(calls, np.array([3], np.int64)))
    assert calls == ["commit", "apply", ("released", [3])]

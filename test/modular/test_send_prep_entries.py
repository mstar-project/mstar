"""The send stage reports stream consumption only for requests that have
stream buffers; the rest would carry an empty dict each."""
from types import SimpleNamespace

from mstar.worker.worker import Worker


class _Buf:
    def __init__(self, consumed):
        self._consumed = consumed


def _worker(infos):
    w = Worker.__new__(Worker)
    w.request_state = SimpleNamespace(per_request_info=infos)
    return w


def test_only_requests_with_stream_buffers_are_reported():
    infos = {
        1: SimpleNamespace(stream_buffers={}),
        2: SimpleNamespace(stream_buffers={"audio": _Buf(7), "text": _Buf(3)}),
        4: SimpleNamespace(stream_buffers={"audio": _Buf(0)}),
    }
    w = _worker(infos)
    out = w._stream_consumption_batch([1, 2, 3, 4])
    assert out == [(2, {"audio": 7, "text": 3}), (4, {"audio": 0})]
    # the per-request form agrees entry by entry
    for rid, consumed in out:
        assert w._stream_consumption(rid) == consumed
    assert w._stream_consumption(1) == {} and w._stream_consumption(3) == {}

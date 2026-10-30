"""The send stage reports stream consumption only for requests that have
stream buffers; the rest would carry an empty dict each."""
from types import SimpleNamespace

from mstar.worker.stream_buffers import StreamBufferManager


class _Buf:
    def __init__(self, consumed):
        self._consumed = consumed


def _streams(infos):
    return StreamBufferManager(SimpleNamespace(per_request_info=infos), None, [])


def test_only_requests_with_stream_buffers_are_reported():
    infos = {
        1: SimpleNamespace(stream_buffers={}),
        2: SimpleNamespace(stream_buffers={"audio": _Buf(7), "text": _Buf(3)}),
        4: SimpleNamespace(stream_buffers={"audio": _Buf(0)}),
    }
    streams = _streams(infos)
    out = streams.consumption_batch([1, 2, 3, 4])
    assert out == [(2, {"audio": 7, "text": 3}), (4, {"audio": 0})]
    # the per-request form agrees entry by entry
    for rid, consumed in out:
        assert streams.consumption(rid) == consumed
    assert streams.consumption(1) == {} and streams.consumption(3) == {}

"""What mstar-server releases once nobody will read a submission.

A client that leaves, before the first chunk or mid-stream, streamed or not,
aborts every submission its request made on the API server
(``abort_submission``), and the bridge keeps nothing of them. ``n > 1`` images
are submitted with seed + i and come back in submit order, and one image's
failure, or the client leaving, releases the others. The OpenAI routes mint
their own ids, so a ``request_id`` still in flight reaches them only as the API
server's 409, which is answered as the status.

Expectations carried over from the removed Python routes' tests
(``test_stream_release``, ``test_images_n_release``, ``test_speech_chunking``,
``test_request_id_reuse``): the real binary and bridge, a stub data plane."""

import asyncio
import base64
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, NamedTuple

import pytest

pytest.importorskip("mstar_rust")

from test_rust_frontend import BINARY, _bridged_stack, _Chunk, _RefusedError, _StubAPIServer

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

# Orpheus cuts an input over 250 characters into pieces of at most 200: six here,
# of which a stream submits the first two up front and one more once it plays.
SENTENCES = (
    "The train to the coast leaves at seven tomorrow morning, so please pack your bag tonight. "
    "She opened the window and let the cool evening air drift into the kitchen. "
    "Our meeting has been moved to Thursday afternoon because the room is being repainted! "
    "A gentle rain fell over the harbor while the fishing boats returned one by one. "
    "Remember to water the tomatoes twice a week during the hottest part of the summer? "
    "The museum's new exhibit traces the history of printing from wooden blocks to modern presses. "
)
LONG_TEXT = SENTENCES * 2


def _until(pred, timeout=5.0, what="the condition"):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.005)


class _HoldingStub(_StubAPIServer):
    """Each submission sends ``head(i)`` (``i`` its place in submit order), then
    holds until it is aborted."""

    def __init__(self, head: Callable[[int], list] = lambda i: []):
        super().__init__()
        self.head = head

    def rids(self):
        return [s["rid"] for s in self.submitted]

    async def iter_result_chunks(self, request_id):
        for chunk in self.head(self.rids().index(request_id)):
            yield chunk
        while request_id not in self.aborted:
            await asyncio.sleep(0.005)


def _post_bytes(path, body: bytes, ctype: str) -> bytes:
    head = f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\n\r\n"
    return head.encode() + body


def _leave(port, path, body: bytes, ctype="application/json", *, once=None, read: bytes | None = None):
    """POST ``body`` on a raw connection and close it unread once ``once()``
    holds, or once ``read`` has arrived in the reply."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as conn:
        conn.sendall(_post_bytes(path, body, ctype))
        if read is None:
            _until(once, what="the submissions")
            return
        got = b""
        while read not in got:
            data = conn.recv(65536)
            assert data, f"the reply ended before {read!r}: {got[-200:]!r}"
            got += data


def _json(body: dict) -> bytes:
    return json.dumps(body).encode()


def _form(**fields) -> bytes:
    return urllib.parse.urlencode(fields).encode()


FORM = "application/x-www-form-urlencoded"


class _Route(NamedTuple):
    model: str
    path: str
    body: bytes
    ctype: str
    # what each submission sends before it holds, mid-stream; and how that reads in the reply
    chunk: _Chunk | None
    marker: bytes | None
    # submissions in flight when the client leaves: before the first chunk, then mid-stream
    before: int
    mid: int | None


_AUDIO = _Chunk("audio", b"\x01\x00\x02\x00\x03\x00\x04\x00")

ROUTES = {
    "chat-sse": _Route(
        "qwen3_omni", "/v1/chat/completions",
        _json({"model": "qwen3_omni", "stream": True, "messages": [{"role": "user", "content": "hi"}]}),
        "application/json", _Chunk("text", b"Hello"), b"Hello", 1, 1),
    "speech-stream": _Route(
        "orpheus", "/v1/audio/speech",
        _json({"model": "orpheus", "input": "hi", "voice": "tara", "stream": True, "response_format": "pcm"}),
        "application/json", _AUDIO, _AUDIO.data, 1, 1),
    "speech-stream-chunked": _Route(
        "orpheus", "/v1/audio/speech",
        _json({"model": "orpheus", "input": LONG_TEXT, "voice": "tara", "stream": True, "response_format": "pcm"}),
        "application/json", _AUDIO, _AUDIO.data, 2, 3),
    "videos-stream": _Route(
        "cosmos3", "/v1/videos/generations",
        _json({"model": "cosmos3", "prompt": "a road", "stream_video": True}),
        "application/json", _Chunk("video", b"MP4"), base64.b64encode(b"MP4"), 1, 1),
    "generate-stream": _Route(
        "qwen3_omni", "/generate", _form(text="hi", output_modalities="text", streaming="true"),
        FORM, _Chunk("text", b"Hello"), base64.b64encode(b"Hello"), 1, 1),
    # non-streamed: nothing reaches the client until every submission is done
    "chat": _Route(
        "qwen3_omni", "/v1/chat/completions",
        _json({"model": "qwen3_omni", "messages": [{"role": "user", "content": "hi"}]}),
        "application/json", None, None, 1, None),
    "speech": _Route(
        "orpheus", "/v1/audio/speech", _json({"model": "orpheus", "input": "hi", "voice": "tara"}),
        "application/json", None, None, 1, None),
    # a whole body reads one piece while the next two generate
    "speech-chunked": _Route(
        "orpheus", "/v1/audio/speech", _json({"model": "orpheus", "input": LONG_TEXT, "voice": "tara"}),
        "application/json", None, None, 3, None),
    "videos": _Route(
        "cosmos3", "/v1/videos/generations", _json({"model": "cosmos3", "prompt": "a road"}),
        "application/json", None, None, 1, None),
    "generate": _Route(
        "qwen3_omni", "/generate", _form(text="hi", output_modalities="text", streaming="false"),
        FORM, None, None, 1, None),
    "images": _Route(
        "cosmos3", "/v1/images/generations", _json({"model": "cosmos3", "prompt": "a red circle"}),
        "application/json", None, None, 1, None),
}

CASES = [(route, "before") for route in ROUTES] + [
    (route, "mid") for route, r in ROUTES.items() if r.mid is not None]


@pytest.mark.parametrize("route,when", CASES, ids=[f"{r}-{w}" for r, w in CASES])
def test_a_client_that_leaves_releases_every_submission(route, when):
    """Closing the connection aborts each submission still in flight, and only
    those submitted: a chunked speech request releases the pieces it read ahead
    and never submits the rest."""
    r = ROUTES[route]
    mid = when == "mid"
    stub = _HoldingStub(lambda i: [r.chunk] if mid else [])
    expected = r.mid if mid else r.before
    with _bridged_stack(r.model, stub) as (port, _up, bridge):
        if mid:
            _leave(port, r.path, r.body, r.ctype, read=r.marker)
        else:
            _leave(port, r.path, r.body, r.ctype, once=lambda: len(stub.submitted) == expected)
        _until(lambda: len(stub.aborted) == expected, what="the aborts")
        _until(lambda: not bridge._keys, what="the bridge to forget the submissions")
        assert sorted(stub.aborted) == sorted(stub.rids())
        assert len(stub.submitted) == expected
    if expected > 1:
        assert [rid.rsplit("-", 1)[1] for rid in stub.rids()] == [str(i) for i in range(expected)]


def test_a_stream_read_to_its_end_is_not_aborted():
    """A finished submission is left alone when its connection closes."""
    stub = _StubAPIServer()
    with _bridged_stack("qwen3_omni", stub) as (port, _up, bridge):
        r = ROUTES["chat-sse"]
        req = urllib.request.Request(f"http://127.0.0.1:{port}{r.path}", data=r.body,
                                     headers={"Content-Type": r.ctype})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert b"[DONE]" in resp.read()
        _until(lambda: not bridge._keys, what="the bridge to forget the submission")
    assert len(stub.submitted) == 1 and stub.aborted == []


# ── n > 1 images ────────────────────────────────────────────────────────────

OK, FAIL, HOLD, REFUSE = "ok", "fail", "hold", "refuse"


class _ImagesStub(_HoldingStub):
    """The i-th image ``plan[i]``: its id as the image (OK), the backend's 500
    once every OK image is done (FAIL), held until aborted (HOLD), or refused at
    submit (REFUSE). With ``reverse``, each OK image waits for the next one, so
    they finish last to first."""

    def __init__(self, plan, reverse=False):
        super().__init__()
        self.plan, self.reverse = plan, reverse
        self.finished = set()

    def submit_request(self, **kwargs):
        rid = super().submit_request(**kwargs)
        if self.plan[len(self.submitted) - 1] == REFUSE:
            raise ValueError("injected submit failure")
        return rid

    def ids(self, *outcomes):
        return sorted(rid for rid, o in zip(self.rids(), self.plan, strict=False) if o in outcomes)

    async def _until(self, pred):
        while not pred():
            await asyncio.sleep(0.002)

    async def iter_result_chunks(self, request_id):
        i = self.rids().index(request_id)
        outcome = self.plan[i]
        if outcome == HOLD:
            await self._until(lambda: request_id in self.aborted)
            return
        if outcome == FAIL:
            await self._until(lambda: self.finished >= set(self.ids(OK)))
            yield _Chunk("error", b"injected image failure", {"status": 500})
            return
        if self.reverse and i + 1 < len(self.plan):
            await self._until(lambda: len(self.submitted) == len(self.plan) and self.rids()[i + 1] in self.finished)
        yield _Chunk("image", request_id.encode())
        self.finished.add(request_id)


def _images(port, n, seed=None, timeout=10):
    body = {"model": "cosmos3", "prompt": "a lighthouse", "n": n}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/images/generations", data=_json(body),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_images_are_seeded_in_turn_and_returned_in_submit_order():
    """Image i runs with seed + i, and the images come back in the order they
    were submitted, though here they finish last to first."""
    stub = _ImagesStub([OK] * 4, reverse=True)
    with _bridged_stack("cosmos3", stub) as (port, _up, _bridge):
        status, out = _images(port, 4, seed=7)
    assert status == 200, out
    assert [base64.b64decode(d["b64_json"]).decode() for d in out["data"]] == stub.rids()
    assert [s["mk"]["seed"] for s in stub.submitted] == [7, 8, 9, 10]
    assert stub.aborted == []


@pytest.mark.parametrize("plan,status,message", [
    # the 2nd image fails once the 1st and 3rd are done; the 4th still runs
    ([OK, FAIL, OK, HOLD], 500, "injected image failure"),
    # the 3rd is refused at submit, the others already run
    ([HOLD, HOLD, REFUSE, HOLD], 400, "injected submit failure"),
], ids=["collect", "submit"])
def test_a_failed_image_releases_its_siblings(plan, status, message):
    """The request fails at once with the image's error, and the images still
    running are aborted; the finished ones are left alone."""
    stub = _ImagesStub(plan)
    with _bridged_stack("cosmos3", stub) as (port, _up, bridge):
        got, out = _images(port, len(plan))
        _until(lambda: len(stub.aborted) == plan.count(HOLD), what="the aborts")
        _until(lambda: not bridge._keys, what="the bridge to forget the images")
    assert (got, out["error"]["message"]) == (status, message)
    assert sorted(stub.aborted) == stub.ids(HOLD)


def test_a_client_that_leaves_releases_every_image():
    stub = _ImagesStub([HOLD, OK, HOLD, HOLD])
    with _bridged_stack("cosmos3", stub) as (port, _up, bridge):
        body = _json({"model": "cosmos3", "prompt": "a lighthouse", "n": 4})
        _leave(port, "/v1/images/generations", body,
               once=lambda: len(stub.submitted) == 4 and stub.finished == set(stub.ids(OK)))
        _until(lambda: len(stub.aborted) == 3, what="the aborts")
        _until(lambda: not bridge._keys, what="the bridge to forget the images")
    assert sorted(stub.aborted) == stub.ids(HOLD)


# ── a request_id in flight ──────────────────────────────────────────────────

IN_USE = "request_id {!r} already has a request in flight; wait for it to finish or abort it first"


class _IdInFlightStub(_StubAPIServer):
    """The API server refusing every submission's id as one still in flight."""

    def submit_request(self, *, request_id=None, **kwargs):
        self.submitted.append(request_id)
        raise _RefusedError(409, IN_USE.format(request_id))


@pytest.mark.parametrize("route", ["chat-sse", "chat", "videos-stream", "videos"])
def test_an_openai_route_answers_an_id_in_flight_with_a_409(route):
    r = ROUTES[route]
    stub = _IdInFlightStub()
    with _bridged_stack(r.model, stub) as (port, _up, _bridge):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{r.path}", data=r.body,
                                     headers={"Content-Type": r.ctype})
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(req, timeout=10)
    (rid,) = stub.submitted
    assert refused.value.code == 409
    assert json.loads(refused.value.read())["error"]["message"] == IN_USE.format(rid)
    assert stub.aborted == []

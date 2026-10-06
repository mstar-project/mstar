"""Sentence chunking on ``/v1/audio/speech``: the splitter and the ordered sub-requests.

The router is mounted on a FastAPI app with a stubbed APIServer (as in
``test_openai_router.py``); every sub-request the handler submits is recorded
so ordering, ids, seeds and playback concatenation can be checked without an
engine.
"""

import asyncio
import sys
import tempfile
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")
pytest.importorskip("httpx")
np = pytest.importorskip("numpy")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mstar.api_server.openai import adapters  # noqa: E402
from mstar.api_server.openai.speech_chunking import split_sentences  # noqa: E402

LONG_TEXT = (
    "The train to the coast leaves at seven tomorrow morning, so please pack your bag tonight. "
    "She opened the window and let the cool evening air drift into the kitchen. "
    "Our meeting has been moved to Thursday afternoon because the room is being repainted! "
    "A gentle rain fell over the harbor while the fishing boats returned one by one. "
    "Remember to water the tomatoes twice a week during the hottest part of the summer? "
    "The museum's new exhibit traces the history of printing from wooden blocks to modern presses."
)


# ---------------------------------------------------------------------------
# splitter
# ---------------------------------------------------------------------------


def test_short_text_is_one_chunk():
    assert split_sentences("Hello there. How are you?", max_chars=400) == ["Hello there. How are you?"]
    assert split_sentences("   ", max_chars=400) == []


def test_sentences_are_grouped_up_to_max_chars_and_never_cut():
    chunks = split_sentences(LONG_TEXT, max_chars=200)
    assert len(chunks) == 3
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks) == " ".join(LONG_TEXT.split())
    # every chunk ends where a sentence ends
    assert all(c[-1] in ".!?" for c in chunks)


def test_cjk_terminators_and_paragraph_breaks_split():
    text = "今天天气很好。我们去公园散步吧！你觉得怎么样？\n\nSecond paragraph here."
    chunks = split_sentences(text, max_chars=8, min_chars=1)
    assert chunks[:3] == ["今天天气很好。", "我们去公园散步吧！", "你觉得怎么样？"]
    assert " ".join(chunks[3:]) == "Second paragraph here."

    paragraphs = split_sentences("First paragraph no period\n\nsecond paragraph no period", max_chars=30, min_chars=1)
    assert paragraphs == ["First paragraph no period", "second paragraph no period"]


def test_quotes_after_terminators_stay_with_their_sentence():
    text = 'He said "Wait!" Then she left. "Really?" she asked. Yes.'
    chunks = split_sentences(text, max_chars=20, min_chars=1)
    assert chunks[0] == 'He said "Wait!"'
    assert chunks[1] == "Then she left."
    assert chunks[2] == '"Really?" she asked.'


def test_overlong_sentence_is_wrapped_at_clauses_then_spaces():
    sentence = "alpha beta gamma, delta epsilon zeta, eta theta iota, kappa lambda mu nu xi omicron"
    chunks = split_sentences(sentence, max_chars=30, min_chars=1)
    assert all(len(c) <= 30 for c in chunks)
    assert " ".join(chunks).replace(" ,", ",") == sentence
    assert chunks[0] == "alpha beta gamma,"


def test_tiny_trailing_fragment_merges_into_previous_chunk():
    text = ("A sentence that is fairly long and goes on for a while to fill the chunk nicely. "
            "Yes.")
    chunks = split_sentences(text, max_chars=90, min_chars=24)
    assert chunks == [" ".join(text.split())]


def test_max_chars_must_be_positive():
    with pytest.raises(ValueError):
        split_sentences("a. b.", max_chars=0)


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------


class _Chunk:
    def __init__(self, modality, data, metadata=None):
        self.modality = modality
        self.data = data
        self.metadata = metadata or {}


class _StubModel:
    def get_output_sample_rate(self, modality="audio"):
        return 24000


def _pcm(*vals):
    return np.array(vals, dtype="<i2").tobytes()


class _RecordingAPI:
    """Each submitted request returns PCM that encodes its submission index."""

    def __init__(self):
        self.model_name = "orpheus"
        self.model = _StubModel()
        self.upload_dir = Path(tempfile.mkdtemp())
        self.submits: list[dict] = []
        self._chunks: dict[str, list] = {}
        self.released: list[str] = []
        self.read_to_end: set[str] = set()

    def submit_request(self, **kw):
        index = len(self.submits)
        self.submits.append(kw)
        self._chunks[kw["request_id"]] = [_Chunk("audio", _pcm(index, index), {"sample_rate": 24000})]
        return kw["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        return self._chunks[request_id]

    async def iter_result_chunks(self, request_id):
        for c in self._chunks[request_id]:
            yield c
        self.read_to_end.add(request_id)

    def release_request(self, request_id):
        self.released.append(request_id)

    def fail(self, index, status=400):
        """Make the ``index``-th submitted piece end with an error chunk."""
        submit = self.submit_request

        def failing_submit(**kw):
            request_id = submit(**kw)
            if len(self.submits) - 1 == index:
                self._chunks[request_id] = [_Chunk("error", b"piece failed", {"status": status})]
            return request_id

        self.submit_request = failing_submit


@pytest.fixture
def client_and_stub(monkeypatch):
    import mstar.api_server

    fake_ep = types.ModuleType("mstar.api_server.entrypoint")
    stub = _RecordingAPI()
    fake_ep.api_server = stub
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake_ep)
    monkeypatch.setattr(mstar.api_server, "entrypoint", fake_ep, raising=False)
    # Opt the Orpheus adapter into chunking for texts of 100+ characters.
    monkeypatch.setattr(adapters.OrpheusAdapter, "speech_chunk_min_chars", 100)
    monkeypatch.setattr(adapters.OrpheusAdapter, "speech_chunk_max_chars", 200)

    from mstar.api_server.openai.router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app), stub


def _wav_pcm(content: bytes) -> bytes:
    return content[44:]


def test_long_input_is_synthesized_as_ordered_sentence_chunks(client_and_stub):
    client, stub = client_and_stub
    r = client.post("/v1/audio/speech", json={"model": "orpheus", "input": LONG_TEXT, "voice": "tara", "seed": 10})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    texts = [s["text"] for s in stub.submits]
    assert texts == split_sentences(LONG_TEXT, max_chars=200)
    # One id per chunk, derived from the request id; seeds advance per chunk;
    # the model kwargs are otherwise identical and carry no chunking flag.
    ids = [s["request_id"] for s in stub.submits]
    assert [i.rsplit("-", 1)[1] for i in ids] == ["0", "1", "2"] and len({i.rsplit("-", 1)[0] for i in ids}) == 1
    assert [s["model_kwargs"]["seed"] for s in stub.submits] == [10, 11, 12]
    for submit in stub.submits:
        assert submit["model_kwargs"]["voice"] == "tara"
        assert "sentence_chunking" not in submit["model_kwargs"]
    # Playback order == submission order.
    assert _wav_pcm(r.content) == _pcm(0, 0) + _pcm(1, 1) + _pcm(2, 2)


def test_streaming_chunks_are_concatenated_in_order(client_and_stub):
    client, stub = client_and_stub
    r = client.post("/v1/audio/speech", json={"model": "orpheus", "input": LONG_TEXT, "stream": True})
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    assert len(stub.submits) == 3 and all(s["streaming"] is True for s in stub.submits)
    assert _wav_pcm(r.content) == _pcm(0, 0) + _pcm(1, 1) + _pcm(2, 2)


def test_short_input_and_opt_out_keep_a_single_request(client_and_stub):
    client, stub = client_and_stub
    r = client.post("/v1/audio/speech", json={"model": "orpheus", "input": "Hi there. All good?"})
    assert r.status_code == 200 and len(stub.submits) == 1
    assert stub.submits[0]["request_id"].startswith("speech-") and "-" not in stub.submits[0]["request_id"][7:]

    stub.submits.clear()
    r = client.post("/v1/audio/speech", json={"model": "orpheus", "input": LONG_TEXT, "sentence_chunking": False})
    assert r.status_code == 200 and len(stub.submits) == 1
    assert stub.submits[0]["text"] == LONG_TEXT and "sentence_chunking" not in stub.submits[0]["model_kwargs"]


def test_client_can_force_chunking_below_the_threshold(client_and_stub, monkeypatch):
    client, stub = client_and_stub
    monkeypatch.setattr(adapters.OrpheusAdapter, "speech_chunk_min_chars", None)
    monkeypatch.setattr(adapters.OrpheusAdapter, "speech_chunk_max_chars", 60)
    text = LONG_TEXT[:150]
    r = client.post("/v1/audio/speech", json={"model": "orpheus", "input": text, "sentence_chunking": True})
    assert r.status_code == 200 and len(stub.submits) >= 2
    assert " ".join(s["text"] for s in stub.submits) == text
    assert all(len(s["text"]) <= 60 for s in stub.submits)


def test_streaming_request_that_fails_up_front_returns_the_error_status(client_and_stub):
    client, stub = client_and_stub

    def failing_submit(**kw):
        stub.submits.append(kw)
        stub._chunks[kw["request_id"]] = [
            _Chunk("error", b"Unsupported Qwen3-TTS speaker 'nobody'", {"status": 400}),
        ]
        return kw["request_id"]

    stub.submit_request = failing_submit
    payload = {"model": "orpheus", "input": "hi there", "voice": "nobody", "stream": True}
    r = client.post("/v1/audio/speech", json=payload)
    assert r.status_code == 400
    assert "nobody" in r.json()["error"]["message"]
    # and the non-streaming path keeps returning the error too
    stub._chunks.clear()

    async def collect(request_id, raw_request=None):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Unsupported Qwen3-TTS speaker 'nobody'")

    stub.collect_results = collect
    r = client.post("/v1/audio/speech", json={"model": "orpheus", "input": "hi there", "voice": "nobody"})
    assert r.status_code == 400


def _ids(stub):
    return [s["request_id"] for s in stub.submits]


def test_chunk_count_is_capped(client_and_stub, monkeypatch):
    client, stub = client_and_stub
    monkeypatch.setattr(adapters.OrpheusAdapter, "speech_chunk_max_pieces", 2)
    r = client.post("/v1/audio/speech", json={"model": "orpheus", "input": LONG_TEXT, "sentence_chunking": True})
    assert r.status_code == 400 and "at most 2" in r.json()["error"]["message"]
    assert stub.submits == []


def test_streaming_error_up_front_releases_only_the_pieces_ahead(client_and_stub):
    client, stub = client_and_stub
    stub.fail(0)
    r = client.post("/v1/audio/speech", json={"model": "orpheus", "input": LONG_TEXT, "stream": True})
    assert r.status_code == 400
    # the failed piece was read to its end (so it isn't aborted); the lookahead is released
    ids = _ids(stub)
    assert ids[0] in stub.read_to_end and stub.released == ids[1:]


def test_non_streaming_error_releases_only_the_pieces_ahead(client_and_stub):
    client, stub = client_and_stub
    from fastapi import HTTPException

    async def collect(request_id, raw_request=None):
        if request_id == _ids(stub)[1]:
            raise HTTPException(status_code=400, detail="piece failed")
        return stub._chunks[request_id]

    stub.collect_results = collect
    r = client.post("/v1/audio/speech", json={"model": "orpheus", "input": LONG_TEXT})
    assert r.status_code == 400
    assert stub.released == _ids(stub)[2:] and len(stub.submits) == 3


def _stream(stub, num_chunks):
    ids = [f"speech-x-{i}" for i in range(num_chunks)]

    def submit(index):
        return stub.submit_request(request_id=ids[index], text="")

    async def start():
        from mstar.api_server.openai import serving_speech

        pending = [submit(0), submit(1)]
        first_iter = stub.iter_result_chunks(pending[0])
        first = await anext(first_iter)
        return serving_speech._stream_pcm(stub, submit, num_chunks, pending, first_iter, first, 24000)

    return ids, start


def test_closing_a_chunked_stream_releases_every_unread_piece():
    stub = _RecordingAPI()
    ids, start = _stream(stub, 4)

    async def run():
        gen = await start()
        assert (await anext(gen))[:4] == b"RIFF"
        assert await anext(gen) == _pcm(0, 0)  # piece 0 playing, piece 2 submitted ahead
        await gen.aclose()  # the client went away

    asyncio.run(run())
    assert stub.released == ids[:3] and len(stub.submits) == 3


def test_mid_stream_error_closes_the_stream_and_releases_the_pieces_ahead():
    from fastapi import HTTPException

    stub = _RecordingAPI()
    stub.fail(1)
    ids, start = _stream(stub, 4)

    async def run():
        gen = await start()
        with pytest.raises(HTTPException):
            async for _ in gen:
                pass

    asyncio.run(run())
    assert ids[1] in stub.read_to_end and stub.released == ids[2:]


def test_non_streaming_disconnect_stops_submitting_and_releases_the_pieces_ahead(client_and_stub):
    _, stub = client_and_stub
    from mstar.api_server.openai.protocol import SpeechRequest
    from mstar.api_server.openai.serving_speech import create_speech

    collected = []

    async def collect(request_id, raw_request=None):
        collected.append(request_id)
        return []  # what collect_results returns once it sees the disconnect

    class _Gone:
        async def is_disconnected(self):
            return True

    stub.collect_results = collect
    req = SpeechRequest(model="orpheus", input=LONG_TEXT)
    asyncio.run(create_speech(stub, "orpheus", adapters.OrpheusAdapter(), req, _Gone()))
    ids = _ids(stub)
    assert collected == ids[:1] and len(ids) == 3 and stub.released == ids[1:]


def test_the_adapter_maps_the_request_off_the_event_loop():
    """A slow ``speech_to_request`` (a clip decode, or an allowed remote fetch)
    must not hold the loop: a concurrent task keeps running while it works."""
    import time

    from mstar.api_server.openai.protocol import SpeechRequest
    from mstar.api_server.openai.serving_speech import create_speech

    class _SlowAdapter(adapters.OrpheusAdapter):
        def speech_to_request(self, req, upload_dir):
            time.sleep(0.3)
            return super().speech_to_request(req, upload_dir)

    stub = _RecordingAPI()
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    async def run():
        task = asyncio.create_task(ticker())
        try:
            await create_speech(stub, "orpheus", _SlowAdapter(), SpeechRequest(model="orpheus", input="Hello."))
        finally:
            task.cancel()
        return ticks

    # on the loop the ticker gets at most a tick or two around the awaits
    assert asyncio.run(run()) >= 10
    assert len(stub.submits) == 1


def test_release_request_aborts_only_requests_still_running():
    import threading

    from mstar.api_server.entrypoint import APIServer

    aborted = []
    done, running = threading.Event(), threading.Event()
    done.set()
    server = types.SimpleNamespace(
        request_lock=threading.Lock(),
        pending_requests={"done": types.SimpleNamespace(event=done),
                          "running": types.SimpleNamespace(event=running)},
        abort_request=aborted.append,
    )
    APIServer.release_request(server, "done")
    APIServer.release_request(server, "running")
    assert aborted == ["running"] and "done" not in server.pending_requests

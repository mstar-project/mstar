"""Sentence chunking on ``/v1/audio/speech``: the splitter and the ordered sub-requests.

The router is mounted on a FastAPI app with a stubbed APIServer (as in
``test_openai_router.py``); every sub-request the handler submits is recorded
so ordering, ids, seeds and playback concatenation can be checked without an
engine.
"""

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

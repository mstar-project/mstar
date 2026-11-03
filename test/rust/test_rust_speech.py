"""``/v1/audio/speech`` on ``mstar-server`` against the Python route, for every
adapter class whose speech it ports.

Both frontends take the same requests against backends that answer with the
same audio: the Rust one through the real binary and bridge, the Python one
through ``serving_speech.create_speech`` (fastapi's TestClient for the HTTP
answer, its streaming body iterator for the pieces it writes). Checked: what
each submits (text, modalities, model kwargs, the piece ids of a chunked
input), the status and body of each refusal, and the response bytes, headers
and HTTP chunks. Skipped unless the ``mstar_rust`` extension, the server
binary and fastapi are installed."""
import asyncio
import json
import random
import re
import socket

import pytest

pytest.importorskip("mstar_rust")
pytest.importorskip("fastapi")

from fastapi import HTTPException
from fastapi.testclient import TestClient
from test_rust_frontend import BINARY, _Chunk, _model_stack, _StubAPIServer
from test_rust_openai_adapters import NOT_PORTED, _classes, _model_of, _post

from mstar.api_server import entrypoint
from mstar.api_server.entrypoint import APIServer
from mstar.api_server.openai import serving_speech
from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
from mstar.api_server.openai.protocol import SpeechRequest
from mstar.api_server.served_models import ServedModels

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

SPEECH_CLASSES = sorted(c for c, a in _classes().items()
                        if a.supports_speech and (c, "speech") not in NOT_PORTED)


def test_the_speech_classes_are_the_ported_ones():
    assert SPEECH_CLASSES == ["OrpheusAdapter", "Qwen3OmniAdapter", "Qwen3TTSAdapter"]


def _audio(text: str) -> list[_Chunk]:
    """The backend's answer to a piece: PCM16 derived from its text, with a
    text chunk and an empty audio chunk the route must skip."""
    pcm = text.encode()[:6].ljust(6, b"\0")
    return [_Chunk("audio", pcm), _Chunk("text", b"spoken"), _Chunk("audio", b""),
            _Chunk("audio", pcm[::-1] * 3)]


def _scripted(text: str) -> list[_Chunk]:
    if text and "refuse" in text:
        return [_Chunk("error", b"input refused by the model", {"status": 400})]
    return _audio(text or "")


class _RustSpeechStub(_StubAPIServer):
    async def iter_result_chunks(self, request_id):
        for c in _scripted(self.texts.get(request_id)):
            yield c


class _PythonSpeechServer:
    """What ``create_speech`` calls on APIServer, answering as the Rust stub does."""

    open_result_stream = APIServer.open_result_stream

    def __init__(self, model_name, upload_dir):
        self.model_name, self.upload_dir, self.model = model_name, upload_dir, None
        self.served_models = ServedModels.of(model_name, None)
        self.submitted, self.texts, self.released = [], {}, []

    def submit_request(self, **kwargs):
        self.submitted.append(kwargs)
        self.texts[kwargs["request_id"]] = kwargs["text"]
        return kwargs["request_id"]

    async def iter_result_chunks(self, request_id):
        for c in _scripted(self.texts[request_id]):
            yield c

    async def collect_results(self, request_id, raw_request=None):
        chunks = _scripted(self.texts[request_id])
        if chunks[0].modality == "error":
            raise HTTPException(status_code=chunks[0].metadata["status"], detail=chunks[0].data.decode())
        return chunks

    def release_request(self, request_id):
        self.released.append(request_id)


_RID = re.compile(r"^speech-[0-9a-f]{32}")


def _python_submits(submitted: list[dict]) -> list[dict]:
    # ``model`` names a LoRA adapter on the Python frontend; the Rust frontend
    # refuses to start with LoRA on, so it sends none.
    return [{"rid": _RID.sub("speech-*", s["request_id"]), "text": s["text"], "in": s["input_modalities"],
             "out": s["output_modalities"], "mk": s["model_kwargs"], "files": s["file_paths"],
             "streaming": s["streaming"]} for s in submitted]


def _rust_submits(submitted: list[dict]) -> list[dict]:
    return [{"rid": _RID.sub("speech-*", s["rid"]), "text": s["text"], "in": s["in"], "out": s["out"],
             "mk": s["mk"], "files": s["files"], "streaming": s["streaming"]} for s in submitted]


def _python_answer(model, body, tmp_path, monkeypatch):
    """(status, content type, cache control, body, submits) from the Python route."""
    api = _PythonSpeechServer(model, tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", api)
    r = TestClient(entrypoint.app).post("/v1/audio/speech", json=body)
    return (r.status_code, r.headers.get("content-type"), r.headers.get("cache-control"), r.content,
            _python_submits(api.submitted))


def _rust_answer(port, stub, body):
    stub.submitted.clear()
    status, headers, data = _http(port, body)
    return (status, headers.get("content-type"), headers.get("cache-control"), b"".join(data),
            _rust_submits(stub.submitted))


def _http(port, body, timeout=15):
    """(status, headers, body chunks) of one POST, the chunked body kept as the
    server framed it."""
    payload = json.dumps(body).encode()
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall(b"POST /v1/audio/speech HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  b"Connection: close\r\nContent-Length: %d\r\n\r\n" % len(payload) + payload)
        raw = b""
        while chunk := s.recv(65536):
            raw += chunk
    head, rest = raw.split(b"\r\n\r\n", 1)
    lines = head.decode().split("\r\n")
    headers = {k.lower(): v.strip() for k, v in (line.split(":", 1) for line in lines[1:])}
    status = int(lines[0].split()[1])
    if headers.get("transfer-encoding") != "chunked":
        return status, headers, [rest]
    chunks = []
    while True:
        size, rest = rest.split(b"\r\n", 1)
        n = int(size, 16)
        if n == 0:
            return status, headers, chunks
        chunks.append(rest[:n])
        rest = rest[n + 2:]


def _sentence(i: int) -> str:
    return f"This is sentence {i}, which the model reads aloud at an even pace."


LONG = " ".join(_sentence(i) for i in range(14))  # past Qwen3-TTS's 600 characters

SPEECH_CASES = {
    "benchmark": {"model": "qwen3_tts_1p7b", "input": "The quick brown fox jumps over the lazy dog.",
                  "voice": "ryan", "language": "English", "response_format": "wav"},
    "all-fields": {"input": "Привет, 世界. ¿Qué tal?", "voice": "vivian", "language": "Auto",
                   "instructions": "Speak slowly.", "temperature": 0.6, "top_p": 0.85, "seed": 11,
                   "speed": 1.5, "top_k": 30, "repetition_penalty": 1.05, "code_predictor_top_k": 40,
                   "non_streaming_mode": False, "max_output_tokens": 512, "ref_text": "unused",
                   "x_vector_only_mode": True, "response_format": "pcm"},
    "instruct-wins": {"input": "Hi.", "instructions": "calm", "instruct": "bright", "speed": None},
    "empty-fields": {"input": "Hi.", "instructions": "", "ref_audio": "", "voice": "", "response_format": ""},
    # strings where Python coerces declared fields; temperature and top_p are left out, since the
    # Rust frontend passes them on as sent and the API server's normalize_generation_kwargs reads both
    "strings": {"input": "Hi.", "seed": "7", "speed": "2", "stream": "false"},
    "chunked": {"input": LONG, "seed": 9223372036854775806, "voice": "ryan"},
    "unchunked": {"input": LONG, "seed": 5, "sentence_chunking": False},
    "forced-short": {"input": "  Hello there.  ", "sentence_chunking": True, "seed": 1},
    "forced-cjk": {"input": "海は広い。" * 90 + "Done!", "sentence_chunking": True},
}


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "full"])
@pytest.mark.parametrize("case", sorted(SPEECH_CASES))
@pytest.mark.parametrize("cls", SPEECH_CLASSES)
def test_both_frontends_submit_and_answer_a_speech_the_same_way(cls, case, stream, tmp_path, monkeypatch):
    model = _model_of(cls)
    body = {**SPEECH_CASES[case], "stream": stream} if case != "strings" else SPEECH_CASES[case]
    stub = _RustSpeechStub()
    with _model_stack(model, stub) as (port, _upload):
        rust = _rust_answer(port, stub, body)
    python = _python_answer(model, body, tmp_path, monkeypatch)
    assert rust == python
    assert rust[0] == 200, rust


def test_a_long_qwen3_tts_input_goes_out_as_pieces_with_offset_seeds(tmp_path, monkeypatch):
    """What the parity case above compares, spelled out once."""
    body = {**SPEECH_CASES["chunked"], "stream": True}
    _, _, _, _, submits = _python_answer("qwen3_tts_1p7b", body, tmp_path, monkeypatch)
    assert [s["rid"] for s in submits] == [f"speech-*-{i}" for i in range(len(submits))]
    assert len(submits) == 3
    assert [s["mk"]["seed"] for s in submits] == [2**63 - 2, 2**63 - 1, -(2**63)]
    assert " ".join(s["text"] for s in submits) == LONG


REFUSALS = {
    "stream-mp3": ("qwen3_tts_1p7b", {"input": "Hi.", "stream": True, "response_format": "MP3"}),
    "stream-flac-orpheus": ("orpheus", {"input": "Hi.", "stream": True, "response_format": "flac"}),
    "too-many-pieces": ("qwen3_tts_1p7b", {"input": " ".join(_sentence(i) for i in range(200))}),
    "forced-too-many": ("orpheus", {"input": " ".join(_sentence(i) for i in range(200)),
                                    "sentence_chunking": "yes", "stream": True}),
    "engine-refusal-stream": ("qwen3_tts_1p7b", {"input": "please refuse", "stream": True}),
    "engine-refusal": ("qwen3_tts_1p7b", {"input": "please refuse"}),
    "second-piece-refused": ("qwen3_tts_1p7b", {"input": LONG + " Now refuse this one."}),
    "no-speech": ("qwen3_5_0.8b", {"input": "Hi."}),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_both_frontends_refuse_a_speech_with_the_same_status_and_text(case, tmp_path, monkeypatch):
    model, body = REFUSALS[case]
    stub = _RustSpeechStub()
    with _model_stack(model, stub) as (port, _upload):
        rust = _post(port, "/v1/audio/speech", body)
    python = _python_answer(model, body, tmp_path, monkeypatch)
    assert rust[0] in (400, 404), rust
    assert (rust[0], json.loads(rust[1])) == (python[0], json.loads(python[3]))


def test_what_only_the_rust_frontend_refuses():
    """It has no audio decoder and no compressed-audio encoder, so a clip to
    clone and a non-streamed mp3 are 400s that point at the Python frontend."""
    stub = _RustSpeechStub()
    with _model_stack("qwen3_tts_1p7b", stub) as (port, _upload):
        clone = _post(port, "/v1/audio/speech", {"input": "Hi.", "ref_audio": "data:audio/wav;base64,UklGRg=="})
        mp3 = _post(port, "/v1/audio/speech", {"input": "Hi.", "response_format": "mp3"})
    assert clone[0] == 400 and "serve without --rust-frontend" in json.loads(clone[1])["error"]["message"]
    assert mp3[0] == 400 and "run the Python frontend" in json.loads(mp3[1])["error"]["message"]
    assert stub.submitted == []


async def _python_pieces(model, body, tmp_path):
    """The byte strings the Python route's streaming body yields, in order."""
    api = _PythonSpeechServer(model, tmp_path)
    resp = await serving_speech.create_speech(api, model, ADAPTER_REGISTRY[model], SpeechRequest(**body))
    return [piece async for piece in resp.body_iterator]


@pytest.mark.parametrize("fmt", ["wav", "pcm"])
@pytest.mark.parametrize("text", ["Short.", LONG], ids=["one-piece", "three-pieces"])
def test_a_streamed_speech_is_framed_as_the_python_route_frames_it(text, fmt, tmp_path):
    """The WAV header, then one HTTP chunk per audio chunk of each piece, in order."""
    body = {"input": text, "stream": True, "response_format": fmt, "voice": "ryan"}
    stub = _RustSpeechStub()
    with _model_stack("qwen3_tts_1p7b", stub) as (port, _upload):
        status, headers, chunks = _http(port, body)
    python = asyncio.run(_python_pieces("qwen3_tts_1p7b", body, tmp_path))
    assert status == 200
    assert chunks == python
    assert len(python) == 2 * len(stub.submitted) + (fmt == "wav")
    if fmt == "wav":
        assert python[0] == (b"RIFF\xff\xff\xff\xffWAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
                             b"\xc0\x5d\x00\x00\x80\xbb\x00\x00\x02\x00\x10\x00data\xff\xff\xff\xff")


def test_split_sentences_matches_python_on_random_long_inputs(tmp_path, monkeypatch):
    """Piece texts from both frontends over inputs built from the splitter's
    edge cases: terminators with and without closers, CJK ends, paragraph
    breaks, clause marks, runs with no spaces, combining marks, and Python's
    whitespace (U+001C and U+00A0 among it)."""
    from mstar.api_server.openai.speech_chunking import split_sentences

    words = ["word", "longerword", "3.5", "end.", 'quote."', "aside!)", "really?'", "semi;", "海は広い。",
             "質問？", "clause,", "colon:", "，中文", "\n\n", "\n \t\n", " ", "\u001c", "áb́",
             "x" * 120, "é" * 450, "e\u0301" * 300, "Done.”", "  "]
    rng = random.Random(1234)
    texts = []
    while len(texts) < 120:
        text = " ".join(rng.choice(words) for _ in range(rng.randint(30, 150)))
        if len(text.strip()) > 400:
            texts.append(text)
    stub = _RustSpeechStub()
    with _model_stack("orpheus", stub) as (port, _upload):
        for text in texts:
            stub.submitted.clear()
            status, data = _post(port, "/v1/audio/speech", {"input": text, "sentence_chunking": True})
            want = split_sentences(text, max_chars=400)
            if len(want) > 32:
                assert (status, json.loads(data)["error"]["message"]) == (
                    400, f"input splits into {len(want)} sentence chunks; at most 32 are allowed")
                continue
            assert status == 200
            assert [s["text"] for s in stub.submitted] == want, text

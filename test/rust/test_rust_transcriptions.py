"""``/v1/audio/transcriptions`` on mstar-server against the Python route.

Each case sends the same multipart body to both frontends, served by one
scripted ASR stub, and compares the response (status, body, stream events) and
every window each submitted: the kwargs, the layout, and the window's length
as the stub hears it. Both decode and cut the upload with the same
``WindowPlanner``, the Rust side over the bridge's ``windows`` round trip.
"""

import asyncio
import contextlib
import io
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("mstar_rust")
pytest.importorskip("fastapi")
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from fastapi.testclient import TestClient
from test_rust_frontend import BINARY, _Chunk, _free_port, _wait_healthy

from mstar.api_server import entrypoint
from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
from mstar.api_server.rust_frontend import (
    OPENAI_ADAPTER_ENV,
    RustFrontendBridge,
    openai_adapter_class,
    transcription_facts,
)
from mstar.utils import generation_kwargs

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

WORDS_MARKER = "<|startoflm|>"


def _wav(seconds: float, rate: int = 16_000, channels: int = 1, quiet=()) -> bytes:
    """A tone of ``seconds`` with silence over each ``quiet`` span, as a WAV."""
    t = np.arange(int(seconds * rate)) / rate
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    x += np.random.default_rng(0).normal(0, 0.003, x.shape)
    for start, end in quiet:
        x[int(start * rate):int(end * rate)] *= 0.001
    if channels > 1:
        x = np.stack([x] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, x, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


SHORT = _wav(6.0)
LONG = _wav(70.0, quiet=[(27.5, 28.4), (55.0, 56.5)])
# 44.1 kHz stereo, which the planner resamples as it decodes
LONG_STEREO = _wav(65.0, rate=44_100, channels=2, quiet=[(28.0, 29.0)])
BALANCED = _wav(25.0, quiet=[(11.0, 12.0)])


def _multipart(fields: list[tuple[str, str]], file: tuple[str, bytes] | None) -> tuple[bytes, str]:
    boundary = "----mstarasr"
    body = b"".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, v in fields
    )
    if file is not None:
        name, data = file
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
                 "Content-Type: audio/wav\r\n\r\n").encode() + data + b"\r\n"
    return body + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def _pieces(text: str, size: int = 5) -> list:
    """``text`` as text chunks of ``size`` bytes, split mid-character as BPE tokens are."""
    data = text.encode()
    return [_Chunk("text", data[i:i + size]) for i in range(0, len(data), size)]


class _Model:
    def __init__(self, request_kwargs):
        self._kwargs = request_kwargs

    def request_kwargs(self):
        return self._kwargs


class _ASRStub:
    """Both frontends' data plane: each submission's chunks come from
    ``script(record, index)``, ``record`` being what was submitted."""

    def __init__(self, model_name: str, upload_dir: Path, request_kwargs=None, cap=None):
        self.model_name = model_name
        self.model = _Model(request_kwargs)
        self.upload_dir = upload_dir
        self.max_concurrent_requests = cap
        self.metrics = None
        self.script = lambda record, i: _pieces("hello")
        self.reset()

    def reset(self):
        self.submits: list[dict] = []
        self._chunks: dict = {}
        self.released: list = []

    def submit_request(self, *, text=None, file_paths=None, input_modalities, output_modalities,
                       model_kwargs=None, prompt_parts=None, streaming=True, request_id=None, model=None):
        path = (file_paths or {}).get("audio", [None])[0]
        try:
            info = sf.info(path)
            seconds = round(info.frames / info.samplerate, 4)
        except Exception:  # noqa: BLE001 — an upload no decoder reads
            seconds = None
        record = {"text": text, "in": input_modalities, "out": output_modalities,
                  "mk": model_kwargs or {}, "streaming": streaming, "seconds": seconds}
        self.submits.append(record)
        self._chunks[request_id] = self.script(record, len(self.submits) - 1)
        return request_id

    async def collect_results(self, request_id, raw_request=None):
        return self._chunks.get(request_id, [])

    async def iter_result_chunks(self, request_id):
        for c in self._chunks.get(request_id, []):
            await asyncio.sleep(0)
            yield c

    def release_request(self, request_id):
        self.released.append(request_id)

    def submission_key(self, request_id):
        return request_id

    def relay_chunks_to(self, key, sink, tag):
        pass

    def abort_submission(self, key):
        self.released.append(key)

    def abort_request(self, request_id):
        self.released.append(request_id)


@contextlib.contextmanager
def _rust(model: str, stub: _ASRStub):
    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="mstar_asr_")
    facts = transcription_facts(stub, ADAPTER_REGISTRY[model])
    env = dict(os.environ, **{OPENAI_ADAPTER_ENV: openai_adapter_class(model)})
    if facts is not None:
        env["MSTAR_TRANSCRIPTION"] = json.dumps(facts)
    proc = subprocess.Popen([BINARY, model, str(port), bridge_dir, str(stub.upload_dir)], env=env)
    bridge = RustFrontendBridge(stub, bridge_dir)
    thread = threading.Thread(target=bridge.run, daemon=True)
    thread.start()
    try:
        assert _wait_healthy(port)
        yield port
    finally:
        bridge.stop()
        proc.terminate()
        proc.wait(timeout=10)


_STACKS: dict = {}


@pytest.fixture(scope="module")
def stacks():
    with contextlib.ExitStack() as exits:
        def get(model: str, request_kwargs=None, cap=None):
            key = (model, None if request_kwargs is None else tuple(sorted(request_kwargs)), cap)
            if key not in _STACKS:
                stub = _ASRStub(model, Path(tempfile.mkdtemp(prefix="mstar_asr_up_")), request_kwargs, cap)
                _STACKS[key] = (stub, exits.enter_context(_rust(model, stub)))
            return _STACKS[key]
        yield get
        _STACKS.clear()


def _read(status: int, ctype: str, body: bytes):
    """A response as compared: JSON parsed, a stream's events parsed, text as is."""
    if ctype.startswith("application/json"):
        return status, json.loads(body)
    if ctype.startswith("text/event-stream"):
        events = []
        for block in body.decode().split("\n\n"):
            if block.startswith("data: "):
                payload = block[len("data: "):]
                events.append(payload if payload == "[DONE]" else json.loads(payload))
        return status, events
    return status, body.decode()


def _empty(directory: Path, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(directory.iterdir()):
            return True
        time.sleep(0.05)
    return False


def _both(stacks, monkeypatch, model, fields, file=("speech.wav", SHORT), script=None, request_kwargs=None, cap=None):
    """The Python route's and mstar-server's (response, submits)."""
    stub, port = stacks(model, request_kwargs, cap)
    if script is not None:
        stub.script = script
    body, ctype = _multipart(fields, file)
    monkeypatch.setattr(entrypoint, "api_server", stub)
    stub.reset()
    py = TestClient(entrypoint.app).post("/v1/audio/transcriptions", content=body, headers={"Content-Type": ctype})
    py_out = (_read(py.status_code, py.headers.get("content-type", ""), py.content), stub.submits)
    assert _empty(stub.upload_dir), "the Python route left files"
    stub.reset()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/audio/transcriptions", data=body,
                                 headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            rust = (r.status, r.headers.get("content-type", ""), r.read())
    except urllib.error.HTTPError as e:
        rust = (e.code, e.headers.get("content-type", ""), e.read())
    rust_out = (_read(*rust), stub.submits)
    assert _empty(stub.upload_dir), "mstar-server left files"
    return py_out, rust_out


# ── scripted models ──────────────────────────────────────────────────────────

def _whisper(stop_inside=False, loop_when_prompted=False, loop_below=None):
    """Whisper's rendered stream for a window: two timed segments when asked for
    timestamps (the second left open with ``stop_inside``), word timings after
    the marker, and a repetition loop at temperature 0 under a prompt, or on
    every second window below temperature ``loop_below``."""
    def script(record, i):
        mk, d = record["mk"], record["seconds"] or 6.0
        head = "" if mk.get("language") else "<|en|>"
        if loop_when_prompted and mk.get("initial_prompt") and not mk.get("temperature"):
            return _pieces(head + " la" * 60)
        if loop_below is not None and round(d) % 2 == 0 and (mk.get("temperature") or 0.0) < loop_below:
            return _pieces(head + " na" * 60)
        if not mk.get("timestamps"):
            return _pieces(f"{head} the window of {d:.2f} s, café")
        a, b = round(d * 0.4, 2), round(d * 0.8, 2)
        text = f"{head}<|0.00|> first part {i}<|{a:.2f}|><|{a:.2f}|> second part<|{b:.2f}|>"
        if stop_inside:
            text += f"<|{b:.2f}|> cut off"
        if mk.get("timestamps") == "word":
            text += f"{WORDS_MARKER}<|0.00|> first<|0.40|><|0.40|> part<|0.90|>"
        return _pieces(text)
    return script


def _assert_same(py, rust):
    assert rust[0] == py[0]
    assert rust[1] == py[1]


WHISPER_CASES = {
    "json": [("model", "whisper_large")],
    "text": [("response_format", "text")],
    "srt": [("response_format", "srt")],
    "vtt": [("response_format", "VTT")],
    "verbose": [("response_format", "verbose_json"), ("timestamp_granularities[]", "segment"),
                ("timestamp_granularities[]", "word")],
    "verbose-words-only": [("response_format", "verbose_json"), ("timestamp_granularities[]", "word")],
    "language-and-prompt": [("language", "de"), ("prompt", "Hallo"), ("temperature", "0.2"), ("seed", "7")],
    "extra-kwargs": [("top_k", "5"), ("beam", '{"n": 2}'), ("note", "plain text")],
    # declared text fields stay text, however they would parse as JSON
    "text-fields-as-sent": [("prompt", "2024"), ("language", "null")],
    # an empty field is json; the timestamp checks read it as sent
    "empty-format": [("response_format", "")],
    "empty-format-words": [("response_format", ""), ("timestamp_granularities[]", "word")],
    "stream": [("stream", "true")],
    "stream-verbose": [("stream", "true"), ("response_format", "verbose_json")],
}


@pytest.mark.parametrize("case", list(WHISPER_CASES))
@pytest.mark.parametrize("audio", ["short", "long", "long-stereo"])
def test_whisper_answers_as_the_python_route(stacks, monkeypatch, case, audio):
    file = ("speech.wav", {"short": SHORT, "long": LONG, "long-stereo": LONG_STEREO}[audio])
    _assert_same(*_both(stacks, monkeypatch, "whisper_large", WHISPER_CASES[case], file, _whisper()))


@pytest.mark.parametrize("fields", [
    [("response_format", "verbose_json")],
    [("response_format", "json")],
    [("stream", "true"), ("response_format", "verbose_json")],
], ids=["verbose", "json", "stream"])
def test_a_parallel_window_that_stops_inside_a_segment_decodes_its_tail_alike(stacks, monkeypatch, fields):
    _assert_same(*_both(stacks, monkeypatch, "whisper_large", fields, ("long.wav", LONG), _whisper(stop_inside=True)))


@pytest.mark.parametrize("fields", [
    [("long_form", "sequential"), ("response_format", "verbose_json")],
    [("long_form", "sequential"), ("response_format", "srt")],
    [("long_form", "sequential")],
    [("long_form", "sequential"), ("temperature", "0.3")],
    [("long_form", "sequential"), ("stream", "true")],
    [("long_form", "sequential"), ("window_seconds", "12")],
], ids=["verbose", "srt", "json", "pinned-temperature", "stream", "window-seconds"])
@pytest.mark.parametrize("stop_inside", [False, True], ids=["closed", "stops-inside"])
def test_sequential_long_form_runs_as_the_python_route(stacks, monkeypatch, fields, stop_inside):
    _assert_same(*_both(stacks, monkeypatch, "whisper_large", fields, ("long.wav", LONG),
                        _whisper(stop_inside=stop_inside, loop_when_prompted=True)))


@pytest.mark.parametrize("fields", [
    [("long_form", "sequential"), ("response_format", "verbose_json")],
    [("long_form", "sequential"), ("stream", "true")],
], ids=["verbose", "stream"])
def test_a_looping_window_warms_up_and_stops_conditioning_alike(stacks, monkeypatch, fields):
    (py, py_submits), rust = _both(stacks, monkeypatch, "whisper_large", fields, ("long.wav", LONG),
                                   _whisper(loop_when_prompted=True, loop_below=0.6))
    _assert_same((py, py_submits), rust)
    assert max(s["mk"].get("temperature", 0.0) for s in py_submits) > 0.5, "the ladder never climbed"


@pytest.mark.parametrize("request_kwargs", [{"language", "temperature", "seed"}, None], ids=["declared", "undeclared"])
def test_the_drivers_add_conditioning_and_timestamps_only_for_a_model_that_reads_them(
        stacks, monkeypatch, request_kwargs):
    fields = [("long_form", "sequential"), ("response_format", "verbose_json")]
    _assert_same(*_both(stacks, monkeypatch, "whisper_large", fields, ("long.wav", LONG),
                        _whisper(stop_inside=True, loop_when_prompted=True), request_kwargs=request_kwargs))


@pytest.mark.parametrize("cap", [2, 64])
@pytest.mark.parametrize("stream", ["false", "true"])
def test_windows_in_flight_follow_the_share_alike(stacks, monkeypatch, cap, stream):
    _assert_same(*_both(stacks, monkeypatch, "whisper_large", [("stream", stream)], ("long.wav", LONG),
                        _whisper(stop_inside=True), cap=cap))


def _failing(at: int, status: int = 503):
    def script(record, i):
        if i == at:
            return [_Chunk("text", b" partial"), _Chunk("error", b"the engine gave up", {"status": status})]
        return _whisper()(record, i)
    return script


@pytest.mark.parametrize("stream", ["false", "true"])
@pytest.mark.parametrize("at", [0, 1])
def test_a_failed_window_fails_the_upload_alike(stacks, monkeypatch, stream, at):
    _assert_same(*_both(stacks, monkeypatch, "whisper_large", [("stream", stream)], ("long.wav", LONG), _failing(at)))


REFUSALS = {
    "bad-format": [("response_format", "xml")],
    "bad-long-form": [("long_form", "batch")],
    "long-form-not-a-string": [("long_form", "[1]")],
    "window-too-short": [("window_seconds", "2")],
    "window-not-a-number": [("window_seconds", "soon")],
    "temperature-not-a-number": [("temperature", "hot")],
    "pydantic-several": [("prompt", "1.5"), ("seed", "x"), ("stream", "maybe")],
    "granularities-not-a-list": [("timestamp_granularities", "word")],
    "seed-fractional": [("seed", "5.5")],
    "seed-past-int64": [("seed", str(2**70))],
    "bad-knob": [("top_p", "2")],
    "bad-knob-order": [("top_k", "1.5"), ("temperature", "-1")],
    "response-format-null": [("response_format", "null")],
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_a_refused_request_reads_as_pythons(stacks, monkeypatch, case):
    (py, py_submits), (rust, rust_submits) = _both(stacks, monkeypatch, "whisper_large", REFUSALS[case])
    assert py[0] == 400, py
    assert rust == py
    assert py_submits == rust_submits == []


# every knob normalize_generation_kwargs checks (the sampler's rules and its own):
# Rust's copy of the tables must refuse each before a stream commits to a 200
KNOBS = [*generation_kwargs._REAL_RULES, *generation_kwargs._INT_RULES, *generation_kwargs._BOOL_KNOBS]


@pytest.mark.parametrize("value", ["nope", "-1000000000"])
@pytest.mark.parametrize("knob", KNOBS)
def test_every_generation_knob_is_refused_alike(stacks, monkeypatch, knob, value):
    fields = [(knob, value), ("stream", "true")]
    (py, py_submits), (rust, rust_submits) = _both(stacks, monkeypatch, "whisper_large", fields)
    assert rust == py
    assert py[0] == 400, py
    assert py_submits == rust_submits == []


@pytest.mark.parametrize("file", [None, ("speech.wav", b"")], ids=["no-file", "empty-file"])
def test_a_missing_or_empty_upload_is_refused_alike(stacks, monkeypatch, file):
    (py, _), (rust, _) = _both(stacks, monkeypatch, "whisper_large", [("model", "whisper_large")], file)
    assert rust == py and py[0] == 400


def test_an_upload_no_decoder_reads_goes_whole_to_the_model_alike(stacks, monkeypatch):
    fields = [("response_format", "verbose_json")]
    (py, py_submits), (rust, rust_submits) = _both(
        stacks, monkeypatch, "whisper_large", fields, ("speech.webm", b"not audio at all"))
    assert rust == py
    assert rust_submits == py_submits and len(py_submits) == 1


def test_an_upload_past_the_cap_is_a_413_alike(stacks, monkeypatch):
    over = _wav(31 * 60, rate=8000)
    (py, _), (rust, _) = _both(stacks, monkeypatch, "whisper_large", [], ("long.wav", over))
    assert py[0] == 413
    assert rust == py


def test_a_model_without_the_route_is_a_404_alike(stacks, monkeypatch):
    (py, _), (rust, _) = _both(stacks, monkeypatch, "kokoro", [])
    assert py[0] == 404
    assert rust == py


# ── the other ASR adapters ───────────────────────────────────────────────────

def _qwen3_asr(record, i):
    mk = record["mk"]
    if mk.get("language"):
        return _pieces(" hello " + "again " * 30 + "world " * 3)
    head = "language None" if i % 2 else "language German"
    return _pieces(f"{head}<asr_text>guten Tag {i} " + "ja" * 25)


OTHERS = {
    "qwen3_asr": [
        [("response_format", "verbose_json")],
        [("language", "en")],
        [("stream", "true")],
        [("window_seconds", "1200"), ("response_format", "verbose_json")],
        [("prompt", "<|audio_pad|>")],
    ],
    "canary": [
        [("language", "de-DE")],
        [("task", "translate"), ("language", "fr")],
        [("target_language", "de"), ("language", "fr")],
        [("language", "xx")],
        [("punctuation", '"yes"')],
        [("task", "summarize")],
        [("timestamp_granularities[]", "word")],
        [("target_language", "5")],
        # an untimed stream: its text is lstripped (_plain's leading space)
        [("stream", "true")],
    ],
    "canary_qwen": [[("language", "en_US")], [("language", "de")], [("task", "translate")], [("stream", "true")]],
    "parakeet": [[("language", "auto")], [("language", "xx")], [("language", "")], [("prompt", "dropped")],
                 [("stream", "true")]],
    "nemotron_asr": [[("language", "xx")], [("task", "transcribe")], [("task", "translate")], [("stream", "true")]],
    "higgs_audio": [[("prompt", "Transcribe, please.")], [("response_format", "srt")]],
    "omnilingual_asr_ctc_300m": [
        [("response_format", "verbose_json")],
        [("window_seconds", "6"), ("response_format", "vtt")],
        [("stream", "true")],
    ],
}


def _plain(record, i):
    return _pieces(f" heard {record['seconds']} s")


@pytest.mark.parametrize("model,fields", [(m, f) for m, cases in OTHERS.items() for f in cases],
                         ids=[f"{m}-{i}" for m, cases in OTHERS.items() for i in range(len(cases))])
def test_every_asr_adapter_answers_as_the_python_route(stacks, monkeypatch, model, fields):
    script = _qwen3_asr if model == "qwen3_asr" else _plain
    audio = BALANCED if model.startswith("omnilingual") else LONG
    _assert_same(*_both(stacks, monkeypatch, model, fields, ("speech.wav", audio), script))


def test_the_launcher_hands_each_asr_adapter_its_tables():
    stub = _ASRStub("parakeet", Path(tempfile.gettempdir()))
    assert transcription_facts(stub, ADAPTER_REGISTRY["parakeet"])["detects_language"] is True
    assert "detects_language" not in transcription_facts(stub, ADAPTER_REGISTRY["nemotron_asr"])
    assert ["en", "English"] in transcription_facts(stub, ADAPTER_REGISTRY["qwen3_asr"])["language_codes"]
    assert "en" in transcription_facts(stub, ADAPTER_REGISTRY["canary"])["languages"]
    assert transcription_facts(stub, ADAPTER_REGISTRY["kokoro"]) is None

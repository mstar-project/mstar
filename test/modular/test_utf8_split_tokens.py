"""Characters split across text chunks come out whole (#517).

Byte-level detokenizers (Cosmos3, Whisper) send each token's raw bytes as
one chunk, so a CJK character or emoji can span two chunks. Every consumer
here (chat completions, ``/v1/realtime``, the Python SDK, the Dynamo bridge)
must join or incrementally decode the bytes instead of decoding each chunk
on its own. The Rust frontend's twins are the tests in
``rust/server/src/{utf8,main,serving,tokenizer}.rs``.
"""

import asyncio
import base64
import io
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from mstar.utils.utf8_stream import Utf8Stream, decode_joined

TEXT = "中文 😀"
DATA = TEXT.encode()
# boundaries fall inside 中 (3 bytes) and inside 😀 (4 bytes)
PIECES = [DATA[:1], DATA[1:4], DATA[4:8], DATA[8:]]
TRUNCATED = PIECES[:-1] + [PIECES[-1][:-1]]  # the stream ends inside 😀


def _chunk(data, modality="text", metadata=None):
    return SimpleNamespace(modality=modality, data=data, metadata=metadata or {})


# ----------------------------------------------------------------------
# the shared helper
# ----------------------------------------------------------------------

def test_utf8_stream_holds_back_an_incomplete_tail():
    s = Utf8Stream()
    assert [s.feed(p) for p in PIECES] == ["", "中", "文 ", "😀"]
    assert s.flush() == ""


def test_utf8_stream_one_byte_at_a_time():
    s = Utf8Stream()
    data = "a中😀b".encode()
    assert "".join(s.feed(data[i:i + 1]) for i in range(len(data))) + s.flush() == "a中😀b"


def test_utf8_stream_flushes_an_unfinished_character_as_one_replacement():
    s = Utf8Stream()
    assert s.feed(b"ok") == "ok"
    assert s.feed("😀".encode()[:3]) == ""
    assert s.flush() == "\ufffd"
    assert s.feed(b"x") == "x"


@pytest.mark.parametrize("bad", [b"a\xffb", b"\xe4\x41", b"\xc0\x80z", b"\xed\xa0\x80!"])
def test_utf8_stream_invalid_bytes_match_a_one_shot_decode(bad):
    s = Utf8Stream()
    assert s.feed(bad) + s.flush() == bad.decode("utf-8", "replace")


def test_decode_joined():
    assert decode_joined(PIECES) == TEXT


# ----------------------------------------------------------------------
# chat completions
# ----------------------------------------------------------------------

class _BatchAPI:
    """Just enough APIServer for ``reply.reply_events``."""

    def __init__(self, batches):
        self._batches = batches

    def submission_key(self, request_id):
        return request_id

    async def iter_result_chunk_batches(self, request_id):
        for batch in self._batches:
            yield batch


def _sse_events(frames):
    events = []
    for frame in frames:
        for line in frame.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[len("data: "):]))
    return events


def _stream_chat(batches):
    pytest.importorskip("fastapi")
    from mstar.api_server.openai import reply
    from mstar.api_server.openai.serving_chat import _stream_events

    async def run():
        return [f async for f in _stream_events(reply.reply_events(_BatchAPI(batches), "r"), "m", "r")]

    events = _sse_events(asyncio.run(run()))
    content = "".join(e["choices"][0]["delta"].get("content") or "" for e in events)
    return content, events


def test_chat_response_joins_text_before_decoding():
    pytest.importorskip("fastapi")
    from mstar.api_server.openai.serving_chat import _build_response

    resp = _build_response("m", "r", [_chunk(p) for p in PIECES], 24000)
    assert resp["choices"][0]["message"]["content"] == TEXT


@pytest.mark.parametrize("batched", [False, True], ids=["one-per-batch", "one-batch"])
def test_chat_stream_deltas_hold_split_characters(batched):
    chunks = [_chunk(p) for p in PIECES]
    content, events = _stream_chat([chunks] if batched else [[c] for c in chunks])
    assert content == TEXT
    assert "\ufffd" not in json.dumps(events, ensure_ascii=False)
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("batched", [False, True], ids=["one-per-batch", "one-batch"])
def test_chat_stream_sends_no_empty_deltas(batched):
    # the first piece completes no character, and an end token decodes to nothing
    chunks = [_chunk(p) for p in PIECES] + [_chunk(b"")]
    content, events = _stream_chat([chunks] if batched else [[c] for c in chunks])
    assert content == TEXT
    deltas = [e["choices"][0]["delta"] for e in events]
    assert deltas == [{"role": "assistant", "content": "中"}, {"content": "文 "}, {"content": "😀"}, {}]


def test_chat_stream_keeps_text_whole_across_an_audio_chunk():
    chunks = [_chunk(PIECES[0]), _chunk(b"\x00\x00", "audio"), *[_chunk(p) for p in PIECES[1:]]]
    content, _ = _stream_chat([[c] for c in chunks])
    assert content == TEXT


def test_chat_stream_flushes_an_unfinished_character_before_stop():
    content, events = _stream_chat([[_chunk(p) for p in TRUNCATED]])
    assert content == "中文 \ufffd"
    assert events[-1]["choices"][0]["delta"] == {"content": "\ufffd"}
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


# ----------------------------------------------------------------------
# /v1/realtime
# ----------------------------------------------------------------------

def test_realtime_step_joins_text_before_decoding(tmp_path):
    pytest.importorskip("fastapi")
    from mstar.api_server.openai.adapters import SubmitArgs
    from mstar.api_server.openai.protocol import TranscriptionRequest
    from mstar.api_server.openai.serving_realtime import RealtimeSession, _run_step

    class API:
        upload_dir = str(tmp_path)

        def submit_request(self, **kwargs):
            pass

        async def collect_results(self, request_id):
            return [_chunk(p) for p in PIECES]

    adapter = SimpleNamespace(realtime_step_request=lambda req, path, prefix: SubmitArgs(
        text="", file_paths={"audio": [path]}, input_modalities=["audio"],
        output_modalities=["text"], model_kwargs={},
    ))
    session = RealtimeSession(session_id="s", request=TranscriptionRequest(model="m"))
    assert asyncio.run(_run_step(API(), adapter, session, "prefix ")) == "prefix " + TEXT


# ----------------------------------------------------------------------
# Python SDK
# ----------------------------------------------------------------------

def _ndjson_response(chunks):
    requests = pytest.importorskip("requests")
    body = "".join(
        json.dumps({"modality": m, "data": base64.b64encode(d).decode(), "metadata": {"i": i}}) + "\n"
        for i, (m, d) in enumerate(chunks)
    ).encode()
    resp = requests.Response()
    resp.status_code = 200
    resp.headers["Content-Type"] = "application/x-ndjson"
    resp.raw = io.BytesIO(body)
    return resp


def _sdk_stream(resp, client=None):
    from mstar.client import MStarClient

    client = client or MStarClient("http://x")
    ctx = mock.MagicMock()
    ctx.__enter__.return_value = resp
    with mock.patch.object(client._session, "post", return_value=ctx):
        return list(client._stream("http://x/generate", {}, None))


def test_sdk_result_joins_text_before_decoding():
    pytest.importorskip("requests")
    from mstar.client import MStarClient

    payload = {"outputs": {"text": [{"data": base64.b64encode(p).decode()} for p in PIECES]}}
    assert MStarClient._parse_result(payload).text == TEXT


def test_sdk_stream_text_chunks_never_end_mid_character():
    events = _sdk_stream(_ndjson_response([("text", p) for p in PIECES]))
    assert [e.text for e in events] == ["", "中", "文 ", "😀"]
    assert [e.metadata for e in events] == [{"i": i} for i in range(4)]


def test_sdk_stream_flushes_an_unfinished_character_at_the_end():
    events = _sdk_stream(_ndjson_response([("text", p) for p in TRUNCATED]))
    assert "".join(e.text for e in events) == "中文 \ufffd"
    assert events[-1].text == "\ufffd"


def test_sdk_binary_stream_text_chunks_never_end_mid_character():
    pytest.importorskip("requests")
    from mstar.api_server.entrypoint import _chunk_to_binary_frame
    from mstar.api_server.request_types import ResultChunk
    from mstar.client import MStarClient
    from mstar.client.media import BINARY_STREAM_MEDIA_TYPE

    body = b"".join(
        b"".join(_chunk_to_binary_frame(ResultChunk(request_id="r", modality="text", data=p, metadata={})))
        for p in PIECES
    )
    resp = _ndjson_response([])
    resp.headers["Content-Type"] = BINARY_STREAM_MEDIA_TYPE
    resp.raw = io.BytesIO(body)
    events = _sdk_stream(resp, MStarClient("http://x", prefer_binary=True))
    assert "".join(e.text for e in events) == TEXT
    assert all("\ufffd" not in e.text for e in events)


# ----------------------------------------------------------------------
# Dynamo bridge
# ----------------------------------------------------------------------

class _DynamoServer:
    def __init__(self, chunks, upload_dir):
        self.chunks = chunks
        self.upload_dir = upload_dir

    def submit_request(self, **kwargs):
        pass

    async def iter_result_chunks(self, request_id):
        for c in self.chunks:
            yield c

    def abort_request(self, request_id):
        pass


class _Ctx:
    def is_stopped(self):
        return False


def _dynamo_chat(chunks, tmp_path):
    pytest.importorskip("fastapi")
    from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
    from mstar.integrations.dynamo.bridges import RequestBridge

    bridge = RequestBridge(_DynamoServer(chunks, tmp_path), ADAPTER_REGISTRY["qwen3_omni"], "q3o")

    async def run():
        return [o async for o in bridge.generate({"messages": [{"role": "user", "content": "hi"}]}, _Ctx())]

    out = asyncio.run(run())
    return "".join(o["choices"][0]["delta"].get("content") or "" for o in out), out


def test_dynamo_chat_deltas_hold_split_characters(tmp_path):
    content, out = _dynamo_chat([_chunk(p) for p in PIECES], tmp_path)
    assert content == TEXT
    assert all("\ufffd" not in (o["choices"][0]["delta"].get("content") or "") for o in out)
    assert out[-1]["choices"][0]["finish_reason"] == "stop"


def test_dynamo_chat_sends_no_empty_deltas(tmp_path):
    content, out = _dynamo_chat([_chunk(p) for p in PIECES] + [_chunk(b"")], tmp_path)
    assert content == TEXT
    assert [o["choices"][0]["delta"]["content"] for o in out[:-1]] == ["中", "文 ", "😀"]
    assert out[-1]["choices"][0]["finish_reason"] == "stop"


def test_dynamo_chat_flushes_an_unfinished_character_before_stop(tmp_path):
    content, out = _dynamo_chat([_chunk(p) for p in TRUNCATED], tmp_path)
    assert content == "中文 \ufffd"
    assert out[-2]["choices"][0]["delta"]["content"] == "\ufffd"


def _dynamo_realtime_text(chunks, tmp_path):
    pytest.importorskip("fastapi")
    from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
    from mstar.integrations.dynamo.bridges import RealtimeBridge

    bridge = RealtimeBridge(_DynamoServer(chunks, tmp_path), ADAPTER_REGISTRY["qwen3_omni"], "q3o")
    events = [
        {"type": "session.update", "session": {"type": "realtime", "model": "q3o", "output_modalities": ["text"]}},
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x00\x10" * 200).decode("ascii")},
        {"type": "input_audio_buffer.commit"},
    ]

    async def client():
        for e in events:
            yield e

    async def run():
        return [e async for e in bridge.generate(client(), _Ctx())]

    out = asyncio.run(run())
    deltas = [e["delta"] for e in out if e["type"] == "response.output_text.delta"]
    done = next(e["text"] for e in out if e["type"] == "response.output_text.done")
    return deltas, done


def test_dynamo_realtime_deltas_hold_split_characters(tmp_path):
    deltas, done = _dynamo_realtime_text([_chunk(p) for p in PIECES], tmp_path)
    assert deltas == ["中", "文 ", "😀"]
    assert done == TEXT


def test_dynamo_realtime_flushes_an_unfinished_character(tmp_path):
    deltas, done = _dynamo_realtime_text([_chunk(p) for p in TRUNCATED], tmp_path)
    assert deltas == ["中", "文 ", "\ufffd"]
    assert done == "中文 \ufffd"

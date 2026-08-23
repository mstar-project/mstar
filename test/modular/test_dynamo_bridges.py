"""Unit tests for the Dynamo request bridges (body translation and the
realtime event translation against a seam double — no server, no Dynamo
runtime). The registration mapping tests skip when the ai-dynamo bindings
aren't installed."""

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
from mstar.api_server.openai.protocol import SpeechRequest, VideoGenerationRequest
from mstar.integrations.dynamo.bridges import (
    RealtimeBridge,
    _clean,
    _image_body,
    _speech_body,
    _video_body,
)

PNG_1X1 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlE"
    "QVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_clean_strips_bookkeeping_keeps_knobs():
    body = _clean({
        "messages": [], "top_k": 5, "repetition_penalty": 1.1,
        "routing": {}, "nvext": {"seed": 1}, "stop_conditions": {}, "user": "u",
    })
    assert "messages" in body and body["top_k"] == 5 and body["repetition_penalty"] == 1.1
    assert not {"routing", "nvext", "stop_conditions", "user"} & body.keys()


def test_image_body_flattens_nvext_and_rejects_url():
    body = _image_body({
        "prompt": "x", "model": "m", "size": "512x512",
        "nvext": {"seed": 3, "guidance_scale": 5.0},
    })
    assert body["seed"] == 3 and body["guidance_scale"] == 5.0 and body["size"] == "512x512"
    with pytest.raises(ValueError, match="media store"):
        _image_body({"prompt": "x", "response_format": "url"})


def test_video_body_derives_num_frames_from_seconds():
    body = _video_body({
        "prompt": "x", "model": "m", "seconds": 2, "nvext": {"fps": 16},
    })
    assert body["num_frames"] == 32 and body["fps"] == 16
    assert "seconds" not in body


def test_video_body_explicit_num_frames_wins():
    body = _video_body({
        "prompt": "x", "seconds": 10, "nvext": {"fps": 16, "num_frames": 8},
    })
    assert body["num_frames"] == 8


def test_video_body_reference_and_no_leakage():
    body = _video_body({
        "prompt": "x", "input_reference": PNG_1X1, "stream": False,
        "output_format": "mp4", "user": "u",
    })
    assert body["image"] == PNG_1X1
    assert not {"input_reference", "stream", "output_format", "user"} & body.keys()
    with pytest.raises(ValueError, match="media store"):
        _video_body({"prompt": "x", "response_format": "url"})
    with pytest.raises(ValueError, match="output_format"):
        _video_body({"prompt": "x", "output_format": "mjpeg"})


def test_video_body_through_cosmos3_adapter():
    body = _video_body({
        "prompt": "a fox", "model": "m", "size": "832x480", "seconds": 2,
        "nvext": {"fps": 16, "num_inference_steps": 20, "seed": 7},
    })
    req = VideoGenerationRequest.model_validate(body)
    args = ADAPTER_REGISTRY["cosmos3"].video_to_request(req, None)
    assert args.output_modalities == ["video"]
    assert args.model_kwargs["num_frames"] == 32
    assert args.model_kwargs["size"] == "832x480"
    assert args.model_kwargs["num_inference_steps"] == 20


def test_video_body_through_wan22_adapter_splits_size():
    body = _video_body({"prompt": "x", "size": "832x480", "nvext": {"num_frames": 49}})
    req = VideoGenerationRequest.model_validate(body)
    args = ADAPTER_REGISTRY["wan22"].video_to_request(req, None)
    assert args.model_kwargs["width"] == 832 and args.model_kwargs["height"] == 480


def test_speech_body_codec_survives_and_url_rejected():
    body = _speech_body({
        "input": "hi", "model": "m", "voice": "tara",
        "response_format": "MP3", "data_source": "b64_json",
    })
    assert body["response_format"] == "mp3" and "data_source" not in body
    req = SpeechRequest.model_validate(body)
    args = ADAPTER_REGISTRY["orpheus"].speech_to_request(req, None)
    assert args.text == "hi" and args.model_kwargs["voice"] == "tara"
    with pytest.raises(ValueError, match="media store"):
        _speech_body({"input": "x", "data_source": "url"})


def test_model_type_mapping():
    pytest.importorskip("dynamo.llm")
    from mstar.api_server.openai.adapters import get_adapter
    from mstar.integrations.dynamo.worker import _model_type

    # The pyo3 ModelType has no __eq__; compare the flag-set string form.
    assert str(_model_type(get_adapter("bagel"))) == "chat,images"
    assert str(_model_type(get_adapter("orpheus"))) == "audios"
    assert str(_model_type(get_adapter("qwen3_omni"))) == "chat,audios"
    # cosmos3's video surface registers on its own endpoint, not in this mask
    assert str(_model_type(get_adapter("cosmos3"))) == "images"
    assert get_adapter("cosmos3").supports_videos


def test_realtime_surface_flags():
    # Realtime registers on its own bidirectional endpoint, gated by the flag.
    assert ADAPTER_REGISTRY["qwen3_omni"].supports_realtime
    assert not ADAPTER_REGISTRY["orpheus"].supports_realtime
    assert not ADAPTER_REGISTRY["bagel"].supports_realtime


# ----------------------------------------------------------------------
# realtime bridge (seam double; no engine)
# ----------------------------------------------------------------------

PCM = b"\x00\x10" * 200  # 200 PCM16 samples


class _Ctx:
    def __init__(self):
        self.stopped = False

    def is_stopped(self):
        return self.stopped


class _FakeRealtimeServer:
    """Seam double: records submits, streams scripted chunks, records aborts.
    Optionally flips ``stop_ctx.stopped`` after ``stop_after`` chunks."""

    def __init__(self, chunks, upload_dir, stop_ctx=None, stop_after=None):
        self.chunks = chunks
        self.upload_dir = Path(upload_dir)
        self.submitted = []
        self.aborted = []
        self._stop_ctx = stop_ctx
        self._stop_after = stop_after

    def submit_request(self, **kwargs):
        self.submitted.append(kwargs)
        return kwargs.get("request_id")

    async def iter_result_chunks(self, request_id):
        for i, chunk in enumerate(self.chunks):
            yield chunk
            if self._stop_ctx is not None and i + 1 == self._stop_after:
                self._stop_ctx.stopped = True

    def abort_request(self, request_id):
        self.aborted.append(request_id)


async def _client_events(items):
    for item in items:
        yield item


def _run_connection(server, events):
    bridge = RealtimeBridge(server, ADAPTER_REGISTRY["qwen3_omni"], "q3o")
    ctx = server._stop_ctx or _Ctx()

    async def collect():
        out = []
        async for event in bridge.generate(_client_events(events), ctx):
            out.append(event)
        return out

    return asyncio.run(collect())


def _chunk(modality, data):
    return SimpleNamespace(modality=modality, data=data, metadata={})


def test_realtime_full_turn(tmp_path):
    server = _FakeRealtimeServer([
        _chunk("text", b"hello"),
        _chunk("audio", b"\x01\x02\x03\x04"),
        _chunk("text", b"<|im_end|>"),  # bare end token: stripped, no delta
    ], tmp_path)
    out = _run_connection(server, [
        {"type": "session.update", "session": {
            "type": "realtime", "model": "q3o",
            "audio": {"output": {"voice": "cherry"}},
        }},
        {"type": "input_audio_buffer.append",
         "audio": base64.b64encode(PCM).decode("ascii")},
        {"type": "input_audio_buffer.commit"},
    ])

    assert [e["type"] for e in out] == [
        "session.updated",
        "input_audio_buffer.committed",
        "response.created",
        "response.output_audio_transcript.delta",
        "response.output_audio.delta",
        "response.output_audio.done",
        "response.done",
    ]
    audio_delta = next(e for e in out if e["type"] == "response.output_audio.delta")
    assert base64.b64decode(audio_delta["delta"]) == b"\x01\x02\x03\x04"
    assert out[-1]["response"]["status"] == "completed"
    # Every content event carries the same response/item ids.
    rids = {e["response_id"] for e in out if "response_id" in e}
    assert rids == {out[-1]["response"]["id"]}

    # The submit is chat-shaped: the committed buffer landed as a WAV file,
    # speech output was requested, and the session voice mapped through.
    sub = server.submitted[0]
    assert sub["input_modalities"] == ["audio"]
    assert sub["output_modalities"] == ["text", "audio"]
    assert sub["model_kwargs"]["voice"] == "cherry"
    wav = Path(sub["file_paths"]["audio"][0]).read_bytes()
    assert wav[:4] == b"RIFF" and PCM in wav
    assert server.aborted  # release runs even on clean completion


def test_realtime_empty_commit_errors(tmp_path):
    server = _FakeRealtimeServer([], tmp_path)
    out = _run_connection(server, [
        {"type": "session.update", "session": {"type": "realtime", "model": "q3o"}},
        {"type": "input_audio_buffer.commit"},
    ])
    assert [e["type"] for e in out] == ["session.updated", "error"]
    assert out[1]["error"]["code"] == "input_audio_buffer_commit_empty"
    assert not server.submitted


def test_realtime_text_only_session(tmp_path):
    server = _FakeRealtimeServer([
        _chunk("text", b"a"),
        _chunk("text", b"b"),
    ], tmp_path)
    out = _run_connection(server, [
        {"type": "session.update", "session": {
            "type": "realtime", "model": "q3o", "output_modalities": ["text"],
        }},
        {"type": "input_audio_buffer.append",
         "audio": base64.b64encode(PCM).decode("ascii")},
        {"type": "input_audio_buffer.commit"},
    ])
    types = [e["type"] for e in out]
    assert types == [
        "session.updated",
        "input_audio_buffer.committed",
        "response.created",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.done",
    ]
    assert next(e for e in out if e["type"] == "response.output_text.done")["text"] == "ab"
    assert out[-1]["response"]["output_modalities"] == ["text"]
    assert server.submitted[0]["output_modalities"] == ["text"]


def test_realtime_bad_input_is_survivable(tmp_path):
    server = _FakeRealtimeServer([], tmp_path)
    out = _run_connection(server, [
        # Non-PCM input format: error event, session stays usable.
        {"type": "session.update", "session": {
            "type": "realtime", "model": "q3o",
            "audio": {"input": {"format": {"type": "audio/pcmu"}}},
        }},
        # Odd-length payload: dropped, so the buffer stays empty.
        {"type": "input_audio_buffer.append",
         "audio": base64.b64encode(b"\x00\x10\x20").decode("ascii")},
        {"type": "input_audio_buffer.commit"},
    ])
    types = [e["type"] for e in out]
    assert types == ["error", "session.updated", "error"]
    assert out[0]["error"]["code"] == "unsupported_audio_format"
    assert out[2]["error"]["code"] == "input_audio_buffer_commit_empty"
    assert not server.submitted


def test_realtime_stop_mid_turn(tmp_path):
    ctx = _Ctx()
    server = _FakeRealtimeServer([
        _chunk("text", b"partial"),
        _chunk("audio", b"\x01\x02"),
    ], tmp_path, stop_ctx=ctx, stop_after=1)
    out = _run_connection(server, [
        {"type": "session.update", "session": {"type": "realtime", "model": "q3o"}},
        {"type": "input_audio_buffer.append",
         "audio": base64.b64encode(PCM).decode("ascii")},
        {"type": "input_audio_buffer.commit"},
    ])
    types = [e["type"] for e in out]
    # The first chunk made it out; the turn then observed the stop and
    # claimed no completed response.
    assert "response.output_audio_transcript.delta" in types
    assert "response.done" not in types
    assert server.aborted

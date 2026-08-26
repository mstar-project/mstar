"""Unit tests for the Dynamo request bridges (body translation and the
realtime event translation against a seam double — no server, no Dynamo
runtime). The registration mapping tests skip when the ai-dynamo bindings
aren't installed."""

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
from mstar.api_server.openai.protocol import SpeechRequest, VideoGenerationRequest
from mstar.integrations.dynamo.bridges import (
    TENSOR_SPECS,
    RealtimeBridge,
    TensorBridge,
    _clean,
    _image_body,
    _speech_body,
    _video_body,
    get_tensor_spec,
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


def test_clean_keeps_ignore_eos_flowing_to_model_kwargs():
    # Models read ignore_eos from model_kwargs (benchmark parity), so it
    # must survive cleaning and reach model_kwargs through the adapter.
    from mstar.api_server.openai.protocol import ChatCompletionRequest

    body = _clean({"messages": [], "ignore_eos": True, "min_tokens": 5})
    assert body["ignore_eos"] is True and "min_tokens" not in body

    req = ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], "ignore_eos": True}
    )
    args = ADAPTER_REGISTRY["qwen3_omni"].chat_to_request(req, None)
    assert args.model_kwargs["ignore_eos"] is True


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


# ----------------------------------------------------------------------
# tensor bridge (seam double; no engine)
# ----------------------------------------------------------------------

def _tensor(name, data_type, shape, values):
    return {
        "metadata": {"name": name, "data_type": data_type, "shape": shape},
        "data": {"data_type": data_type, "values": values},
    }


def _pi05_frames(cameras=2, h=4, w=5):
    n = cameras * h * w * 3
    return (np.arange(n, dtype=np.int64) % 251).astype(np.uint8).reshape(
        cameras, h, w, 3
    )


def _pi05_request(frames, prompt=b"pick up the block", state=None, req_id="req-1"):
    tensors = [
        _tensor("images", "Uint8", list(frames.shape), frames.flatten().tolist()),
        _tensor("prompt", "Bytes", [1], [list(prompt)]),
    ]
    if state is not None:
        tensors.append(_tensor("state", "Float32", [len(state)], state))
    return {"id": req_id, "model": "pi05", "parameters": {}, "tensors": tensors}


def _run_tensor(server, request, ctx=None):
    bridge = TensorBridge(server, TENSOR_SPECS["pi05"], "pi05")
    ctx = ctx or server._stop_ctx or _Ctx()

    async def collect():
        out = []
        async for response in bridge.generate(request, ctx):
            out.append(response)
        return out

    return asyncio.run(collect())


def test_tensor_spec_registry_and_config():
    assert get_tensor_spec("pi05") is TENSOR_SPECS["pi05"]
    assert get_tensor_spec("qwen3_omni") is None
    config = TENSOR_SPECS["pi05"].model_config("served-name")
    assert config["name"] == "served-name"
    assert [(t["name"], t["data_type"], t["shape"]) for t in config["inputs"]] == [
        ("images", "Uint8", [-1, -1, -1, 3]),
        ("state", "Float32", [-1]),
        ("prompt", "Bytes", [1]),
    ]
    assert config["outputs"] == [
        {"name": "actions", "data_type": "Float32", "shape": [-1, 32]},
    ]


def test_tensor_full_request(tmp_path):
    frames = _pi05_frames()
    state = [float(i) / 10 for i in range(32)]
    actions = (np.arange(50 * 32, dtype="<f4") / 100).reshape(50, 32)
    server = _FakeRealtimeServer([_chunk("action", actions.tobytes())], tmp_path)

    out = _run_tensor(server, _pi05_request(frames, state=state))

    assert len(out) == 1
    assert out[0]["id"] == "req-1" and out[0]["model"] == "pi05"
    (result,) = out[0]["tensors"]
    assert result["metadata"] == {
        "name": "actions", "data_type": "Float32", "shape": [50, 32],
    }
    got = np.array(result["data"]["values"], dtype="<f4").reshape(50, 32)
    assert np.array_equal(got, actions)

    sub = server.submitted[0]
    assert sub["text"] == "pick up the block"
    assert sub["input_modalities"] == ["image", "text"]
    assert sub["output_modalities"] == ["action"]
    assert sub["model_kwargs"]["robot_state"] == state
    paths = sub["file_paths"]["image"]
    assert len(paths) == 2
    from PIL import Image
    for i, path in enumerate(paths):
        assert np.array_equal(np.asarray(Image.open(path)), frames[i])
    assert server.aborted


def test_tensor_state_is_optional(tmp_path):
    server = _FakeRealtimeServer(
        [_chunk("action", np.zeros(32, dtype="<f4").tobytes())], tmp_path
    )
    out = _run_tensor(server, _pi05_request(_pi05_frames(cameras=1)))
    assert out[0]["tensors"][0]["metadata"]["shape"] == [1, 32]
    assert "robot_state" not in server.submitted[0]["model_kwargs"]


def test_tensor_prompt_as_bytes(tmp_path):
    # The request-plane codec may deliver Bytes elements as byte strings
    # instead of int lists; both must decode.
    server = _FakeRealtimeServer(
        [_chunk("action", np.zeros(32, dtype="<f4").tobytes())], tmp_path
    )
    request = _pi05_request(_pi05_frames(cameras=1))
    request["tensors"][1]["data"]["values"] = [b"open the drawer"]
    _run_tensor(server, request)
    assert server.submitted[0]["text"] == "open the drawer"


def test_tensor_missing_inputs_error(tmp_path):
    server = _FakeRealtimeServer([], tmp_path)
    request = {"id": "r", "model": "pi05", "tensors": [
        _tensor("prompt", "Bytes", [1], [list(b"x")]),
    ]}
    with pytest.raises(ValueError, match="images"):
        _run_tensor(server, request)
    assert not server.submitted and not server.aborted


def test_tensor_bad_images_shape_error(tmp_path):
    server = _FakeRealtimeServer([], tmp_path)
    request = _pi05_request(_pi05_frames(cameras=1))
    request["tensors"][0]["metadata"]["shape"] = [4, 5, 3]  # missing camera dim
    with pytest.raises(ValueError, match="cameras"):
        _run_tensor(server, request)
    assert not server.submitted


def test_tensor_error_chunk_raises(tmp_path):
    server = _FakeRealtimeServer([_chunk("error", b"engine exploded")], tmp_path)
    with pytest.raises(RuntimeError, match="engine exploded"):
        _run_tensor(server, _pi05_request(_pi05_frames(cameras=1)))
    assert server.aborted


def test_tensor_misaligned_action_bytes(tmp_path):
    server = _FakeRealtimeServer([_chunk("action", b"\x00" * 4)], tmp_path)
    with pytest.raises(ValueError, match="multiple of 32"):
        _run_tensor(server, _pi05_request(_pi05_frames(cameras=1)))
    assert server.aborted


def test_tensor_no_output_raises(tmp_path):
    server = _FakeRealtimeServer([_chunk("text", b"stray")], tmp_path)
    with pytest.raises(RuntimeError, match="no tensor output"):
        _run_tensor(server, _pi05_request(_pi05_frames(cameras=1)))
    assert server.aborted


def test_tensor_stop_before_output(tmp_path):
    ctx = _Ctx()
    server = _FakeRealtimeServer([
        _chunk("text", b"ignored"),
        _chunk("action", np.zeros(32, dtype="<f4").tobytes()),
    ], tmp_path, stop_ctx=ctx, stop_after=1)
    out = _run_tensor(server, _pi05_request(_pi05_frames(cameras=1)))
    # Stopped after the first (non-action) chunk: no response, no
    # "no output" error — a cancel is not a failure — and the abort ran.
    assert out == []
    assert server.aborted


# ----------------------------------------------------------------------
# tensor bridge: vjepa2_ac (seam double; no engine)
# ----------------------------------------------------------------------

VJEPA2_D = 1408


def _vjepa2_trajectories(rows=4, dim=7):
    # Integer-valued floats survive the float32 round-trip exactly.
    actions = [[float(i * 10 + j) for j in range(dim)] for i in range(rows)]
    states = [[float(i * 10 + j + 100) for j in range(dim)] for i in range(rows)]
    return actions, states


def _vjepa2_request(video=b"ftyp-not-a-real-mp4", actions=None, states=None,
                    parameters=None, req_id="vj-1"):
    default_a, default_s = _vjepa2_trajectories()
    actions = actions if actions is not None else default_a
    states = states if states is not None else default_s
    a = np.asarray(actions, dtype="<f4")
    s = np.asarray(states, dtype="<f4")
    tensors = [
        _tensor("video", "Bytes", [1], [list(video)]),
        _tensor("actions", "Float32", list(a.shape), a.flatten().tolist()),
        _tensor("states", "Float32", list(s.shape), s.flatten().tolist()),
    ]
    return {"id": req_id, "model": "vjepa2_ac",
            "parameters": parameters or {}, "tensors": tensors}


def _run_vjepa2(server, request, ctx=None):
    bridge = TensorBridge(server, TENSOR_SPECS["vjepa2_ac"], "vjepa2_ac")
    ctx = ctx or server._stop_ctx or _Ctx()

    async def collect():
        out = []
        async for response in bridge.generate(request, ctx):
            out.append(response)
        return out

    return asyncio.run(collect())


def test_vjepa2_spec_registry_and_config():
    assert get_tensor_spec("vjepa2_ac") is TENSOR_SPECS["vjepa2_ac"]
    config = TENSOR_SPECS["vjepa2_ac"].model_config("served-name")
    assert config["name"] == "served-name"
    assert [(t["name"], t["data_type"], t["shape"]) for t in config["inputs"]] == [
        ("video", "Bytes", [1]),
        ("actions", "Float32", [-1, -1]),
        ("states", "Float32", [-1, -1]),
    ]
    assert config["outputs"] == [
        {"name": "latents", "data_type": "Float32", "shape": [-1, VJEPA2_D]},
    ]


def test_vjepa2_full_request(tmp_path):
    latents = (np.arange(6 * VJEPA2_D, dtype="<f4") / 1000).reshape(6, VJEPA2_D)
    server = _FakeRealtimeServer([_chunk("video", latents.tobytes())], tmp_path)
    video = b"ftyp-not-a-real-mp4" * 3
    actions, states = _vjepa2_trajectories()
    request = _vjepa2_request(video=video,
                              parameters={"rollout_horizon": {"int64": 3}})

    out = _run_vjepa2(server, request)

    assert len(out) == 1
    assert out[0]["id"] == "vj-1" and out[0]["model"] == "vjepa2_ac"
    (result,) = out[0]["tensors"]
    assert result["metadata"] == {
        "name": "latents", "data_type": "Float32", "shape": [6, VJEPA2_D],
    }
    got = np.array(result["data"]["values"], dtype="<f4").reshape(6, VJEPA2_D)
    assert np.array_equal(got, latents)

    sub = server.submitted[0]
    assert sub["text"] is None
    assert sub["input_modalities"] == ["video"]
    assert sub["output_modalities"] == ["video"]
    assert sub["model_kwargs"]["actions"] == actions
    assert sub["model_kwargs"]["states"] == states
    assert sub["model_kwargs"]["rollout_horizon"] == 3
    (path,) = sub["file_paths"]["video"]
    assert path.endswith(".mp4")
    assert Path(path).read_bytes() == video
    assert Path(path).parent == Path(tmp_path)
    assert server.aborted


def test_vjepa2_batched_rollout_aggregates(tmp_path):
    # The engine emits one chunk per rollout iteration even in the batched
    # walk; without stream_rollout they must fold into ONE response.
    steps = [np.full((2, VJEPA2_D), float(i), dtype="<f4") for i in range(3)]
    server = _FakeRealtimeServer(
        [_chunk("video", s.tobytes()) for s in steps], tmp_path
    )
    out = _run_vjepa2(server, _vjepa2_request(
        parameters={"rollout_horizon": {"int64": 3}}
    ))
    assert len(out) == 1
    (result,) = out[0]["tensors"]
    assert result["metadata"]["shape"] == [6, VJEPA2_D]
    got = np.array(result["data"]["values"], dtype="<f4").reshape(3, 2, VJEPA2_D)
    for i in range(3):
        assert np.all(got[i] == float(i))


def test_vjepa2_streaming_rollout(tmp_path):
    steps = [np.full((2, VJEPA2_D), float(i), dtype="<f4") for i in range(3)]
    server = _FakeRealtimeServer(
        [_chunk("video", s.tobytes()) for s in steps], tmp_path
    )
    out = _run_vjepa2(server, _vjepa2_request(parameters={
        "rollout_horizon": {"int64": 3}, "stream_rollout": {"bool": True},
    }))
    assert len(out) == 3
    for i, response in enumerate(out):
        (result,) = response["tensors"]
        assert result["metadata"]["shape"] == [2, VJEPA2_D]
        assert result["data"]["values"][0] == float(i)
    kwargs = server.submitted[0]["model_kwargs"]
    assert kwargs["rollout_horizon"] == 3
    assert kwargs["stream_rollout"] is True


def test_vjepa2_plain_parameter_values(tmp_path):
    # Untagged scalars (tests, future HTTP route) unwrap the same way.
    server = _FakeRealtimeServer(
        [_chunk("video", np.zeros(VJEPA2_D, dtype="<f4").tobytes())], tmp_path
    )
    _run_vjepa2(server, _vjepa2_request(
        parameters={"rollout_horizon": 2, "stream_rollout": True}
    ))
    kwargs = server.submitted[0]["model_kwargs"]
    assert kwargs["rollout_horizon"] == 2
    assert kwargs["stream_rollout"] is True


def test_vjepa2_defaults_to_single_pass(tmp_path):
    server = _FakeRealtimeServer(
        [_chunk("video", np.zeros(VJEPA2_D, dtype="<f4").tobytes())], tmp_path
    )
    out = _run_vjepa2(server, _vjepa2_request())
    assert out[0]["tensors"][0]["metadata"]["shape"] == [1, VJEPA2_D]
    assert set(server.submitted[0]["model_kwargs"]) == {"actions", "states"}


def test_vjepa2_video_bytes_as_bytestring(tmp_path):
    server = _FakeRealtimeServer(
        [_chunk("video", np.zeros(VJEPA2_D, dtype="<f4").tobytes())], tmp_path
    )
    request = _vjepa2_request()
    request["tensors"][0]["data"]["values"] = [b"raw-container-bytes"]
    _run_vjepa2(server, request)
    (path,) = server.submitted[0]["file_paths"]["video"]
    assert Path(path).read_bytes() == b"raw-container-bytes"


def test_vjepa2_missing_inputs_error(tmp_path):
    server = _FakeRealtimeServer([], tmp_path)
    request = _vjepa2_request()
    request["tensors"] = request["tensors"][:2]  # drop states
    with pytest.raises(ValueError, match="states"):
        _run_vjepa2(server, request)
    assert not server.submitted and not server.aborted


def test_vjepa2_bad_trajectory_shape_error(tmp_path):
    server = _FakeRealtimeServer([], tmp_path)
    request = _vjepa2_request()
    request["tensors"][1]["metadata"]["shape"] = [28]  # flattened, not [n, d]
    with pytest.raises(ValueError, match="timesteps"):
        _run_vjepa2(server, request)
    assert not server.submitted


def test_vjepa2_trajectory_mismatch_error(tmp_path):
    server = _FakeRealtimeServer([], tmp_path)
    actions, states = _vjepa2_trajectories(rows=3)
    request = _vjepa2_request(actions=actions, states=states[:2])
    with pytest.raises(ValueError, match="share"):
        _run_vjepa2(server, request)
    assert not server.submitted


def test_vjepa2_empty_video_error(tmp_path):
    server = _FakeRealtimeServer([], tmp_path)
    with pytest.raises(ValueError, match="empty"):
        _run_vjepa2(server, _vjepa2_request(video=b""))
    assert not server.submitted


def test_vjepa2_misaligned_latent_bytes(tmp_path):
    server = _FakeRealtimeServer([_chunk("video", b"\x00" * 4)], tmp_path)
    with pytest.raises(ValueError, match="multiple of 1408"):
        _run_vjepa2(server, _vjepa2_request())
    assert server.aborted

"""Unit tests for the Dynamo request bridges (body translation only — no
server, no Dynamo runtime). The registration mapping tests skip when the
ai-dynamo bindings aren't installed."""

import pytest

from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
from mstar.api_server.openai.protocol import SpeechRequest, VideoGenerationRequest
from mstar.integrations.dynamo.bridges import (
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

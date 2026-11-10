"""The images / videos response body: ``MediaJSONResponse`` writes the base64 payloads the
handlers leave as bytes, and the body is the one ``JSONResponse`` renders for str."""

from __future__ import annotations

import asyncio
import base64
import os

import pytest
from starlette.responses import JSONResponse

from mstar.api_server.openai._util import MediaJSONResponse


def _as_str(value):
    if isinstance(value, bytes):
        return value.decode("ascii")
    if isinstance(value, dict):
        return {k: _as_str(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_str(v) for v in value]
    return value


@pytest.mark.parametrize("sizes", [[0], [1], [2], [3], [300_001], [5, 70_000, 2]])
def test_the_body_is_the_one_json_response_renders(sizes):
    blobs = [os.urandom(n) for n in sizes]
    content = {"created": 1_730_000_000, "data": [{"b64_json": base64.b64encode(b), "url": None} for b in blobs]}
    response = MediaJSONResponse(content)
    expected = JSONResponse(_as_str(content))
    assert response.body == expected.body
    assert response.headers["content-type"] == expected.headers["content-type"]
    assert response.headers["content-length"] == expected.headers["content-length"]


def test_values_other_than_bytes_render_as_json_response_renders_them():
    content = {"created": 1, "data": [], "x": {"é": [1.5, True, None, "q\"\n\\", -3]}, "": "ü"}
    assert MediaJSONResponse(content).body == JSONResponse(content).body
    with pytest.raises(ValueError):
        MediaJSONResponse({"x": float("nan")})  # JSONResponse's allow_nan=False


class _Api:
    upload_dir = "/tmp"

    def __init__(self, data: bytes):
        self.data = data

    def submit_request(self, **kwargs):
        pass

    async def collect_results(self, request_id, raw_request=None):
        from mstar.api_server.request_types import ResultChunk

        return [ResultChunk(request_id=request_id, modality="video", data=self.data, metadata={})]


class _Adapter:
    def video_to_request(self, req, upload_dir):
        from types import SimpleNamespace

        return SimpleNamespace(text="x", file_paths={}, input_modalities=["text"], output_modalities=["video"],
                               model_kwargs={})


def test_a_video_handler_result_renders_to_the_old_body():
    from mstar.api_server.openai import serving_videos

    mp4 = os.urandom(123_457)
    from mstar.api_server.openai.protocol import VideoGenerationRequest

    req = VideoGenerationRequest(model="m", prompt="x")
    result = asyncio.run(serving_videos.create_videos(_Api(mp4), "m", _Adapter(), req))
    assert base64.b64decode(result["data"][0]["b64_json"]) == mp4
    old = {"created": result["created"], "data": [{"b64_json": base64.b64encode(mp4).decode("ascii"), "url": None}]}
    assert MediaJSONResponse(result).body == JSONResponse(old).body

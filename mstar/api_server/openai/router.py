"""FastAPI routes for the OpenAI-compatible API.

Endpoints stay model-agnostic: each looks up the loaded model's adapter, checks
the surface is supported, and delegates to a serving handler. The native
``/generate`` endpoint is unaffected.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mstar.api_server.openai import serving_chat, serving_images, serving_speech
from mstar.api_server.openai._util import now
from mstar.api_server.openai.adapters import get_adapter
from mstar.api_server.openai.protocol import (
    ChatCompletionRequest,
    ImageGenerationRequest,
    ModelCard,
    ModelList,
    SpeechRequest,
)

router = APIRouter()


def _api():
    # The server runs either as the package module (console scripts ``mstar`` /
    # ``mstar-serve``) or as ``__main__`` (``python mstar/api_server/entrypoint.py``,
    # used by the test/*/launch_server_*.sh scripts). main() sets ``api_server``
    # on whichever module it runs in, so resolve the live instance from both —
    # via sys.modules (importing would re-execute the entrypoint module).
    import sys

    for name in ("mstar.api_server.entrypoint", "__main__"):
        mod = sys.modules.get(name)
        api = getattr(mod, "api_server", None) if mod is not None else None
        if api is not None:
            return api
    return None


def _error(status: int, message: str, type_: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "code": status}},
    )


def _resolve(require: str):
    """Return (api, model_name, adapter, error_response). ``error`` is non-None
    when the loaded model can't serve ``require`` (e.g. 'supports_chat')."""
    api = _api()
    if api is None:
        return None, None, None, _error(503, "Server not ready", "server_error")
    adapter = get_adapter(api.model_name)
    if adapter is None:
        return api, api.model_name, None, _error(
            404, f"Model {api.model_name!r} has no OpenAI-compatible adapter; use POST /generate", "model_not_found"
        )
    if not getattr(adapter, require, False):
        return api, api.model_name, adapter, _error(
            404, f"Model {api.model_name!r} does not support this endpoint"
        )
    return api, api.model_name, adapter, None


@router.get("/v1/models")
async def list_models():
    api = _api()
    name = api.model_name if api is not None else "unknown"
    return JSONResponse(ModelList(data=[ModelCard(id=name, created=now())]).model_dump())


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    api, model_name, adapter, err = _resolve("supports_chat")
    if err is not None:
        return err
    try:
        result = await serving_chat.create_chat_completion(api, model_name, adapter, request)
    except Exception as e:  # noqa: BLE001 — surface as an OpenAI error envelope
        return _error(getattr(e, "status_code", 500), str(getattr(e, "detail", e)), "server_error")
    if request.stream:
        return StreamingResponse(
            result, media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )
    return JSONResponse(result)


@router.post("/v1/audio/speech")
async def audio_speech(request: SpeechRequest):
    api, model_name, adapter, err = _resolve("supports_speech")
    if err is not None:
        return err
    try:
        return await serving_speech.create_speech(api, model_name, adapter, request)
    except Exception as e:  # noqa: BLE001
        return _error(getattr(e, "status_code", 500), str(getattr(e, "detail", e)), "server_error")


@router.post("/v1/images/generations")
async def images_generations(request: ImageGenerationRequest):
    api, model_name, adapter, err = _resolve("supports_images")
    if err is not None:
        return err
    try:
        result = await serving_images.create_images(api, model_name, adapter, request)
    except Exception as e:  # noqa: BLE001
        return _error(getattr(e, "status_code", 500), str(getattr(e, "detail", e)), "server_error")
    return JSONResponse(result)


@router.post("/v1/images/edits")
async def images_edits(request: Request):
    # Multipart (image file + prompt + passthrough fields), parsed manually so
    # arbitrary model knobs (e.g. cfg_*_scale) flow through as model_kwargs.
    api, model_name, adapter, err = _resolve("supports_images")
    if err is not None:
        return err
    try:
        form = await request.form()
        image = form.get("image")
        if image is None or not hasattr(image, "read"):
            return _error(400, "images/edits requires an 'image' file upload")
        image_bytes = await image.read()
        prompt = form.get("prompt") or ""
        known = {"image", "prompt", "model", "n", "size", "response_format"}
        extra: dict = {}
        for key, value in form.multi_items():
            if key in known or hasattr(value, "filename"):
                continue
            try:
                extra[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                extra[key] = value
        result = await serving_images.create_image_edit(
            api,
            model_name,
            adapter,
            prompt=prompt,
            image_bytes=image_bytes,
            image_filename=getattr(image, "filename", None),
            model_kwargs=extra,
        )
    except Exception as e:  # noqa: BLE001
        return _error(getattr(e, "status_code", 500), str(getattr(e, "detail", e)), "server_error")
    return JSONResponse(result)

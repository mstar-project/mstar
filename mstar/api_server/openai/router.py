"""FastAPI routes for the OpenAI-compatible API.

Endpoints stay model-agnostic: each looks up the loaded model's adapter, checks
the surface is supported, and delegates to a serving handler. The native
``/generate`` endpoint is unaffected.
"""

from __future__ import annotations

import json
import logging
import math

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse

from mstar.api_server.body_limit import max_body_bytes
from mstar.api_server.openai import (
    serving_chat,
    serving_images,
    serving_messages,
    serving_realtime,
    serving_responses,
    serving_speech,
    serving_transcriptions,
    serving_videos,
)
from mstar.api_server.openai._util import MediaJSONResponse, check_inline_result, error_type, now
from mstar.api_server.openai.adapters import get_adapter
from mstar.api_server.openai.protocol import (
    ChatCompletionRequest,
    ImageGenerationRequest,
    LoadLoraAdapterRequest,
    ModelCard,
    ModelList,
    SpeechRequest,
    TranscriptionRequest,
    UnloadLoraAdapterRequest,
    VideoGenerationRequest,
)
from mstar.utils.client_errors import exception_status_text

logger = logging.getLogger(__name__)

router = APIRouter()

# a transcription form's declared text fields stay text: a prompt of "2024" is not a number
_TRANSCRIPTION_TEXT_FIELDS = frozenset(
    name for name, f in TranscriptionRequest.model_fields.items() if f.annotation in (str, str | None)
)


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


def _error(status: int, message: str, type_: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_ or error_type(status), "code": status}},
    )


class RouteError(Exception):
    """Why a route refuses a request, before any protocol shapes it: ``_error``
    writes OpenAI's envelope and ``_anthropic_error`` Anthropic's. ``type`` is
    OpenAI's error type where it is not the status's own."""

    def __init__(self, status: int, message: str, type_: str | None = None):
        super().__init__(message)
        self.status, self.message, self.type = status, message, type_


def _exception_error(e: Exception, route: str) -> JSONResponse:
    """OpenAI's error envelope for an exception a route's handler raised (:func:`_route_error`)."""
    err = _route_error(e, route)
    return _error(err.status, err.message, err.type)


def _route_error(e: Exception, route: str) -> RouteError:
    """The status and message for an exception a route's handler raised.

    An ``HTTPException`` was raised on purpose and keeps its status and
    detail. A ``ValueError`` is a refusal: a 400 with server paths cut to
    base names, logged as one INFO line, since the client reads why (a
    client that sends ``stop`` on every call would otherwise log a traceback
    per request). Anything else goes through :func:`client_error_text`: a
    ``TypeError`` is a 400 too, but may be a bug in the route, so it keeps its
    traceback; any other error is a 500 that shows the client only an error
    id, with the full error logged under it. An exception class that names
    its own ``status_code`` (``AudioTooLongError`` is a 413) keeps that status.
    """
    return RouteError(*exception_status_text(e, route, log=logger))


def _serving(require: str):
    """(api, model_name, adapter) of the loaded model, or a :class:`RouteError`
    when it can't serve ``require`` (e.g. 'supports_chat')."""
    api = _api()
    if api is None:
        raise RouteError(503, "Server not ready")
    adapter = get_adapter(api.model_name)
    if adapter is None:
        raise RouteError(
            404, f"Model {api.model_name!r} has no OpenAI-compatible adapter; use POST /generate", "model_not_found"
        )
    if not getattr(adapter, require, False):
        raise RouteError(404, f"Model {api.model_name!r} does not support this endpoint")
    return api, api.model_name, adapter


def _resolve(require: str):
    """``_serving`` for a route that answers in OpenAI's shape: (api, model_name,
    adapter, error_response), the error non-None when the model can't serve it."""
    try:
        return (*_serving(require), None)
    except RouteError as e:
        return None, None, None, _error(e.status, e.message, e.type)


@router.get("/v1/models")
async def list_models():
    api = _api()
    if api is None:
        return JSONResponse(ModelList(data=[ModelCard(id="unknown", created=now())]).model_dump())
    created = now()
    # served LoRA adapters beside the base model, each naming it as its parent (vLLM's shape)
    cards = [ModelCard(id=api.model_name, created=created)] + [
        ModelCard(id=name, created=created, parent=api.model_name)
        for name in api.served_models.adapters
    ]
    return JSONResponse(ModelList(data=cards).model_dump(exclude_none=True))


async def _lora_admin_call(call, route: str):
    """A ``LoraAdmin`` call's answer, or its failure as an OpenAI error envelope
    (:func:`_exception_error`)."""
    api = _api()
    if api is None:
        return _error(503, "Server not ready")
    try:
        return PlainTextResponse(await call(api.lora_admin))
    except Exception as e:  # noqa: BLE001 — surface as an OpenAI error envelope
        return _exception_error(e, route)


@router.post("/v1/load_lora_adapter")
async def load_lora_adapter(request: LoadLoraAdapterRequest):
    return await _lora_admin_call(
        lambda admin: admin.load(request.lora_name, request.lora_path), "/v1/load_lora_adapter",
    )


@router.post("/v1/unload_lora_adapter")
async def unload_lora_adapter(request: UnloadLoraAdapterRequest):
    return await _lora_admin_call(lambda admin: admin.unload(request.lora_name), "/v1/unload_lora_adapter")


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    api, model_name, adapter, err = _resolve("supports_chat")
    if err is not None:
        return err
    try:
        result = await serving_chat.create_chat_completion(api, model_name, adapter, request, raw_request)
    except Exception as e:  # noqa: BLE001 — surface as an OpenAI error envelope
        return _exception_error(e, "/v1/chat/completions")
    if request.stream:
        return result  # serving_chat builds the streaming response
    return JSONResponse(result)


def _json_body(raw: bytes):
    """The body as JSON proper: Python's json also reads NaN, Infinity and 1e400
    (as inf), which no JSON encoder writes back, and a response object echoes
    its request."""

    def finite(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"{text} is out of range")
        return value

    def constant(name: str):
        raise ValueError(f"{name} is not JSON")

    try:
        return json.loads(raw, parse_float=finite, parse_constant=constant)
    except RecursionError:
        # the decoder recurses once per nesting level
        raise ValueError("it nests too deeply") from None


def _anthropic_error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=serving_messages.error_body(status, message))


async def _anthropic(require: str, raw_request: Request, handle):
    """Runs ``handle(api, model_name, adapter, body)`` for an Anthropic route,
    its errors in Anthropic's shape."""
    try:
        api, model_name, adapter = _serving(require)
    except RouteError as e:
        return _anthropic_error(e.status, e.message)
    try:
        body = _json_body(await raw_request.body())
    except ValueError as e:
        return _anthropic_error(400, f"the request body is not JSON: {e}")
    try:
        result = await handle(api, model_name, adapter, body)
    except Exception as e:  # noqa: BLE001 — surface as an Anthropic error
        err = _route_error(e, raw_request.url.path)
        return _anthropic_error(err.status, err.message)
    if isinstance(result, dict):
        return JSONResponse(result)
    return result  # serving_messages builds the streaming response


@router.post("/v1/messages")
async def messages(raw_request: Request):
    """Anthropic's Messages API, on the chat path (see ``serving_messages``)."""
    return await _anthropic("supports_messages", raw_request, lambda api, name, adapter, body: (
        serving_messages.create_message(api, name, adapter, body, raw_request)))


@router.post("/v1/messages/count_tokens")
async def messages_count_tokens(raw_request: Request):
    return await _anthropic("supports_messages", raw_request, lambda api, name, adapter, body: (
        serving_messages.count_tokens(api, adapter, body)))


@router.post("/v1/responses")
async def responses(raw_request: Request):
    """OpenAI's Responses API, stateless, on the chat path (see ``serving_responses``)."""
    api, model_name, adapter, err = _resolve("supports_responses")
    if err is not None:
        return err
    try:
        body = _json_body(await raw_request.body())
    except ValueError as e:
        return _error(400, f"the request body is not JSON: {e}")
    try:
        result = await serving_responses.create_response(api, model_name, adapter, body, raw_request)
    except Exception as e:  # noqa: BLE001 — surface as an OpenAI error envelope
        return _exception_error(e, "/v1/responses")
    if isinstance(result, dict):
        return JSONResponse(result)
    return result  # serving_responses builds the streaming response


@router.post("/v1/audio/speech")
async def audio_speech(request: SpeechRequest, raw_request: Request):
    api, model_name, adapter, err = _resolve("supports_speech")
    if err is not None:
        return err
    try:
        return await serving_speech.create_speech(api, model_name, adapter, request, raw_request)
    except Exception as e:  # noqa: BLE001
        return _exception_error(e, "/v1/audio/speech")


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request):
    # Multipart (audio file + the OpenAI fields), parsed manually like
    # images/edits so unknown fields flow through as model_kwargs.
    api, model_name, adapter, err = _resolve("supports_transcriptions")
    if err is not None:
        return err
    try:
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return _error(400, "audio/transcriptions requires a 'file' upload")
        limit = serving_transcriptions.MAX_UPLOAD_BYTES
        if (getattr(upload, "size", None) or 0) > limit:
            return _error(413, f"audio/transcriptions accepts files up to {limit // 2**20} MiB")
        audio_bytes = await upload.read()
        if not audio_bytes:
            return _error(400, "audio/transcriptions received an empty 'file'")
        fields: dict = {}
        for key, value in form.multi_items():
            if hasattr(value, "filename"):
                continue
            # OpenAI clients send list fields as repeated ``name[]`` keys.
            if key.endswith("[]"):
                fields.setdefault(key[:-2], []).append(value)
                continue
            if key in _TRANSCRIPTION_TEXT_FIELDS:
                fields[key] = value
                continue
            try:
                fields[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                fields[key] = value
        try:
            req = TranscriptionRequest(**fields)
        except Exception as e:  # noqa: BLE001 — pydantic validation
            return _error(400, str(e))
        return await serving_transcriptions.create_transcription(
            api, model_name, adapter, req, audio_bytes,
            getattr(upload, "filename", None), raw_request=request,
        )
    except Exception as e:  # noqa: BLE001
        return _exception_error(e, "/v1/audio/transcriptions")


@router.websocket("/v1/realtime")
async def realtime(websocket: WebSocket):
    """OpenAI Realtime API. A full-duplex model (one whose client stream
    declares ``ClientAudio``) serves a conversation; ``?intent=transcription``
    streams speech-to-text with partial results, for a model whose adapter
    can continue a hypothesis. An omitted ``intent`` is the one the model
    serves. See ``serving_realtime``."""
    intent = websocket.query_params.get("intent")
    api = _api()
    audio = api.client_audio if api is not None else []
    # why a model with an audio client stream can't hold a conversation (None: it can)
    no_conversation = None
    if audio:
        channels = api.model.get_output_audio_channels("audio")
        if api.model.turn_tokens is None:
            no_conversation = "this model has no turn tokens, so it can't serve OpenAI Realtime conversations"
        elif len(audio) > 1:
            no_conversation = f"this model takes audio on {len(audio)} client streams, and a conversation feeds one"
        elif channels != 1:
            no_conversation = f"this model speaks in {channels} channels, and a conversation's audio is mono"
        elif intent is None:
            await serving_realtime.RealtimeConversation(api, api.model_name, websocket, audio[0]).run()
            return
    converses = bool(audio) and no_conversation is None
    reason = None
    if intent not in (None, "transcription"):
        reason = f"unsupported intent {intent!r}; " + ("omit intent for a conversation" if converses
                                                        else "use intent=transcription")
    else:
        api, model_name, adapter, err = _resolve("supports_realtime_transcription")
        if err is not None:
            reason = ("this model serves /v1/realtime conversations; omit intent" if converses
                      else f"{no_conversation}; use /generate/ws" if audio
                      else json.loads(bytes(err.body))["error"]["message"])
    if reason is not None:
        # 1008 is policy violation, the served model has no such endpoint
        await websocket.accept()
        await websocket.close(code=1008, reason=reason)
        return
    await serving_realtime.RealtimeTranscription(api, model_name, adapter, websocket).run()


@router.get("/v1/audio/voices")
async def audio_voices():
    api, model_name, adapter, err = _resolve("supports_speech")
    if err is not None:
        return err
    voices = serving_speech.list_voices(api)
    if voices is None:
        return _error(404, f"Model {model_name!r} does not publish a voice list")
    return JSONResponse(voices.model_dump())


@router.post("/v1/images/generations")
async def images_generations(request: ImageGenerationRequest, raw_request: Request):
    api, model_name, adapter, err = _resolve("supports_images")
    if err is not None:
        return err
    try:
        result = await serving_images.create_images(api, model_name, adapter, request, raw_request)
    except Exception as e:  # noqa: BLE001
        return _exception_error(e, "/v1/images/generations")
    return MediaJSONResponse(result)


@router.post("/v1/videos/generations")
async def videos_generations(request: VideoGenerationRequest, raw_request: Request):
    api, model_name, adapter, err = _resolve("supports_videos")
    if err is not None:
        return err
    try:
        result = await serving_videos.create_videos(api, model_name, adapter, request, raw_request)
    except Exception as e:  # noqa: BLE001
        return _exception_error(e, "/v1/videos/generations")
    if (request.model_extra or {}).get("stream_video"):
        return result  # serving_videos builds the streaming response
    return MediaJSONResponse(result)


@router.post("/v1/images/edits")
async def images_edits(request: Request):
    # Multipart (image file + prompt + passthrough fields), parsed manually so
    # arbitrary model knobs (e.g. cfg_*_scale) flow through as model_kwargs.
    api, model_name, adapter, err = _resolve("supports_images")
    if err is not None:
        return err
    try:
        form = await request.form()
        # One image, edited whole: a mask or a second image would be dropped. ``image[]``
        # is how OpenAI clients name the image list.
        images = []
        for key, value in form.multi_items():
            if not hasattr(value, "filename"):
                continue
            if key not in ("image", "image[]"):
                return _error(400, f"images/edits does not take a {key!r} upload; the whole image is edited")
            images.append(value)
        if len(images) > 1:
            return _error(400, f"images/edits takes one image, got {len(images)}")
        image = images[0] if images else None
        if image is None:
            return _error(400, "images/edits requires an 'image' file upload")
        n = form.get("n")
        check_inline_result(form.get("response_format"), int(n) if n else None)
        limit = max_body_bytes()
        if (getattr(image, "size", None) or 0) > limit:
            return _error(413, f"images/edits accepts files up to {limit // 2**20} MiB (MSTAR_MAX_BODY_MB)")
        image_bytes = await image.read()
        prompt = form.get("prompt") or ""
        known = {"image", "image[]", "prompt", "model", "n", "size", "response_format"}
        extra: dict = {}
        for key, value in form.multi_items():
            if key in known or hasattr(value, "filename"):
                continue
            try:
                extra[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                extra[key] = value
        if form.get("size"):
            extra["size"] = form.get("size")  # the adapter maps "WxH" to the model's width / height
        result = await serving_images.create_image_edit(
            api,
            model_name,
            adapter,
            prompt=prompt,
            image_bytes=image_bytes,
            image_filename=getattr(image, "filename", None),
            model_kwargs=extra,
            requested_model=form.get("model"),
            raw_request=request,
        )
    except Exception as e:  # noqa: BLE001
        return _exception_error(e, "/v1/images/edits")
    return MediaJSONResponse(result)

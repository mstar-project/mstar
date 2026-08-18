"""Bridge Dynamo endpoint requests onto an embedded APIServer.

The Dynamo frontend forwards OpenAI request bodies to Text-input backends
(the backend owns templating/tokenization). Requests are translated through
the same per-model adapters the native ``/v1`` routes use, submitted via
``APIServer.submit_request``, and results stream back from
``iter_result_chunks``:

- chat: OpenAI ``chat.completion.chunk`` dicts, one per text delta
  (the envelope the frontend expects from Text/Chat backends);
- images: a single ``{"created": ..., "data": [{"b64_json": ...}]}``
  response after all image chunks arrive;
- speech: a single ``NvAudioSpeechResponse``-shaped dict; the frontend
  unwraps ``b64_json`` into a raw audio-bytes response;
- videos: a single ``NvVideosResponse``-shaped dict carrying the mp4 as
  ``b64_json``, served on its own endpoint (a minimal video body is not
  distinguishable from an image body, so they can't share one).
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from typing import TYPE_CHECKING

from mstar.api_server.openai.adapters import OpenAIAdapter, SubmitArgs
from mstar.api_server.openai.protocol import (
    ChatCompletionRequest,
    ImageGenerationRequest,
    SpeechRequest,
    VideoGenerationRequest,
)

if TYPE_CHECKING:
    from mstar.api_server.entrypoint import APIServer

logger = logging.getLogger(__name__)

# Router/frontend bookkeeping fields and OpenAI-standard fields with no M*
# mapping. Both are dropped before validation so they don't leak into
# model_kwargs through the extra-field passthrough (documented extra_body
# knobs such as top_k / repetition_penalty still flow).
_STRIP_KEYS = {
    "routing", "output_options", "sampling_options", "stop_conditions",
    "token_ids", "batch_token_ids", "bootstrap_info", "multi_modal_data",
    "guided_decoding", "chat_template_kwargs", "chat_template_args",
    "nvext", "extra_args", "annotations", "trace", "disaggregated_params",
    "backend_instance_id", "ignore_eos", "min_tokens", "eos_token_ids",
    "stream_options", "frequency_penalty", "presence_penalty", "logit_bias",
    "logprobs", "top_logprobs", "tools", "tool_choice", "parallel_tool_calls",
    "response_format", "user", "store", "metadata", "service_tier",
}

# Image-request fields that ride nvext (mirrors Dynamo's NvCreateImageRequest);
# flattened onto the adapter request so they pass through as model_kwargs.
_IMAGE_NVEXT_KEYS = ("negative_prompt", "num_inference_steps", "guidance_scale", "seed")

# Video-request fields that ride nvext (mirrors Dynamo's NvCreateVideoRequest).
# fps / num_frames are first-class fields on the M* video request; the rest
# pass through as model_kwargs.
_VIDEO_NVEXT_KEYS = (
    "negative_prompt", "num_inference_steps", "guidance_scale", "seed",
    "fps", "num_frames",
)


def _clean(request: dict) -> dict:
    return {k: v for k, v in request.items() if k not in _STRIP_KEYS}


def _image_body(request: dict) -> dict:
    """NvCreateImageRequest -> body for the native image request model."""
    if request.get("response_format") == "url":
        raise ValueError("response_format 'url' needs a media store; use 'b64_json'")
    body = _clean(request)
    for key in _IMAGE_NVEXT_KEYS:
        value = (request.get("nvext") or {}).get(key)
        if value is not None:
            body.setdefault(key, value)
    return body


def _video_body(request: dict) -> dict:
    """NvCreateVideoRequest -> body for the native video request model."""
    if request.get("response_format") == "url":
        raise ValueError("response_format 'url' needs a media store; use 'b64_json'")
    if request.get("output_format") not in (None, "mp4"):
        raise ValueError(f"unsupported output_format {request['output_format']!r}; only 'mp4'")
    body = _clean(request)
    for key in _VIDEO_NVEXT_KEYS:
        value = (request.get("nvext") or {}).get(key)
        if value is not None:
            body.setdefault(key, value)
    # An explicit num_frames wins; otherwise derive it from fps * seconds
    # when both are present. With neither, the model's defaults apply.
    seconds = body.pop("seconds", None)
    if body.get("num_frames") is None and seconds and body.get("fps"):
        body["num_frames"] = int(round(body["fps"] * seconds))
    body.pop("output_format", None)
    body.pop("stream", None)
    reference = body.pop("input_reference", None)
    if reference:
        body["image"] = reference  # image-to-video conditioning
    return body


def _speech_body(request: dict) -> dict:
    """NvCreateAudioSpeechRequest -> body for the native speech request model."""
    if request.get("data_source") == "url":
        raise ValueError("data_source 'url' needs a media store; use 'b64_json'")
    # For speech, response_format is the audio codec (wav/mp3/...), not the
    # chat-style field _clean strips — reattach it after cleaning.
    fmt = (request.get("response_format") or "wav").lower()
    body = _clean(request)
    body.pop("data_source", None)
    body["response_format"] = fmt
    return body


class RequestBridge:
    """Translate one endpoint's requests for a single embedded server."""

    def __init__(self, server: APIServer, adapter: OpenAIAdapter, served_model_name: str):
        self.server = server
        self.adapter = adapter
        self.served_model_name = served_model_name

    async def generate(self, request: dict, context=None):
        """Endpoint handler. Chat, speech, and image requests share the
        endpoint; their bodies are distinguishable — chat carries
        ``messages``, speech carries ``input``, images carry ``prompt``."""
        if "messages" in request:
            async for out in self._chat(request, context):
                yield out
        elif "input" in request:
            yield await self._speech(request, context)
        elif "prompt" in request:
            yield await self._images(request, context)
        else:
            raise ValueError(
                "request has none of 'messages' (chat), 'input' (speech), 'prompt' (images)"
            )

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------

    async def _chat(self, request: dict, context):
        req = ChatCompletionRequest.model_validate(_clean(request))
        args = self.adapter.chat_to_request(req, self.server.upload_dir)
        rid = self._submit(args, prefix="chatcmpl")
        created = int(time.time())

        def envelope(delta: dict, finish: str | None = None) -> dict:
            return {
                "id": rid,
                "created": created,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                "model": self.served_model_name,
                "object": "chat.completion.chunk",
            }

        cancelled = False
        try:
            async for chunk in self.server.iter_result_chunks(rid):
                if context is not None and context.is_stopped():
                    cancelled = True
                    break
                if chunk.modality == "text":
                    yield envelope({
                        "role": "assistant",
                        "content": chunk.data.decode("utf-8", "replace"),
                    })
                elif chunk.modality == "error":
                    raise RuntimeError(chunk.data.decode("utf-8", "replace"))
                # audio/image chat outputs need a richer delta mapping; the
                # chat bridge is text-first for now.
            if not cancelled:
                yield envelope({"role": "assistant", "content": ""}, finish="stop")
        finally:
            self.server.abort_request(rid)
            logger.info("request %s finished (cancelled=%s)", rid, cancelled)

    # ------------------------------------------------------------------
    # images
    # ------------------------------------------------------------------

    async def _images(self, request: dict, context) -> dict:
        body = _image_body(request)
        req = ImageGenerationRequest.model_validate(body)

        reference = body.get("input_reference")
        if reference:
            from mstar.api_server import media_io
            _, path = media_io.resolve_media_ref(reference, self.server.upload_dir)
            extra = dict(req.model_extra or {})
            extra.pop("input_reference", None)
            args = self.adapter.image_edit_to_request(req.prompt, path, extra)
        else:
            args = self.adapter.image_to_request(req, self.server.upload_dir)

        rid = self._submit(args, prefix="img")
        images: list[bytes] = []
        try:
            async for chunk in self.server.iter_result_chunks(rid):
                if context is not None and context.is_stopped():
                    break
                if chunk.modality == "image":
                    images.append(chunk.data)
                elif chunk.modality == "error":
                    raise RuntimeError(chunk.data.decode("utf-8", "replace"))
        finally:
            self.server.abort_request(rid)

        if not images:
            raise RuntimeError("no image produced")
        return {
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(img).decode("ascii")} for img in images],
        }

    # ------------------------------------------------------------------
    # speech
    # ------------------------------------------------------------------

    async def _speech(self, request: dict, context) -> dict:
        """``/v1/audio/speech`` bodies (``NvCreateAudioSpeechRequest`` in, one
        ``NvAudioSpeechResponse`` out; the frontend unwraps ``b64_json`` into
        a raw audio-bytes response with the codec's content type)."""
        started = time.monotonic()
        body = _speech_body(request)
        fmt = body["response_format"]
        req = SpeechRequest.model_validate(body)
        args = self.adapter.speech_to_request(req, self.server.upload_dir)

        rid = self._submit(args, prefix="speech")
        pcm = bytearray()
        try:
            async for chunk in self.server.iter_result_chunks(rid):
                if context is not None and context.is_stopped():
                    break
                if chunk.modality == "audio":
                    pcm.extend(chunk.data)
                elif chunk.modality == "error":
                    raise RuntimeError(chunk.data.decode("utf-8", "replace"))
        finally:
            self.server.abort_request(rid)

        if not pcm:
            raise RuntimeError("no audio produced")
        from mstar.api_server import media_io
        model = self.server.model
        sample_rate = model.get_output_sample_rate("audio") if model is not None else 24000
        audio_bytes, _ = media_io.pcm16_to_container(bytes(pcm), sample_rate, fmt)
        return {
            "id": rid,
            "object": "audio.speech",
            "model": self.served_model_name,
            "status": "completed",
            "progress": 100,
            "created": int(time.time()),
            "data": [{
                "output_format": fmt,
                "b64_json": base64.b64encode(audio_bytes).decode("ascii"),
            }],
            "inference_time_s": round(time.monotonic() - started, 3),
        }

    # ------------------------------------------------------------------
    # videos
    # ------------------------------------------------------------------

    async def videos(self, request: dict, context=None):
        """Dedicated endpoint handler for ``/v1/videos`` bodies
        (``NvCreateVideoRequest`` in, one ``NvVideosResponse`` out)."""
        started = time.monotonic()
        body = _video_body(request)
        req = VideoGenerationRequest.model_validate(body)
        args = self.adapter.video_to_request(req, self.server.upload_dir)

        rid = self._submit(args, prefix="vid")
        videos: list[bytes] = []
        audios = []
        try:
            async for chunk in self.server.iter_result_chunks(rid):
                if context is not None and context.is_stopped():
                    break
                if chunk.modality == "video":
                    videos.append(chunk.data)
                elif chunk.modality == "audio":
                    audios.append(chunk)
                elif chunk.modality == "error":
                    raise RuntimeError(chunk.data.decode("utf-8", "replace"))
        finally:
            self.server.abort_request(rid)

        if not videos:
            raise RuntimeError("no video produced")
        # Mirror the native videos route: a sound request also emits a raw PCM
        # chunk per video, muxed into the mp4 as an AAC track and rescaled to
        # the request frame rate; a failed mux degrades to the plain video.
        request_fps = args.model_kwargs.get("fps")
        data = []
        for i, video in enumerate(videos):
            if i < len(audios):
                from mstar.api_server import media_io
                audio = audios[i]
                try:
                    video = media_io.mux_mp4_with_pcm16(
                        video,
                        audio.data,
                        sample_rate=int((audio.metadata or {}).get("sample_rate", 48000)),
                        num_channels=int((audio.metadata or {}).get("num_channels", 2)),
                        video_fps=float(request_fps) if request_fps else None,
                    )
                except Exception:  # noqa: BLE001 — degrade to video-only, keep serving
                    logger.exception("Muxing generated audio into the mp4 failed; returning video only")
            data.append({
                "output_format": "mp4",
                "b64_json": base64.b64encode(video).decode("ascii"),
            })
        yield {
            "id": rid,
            "object": "video",
            "model": self.served_model_name,
            "status": "completed",
            "progress": 100,
            "created": int(time.time()),
            "data": data,
            "inference_time_s": round(time.monotonic() - started, 3),
        }

    # ------------------------------------------------------------------

    def _submit(self, args: SubmitArgs, prefix: str) -> str:
        rid = f"{prefix}-{uuid.uuid4()}"
        self.server.submit_request(
            text=args.text,
            file_paths=args.file_paths,
            input_modalities=args.input_modalities,
            output_modalities=args.output_modalities,
            model_kwargs=args.model_kwargs,
            streaming=True,
            request_id=rid,
        )
        logger.info("request %s submitted (out=%s)", rid, args.output_modalities)
        return rid

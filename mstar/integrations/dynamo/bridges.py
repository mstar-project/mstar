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
  distinguishable from an image body, so they can't share one);
- realtime: OpenAI Realtime events over a bidirectional endpoint
  (:class:`RealtimeBridge`) — appended audio is committed into independent
  chat-shaped requests whose text/audio chunks stream back as
  ``response.*`` events.
- tensor: KServe Predict v2 requests (the frontend's ``ModelInfer`` /
  ``ModelStreamInfer`` gRPC route) translated through a per-model
  :data:`TENSOR_SPECS` entry (:class:`TensorBridge`); pi05 maps camera
  frames + state + prompt onto the multipart shape ``/generate`` takes
  and returns the action trajectory as one Float32 tensor; vjepa2_ac
  maps an encoded context clip + action/state trajectories onto the
  same shape and returns predicted latent grids — one batched
  response by default, one response per rollout step when the
  request parameters ask for ``stream_rollout``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from pathlib import Path
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
        return _submit(self.server, args, prefix)


def _submit(server: APIServer, args: SubmitArgs, prefix: str) -> str:
    rid = f"{prefix}-{uuid.uuid4()}"
    server.submit_request(
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


# ----------------------------------------------------------------------
# realtime
# ----------------------------------------------------------------------

# Template end-of-turn tokens stripped from transcript deltas (they close the
# model's chat turn; they are not speech transcript).
_TEMPLATE_END_TOKENS = ("<|im_end|>",)


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def _decode_pcm16_b64(audio_b64) -> bytes | None:
    """Base64 PCM16 append payload -> raw bytes; None when empty or malformed.

    PCM16 is 2-byte aligned; a bad frame is dropped rather than tearing down
    the whole session.
    """
    if not audio_b64 or not isinstance(audio_b64, str):
        return None
    try:
        raw = base64.b64decode(audio_b64)
    except Exception:  # noqa: BLE001 — drop the malformed chunk, keep the session
        return None
    if not raw or len(raw) % 2:
        return None
    return raw


def _drain(queue: asyncio.Queue) -> None:
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break


class _RealtimeTurn:
    """One committed utterance and the response events it produces.

    ``events`` buffers this turn's server events for in-order forwarding
    (``None`` marks the end of the turn's response); bounded so a turn
    waiting for its forwarding slot backpressures its own engine drain
    instead of buffering without limit.
    """

    def __init__(self) -> None:
        self.response_id = f"resp_{uuid.uuid4().hex}"
        self.item_id = f"item_{uuid.uuid4().hex}"
        self.events: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.task: asyncio.Task | None = None


class RealtimeBridge:
    """Serve one ``/v1/realtime`` connection per handler invocation.

    The frontend parses client frames into typed OpenAI Realtime events and
    forwards them as dicts on ``request_stream``; every yielded dict must
    deserialize into the frontend's typed server-event enum, so only
    spec-shaped events leave here. Turn model (single utterance): appended
    audio buffers until ``input_audio_buffer.commit``; each commit becomes
    one independent chat-shaped engine request (audio in, text+audio out)
    whose chunks are translated to ``response.*`` events. Turns may overlap
    in the engine, but their responses are forwarded strictly in commit
    order. ``conversation.item.*`` / ``response.*`` client events are
    accepted and ignored.
    """

    def __init__(
        self,
        server: APIServer,
        adapter,
        served_model_name: str,
        max_concurrent_turns: int = 2,
    ) -> None:
        self.server = server
        self.adapter = adapter
        self.served_model_name = served_model_name
        self.max_concurrent_turns = max_concurrent_turns

    # -- server-event shapes -------------------------------------------

    @staticmethod
    def _session_updated(session) -> dict:
        return {"type": "session.updated", "event_id": _event_id(), "session": session}

    @staticmethod
    def _committed(turn: _RealtimeTurn) -> dict:
        return {
            "type": "input_audio_buffer.committed",
            "event_id": _event_id(),
            "previous_item_id": None,
            "item_id": turn.item_id,
        }

    @staticmethod
    def _error(message: str, *, etype: str = "server_error", code: str | None = None) -> dict:
        return {
            "type": "error",
            "event_id": _event_id(),
            "error": {"type": etype, "code": code, "message": message},
        }

    @staticmethod
    def _response(turn: _RealtimeTurn, status: str, modalities: list[str],
                  status_details: dict | None = None) -> dict:
        response: dict = {
            "id": turn.response_id,
            "object": "realtime.response",
            "max_output_tokens": "inf",
            "output": [],
            "output_modalities": modalities,
            "status": status,
        }
        if status_details is not None:
            response["status_details"] = status_details
        return response

    def _response_created(self, turn: _RealtimeTurn, modalities: list[str]) -> dict:
        return {"type": "response.created", "event_id": _event_id(),
                "response": self._response(turn, "in_progress", modalities)}

    def _response_done(self, turn: _RealtimeTurn, modalities: list[str],
                       status: str = "completed",
                       status_details: dict | None = None) -> dict:
        return {"type": "response.done", "event_id": _event_id(),
                "response": self._response(turn, status, modalities, status_details)}

    @staticmethod
    def _content_event(etype: str, turn: _RealtimeTurn, **fields) -> dict:
        return {
            "type": etype,
            "event_id": _event_id(),
            "response_id": turn.response_id,
            "item_id": turn.item_id,
            "output_index": 0,
            "content_index": 0,
            **fields,
        }

    # -- utterance -> engine -------------------------------------------

    def _utterance_args(self, session, pcm: bytes) -> SubmitArgs:
        """Build the chat-shaped submit args for one committed utterance.

        The body is exactly what the native ``/v1/chat/completions`` route
        would see (one user turn with an ``input_audio`` part), so the
        model's chat adapter does the whole translation.
        """
        from mstar.api_server import media_io

        session = session if isinstance(session, dict) else {}
        audio_cfg = session.get("audio") or {}
        fmt = (audio_cfg.get("input") or {}).get("format") or {}
        rate = int(fmt.get("rate") or 24000) if isinstance(fmt, dict) else 24000
        wav = media_io.pcm16_to_wav_bytes(pcm, rate)

        body: dict = {
            "model": self.served_model_name,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(wav).decode("ascii"),
                        "format": "wav",
                    },
                }],
            }],
        }
        modalities = session.get("output_modalities")
        text_only = isinstance(modalities, list) and modalities and "audio" not in modalities
        if not text_only:
            body["modalities"] = ["audio"]
        voice = (audio_cfg.get("output") or {}).get("voice")
        if voice:
            body["audio"] = {"voice": voice}
        max_tokens = session.get("max_output_tokens")
        if isinstance(max_tokens, int):
            body["max_completion_tokens"] = max_tokens

        req = ChatCompletionRequest.model_validate(body)
        return self.adapter.chat_to_request(req, self.server.upload_dir)

    async def _run_turn(self, turn: _RealtimeTurn, args: SubmitArgs, context) -> None:
        """Drive one utterance through the engine and buffer its events.

        Always closes ``turn.events`` with the ``None`` sentinel; failures
        emit ``error`` plus a terminal ``response.done(status=failed)`` that
        carries the response id, so the client can correlate.
        """
        events = turn.events
        want_audio = "audio" in args.output_modalities
        modalities = ["audio"] if want_audio else ["text"]
        try:
            await events.put(self._response_created(turn, modalities))
            rid = _submit(self.server, args, prefix="rt")
            sent_audio = False
            stopped = False
            text_parts: list[str] = []
            try:
                async for chunk in self.server.iter_result_chunks(rid):
                    if context is not None and context.is_stopped():
                        stopped = True
                        break
                    if chunk.modality == "text":
                        delta = chunk.data.decode("utf-8", "replace")
                        for token in _TEMPLATE_END_TOKENS:
                            delta = delta.replace(token, "")
                        if not delta:
                            continue
                        text_parts.append(delta)
                        etype = ("response.output_audio_transcript.delta" if want_audio
                                 else "response.output_text.delta")
                        await events.put(self._content_event(etype, turn, delta=delta))
                    elif chunk.modality == "audio":
                        sent_audio = True
                        await events.put(self._content_event(
                            "response.output_audio.delta", turn,
                            delta=base64.b64encode(chunk.data).decode("ascii")))
                    elif chunk.modality == "error":
                        raise RuntimeError(chunk.data.decode("utf-8", "replace"))
            finally:
                self.server.abort_request(rid)
            if stopped:
                # Connection torn down mid-turn; claim no completed response.
                return
            if sent_audio:
                await events.put(self._content_event("response.output_audio.done", turn))
            if not want_audio:
                await events.put(self._content_event(
                    "response.output_text.done", turn, text="".join(text_parts)))
            await events.put(self._response_done(turn, modalities))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — engine failures travel in-band
            logger.exception("realtime turn %s failed", turn.response_id)
            await events.put(self._error(str(exc), code="generation_error"))
            await events.put(self._response_done(
                turn, modalities, status="failed",
                status_details={
                    "type": "failed",
                    "error": {"code": "generation_error", "type": "server_error"},
                }))
        finally:
            await events.put(None)

    # -- connection orchestration --------------------------------------

    async def generate(self, request_stream, context=None):
        """Bidirectional endpoint handler; one invocation per connection.

        A pump task demuxes client events and buffers audio; each commit
        spawns a turn task (capped by a semaphore) whose buffered events the
        loop below forwards in commit order, one turn's response completing
        before the next begins. Request-stream end is not cancellation —
        the turns already in flight finish and their responses flush.
        """
        out_stream: asyncio.Queue = asyncio.Queue()
        turns: list[_RealtimeTurn] = []
        session = None
        pcm = bytearray()
        slots = asyncio.Semaphore(self.max_concurrent_turns)

        async def pump() -> None:
            nonlocal session, pcm
            try:
                async for event in request_stream:
                    if context is not None and context.is_stopped():
                        break
                    etype = event.get("type") if isinstance(event, dict) else None
                    if etype == "session.update":
                        session = event.get("session")
                        fmt = {}
                        if isinstance(session, dict):
                            fmt = ((session.get("audio") or {}).get("input") or {}).get("format") or {}
                        if isinstance(fmt, dict) and fmt.get("type") not in (None, "audio/pcm"):
                            out_stream.put_nowait(self._error(
                                f"unsupported input audio format {fmt.get('type')!r}; "
                                "only audio/pcm is supported",
                                etype="invalid_request_error",
                                code="unsupported_audio_format"))
                        out_stream.put_nowait(self._session_updated(session))
                    elif etype == "input_audio_buffer.append":
                        raw = _decode_pcm16_b64(event.get("audio", ""))
                        if raw is None:
                            logger.warning("realtime: dropping malformed audio append")
                        else:
                            pcm.extend(raw)
                    elif etype == "input_audio_buffer.commit":
                        if not pcm:
                            out_stream.put_nowait(self._error(
                                "input audio buffer is empty",
                                etype="invalid_request_error",
                                code="input_audio_buffer_commit_empty"))
                            continue
                        try:
                            args = self._utterance_args(session, bytes(pcm))
                        except Exception as exc:  # noqa: BLE001 — keep the session usable
                            logger.exception("realtime: building the utterance request failed")
                            out_stream.put_nowait(self._error(
                                str(exc), etype="invalid_request_error"))
                            pcm = bytearray()
                            continue
                        pcm = bytearray()
                        await slots.acquire()
                        turn = _RealtimeTurn()
                        turns.append(turn)
                        out_stream.put_nowait(self._committed(turn))
                        out_stream.put_nowait(turn)
                        turn.task = asyncio.create_task(self._run_turn(turn, args, context))
                        turn.task.add_done_callback(lambda _: slots.release())
                    elif etype == "input_audio_buffer.clear":
                        pcm = bytearray()
                    else:
                        logger.debug("realtime: ignoring client event %s", etype)
            finally:
                # No more turns/events will be queued; the forwarder stops
                # after draining every turn already on out_stream.
                out_stream.put_nowait(None)

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                item = await out_stream.get()
                if item is None:
                    break
                if isinstance(item, _RealtimeTurn):
                    while True:
                        event = await item.events.get()
                        if event is None:
                            break
                        yield event
                    turns.remove(item)
                else:
                    yield item
        finally:
            pump_task.cancel()
            for turn in turns:
                if turn.task is not None:
                    turn.task.cancel()
            # Unblock turn tasks parked on a full events queue so their
            # cancellation propagates, then drive everything to completion.
            for turn in turns:
                _drain(turn.events)
            await asyncio.gather(
                pump_task,
                *(turn.task for turn in turns if turn.task is not None),
                return_exceptions=True,
            )


# ----------------------------------------------------------------------
# tensor (KServe Predict v2 models)
# ----------------------------------------------------------------------

# The frontend's tensor route parses KServe ModelInfer requests into dicts
# shaped like its internal tensor protocol: {"id", "model", "parameters",
# "tensors": [{"metadata": {"name", "data_type", "shape", "parameters"},
# "data": {"data_type", "values"}}]}. Responses take the same shape back
# (validated: dtype must match the data variant, shape must multiply out
# to the element count), and a unary ModelInfer requires exactly one
# response. Dtype names in these dicts are the internal variant names
# ("Uint8"/"Float32"/"Bytes"), not the KServe wire strings ("UINT8"/...).


class Pi05TensorSpec:
    """pi05: camera frames + robot state + task prompt in, one action
    trajectory out. Translated onto the multipart shape ``/generate``
    takes — encoded image files, prompt text, ``robot_state`` in
    model_kwargs — so the bridge adds no new native capability."""

    # Baked into the pi05 weights (action_in/out_proj); the trajectory
    # length varies with the server's action_horizon, the width does not.
    action_dim = 32

    def model_config(self, served_model_name: str) -> dict:
        return {
            "name": served_model_name,
            "inputs": [
                # cameras x H x W x RGB; all cameras equal size per request,
                # images[0] is the primary camera.
                {"name": "images", "data_type": "Uint8", "shape": [-1, -1, -1, 3]},
                {"name": "state", "data_type": "Float32", "shape": [-1]},
                {"name": "prompt", "data_type": "Bytes", "shape": [1]},
            ],
            "outputs": [
                {"name": "actions", "data_type": "Float32", "shape": [-1, self.action_dim]},
            ],
        }

    def submit_args(self, tensors: dict[str, dict], upload_dir, parameters=None) -> SubmitArgs:
        import numpy as np
        from PIL import Image

        images = tensors.get("images")
        prompt = tensors.get("prompt")
        if images is None or prompt is None:
            raise ValueError("pi05 needs 'images' and 'prompt' input tensors")

        prompt_values = prompt["data"]["values"]
        if len(prompt_values) != 1:
            raise ValueError("'prompt' must hold exactly one string element")
        text = bytes(prompt_values[0]).decode("utf-8")

        shape = [int(d) for d in images["metadata"]["shape"]]
        if len(shape) != 4 or shape[3] != 3:
            raise ValueError(
                f"'images' must be [cameras, height, width, 3] uint8, got shape {shape}"
            )
        values = images["data"]["values"]
        if isinstance(values, (bytes, bytearray)):
            frames = np.frombuffer(bytes(values), dtype=np.uint8).reshape(shape)
        else:
            frames = np.asarray(values, dtype=np.uint8).reshape(shape)

        upload_dir = Path(upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex
        paths = []
        for i, frame in enumerate(frames):
            path = upload_dir / f"tensor_{tag}_cam{i}.png"
            Image.fromarray(np.ascontiguousarray(frame), mode="RGB").save(path)
            paths.append(str(path))

        model_kwargs = {}
        state = tensors.get("state")
        if state is not None:
            model_kwargs["robot_state"] = [float(v) for v in state["data"]["values"]]

        return SubmitArgs(
            text=text,
            file_paths={"image": paths},
            input_modalities=["image", "text"],
            output_modalities=["action"],
            model_kwargs=model_kwargs,
        )

    def output_tensors(self, chunk) -> list[dict] | None:
        if chunk.modality != "action":
            return None
        import numpy as np

        actions = np.frombuffer(chunk.data, dtype="<f4")
        if actions.size == 0 or actions.size % self.action_dim:
            raise ValueError(
                f"action chunk holds {actions.size} float32 values, "
                f"not a non-empty multiple of {self.action_dim}"
            )
        steps = actions.size // self.action_dim
        return [{
            "metadata": {
                "name": "actions",
                "data_type": "Float32",
                "shape": [steps, self.action_dim],
            },
            "data": {"data_type": "Float32", "values": actions.tolist()},
        }]

    def aggregate_chunks(self, parameters) -> bool:
        return False


def _parameter_value(value):
    """Unwrap a KServe request parameter. The frontend forwards values
    tagged by variant ({"int64": 4}, {"bool": True}, ...); plain scalars
    pass through unchanged."""
    if isinstance(value, dict) and len(value) == 1:
        return next(iter(value.values()))
    return value


class VJepa2AcTensorSpec:
    """vjepa2_ac: a context clip plus per-timestep action/state
    trajectories in, predicted latent grids out. The encoded video file
    rides a Bytes tensor verbatim into the multipart shape ``/generate``
    takes; ``rollout_horizon`` and ``stream_rollout`` ride the request
    parameters map into model_kwargs. Emission defaults to one batched
    response (safe for unary ModelInfer, which rejects multiple
    responses); ``stream_rollout`` selects one response per rollout step
    for ModelStreamInfer clients."""

    # Baked into the AC ViT-g weights; latent chunks reshape to
    # [n_tokens, hidden_size].
    hidden_size = 1408

    def model_config(self, served_model_name: str) -> dict:
        return {
            "name": served_model_name,
            "inputs": [
                # One element: the encoded video file bytes (any container
                # the server's decoder can probe).
                {"name": "video", "data_type": "Bytes", "shape": [1]},
                {"name": "actions", "data_type": "Float32", "shape": [-1, -1]},
                {"name": "states", "data_type": "Float32", "shape": [-1, -1]},
            ],
            "outputs": [
                {"name": "latents", "data_type": "Float32", "shape": [-1, self.hidden_size]},
            ],
        }

    @staticmethod
    def _trajectory(tensor: dict, name: str) -> list[list[float]]:
        shape = [int(d) for d in tensor["metadata"]["shape"]]
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
            raise ValueError(f"'{name}' must be [timesteps, dim] float32, got shape {shape}")
        rows, dim = shape
        values = [float(v) for v in tensor["data"]["values"]]
        return [values[i * dim : (i + 1) * dim] for i in range(rows)]

    def submit_args(self, tensors: dict[str, dict], upload_dir, parameters=None) -> SubmitArgs:
        video = tensors.get("video")
        actions = tensors.get("actions")
        states = tensors.get("states")
        if video is None or actions is None or states is None:
            raise ValueError("vjepa2_ac needs 'video', 'actions' and 'states' input tensors")

        video_values = video["data"]["values"]
        if len(video_values) != 1:
            raise ValueError("'video' must hold exactly one encoded-file element")
        video_bytes = bytes(video_values[0])
        if not video_bytes:
            raise ValueError("'video' element is empty")

        action_rows = self._trajectory(actions, "actions")
        state_rows = self._trajectory(states, "states")
        if (len(action_rows), len(action_rows[0])) != (len(state_rows), len(state_rows[0])):
            raise ValueError(
                "'actions' and 'states' must share one [timesteps, dim] shape, got "
                f"{actions['metadata']['shape']} vs {states['metadata']['shape']}"
            )

        upload_dir = Path(upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / f"tensor_{uuid.uuid4().hex}.mp4"
        path.write_bytes(video_bytes)

        model_kwargs = {"actions": action_rows, "states": state_rows}
        parameters = parameters or {}
        horizon = _parameter_value(parameters.get("rollout_horizon"))
        if horizon is not None:
            model_kwargs["rollout_horizon"] = int(horizon)
        stream = _parameter_value(parameters.get("stream_rollout"))
        if stream is not None:
            model_kwargs["stream_rollout"] = bool(stream)

        return SubmitArgs(
            file_paths={"video": [str(path)]},
            input_modalities=["video"],
            output_modalities=["video"],
            model_kwargs=model_kwargs,
        )

    def output_tensors(self, chunk) -> list[dict] | None:
        if chunk.modality != "video":
            return None
        import numpy as np

        latents = np.frombuffer(chunk.data, dtype="<f4")
        if latents.size == 0 or latents.size % self.hidden_size:
            raise ValueError(
                f"latent chunk holds {latents.size} float32 values, "
                f"not a non-empty multiple of {self.hidden_size}"
            )
        tokens = latents.size // self.hidden_size
        return [{
            "metadata": {
                "name": "latents",
                "data_type": "Float32",
                "shape": [tokens, self.hidden_size],
            },
            "data": {"data_type": "Float32", "values": latents.tolist()},
        }]

    def aggregate_chunks(self, parameters) -> bool:
        # The engine emits one latent chunk per rollout iteration in both
        # walks; only a stream_rollout client wants them as separate
        # responses. Everyone else gets one merged response, which is what
        # a unary ModelInfer requires.
        return not bool(_parameter_value((parameters or {}).get("stream_rollout")))


TENSOR_SPECS = {
    "pi05": Pi05TensorSpec(),
    "vjepa2_ac": VJepa2AcTensorSpec(),
}


def get_tensor_spec(model_name: str):
    return TENSOR_SPECS.get(model_name)


def _merge_tensor(merged: dict, tensor: dict) -> None:
    """Concatenate a chunk's tensor onto the accumulated one of the same
    name (leading dimension grows; trailing dims and dtype must match)."""
    name = tensor["metadata"]["name"]
    existing = merged.get(name)
    if existing is None:
        merged[name] = tensor
        return
    old_meta, new_meta = existing["metadata"], tensor["metadata"]
    if (old_meta["shape"][1:] != new_meta["shape"][1:]
            or old_meta["data_type"] != new_meta["data_type"]):
        raise ValueError(
            f"cannot aggregate chunks of {name!r}: "
            f"{old_meta['shape']} vs {new_meta['shape']}"
        )
    old_meta["shape"] = [
        old_meta["shape"][0] + new_meta["shape"][0], *old_meta["shape"][1:]
    ]
    existing["data"]["values"].extend(tensor["data"]["values"])


class TensorBridge:
    """Serve one tensor model's KServe requests against an embedded server."""

    def __init__(self, server: APIServer, spec, served_model_name: str):
        self.server = server
        self.spec = spec
        self.served_model_name = served_model_name

    async def generate(self, request: dict, context=None):
        named = {t["metadata"]["name"]: t for t in request.get("tensors", [])}
        parameters = request.get("parameters") or {}
        args = self.spec.submit_args(named, self.server.upload_dir, parameters)
        # An aggregating spec folds every mapped chunk into ONE response —
        # required for unary ModelInfer when the engine emits multiple
        # chunks per request. Non-aggregating requests get one response
        # per mapped chunk (ModelStreamInfer territory).
        aggregate = self.spec.aggregate_chunks(parameters)
        rid = _submit(self.server, args, prefix="tensor")
        produced = False
        cancelled = False
        merged: dict[str, dict] = {}
        try:
            async for chunk in self.server.iter_result_chunks(rid):
                if context is not None and context.is_stopped():
                    cancelled = True
                    break
                if chunk.modality == "error":
                    raise RuntimeError(chunk.data.decode("utf-8", "replace"))
                tensors = self.spec.output_tensors(chunk)
                if not tensors:
                    continue
                if aggregate:
                    for tensor in tensors:
                        _merge_tensor(merged, tensor)
                else:
                    produced = True
                    yield {
                        "id": request.get("id"),
                        "model": request.get("model") or self.served_model_name,
                        "tensors": tensors,
                    }
        finally:
            self.server.abort_request(rid)
        if merged and not cancelled:
            produced = True
            yield {
                "id": request.get("id"),
                "model": request.get("model") or self.served_model_name,
                "tensors": list(merged.values()),
            }
        if not produced and not cancelled:
            raise RuntimeError("no tensor output produced")

"""Bridge Dynamo endpoint requests onto an embedded APIServer.

The Dynamo frontend forwards OpenAI request bodies to Text-input backends
(the backend owns templating/tokenization). Requests are translated through
the same per-model adapters the native ``/v1`` routes use, submitted via
``APIServer.submit_request``, and results stream back from
``iter_result_chunks``:

- chat: OpenAI ``chat.completion.chunk`` dicts, one per text delta
  (the envelope the frontend expects from Text/Chat backends);
- images: a single ``{"created": ..., "data": [{"b64_json": ...}]}``
  response after all image chunks arrive.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid

from mstar.api_server.entrypoint import APIServer
from mstar.api_server.openai.adapters import OpenAIAdapter, SubmitArgs
from mstar.api_server.openai.protocol import ChatCompletionRequest, ImageGenerationRequest

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


def _clean(request: dict) -> dict:
    return {k: v for k, v in request.items() if k not in _STRIP_KEYS}


class RequestBridge:
    """Translate one endpoint's requests for a single embedded server."""

    def __init__(self, server: APIServer, adapter: OpenAIAdapter, served_model_name: str):
        self.server = server
        self.adapter = adapter
        self.served_model_name = served_model_name

    async def generate(self, request: dict, context=None):
        """Endpoint handler. Chat and image requests share the endpoint;
        chat bodies carry ``messages``, image bodies carry ``prompt``."""
        if "messages" in request:
            async for out in self._chat(request, context):
                yield out
        elif "prompt" in request:
            yield await self._images(request, context)
        else:
            raise ValueError("request has neither 'messages' (chat) nor 'prompt' (images)")

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
        if request.get("response_format") == "url":
            raise ValueError("response_format 'url' needs a media store; use 'b64_json'")

        body = _clean(request)
        for key in _IMAGE_NVEXT_KEYS:
            value = (request.get("nvext") or {}).get(key)
            if value is not None:
                body.setdefault(key, value)
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

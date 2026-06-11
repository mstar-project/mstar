"""Per-model translation between OpenAI-shaped requests and mstar's request path.

The OpenAI endpoints are model-agnostic; everything model-specific lives here.
An adapter translates an OpenAI request into :class:`SubmitArgs` (the arguments
``APIServer.submit_request`` expects) and declares which OpenAI surfaces the
model supports. Output chunks are translated back to OpenAI shapes by the
serving handlers, which are generic across models.

``model_kwargs`` is non-standardized across models, so each adapter maps the
standard OpenAI fields (``temperature``, ``top_p``, ``max_tokens``, ``seed``,
``voice``, ``modalities`` …) onto the keys its model actually honors (see
:func:`_apply_sampling`). Knobs that aren't OpenAI-standard (``top_k``,
``repetition_penalty``, or model-namespaced keys like ``talker_top_p``) are not
first-class fields — pass them via the OpenAI client's ``extra_body`` and they
flow through verbatim as model_kwargs (see :func:`_passthrough`).

Models whose outputs have no OpenAI-standard representation (robot actions from
Pi0.5, world-model latents from V-JEPA 2) intentionally have **no** adapter:
they are served only through the native ``/generate`` endpoint and the Python
SDK, and ``/v1/*`` returns 404 for them (they do not fall back to chat). New
OpenAI-capable models opt in by adding an adapter and registering it in
:data:`ADAPTER_REGISTRY`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from mstar.api_server import media_io

if TYPE_CHECKING:  # for type checkers / IDEs only (annotations are lazy via __future__)
    from mstar.api_server.openai.protocol import (
        ChatCompletionRequest,
        ImageGenerationRequest,
        SpeechRequest,
    )


@dataclass
class SubmitArgs:
    text: str | None = None
    file_paths: dict[str, list[str]] | None = None
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    model_kwargs: dict = field(default_factory=dict)


def flatten_messages(
    messages: list, upload_dir: Path, allow_remote: bool = True
) -> tuple[str | None, dict[str, list[str]], list[str]]:
    """Flatten OpenAI chat ``messages`` into (text, file_paths, input_modalities).

    Text parts across all messages are concatenated (newline-joined). Image /
    audio / video content parts are persisted under ``upload_dir`` and grouped
    by modality. (Multi-turn role structure is flattened — a v1 simplification;
    the models here apply their own prompt formatting in ``process_prompt``.)
    """
    text_parts: list[str] = []
    file_paths: dict[str, list[str]] = {}

    def add_file(modality: str, path: str) -> None:
        file_paths.setdefault(modality, []).append(path)

    for msg in messages or []:
        # Messages may be pydantic ChatMessage objects or plain dicts.
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if content is None:
            continue
        if isinstance(content, str):
            if content:
                text_parts.append(content)
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                if part.get("text"):
                    text_parts.append(part["text"])
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url:
                    mod, path = media_io.resolve_media_ref(url, upload_dir, allow_remote=allow_remote)
                    add_file(mod or "image", path)
            elif ptype == "video_url":  # extension for video-capable models
                url = (part.get("video_url") or {}).get("url", "")
                if url:
                    mod, path = media_io.resolve_media_ref(url, upload_dir, allow_remote=allow_remote)
                    add_file(mod or "video", path)
            elif ptype == "audio_url":  # data:/http audio input (vllm-omni content style)
                url = (part.get("audio_url") or {}).get("url", "")
                if url:
                    mod, path = media_io.resolve_media_ref(url, upload_dir, allow_remote=allow_remote)
                    add_file(mod or "audio", path)
            elif ptype == "input_audio":  # OpenAI-native audio input (base64 + format)
                ia = part.get("input_audio") or {}
                data, fmt = ia.get("data"), ia.get("format", "wav")
                if data:
                    mod, path = media_io.save_base64(data, fmt, "audio", upload_dir)
                    add_file(mod, path)

    input_modalities = list(file_paths.keys())
    text = "\n".join(text_parts) if text_parts else None
    if text is not None:
        input_modalities.append("text")
    return text, file_paths, input_modalities


def _passthrough(req) -> dict:
    """Unknown request fields (e.g. from the OpenAI client's ``extra_body``)
    flow through verbatim as model_kwargs."""
    extra = getattr(req, "model_extra", None) or {}
    return dict(extra)


def _apply_sampling(
    req,
    mk: dict,
    *,
    temperature_key: str = "temperature",
    top_p_key: str = "top_p",
    max_tokens_key: str | None = "max_output_tokens",
) -> dict:
    """Map the OpenAI-standard sampling fields onto a model's ``model_kwargs``.

    Behavior is common across models, but the target key names differ (e.g.
    Qwen3-Omni's Thinker uses ``thinker_temperature`` and its Talker
    ``talker_temperature``), so callers pass them in. ``setdefault`` lets an
    explicit ``extra_body`` value win over the standard field.

    Handled (the OpenAI-standard scalars): ``temperature``, ``top_p``, ``seed``,
    and ``max_tokens`` / ``max_completion_tokens``. Non-standard knobs
    (``top_k``, ``repetition_penalty``, model-namespaced keys) are not OpenAI
    fields — send them via ``extra_body`` and they pass through :func:`_passthrough`.
    ``seed`` is honored by the conductor, which uses it in place of the
    request-id-derived RNG seed.
    """
    temperature = getattr(req, "temperature", None)
    if temperature is not None:
        mk.setdefault(temperature_key, temperature)
    top_p = getattr(req, "top_p", None)
    if top_p is not None:
        mk.setdefault(top_p_key, top_p)
    seed = getattr(req, "seed", None)
    if seed is not None:
        mk.setdefault("seed", seed)
    if max_tokens_key:
        max_tokens = getattr(req, "max_completion_tokens", None) or getattr(req, "max_tokens", None)
        if max_tokens is not None:
            mk.setdefault(max_tokens_key, max_tokens)
    return mk


class OpenAIAdapter:
    """Base adapter. A model subclasses this and implements the surfaces it
    supports; an unimplemented surface raises and the endpoint returns 404.
    Models with no OpenAI-standard output have no adapter (see the module
    docstring) and are reached only via ``/generate`` / the SDK.
    """

    # Each flag gates exactly one OpenAI surface:
    supports_chat: bool = False     # POST /v1/chat/completions
    supports_speech: bool = False   # POST /v1/audio/speech
    supports_images: bool = False   # POST /v1/images/generations and /v1/images/edits

    def chat_to_request(self, req: ChatCompletionRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        # Output modalities vary by model: e.g. Qwen3-Omni speech output also
        # emits text, whereas BAGEL chat is text-only.
        raise NotImplementedError("chat is not supported by this model")

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        raise NotImplementedError("audio/speech is not supported by this model")

    def image_to_request(self, req: ImageGenerationRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        raise NotImplementedError("image generation is not supported by this model")

    def image_edit_to_request(self, prompt: str, image_path: str, extra_kwargs: dict) -> SubmitArgs:  # noqa: ARG002
        raise NotImplementedError("image editing is not supported by this model")


class BagelAdapter(OpenAIAdapter):
    """BAGEL: text chat (text out) + text-to-image / image editing.

    BAGEL's ``get_sampling_config`` reads the model config, so per-request
    ``temperature`` / ``top_p`` are not honored; ``max_output_tokens`` and
    ``seed`` are.
    """

    supports_chat = True
    supports_images = True

    def chat_to_request(self, req: ChatCompletionRequest, upload_dir: Path) -> SubmitArgs:
        text, file_paths, in_mods = flatten_messages(req.messages, upload_dir)
        mk = _passthrough(req)
        _apply_sampling(req, mk)
        return SubmitArgs(
            text=text,
            file_paths=file_paths or None,
            input_modalities=in_mods,
            output_modalities=["text"],
            model_kwargs=mk,
        )

    def image_to_request(self, req: ImageGenerationRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        mk = _passthrough(req)
        if getattr(req, "seed", None) is not None:
            mk.setdefault("seed", req.seed)
        return SubmitArgs(
            text=req.prompt,
            input_modalities=["text"],
            output_modalities=["image"],
            model_kwargs=mk,
        )

    def image_edit_to_request(self, prompt: str, image_path: str, extra_kwargs: dict) -> SubmitArgs:
        # Image editing: the input image + prompt produce an edited image
        # (BAGEL's I2I path). Extra kwargs (e.g. cfg_*_scale, seed) pass through.
        return SubmitArgs(
            text=prompt,
            file_paths={"image": [image_path]},
            input_modalities=["image", "text"],
            output_modalities=["image"],
            model_kwargs=dict(extra_kwargs or {}),
        )


class Qwen3OmniAdapter(OpenAIAdapter):
    """Qwen3-Omni: multimodal chat (text, optionally + speech) and TTS.

    Sampling is split across two stages: the Thinker (text) takes
    ``thinker_*`` keys, the Talker (speech) ``talker_*``. The other Talker knobs
    (``talker_top_k``, ``talker_repetition_penalty``) aren't OpenAI fields — pass
    them via ``extra_body``.
    """

    supports_chat = True
    supports_speech = True

    def _voice(self, req) -> str | None:
        audio_cfg = getattr(req, "audio", None) or {}
        if isinstance(audio_cfg, dict) and audio_cfg.get("voice"):
            return audio_cfg["voice"]
        return getattr(req, "voice", None)

    def chat_to_request(self, req: ChatCompletionRequest, upload_dir: Path) -> SubmitArgs:
        text, file_paths, in_mods = flatten_messages(req.messages, upload_dir)
        mk = _passthrough(req)
        # Speech output also emits text, so request both modalities when audio is asked for.
        want_audio = bool(req.modalities and "audio" in req.modalities)
        out_mods = ["text", "audio"] if want_audio else ["text"]
        _apply_sampling(req, mk, temperature_key="thinker_temperature", top_p_key="thinker_top_p")
        voice = self._voice(req)
        if voice:
            mk["voice"] = voice
        return SubmitArgs(
            text=text,
            file_paths=file_paths or None,
            input_modalities=in_mods,
            output_modalities=out_mods,
            model_kwargs=mk,
        )

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        # Qwen3-Omni is a chat model; /v1/audio/speech returns the audio of its
        # spoken response to ``input`` (the handler keeps only the audio).
        mk = _passthrough(req)
        if getattr(req, "voice", None):
            mk["voice"] = req.voice
        # Talker (speech) sampling; max_tokens is not an OpenAI speech field.
        _apply_sampling(req, mk, temperature_key="talker_temperature", top_p_key="talker_top_p", max_tokens_key=None)
        return SubmitArgs(
            text=req.input,
            input_modalities=["text"],
            output_modalities=["text", "audio"],
            model_kwargs=mk,
        )


class OrpheusAdapter(OpenAIAdapter):
    """Orpheus: text-to-speech (audio out only). Honors temperature/top_p/seed
    (its ``get_sampling_config`` reads model_kwargs)."""

    supports_speech = True

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        mk = _passthrough(req)
        if getattr(req, "voice", None):
            mk["voice"] = req.voice
        _apply_sampling(req, mk, temperature_key="temperature", top_p_key="top_p", max_tokens_key=None)
        return SubmitArgs(
            text=req.input,
            input_modalities=["text"],
            output_modalities=["audio"],
            model_kwargs=mk,
        )


# Only models with an OpenAI-standard surface are registered. Action/world-model
# models (pi05, vjepa2) are deliberately absent → /v1/* 404s; use /generate.
ADAPTER_REGISTRY: dict[str, OpenAIAdapter] = {
    "bagel": BagelAdapter(),
    "qwen3_omni": Qwen3OmniAdapter(),
    "orpheus": OrpheusAdapter(),
}


def get_adapter(model_name: str) -> OpenAIAdapter | None:
    return ADAPTER_REGISTRY.get(model_name)

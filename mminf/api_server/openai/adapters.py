"""Per-model translation between OpenAI-shaped requests and mminf's request path.

The OpenAI endpoints are model-agnostic; everything model-specific lives here.
An adapter translates an OpenAI request into :class:`SubmitArgs` (the arguments
``APIServer.submit_request`` expects) and declares which OpenAI surfaces the
model supports. Output chunks are translated back to OpenAI shapes by the
serving handlers, which are generic across models.

``model_kwargs`` is non-standardized across models, so each adapter maps the
standard OpenAI fields (``temperature``, ``max_tokens``, ``voice``,
``modalities`` …) onto the keys its model actually honors. New models opt in by
adding an adapter and registering it in :data:`ADAPTER_REGISTRY`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mminf.api_server import media_io


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
            elif ptype == "input_audio":
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


class OpenAIAdapter:
    """Base adapter. Models override the methods for surfaces they support."""

    supports_chat: bool = False
    supports_speech: bool = False
    supports_images: bool = False

    def chat_to_request(self, req, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        raise NotImplementedError("chat is not supported by this model")

    def speech_to_request(self, req, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        raise NotImplementedError("audio/speech is not supported by this model")

    def image_to_request(self, req, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        raise NotImplementedError("image generation is not supported by this model")


class BagelAdapter(OpenAIAdapter):
    """BAGEL: text chat + text-to-image. (Sampling uses model config defaults.)"""

    supports_chat = True
    supports_images = True

    def chat_to_request(self, req, upload_dir: Path) -> SubmitArgs:
        text, file_paths, in_mods = flatten_messages(req.messages, upload_dir)
        mk = _passthrough(req)
        max_tokens = req.max_completion_tokens or req.max_tokens
        if max_tokens is not None:
            mk["max_output_tokens"] = max_tokens
        return SubmitArgs(
            text=text,
            file_paths=file_paths or None,
            input_modalities=in_mods,
            output_modalities=["text"],
            model_kwargs=mk,
        )

    def image_to_request(self, req, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        mk = _passthrough(req)
        return SubmitArgs(
            text=req.prompt,
            input_modalities=["text"],
            output_modalities=["image"],
            model_kwargs=mk,
        )


class Qwen3OmniAdapter(OpenAIAdapter):
    """Qwen3-Omni: multimodal chat (text + optional speech out) and TTS."""

    supports_chat = True
    supports_speech = True

    def _voice(self, req) -> str | None:
        audio_cfg = getattr(req, "audio", None) or {}
        if isinstance(audio_cfg, dict) and audio_cfg.get("voice"):
            return audio_cfg["voice"]
        return getattr(req, "voice", None)

    def chat_to_request(self, req, upload_dir: Path) -> SubmitArgs:
        text, file_paths, in_mods = flatten_messages(req.messages, upload_dir)
        mk = _passthrough(req)
        want_audio = bool(req.modalities and "audio" in req.modalities)
        out_mods = ["text", "audio"] if want_audio else ["text"]
        if req.temperature is not None:
            mk.setdefault("thinker_temperature", req.temperature)
        if req.top_p is not None:
            mk.setdefault("thinker_top_p", req.top_p)
        voice = self._voice(req)
        if voice:
            mk["voice"] = voice
        max_tokens = req.max_completion_tokens or req.max_tokens
        if max_tokens is not None:
            mk["max_output_tokens"] = max_tokens
        return SubmitArgs(
            text=text,
            file_paths=file_paths or None,
            input_modalities=in_mods,
            output_modalities=out_mods,
            model_kwargs=mk,
        )

    def speech_to_request(self, req, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        # Qwen3-Omni is a chat model; /v1/audio/speech returns the audio of its
        # spoken response to ``input``. Text output is discarded by the handler.
        mk = _passthrough(req)
        if getattr(req, "voice", None):
            mk["voice"] = req.voice
        if getattr(req, "temperature", None) is not None:
            mk.setdefault("talker_temperature", req.temperature)
        return SubmitArgs(
            text=req.input,
            input_modalities=["text"],
            output_modalities=["text", "audio"],
            model_kwargs=mk,
        )


class OrpheusAdapter(OpenAIAdapter):
    """Orpheus: text-to-speech (audio out only)."""

    supports_speech = True

    def speech_to_request(self, req, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        mk = _passthrough(req)
        if getattr(req, "voice", None):
            mk["voice"] = req.voice
        # Honored by Orpheus after the get_sampling_config patch.
        if getattr(req, "temperature", None) is not None:
            mk.setdefault("temperature", req.temperature)
        if getattr(req, "top_p", None) is not None:
            mk.setdefault("top_p", req.top_p)
        return SubmitArgs(
            text=req.input,
            input_modalities=["text"],
            output_modalities=["audio"],
            model_kwargs=mk,
        )


ADAPTER_REGISTRY: dict[str, OpenAIAdapter] = {
    "bagel": BagelAdapter(),
    "qwen3_omni": Qwen3OmniAdapter(),
    "orpheus": OrpheusAdapter(),
}


def get_adapter(model_name: str) -> OpenAIAdapter | None:
    return ADAPTER_REGISTRY.get(model_name)

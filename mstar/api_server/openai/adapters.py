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

import http.client
import logging
import re
import ssl
import urllib.error
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from mstar.api_server import media_io
from mstar.api_server.openai import glm_chat
from mstar.api_server.openai.speech_chunking import _cut_evenly, split_sentences
from mstar.model.multimodal import PromptPart
from mstar.model.qwen3_asr.constants import ASR_TEXT_TAG
from mstar.model.registry import qwen_3_5_dense_sizes as QWEN_3_5_DENSE_SIZES
from mstar.model.whisper.config import WORDS_MARKER
from mstar.utils.generation_kwargs import normalize_generation_kwargs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # for type checkers / IDEs only (annotations are lazy via __future__)
    from mstar.api_server.openai.protocol import (
        ChatCompletionRequest,
        ImageGenerationRequest,
        SpeechRequest,
        TranscriptionRequest,
        VideoGenerationRequest,
    )


@dataclass
class SubmitArgs:
    text: str | None = None
    file_paths: dict[str, list[str]] | None = None
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    model_kwargs: dict = field(default_factory=dict)
    # Ordered text/attachment sequence. None from entrypoints with no ordering
    # to preserve, which keep the legacy attachments-then-text layout.
    prompt_parts: list[PromptPart] | None = None


@dataclass
class Transcript:
    """A finished transcription as the serving layer reports it.

    ``text`` is the clean transcript. ``language`` is an ISO-639-1 code when the
    model reported one (detected or forced). ``segments`` / ``words`` carry
    ``{"start", "end", "text"}`` / ``{"start", "end", "word"}`` in seconds when
    the model emitted timestamps; empty otherwise. ``unfinished`` says the
    output stopped inside a segment (a start timestamp with no end): the text
    after the last closed segment is provisional, and a long-form driver
    re-decodes that audio from the last segment end.
    """
    text: str
    language: str | None = None
    segments: list[dict] = field(default_factory=list)
    words: list[dict] = field(default_factory=list)
    unfinished: bool = False


def flatten_messages(
    messages: list, upload_dir: Path, allow_remote: bool = False
) -> tuple[str | None, dict[str, list[str]], list[str], list[PromptPart]]:
    """Flatten OpenAI chat ``messages`` into (text, file_paths, input_modalities, parts).

    ``parts`` is the ordered sequence as written; the other three derive from
    it — text newline-joined, attachments persisted under ``upload_dir`` and
    grouped by modality, ``input_modalities`` the per-part modality sequence.
    So an attachment's position and a repeated modality both survive. Each
    text part carries its message's role, so a reply stays its own turn.
    """
    parts: list[PromptPart] = []
    file_paths: dict[str, list[str]] = {}

    def add_file(modality: str, path: str) -> None:
        # shortcut: always a user turn, so the model can't tell an image it made from one it was
        # sent, and a reply that is only an image leaves no assistant turn; a feature like
        # multi-turn editing would give the turn break between two attachments a text slot
        paths = file_paths.setdefault(modality, [])
        parts.append(PromptPart(modality=modality, index=len(paths), role="user"))
        paths.append(path)

    def add_text(text: str, role: str) -> None:
        # Adjacent text parts were newline-joined before ordering was kept;
        # merge them here, across messages of one role too (the layout has no
        # slot for the boundary between them), so only text an attachment or a
        # role change separates gets its own part, and the rendered prompt
        # keeps the separator it used to have.
        if parts and parts[-1].modality == "text" and parts[-1].role == role:
            parts[-1] = replace(parts[-1], text=f"{parts[-1].text}\n{text}")
            return
        parts.append(PromptPart(modality="text", text=text, role=role))

    try:
        for msg in messages or []:
            # Messages may be pydantic ChatMessage objects or plain dicts.
            if isinstance(msg, dict):
                role, content = msg.get("role"), msg.get("content")
            else:
                role, content = getattr(msg, "role", None), getattr(msg, "content", None)
            if role == "developer":  # OpenAI's newer name for system
                role = "system"
            elif role not in ("system", "user", "assistant"):
                raise ValueError(f"a message's role must be system, developer, user or assistant, not {role!r}")
            if content is None:
                continue
            if isinstance(content, str):
                if content:
                    add_text(content, role)
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    if part.get("text"):
                        add_text(part["text"], role)
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
                elif ptype == "audio_url":  # data:/http audio input in an ``audio_url`` content part
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
    except BaseException:
        # A refused chat is never submitted, so no deferred cleanup covers
        # the attachments saved before the refusal.
        media_io.remove_uploads(file_paths, upload_dir)
        raise

    text_parts = [p.text for p in parts if p.modality == "text"]
    text = "\n".join(text_parts) if text_parts else None
    return text, file_paths, [p.modality for p in parts], parts


def _passthrough(req) -> dict:
    """Unknown request fields (e.g. from the OpenAI client's ``extra_body``)
    flow through verbatim as model_kwargs."""
    extra = getattr(req, "model_extra", None) or {}
    return dict(extra)


def _forward_sent_speed(req, mk: dict) -> None:
    """A client-sent ``speed`` into ``model_kwargs`` for a model with no speed
    control, so the server logs it as ignored. ``SpeechRequest.speed`` defaults
    to 1.0, so an unsent one stays out; it would warn on every request."""
    if "speed" in getattr(req, "model_fields_set", ()):
        mk.setdefault("speed", req.speed)


def _refuse_tool_calls(req) -> None:
    """Refuse a chat that asks the model to call a tool.

    No adapter here parses a tool call out of a reply, so a client sent plain
    text takes it for the model declining every tool. An empty ``tools`` and a
    ``tool_choice`` of ``none`` or ``auto`` ask for no call; clients send them
    by default. A ``tool_choice`` of ``none`` asks for no call even when
    ``tools`` lists some. Their older names, ``functions`` and
    ``function_call``, follow the same rules.
    """
    extra = req.model_extra or {}
    choice = extra.get("tool_choice")
    function_call = extra.get("function_call")
    if (
        (extra.get("tools") not in (None, []) and choice != "none")
        or (extra.get("functions") not in (None, []) and function_call != "none")
        or choice == "required"
        or isinstance(choice, dict)
        or function_call not in (None, "none", "auto")
    ):
        raise ValueError("tool calling is not supported for this model")


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
    Qwen3-Omni's speech route targets its Talker with ``talker_temperature``;
    a plain key sets a model's primary stage), so callers pass them in. ``setdefault`` lets an
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
        # is-not-None, not ``or``: an explicit 0 must reach validation (a 400), not fall through
        max_tokens = getattr(req, "max_completion_tokens", None)
        if max_tokens is None:
            max_tokens = getattr(req, "max_tokens", None)
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
    supports_videos: bool = False   # POST /v1/videos/generations
    supports_realtime: bool = False  # /v1/realtime (bidirectional speech WebSocket)
    supports_transcriptions: bool = False  # POST /v1/audio/transcriptions
    # POST /v1/messages (Anthropic) and /v1/responses, translated onto chat: a
    # text-only chat that serves roles and tools
    supports_messages: bool = False
    supports_responses: bool = False
    # Longest clip the model transcribes in one request. Longer uploads are cut
    # into windows of this length by the transcription route (None means unlimited).
    max_audio_seconds: float | None = None
    # Window length for long uploads when shorter than the clip bound (None:
    # the bound); a request can override it with ``window_seconds`` in extra_body.
    window_seconds: float | None = None
    # Cut a long upload into the fewest windows of at most the window length,
    # of about equal length, instead of full windows and a remainder.
    balanced_windows: bool = False
    # Uploads up to this length run as one window even when longer than the
    # window length (None: the window length). A request's ``window_seconds``
    # sets both.
    whole_seconds: float | None = None
    # How far back from its nominal boundary a cut may move to the quietest
    # frame, and the frame length compared (None: the route's defaults).
    window_search_seconds: float | None = None
    window_frame_seconds: float | None = None
    # How the windows of a long upload run: "sequential" feeds each window the
    # previous transcript as ``initial_prompt`` (openai-whisper's carry-over),
    # "parallel" submits them all at once. A request can override with
    # ``long_form`` in extra_body.
    long_form: str = "sequential"
    # Sequential long form: decode every window with timestamps and start the
    # next one where the last closed segment ended, instead of at a fixed
    # boundary (openai-whisper's seek). Needs ``parse_transcript`` to report
    # segments and ``unfinished``.
    seeks_by_timestamps: bool = False
    # Sequential long form: a window whose text compresses better than this
    # (gzip bytes ratio) is a repetition loop and is decoded again at rising
    # temperatures (openai-whisper's fallback). None disables the check.
    compression_ratio_threshold: float | None = 2.4
    # Streaming transcription over /v1/realtime: the model must be able to
    # continue a hypothesis it is handed (see ``realtime_step_request``).
    supports_realtime_transcription: bool = False

    # ``/v1/audio/speech`` sentence chunking (``serving_speech``): inputs of at
    # least ``speech_chunk_min_chars`` characters are split into sentence
    # groups of about ``speech_chunk_max_chars`` and synthesized as ordered
    # sub-requests. ``None`` keeps the whole text in one request unless the
    # client sends ``sentence_chunking: true``.
    speech_chunk_min_chars: int | None = None
    speech_chunk_max_chars: int = 400
    # sub-requests kept in flight ahead of the one being streamed
    speech_chunk_lookahead: int = 2
    # most sub-requests one input may split into; longer inputs get a 400
    speech_chunk_max_pieces: int = 32

    def split_speech_text(self, text: str) -> list[str]:
        """The sentence chunks of a ``/v1/audio/speech`` input that chunking applies to."""
        return split_sentences(text, max_chars=self.speech_chunk_max_chars)

    def chat_to_request(self, req: ChatCompletionRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        # Output modalities vary by model: e.g. Qwen3-Omni speech output also
        # emits text, whereas BAGEL chat is text-only.
        raise NotImplementedError("chat is not supported by this model")

    def make_output_parser(self, req: ChatCompletionRequest):  # noqa: ARG002
        """A parser for a chat reply with structure inside its text (reasoning,
        tool calls), or None to return the text as ``content``."""
        return None

    def stop_strings(self, req: ChatCompletionRequest) -> list[str]:  # noqa: ARG002
        """The strings the chat handler ends the reply at (``stop``). An adapter
        that does not serve them returns none."""
        return []

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        raise NotImplementedError("audio/speech is not supported by this model")

    def image_to_request(self, req: ImageGenerationRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        raise NotImplementedError("image generation is not supported by this model")

    def video_to_request(self, req: VideoGenerationRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        raise NotImplementedError("video generation is not supported by this model")

    def image_edit_to_request(self, prompt: str, image_path: str, extra_kwargs: dict) -> SubmitArgs:  # noqa: ARG002
        raise NotImplementedError("image editing is not supported by this model")

    def transcription_to_request(self, req: TranscriptionRequest, audio_path: str) -> SubmitArgs:  # noqa: ARG002
        raise NotImplementedError("audio/transcriptions is not supported by this model")

    def parse_transcript(self, text: str, req: TranscriptionRequest) -> Transcript:  # noqa: ARG002
        """Structured view of a model's raw text stream. The default is plain
        text; a model whose stream carries control tokens (Whisper's language
        and timestamp tokens, an LLM decoder's tags) parses them here."""
        return Transcript(text=text.strip())

    def final_text(self, text: str) -> str:
        """The finished transcript's text as the client gets it (a response,
        or the stream's ``transcript.text.done``); not applied to partial
        text, which only grows. The default passes it through."""
        return text

    def stream_text(self, raw: str) -> str:
        """The client-facing text of the raw stream so far. It must only
        grow as ``raw`` does, so a model with control tokens strips them and
        holds back a tail that may still turn out to be one. The default
        passes the text through."""
        return raw

    def realtime_step_request(self, req: TranscriptionRequest, audio_path: str, prefix: str) -> SubmitArgs:  # noqa: ARG002
        """One streaming step: transcribe ``audio_path`` (everything heard so
        far) continuing from ``prefix``, the stable part of the previous raw
        hypothesis. Models that cannot continue a hypothesis leave this
        unimplemented and stay off ``/v1/realtime``."""
        raise NotImplementedError("realtime transcription is not supported by this model")


def _size_kwargs(size: str | None, mk: dict) -> None:
    """OpenAI ``size`` (``WxH``) onto the model's ``width`` / ``height``."""
    if not size:
        return
    try:
        width, height = (int(v) for v in size.lower().split("x"))
    except ValueError:
        raise ValueError(f"size must be 'WxH' (e.g. '1024x1024'); got {size!r}") from None
    mk.setdefault("width", width)
    mk.setdefault("height", height)


class BagelAdapter(OpenAIAdapter):
    """BAGEL: text chat (text out) + text-to-image / image editing.

    Chat honors ``temperature`` / ``top_p`` / ``max_tokens`` / ``seed``, and
    ``top_k`` / ``repetition_penalty`` / ``penalize_prompt`` via ``extra_body``
    (``min_p`` is refused). Images map ``size`` (``WxH``) onto ``width`` /
    ``height``; on an edit they bound the output, keeping the input's aspect.
    """

    supports_chat = True
    supports_images = True

    def chat_to_request(self, req: ChatCompletionRequest, upload_dir: Path) -> SubmitArgs:
        _refuse_tool_calls(req)
        text, file_paths, in_mods, parts = flatten_messages(req.messages, upload_dir)
        mk = _passthrough(req)
        _apply_sampling(req, mk)
        return SubmitArgs(
            text=text,
            file_paths=file_paths or None,
            input_modalities=in_mods,
            output_modalities=["text"],
            model_kwargs=mk,
            prompt_parts=parts,
        )

    def image_to_request(self, req: ImageGenerationRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        mk = _passthrough(req)
        _size_kwargs(getattr(req, "size", None), mk)
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
        mk = dict(extra_kwargs or {})
        _size_kwargs(mk.pop("size", None), mk)
        return SubmitArgs(
            text=prompt,
            file_paths={"image": [image_path]},
            input_modalities=["image", "text"],
            output_modalities=["image"],
            model_kwargs=mk,
        )


class Qwen3OmniAdapter(OpenAIAdapter):
    """Qwen3-Omni: multimodal chat (text, optionally + speech) and TTS.

    Sampling is split across three stages: the Thinker (text) takes plain or
    ``thinker_*`` keys, the Talker (speech) ``talker_*``, and the Talker's
    CodePredictor (residual codec groups) ``code_predictor_*``; a stage key wins
    over a plain one. Everything
    except temperature/top_p is not an OpenAI field (``talker_top_k``,
    ``talker_repetition_penalty``, ``code_predictor_top_k``, …) — pass those
    via ``extra_body``. A speech request's ``speed`` is reported as ignored.
    """

    supports_chat = True
    supports_speech = True
    supports_realtime = True

    def _voice(self, req) -> str | None:
        audio_cfg = getattr(req, "audio", None) or {}
        if isinstance(audio_cfg, dict) and audio_cfg.get("voice") is not None:
            return audio_cfg["voice"]
        return getattr(req, "voice", None)

    def chat_to_request(self, req: ChatCompletionRequest, upload_dir: Path) -> SubmitArgs:
        _refuse_tool_calls(req)
        text, file_paths, in_mods, parts = flatten_messages(req.messages, upload_dir)
        mk = _passthrough(req)
        # Speech output also emits text, so request both modalities when audio is asked for.
        want_audio = bool(req.modalities and "audio" in req.modalities)
        out_mods = ["text", "audio"] if want_audio else ["text"]
        _apply_sampling(req, mk, temperature_key="thinker_temperature", top_p_key="thinker_top_p")
        voice = self._voice(req)
        # is-not-None: an empty voice reaches the model, which refuses it
        if voice is not None:
            mk["voice"] = voice
        return SubmitArgs(
            text=text,
            file_paths=file_paths or None,
            input_modalities=in_mods,
            output_modalities=out_mods,
            model_kwargs=mk,
            prompt_parts=parts,
        )

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        # Qwen3-Omni is a chat model; /v1/audio/speech returns the audio of its
        # spoken response to ``input`` (the handler keeps only the audio).
        mk = _passthrough(req)
        if getattr(req, "voice", None) is not None:
            mk["voice"] = req.voice
        # Talker (speech) sampling; max_tokens is not an OpenAI speech field.
        _apply_sampling(req, mk, temperature_key="talker_temperature", top_p_key="talker_top_p", max_tokens_key=None)
        _forward_sent_speed(req, mk)
        return SubmitArgs(
            text=req.input,
            input_modalities=["text"],
            output_modalities=["text", "audio"],
            model_kwargs=mk,
        )


class Qwen3_5Adapter(OpenAIAdapter):
    """Qwen3.5: text, image and video chat (``image_url`` / ``video_url`` parts), text out.

    Chat only; the other ``/v1/*`` surfaces stay 404. Sampling keys:
    ``temperature``, ``top_p``, ``max_output_tokens`` and ``seed``, plus
    non-OpenAI fields via ``extra_body``:

    * ``top_k``; ``min_p`` is refused (the sampler has no min-p filter).
    * ``repetition_penalty`` — counts the prompt's tokens as well as the
      generated ones unless ``penalize_prompt`` is false.
    * ``enable_thinking`` (default true) — the chat template opens a
      ``<think>`` block. Set it false for short answers.
    """

    supports_chat = True

    def chat_to_request(self, req: ChatCompletionRequest, upload_dir: Path) -> SubmitArgs:
        _refuse_tool_calls(req)
        text, file_paths, in_mods, parts = flatten_messages(req.messages, upload_dir)
        mk = _passthrough(req)
        _apply_sampling(req, mk)
        return SubmitArgs(
            text=text,
            file_paths=file_paths or None,
            input_modalities=in_mods,
            output_modalities=["text"],
            model_kwargs=mk,
            prompt_parts=parts,
        )


class OrpheusAdapter(OpenAIAdapter):
    """Orpheus: text-to-speech (audio out only). ``temperature`` / ``top_p`` /
    ``seed`` map directly; ``top_k``, ``repetition_penalty`` and
    ``penalize_prompt`` travel through ``extra_body``. ``speed`` is not an
    Orpheus control and is reported as ignored."""

    supports_speech = True

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        mk = _passthrough(req)
        if getattr(req, "voice", None) is not None:
            mk["voice"] = req.voice
        _apply_sampling(req, mk, temperature_key="temperature", top_p_key="top_p", max_tokens_key=None)
        _forward_sent_speed(req, mk)
        return SubmitArgs(
            text=req.input,
            input_modalities=["text"],
            output_modalities=["audio"],
            model_kwargs=mk,
        )


class OmniVoiceAdapter(OpenAIAdapter):
    """OmniVoice: zero-shot TTS in three modes over one endpoint.

    Mode follows from what the caller sends, as it does in the reference:
    ``ref_audio`` + ``ref_text`` clones a voice, ``instruct`` designs one from a
    description, and neither lets the model pick. ``voice`` is accepted as an
    alias for ``instruct`` so an OpenAI client that only knows the standard
    field can still reach voice design.

    ``speed`` is a real OmniVoice knob (it scales the estimated duration), so
    unlike the other speech adapters it is forwarded rather than dropped. The
    decoding knobs -- ``num_step``, ``guidance_scale``, ``t_shift``,
    ``language``, ``duration``, the two temperatures -- arrive through
    ``extra_body`` and stay per-request all the way to the unmask loop.

    ``temperature`` / ``top_p`` are NOT mapped: OmniVoice does not sample one
    token per position, it ranks the whole canvas by confidence. The nearest
    knobs are ``class_temperature`` (which token) and ``position_temperature``
    (which cell reveals next), and silently aliasing the OpenAI fields onto
    either would mislead.
    """

    supports_speech = True

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:
        mk = _passthrough(req)
        # is-not-None, so an empty voice reaches the model and is refused there
        if getattr(req, "voice", None) is not None and mk.get("instruct") is None:
            mk["instruct"] = req.voice
        mk.pop("voice", None)
        if getattr(req, "speed", None) is not None:
            mk.setdefault("speed", req.speed)
        if getattr(req, "seed", None) is not None:
            mk.setdefault("seed", req.seed)

        file_paths = None
        ref_audio = mk.pop("ref_audio", None)
        input_modalities = ["text"]
        if ref_audio:
            path = _resolve_ref_audio(ref_audio, upload_dir, decoder="soundfile+torchcodec")
            # Keyed by modality: the data worker iterates the dict and
            # loads each group through the matching loader.
            file_paths = {"audio": [path]}
            input_modalities = ["text", "audio"]

        return SubmitArgs(
            text=req.input,
            file_paths=file_paths,
            input_modalities=input_modalities,
            output_modalities=["audio"],
            model_kwargs=mk,
        )


class Qwen3TTSAdapter(OpenAIAdapter):
    """Qwen3-TTS (CustomVoice / VoiceDesign / Base): text-to-speech, audio only.

    ``voice`` selects a built-in speaker (CustomVoice). ``instructions`` (the
    OpenAI field; ``instruct`` also accepted) carries the style or voice
    description. Non-standard knobs travel through ``extra_body``:
    ``language``, ``non_streaming_mode``, ``top_k``, ``repetition_penalty``,
    the residual-group ``code_predictor_*`` sampling, ``max_output_tokens``.
    ``temperature`` / ``top_p`` / ``seed`` map onto the Talker sampler.
    ``speed`` is not a Qwen3-TTS control and is reported as ignored.

    Inputs of 600+ characters are synthesized as ordered sentence chunks
    (``serving_speech``): each Talker request stays near the lengths the
    model was trained on and the chunks batch like independent requests.
    Benchmark sentences (5-40 words) never reach the threshold.
    """

    supports_speech = True
    speech_chunk_min_chars = 600
    speech_chunk_max_chars = 400

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:
        mk = _passthrough(req)
        if getattr(req, "voice", None) is not None:
            mk["voice"] = req.voice
        # OpenAI's field is ``instructions``; the model reads ``instruct``.
        instructions = mk.pop("instructions", None)
        if instructions:
            mk.setdefault("instruct", instructions)
        _apply_sampling(req, mk, temperature_key="temperature", top_p_key="top_p", max_tokens_key=None)
        _forward_sent_speed(req, mk)
        # Voice clone (Base): ``ref_audio`` is a data URL, an http(s) URL (when
        # the server allows it) or bare base64. It becomes the
        # request's audio input for one request; ``ref_text`` and
        # ``x_vector_only_mode`` ride along as model kwargs. Named, persisted
        # voices come with the shared voice registry (engine/voice-registry).
        ref_audio = mk.pop("ref_audio", None)
        if ref_audio:
            path = _resolve_ref_audio(ref_audio, upload_dir, decoder="soundfile")
            return SubmitArgs(
                text=req.input,
                file_paths={"audio": [path]},
                input_modalities=["audio", "text"],
                output_modalities=["audio"],
                model_kwargs=mk,
            )
        return SubmitArgs(
            text=req.input,
            input_modalities=["text"],
            output_modalities=["audio"],
            model_kwargs=mk,
        )


class KokoroAdapter(OpenAIAdapter):
    """Kokoro-82M: text-to-speech, audio out only.

    ``voice`` is a bundled voice or a blend (``af_bella+af_sky``,
    ``af_bella(2)+af_sky(1)``) and ``speed`` scales the predicted durations;
    both are real model knobs and are forwarded. Kokoro does not sample, so
    ``temperature`` / ``top_p`` are ignored. ``lang_code`` (a G2P language
    override; the default follows the voice's prefix) and ``phonemes`` (skip
    G2P) arrive through ``extra_body``.
    """

    supports_speech = True

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        mk = _passthrough(req)
        if getattr(req, "voice", None) is not None:
            mk.setdefault("voice", req.voice)
        if getattr(req, "speed", None) is not None:
            mk.setdefault("speed", req.speed)
        if getattr(req, "seed", None) is not None:
            mk.setdefault("seed", req.seed)
        return SubmitArgs(
            text=req.input,
            input_modalities=["text"],
            output_modalities=["audio"],
            model_kwargs=mk,
        )


# A ref_audio URL that did not deliver: an HTTP error status or a refused
# redirect (HTTPError), DNS or connect failures (URLError), a timeout, a dropped
# connection, a TLS failure or a body cut short. These are the request's fault.
_REF_AUDIO_FETCH_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    TimeoutError,
    ConnectionError,
    ssl.SSLError,
)


def _ref_audio_fetch_failure(exc: BaseException) -> str:
    """What went wrong with a ref_audio fetch, in words that name no server path."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    return type(exc).__name__


def _resolve_ref_audio(ref: str, upload_dir: Path, decoder: str) -> str:
    """A reference voice clip from the request: a data URL, an http(s) URL (only
    when the server allows remote fetch), or bare base64 that ``decoder`` (the
    model's audio loader) can open. Returns its path. Every form gets the same
    ``check_ref_audio``, so a clip the data worker would fail on, or one longer
    than ``MAX_REF_AUDIO_SECONDS``, is a 400 here.

    A clip that is not audio, malformed base64 and a failed fetch are 400s.
    Failing to store the clip (a full disk, ``upload_dir`` permissions) is the
    server's fault: a 500 whose text, like the 400s, names no server path."""
    if not isinstance(ref, str):
        raise ValueError(f"ref_audio must be a data URL or base64 string, got {type(ref).__name__}")
    try:
        if ref.startswith("data:") or ref.split(":", 1)[0].lower() in ("http", "https"):
            try:
                path = media_io.resolve_media_ref(ref, upload_dir)[1]
            except _REF_AUDIO_FETCH_ERRORS as exc:
                raise ValueError(f"could not fetch ref_audio ({_ref_audio_fetch_failure(exc)})") from exc
            media_io.check_ref_audio(path, decoder)
            return path
        path = media_io.save_base64_audio(ref, upload_dir, decoder)
    except OSError as exc:
        logger.error("could not store a ref_audio upload under %s: %r", upload_dir, exc)
        raise RuntimeError("the server could not store ref_audio") from None
    if path:
        return path
    raise ValueError(
        "server file paths are refused; ref_audio must be a data URL, an http(s) URL "
        "or base64 audio the model can decode"
    )


class ChatterboxAdapter(OpenAIAdapter):
    """Chatterbox / Chatterbox-Turbo: zero-shot TTS with voice cloning.

    ``voice`` is either ``"default"`` (the voice shipped with the checkpoint)
    or the name of a preset clip in the deployment's ``voices_dir``. A
    reference clip for cloning comes through ``ref_audio`` in ``extra_body``
    (a data URL, bare base64 or, when the server allows it, an http(s) URL;
    paths on the server are refused, and so is a clip over 60 s or below
    8 kHz) and is loaded by the worker like an uploaded file.
    The model's own knobs (``exaggeration``, ``cfg_weight``, ``min_p``,
    ``repetition_penalty``, ``top_k``, ``n_cfm_timesteps``, ``watermark``,
    ``max_output_tokens`` and, for the multilingual checkpoint,
    ``language_id``) pass through ``extra_body`` verbatim;
    ``temperature``/``top_p``/``seed`` are the standard fields. Turbo refuses a
    non-zero ``cfg_weight`` / ``exaggeration`` / ``min_p``. ``speed`` is not a
    Chatterbox control and is reported as ignored.
    """

    supports_speech = True

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:
        mk = _passthrough(req)
        if getattr(req, "voice", None) is not None:
            mk.setdefault("voice", req.voice)
        _apply_sampling(req, mk, temperature_key="temperature", top_p_key="top_p", max_tokens_key=None)
        _forward_sent_speed(req, mk)
        file_paths = None
        input_modalities = ["text"]
        ref_audio = mk.pop("ref_audio", None)
        if ref_audio:
            path = _resolve_ref_audio(ref_audio, upload_dir, decoder="soundfile+torchcodec")
            file_paths = {"audio": [path]}
            input_modalities = ["text", "audio"]
        return SubmitArgs(
            text=req.input,
            file_paths=file_paths,
            input_modalities=input_modalities,
            output_modalities=["audio"],
            model_kwargs=mk,
        )


class FishS2ProAdapter(OpenAIAdapter):
    """Fish Audio S2-Pro: text-to-speech with optional reference-audio voice cloning.

    The voice comes from a reference clip: ``ref_audio`` in ``extra_body`` (a
    data URL, bare base64, or, when the server allows remote fetch, an http(s)
    URL; paths on the server are refused) with its transcript in ``ref_text``. Without
    one the model picks a voice. ``voice`` and ``speed`` are not S2-Pro
    controls and are dropped; inline ``[tag]`` prosody control is part of
    the input text. ``temperature`` / ``top_p`` / ``seed`` are the standard
    fields; ``top_k``, ``repetition_penalty``, ``do_sample`` and
    ``max_new_tokens`` pass through ``extra_body``. Long inputs are
    synthesized as ordered sentence chunks, as for Qwen3-TTS.
    """

    supports_speech = True
    speech_chunk_min_chars = 600
    speech_chunk_max_chars = 400

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:
        mk = _passthrough(req)
        _apply_sampling(req, mk, temperature_key="temperature", top_p_key="top_p", max_tokens_key=None)
        ref_audio = mk.pop("ref_audio", None)
        if ref_audio:
            path = _resolve_ref_audio(ref_audio, upload_dir, decoder="soundfile")
            return SubmitArgs(
                text=req.input,
                file_paths={"audio": [path]},
                input_modalities=["text", "audio"],
                output_modalities=["audio"],
                model_kwargs=mk,
            )
        mk.pop("ref_text", None)
        return SubmitArgs(
            text=req.input,
            input_modalities=["text"],
            output_modalities=["audio"],
            model_kwargs=mk,
        )


class CosyVoice3Adapter(OpenAIAdapter):
    """Fun-CosyVoice3: TTS that always clones a voice.

    ``voice`` names a preset in the deployment's ``voices_dir`` (with its
    transcript in ``<voice>.txt``); ``ref_audio`` in ``extra_body`` uploads a
    clip instead (a data URL, an http(s) URL when the server allows remote
    media, or bare base64 audio; never a server path), with its transcript in
    ``ref_text``. ``instructions`` becomes ``instruct``. ``temperature``,
    ``top_p`` and ``seed`` are the standard fields; ``top_k``, ``ras_window``,
    ``ras_tau``, ``n_cfm_timesteps`` and ``max_new_tokens`` (25 Hz speech
    tokens) pass through ``extra_body``. ``speed`` is dropped.

    Long text is synthesized in sentence chunks, as the reference frontend
    (``text_normalize``) does: pieces of about 80 tokens, or 60 to 80
    characters for text with Chinese, cut at sentence ends (and at ``：`` and
    ``、`` with Chinese), with a short tail merged into the piece before it.
    A Chinese piece over 300 characters is cut again at its commas.
    In one request the LM ends long text early or garbles its end. Without Chinese, an estimate of the Qwen2 BPE token
    count stands in for the reference's real one, so Hangul, kana and
    Cyrillic pieces stay near 80 tokens as English ones do.
    """

    supports_speech = True
    _CHARS_PER_TOKEN = 4
    speech_chunk_max_chars = 80 * _CHARS_PER_TOKEN
    # nothing shorter than the smallest piece (80 characters of Chinese) is split
    speech_chunk_min_chars = 81

    # A sentence of the reference's ``split_paragraph(lang="zh")``: up to a CJK
    # sentence or clause mark (with a closing quote), or a Latin one before a
    # space or the end, so "3.5" stays whole.
    _ZH_SENTENCE = re.compile(r".*?(?:[。？！；：、][\"”]?|[.?!;](?=\s|\Z)|\Z)", re.DOTALL)
    # The same with commas, for a piece too long for the LM.
    _ZH_CLAUSE = re.compile(r".*?(?:[。？！；：、，][\"”]?|[.?!;,](?=\s|\Z)|\Z)", re.DOTALL)
    # A Han character is one token of this tokenizer, and Latin text fewer per
    # character, so a piece's length bounds its tokens. The LM takes 512 tokens
    # of prompt text and text: 7 for the system prompt, at most about 200 for
    # the transcript of a 30 s reference clip, and 300 for the piece.
    _ZH_MAX_PIECE_CHARS = 300

    def split_speech_text(self, text: str) -> list[str]:
        if re.search(r"[\u4e00-\u9fff]", text):  # the reference's ``contains_chinese``
            return self._split_chinese(text)
        # The reference counts real tokens here; characters per token differ
        # by script (Hangul or kana about 1.5, Cyrillic about 2.7, English about 4.7).
        chars_per_token = len(text) / max(1.0, self._estimated_tokens(text))
        max_chars = min(self.speech_chunk_max_chars, max(80, int(80 * chars_per_token)))
        return split_sentences(text, max_chars=max_chars, min_chars=int(20 * chars_per_token))

    @classmethod
    def _split_chinese(cls, text: str) -> list[str]:
        """Text with Chinese cut as the reference cuts it: sentences joined with
        nothing between them until a piece is past 60 characters and the next
        sentence would take it past 80, a tail under 20 merged into the piece
        before it, and a longer sentence kept whole rather than wrapped.

        A piece over ``_ZH_MAX_PIECE_CHARS``, which the reference would send
        whole but which could take a request past the LM's 512 tokens (a 400,
        or audio cut short when it is not the first piece), is cut again at
        commas as well, as ``split_paragraph(comma_split=True)`` cuts, after a clause over
        80 characters (a run with no punctuation) is cut into even parts; every
        piece of that is under 160 characters."""
        pieces: list[str] = []
        for piece in cls._group_chinese(cls._ZH_SENTENCE.findall(text)):
            if len(piece) <= cls._ZH_MAX_PIECE_CHARS:
                pieces.append(piece)
                continue
            clauses = cls._ZH_CLAUSE.findall(piece)
            pieces += cls._group_chinese([c for clause in clauses for c in _cut_evenly(clause, 80)])
        return pieces

    @staticmethod
    def _group_chinese(sentences: list[str]) -> list[str]:
        """The grouping of the reference's ``split_paragraph(lang="zh")``."""
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if len(current + sentence) > 80 and len(current) > 60:
                pieces.append(current)
                current = ""
            current += sentence
        if current.strip():
            if len(current) < 20 and pieces:
                pieces[-1] += current
            else:
                pieces.append(current)
        return [piece.strip() for piece in pieces]

    @classmethod
    def _estimated_tokens(cls, text: str) -> float:
        """An upper estimate of the Qwen2 BPE token count of ``text``."""
        return sum(
            1 / cls._CHARS_PER_TOKEN if ord(ch) < 0x80 else 0.5 if ord(ch) < 0x800 else 1.0 for ch in text
        )

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:
        mk = _passthrough(req)
        if getattr(req, "voice", None):
            mk.setdefault("voice", req.voice)
        instructions = mk.pop("instructions", None)
        if instructions:
            mk.setdefault("instruct", instructions)
        _apply_sampling(req, mk, temperature_key="temperature", top_p_key="top_p", max_tokens_key=None)
        file_paths = None
        input_modalities = ["text"]
        ref_audio = mk.pop("ref_audio", None)
        if ref_audio:
            file_paths = {"audio": [_resolve_ref_audio(ref_audio, upload_dir, decoder="soundfile")]}
            input_modalities = ["text", "audio"]
        return SubmitArgs(
            text=req.input,
            file_paths=file_paths,
            input_modalities=input_modalities,
            output_modalities=["audio"],
            model_kwargs=mk,
        )


class Cosmos3Adapter(OpenAIAdapter):
    """NVIDIA Cosmos3: text-to-image and text/image-to-video generation.

    ``size`` ("WxH") maps to the generation resolution; ``seed`` and any
    extra knobs (``guidance_scale``, ``num_inference_steps``, ``negative_prompt``,
    and for video ``num_frames`` / ``fps``) pass through via ``extra_body``.
    """

    supports_images = True
    supports_videos = True

    def image_to_request(self, req: ImageGenerationRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        mk = _passthrough(req)
        if getattr(req, "size", None):
            mk.setdefault("size", req.size)
        if getattr(req, "seed", None) is not None:
            mk.setdefault("seed", req.seed)
        return SubmitArgs(
            text=req.prompt,
            input_modalities=["text"],
            output_modalities=["image"],
            model_kwargs=mk,
        )

    def video_to_request(self, req: VideoGenerationRequest, upload_dir: Path) -> SubmitArgs:
        mk = _passthrough(req)
        if getattr(req, "size", None):
            mk.setdefault("size", req.size)
        if getattr(req, "seed", None) is not None:
            mk.setdefault("seed", req.seed)
        # num_frames / fps are first-class video fields (not in extra_body).
        if getattr(req, "num_frames", None) is not None:
            mk.setdefault("num_frames", req.num_frames)
        if getattr(req, "fps", None) is not None:
            mk.setdefault("fps", req.fps)
        # Image-to-video: the conditioning frame (URL / data URI) is persisted and
        # loaded by the worker, which VAE-encodes it into the clean frame-0 anchor.
        # Video-to-video: the conditioning video is persisted the same way; the
        # worker VAE-encodes its prefix and pins the requested clean latent
        # frames (condition_frame_indexes_vision / condition_video_keep via
        # extra_body).
        image = getattr(req, "image", None)
        video = getattr(req, "video", None)
        if image and video:
            raise ValueError("Provide either 'image' or 'video' conditioning, not both.")
        if image:
            _, path = media_io.resolve_media_ref(image, upload_dir)
            return SubmitArgs(
                text=req.prompt,
                file_paths={"image": [path]},
                input_modalities=["image", "text"],
                output_modalities=["video"],
                model_kwargs=mk,
            )
        if video:
            _, path = media_io.resolve_media_ref(video, upload_dir)
            return SubmitArgs(
                text=req.prompt,
                file_paths={"video": [path]},
                input_modalities=["video", "text"],
                output_modalities=["video"],
                model_kwargs=mk,
            )
        return SubmitArgs(
            text=req.prompt,
            input_modalities=["text"],
            output_modalities=["video"],
            model_kwargs=mk,
        )


class Cosmos3EdgeAdapter(Cosmos3Adapter):
    """Cosmos3-Edge: the generator surfaces of :class:`Cosmos3Adapter` plus
    chat (the reasoner: the understanding tower served as a VLM).

    Chat requests take the OpenAI ``messages`` layout with image / video
    attachments in order; ``temperature`` / ``top_p`` / ``max_tokens`` map to
    the reasoner's sampler (``max_tokens`` defaults to 2048), and ``extra_body``
    knobs (``enable_thinking``, ``top_k``, ``repetition_penalty``,
    ``penalize_prompt``, ``video_fps`` / ``video_num_frames``) pass through;
    ``min_p`` is refused.
    """

    supports_chat = True

    def chat_to_request(self, req: ChatCompletionRequest, upload_dir: Path) -> SubmitArgs:
        _refuse_tool_calls(req)
        mk = _passthrough(req)
        _apply_sampling(req, mk)
        # Clients written for the Edge model's serving recipe toggle thinking
        # through ``chat_template_kwargs``; accept it alongside a flat
        # ``enable_thinking``. Read before the attachments are saved, since a
        # malformed value refuses the request.
        template_kwargs = mk.pop("chat_template_kwargs", None) or {}
        if "enable_thinking" in template_kwargs:
            # raw, not bool(): the model refuses a non-boolean rather than reading "false" as true
            mk.setdefault("enable_thinking", template_kwargs["enable_thinking"])
        text, file_paths, in_mods, parts = flatten_messages(req.messages, upload_dir)
        return SubmitArgs(
            text=text,
            file_paths=file_paths or None,
            input_modalities=in_mods,
            output_modalities=["text"],
            model_kwargs=mk,
            prompt_parts=parts,
        )


class Wan22Adapter(OpenAIAdapter):
    """Wan2.2-TI2V-5B: text/image-to-video generation (video only).

    ``size`` ("WxH") maps to the model's ``width`` / ``height`` kwargs; ``seed``
    and any extra knobs (``guidance_scale``, ``num_inference_steps``,
    ``negative_prompt``) pass through via ``extra_body``. ``fps`` is a playback
    rate (the mp4 container rate), not a generation knob — see
    ``Wan22Model.postprocess``.
    """

    supports_videos = True

    def video_to_request(self, req: VideoGenerationRequest, upload_dir: Path) -> SubmitArgs:
        mk = _passthrough(req)
        if getattr(req, "size", None):
            # Unlike cosmos3, the model has no "size" kwarg — its generation
            # knobs are width/height, so the adapter does the split here.
            try:
                width, height = (int(v) for v in req.size.lower().split("x"))
            except ValueError:
                raise ValueError(f"size must be 'WxH' (e.g. '832x480'); got {req.size!r}") from None
            mk.setdefault("width", width)
            mk.setdefault("height", height)
        if getattr(req, "seed", None) is not None:
            mk.setdefault("seed", req.seed)
        # num_frames / fps are first-class video fields (not in extra_body).
        if getattr(req, "num_frames", None) is not None:
            mk.setdefault("num_frames", req.num_frames)
        if getattr(req, "fps", None) is not None:
            mk.setdefault("fps", req.fps)
        # Image-to-video: the conditioning frame (URL / data URI) is persisted
        # and loaded by the worker, which VAE-encodes it into the frame-0
        # anchor. Wan2.2 has no video-conditioned mode.
        if getattr(req, "video", None):
            raise ValueError("Wan2.2 does not support 'video' conditioning; provide 'image' or neither.")
        image = getattr(req, "image", None)
        if image:
            _, path = media_io.resolve_media_ref(image, upload_dir)
            return SubmitArgs(
                text=req.prompt,
                file_paths={"image": [path]},
                input_modalities=["image", "text"],
                output_modalities=["video"],
                model_kwargs=mk,
            )
        return SubmitArgs(
            text=req.prompt,
            input_modalities=["text"],
            output_modalities=["video"],
            model_kwargs=mk,
        )


class LTX25Adapter(OpenAIAdapter):
    """LTX-2.5: text-to-audio+video generation.

    Returns one mp4 with the generated audio muxed in as an AAC track by default;
    ``audio: false`` in ``extra_body`` asks for video alone. ``size`` ("WxH") maps to
    the model's ``width`` / ``height``; ``seed``, ``num_frames`` and ``fps`` are
    first-class fields, and the model's other knobs (``recipe``) pass through via
    ``extra_body``. No image or video conditioning yet.
    """

    supports_videos = True

    def video_to_request(self, req: VideoGenerationRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        if getattr(req, "image", None) or getattr(req, "video", None):
            raise ValueError("LTX-2.5 (this port) is text-to-video only; omit 'image' and 'video'.")
        mk = _passthrough(req)
        with_audio = mk.pop("audio", True)
        if getattr(req, "size", None):
            try:
                width, height = (int(v) for v in req.size.lower().split("x"))
            except ValueError:
                raise ValueError(f"size must be 'WxH' (e.g. '960x544'); got {req.size!r}") from None
            mk.setdefault("width", width)
            mk.setdefault("height", height)
        for name in ("seed", "num_frames", "fps"):
            if getattr(req, name, None) is not None:
                mk.setdefault(name, getattr(req, name))
        return SubmitArgs(
            text=req.prompt,
            input_modalities=["text"],
            output_modalities=["video", "audio"] if with_audio else ["video"],
            model_kwargs=mk,
        )


class DiffusionImageAdapter(OpenAIAdapter):
    """Generic adapter for the DiT-scaffold image models (FLUX.2 klein, Z-Image, ...):
    text-to-image and reference-image editing.

    ``size`` ("WxH") maps to the models' ``width`` / ``height`` kwargs; ``seed`` is
    honoured by the conductor; every other knob (``num_inference_steps``,
    ``guidance_scale`` on models that take it, ``negative_prompt``) passes through
    ``extra_body``. ``n`` is served as ``n`` concurrent requests by the images handler,
    which the equal-shape batching then runs together.
    """

    supports_images = True

    def image_to_request(self, req: ImageGenerationRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        mk = _passthrough(req)
        _size_kwargs(getattr(req, "size", None), mk)
        if getattr(req, "seed", None) is not None:
            mk.setdefault("seed", req.seed)
        return SubmitArgs(
            text=req.prompt,
            input_modalities=["text"],
            output_modalities=["image"],
            model_kwargs=mk,
        )

    def image_edit_to_request(self, prompt: str, image_path: str, extra_kwargs: dict) -> SubmitArgs:
        mk = dict(extra_kwargs or {})
        _size_kwargs(mk.pop("size", None), mk)
        return SubmitArgs(
            text=prompt,
            file_paths={"image": [image_path]},
            input_modalities=["image", "text"],
            output_modalities=["image"],
            model_kwargs=mk,
        )

# Whisper-style control tokens: ``<|en|>`` (language), ``<|12.34|>`` (timestamp),
# and the task/format markers. ASR models render the ones that carry
# information (language, timestamps) into their text stream so the adapter can
# lift them out here. Everything else is dropped.
_CONTROL_TOKEN = re.compile(r"<\|([^|<>]*)\|>")
# a control token cut off by the end of the stream so far
_PARTIAL_CONTROL_TOKEN = re.compile(r"<(\|[^|<>]*\|?)?$")
_TIMESTAMP = re.compile(r"^\d+\.\d{2}$")
_LANGUAGE = re.compile(r"^[a-z]{2,3}$")


def _transcription_kwargs(req: TranscriptionRequest) -> dict:
    """Map the OpenAI transcription fields shared by every ASR model onto
    ``model_kwargs``. ``language`` / ``temperature`` / ``seed`` pass through
    under their own names; ``prompt`` becomes ``initial_prompt`` (the
    conditioning text, named as faster-whisper does — ``prompt`` itself is
    the request's text argument); a timestamped ``response_format`` or an
    explicit granularity asks the model for timestamps. Neither served ASR
    model reads ``initial_prompt`` or ``timestamps`` yet (and Higgs-Audio not
    ``language``), so those are logged as ignored."""
    mk = _passthrough(req)
    if req.language:
        mk.setdefault("language", req.language)
    if req.prompt:
        mk.setdefault("initial_prompt", req.prompt)
    if req.temperature is not None:
        mk.setdefault("temperature", req.temperature)
    if req.seed is not None:
        mk.setdefault("seed", req.seed)
    granularities = set(req.timestamp_granularities or ())
    if "word" in granularities:
        mk.setdefault("timestamps", "word")
    elif granularities or req.response_format in ("verbose_json", "srt", "vtt"):
        mk.setdefault("timestamps", "segment")
    # checked here too, so a bad knob is a 400 before a stream commits to a 200
    return normalize_generation_kwargs(mk)


class WhisperAdapter(OpenAIAdapter):
    """Whisper (large-v3, large-v3-turbo): speech-to-text.

    The transcript stream is Whisper's own token stream rendered as text.
    ``parse_transcript`` lifts ``<|xx|>`` language and ``<|s.ss|>`` timestamp
    tokens into :class:`Transcript` and ``stream_text`` hides them, but the model
    today forces ``language`` (default ``en``, no detection) and
    ``<|notimestamps|>``, so neither appears and long uploads are cut at fixed
    window boundaries.
    """

    supports_transcriptions = True
    # Whisper hears one 30 s window. The route cuts longer uploads into them,
    # each starting where the previous window's last segment closed.
    max_audio_seconds = 30.0
    seeks_by_timestamps = True

    def transcription_to_request(self, req: TranscriptionRequest, audio_path: str) -> SubmitArgs:
        return SubmitArgs(
            # Whisper is conditioned by its forced token prompt, not free text, so
            # ``prompt`` reaches the model as the ``initial_prompt`` kwarg
            # (``<|startofprev|>`` context).
            text="",
            file_paths={"audio": [audio_path]},
            input_modalities=["audio", "text"],
            output_modalities=["text"],
            model_kwargs=_transcription_kwargs(req),
        )

    def stream_text(self, raw: str) -> str:
        raw = raw.partition(WORDS_MARKER)[0]  # word timings follow the transcript
        return _CONTROL_TOKEN.sub("", _PARTIAL_CONTROL_TOKEN.sub("", raw))

    def parse_transcript(self, text: str, req: TranscriptionRequest) -> Transcript:
        text, _, timed_words = text.partition(WORDS_MARKER)
        language, segments, parts, unfinished = _parse_timestamped(text)
        clean = " ".join(p.strip() for p in parts if p.strip())
        for idx, seg in enumerate(segments):
            seg["id"] = idx
        words = [
            {"word": seg["text"], "start": seg["start"], "end": seg["end"]}
            for seg in _parse_timestamped(timed_words)[1]
        ] if timed_words else []
        return Transcript(
            text=clean, language=language or req.language, segments=segments, words=words,
            unfinished=unfinished,
        )


def _parse_timestamped(text: str) -> tuple[str | None, list[dict], list[str], bool]:
    """Walk Whisper's rendered stream: ``(language, closed segments, text
    pieces in order, whether a segment was left open)``. A segment is the
    text between a start and an end timestamp; text outside timestamps is
    kept as is."""
    language: str | None = None
    segments: list[dict] = []
    parts: list[str] = []
    start: float | None = None
    buffer: list[str] = []
    cursor = 0
    for match in _CONTROL_TOKEN.finditer(text):
        span = text[cursor:match.start()]
        cursor = match.end()
        (buffer if start is not None else parts).append(span)
        token = match.group(1)
        if _TIMESTAMP.match(token):
            stamp = float(token)
            if start is None:
                start = stamp
            else:
                segment_text = "".join(buffer).strip()
                if segment_text:
                    segments.append({"start": start, "end": stamp, "text": segment_text})
                    parts.append(segment_text)
                buffer = []
                start = None
        elif language is None and _LANGUAGE.match(token):
            language = token
    tail = text[cursor:]
    (buffer if start is not None else parts).append(tail)
    if start is not None and "".join(buffer).strip():
        # an open segment at the end of the stream: keep its text, no end
        parts.append("".join(buffer).strip())
    return language, segments, parts, start is not None


class Qwen3ASRAdapter(OpenAIAdapter):
    """Qwen3-ASR: an LLM decoder that writes ``language {Name}<asr_text>{text}``
    (or the text alone when the language is forced). Hears up to 20 minutes
    per request, longer uploads are windowed at that; continues a hypothesis through
    ``assistant_prefix``, which is what the realtime session needs."""

    supports_transcriptions = True
    supports_realtime_transcription = True
    max_audio_seconds = 1200.0
    # 2.4 is Whisper's bound for a 30 s window. The ratio grows with length,
    # so a 20 minute transcript of ordinary speech can pass it.
    compression_ratio_threshold = None

    def transcription_to_request(self, req: TranscriptionRequest, audio_path: str) -> SubmitArgs:
        kwargs = _transcription_kwargs(req)
        # checked here too so a streaming request gets a 400 before it starts
        if "<|audio_pad|>" in str(kwargs.get("initial_prompt") or ""):
            raise ValueError("the prompt must not contain <|audio_pad|>")
        return SubmitArgs(
            text="",
            file_paths={"audio": [audio_path]},
            input_modalities=["audio", "text"],
            output_modalities=["text"],
            model_kwargs=kwargs,
        )

    def realtime_step_request(self, req: TranscriptionRequest, audio_path: str, prefix: str) -> SubmitArgs:
        args = self.transcription_to_request(req, audio_path)
        if prefix:
            args.model_kwargs["assistant_prefix"] = prefix
        return args

    def stream_text(self, raw: str) -> str:
        # A detected language opens the stream as ``language X<asr_text>``.
        # Hold the text back until the tag has arrived and show what follows
        # it. A forced language skips the line, so plain text streams as is.
        if ASR_TEXT_TAG in raw:
            return raw.split(ASR_TEXT_TAG, 1)[1]
        head = raw.lstrip()
        if head.startswith("language") and len(head) < 40 and "\n" not in head:
            return ""
        return raw

    def final_text(self, text: str) -> str:
        from mstar.model.qwen3_asr.repetition import fix_repetitions

        # the reference SDK collapses repetition loops in every finished transcript
        return fix_repetitions(text)

    def parse_transcript(self, text: str, req: TranscriptionRequest) -> Transcript:
        raw = text.strip()
        if ASR_TEXT_TAG not in raw:
            return Transcript(text=raw, language=req.language)
        meta, body = raw.split(ASR_TEXT_TAG, 1)
        language = None
        for line in meta.splitlines():
            line = line.strip()
            if line.lower().startswith("language "):
                language = line[len("language "):].strip() or None
                break
        if language and language.lower() == "none":
            language = None
        return Transcript(text=body.strip(), language=req.language or _qwen3_language_code(language))


def _qwen3_language_code(name: str | None) -> str | None:
    """The ISO code clients expect (what Whisper reports too) for a language
    name Qwen3-ASR wrote; a name outside its list passes through."""
    if not name:
        return None
    from mstar.model.qwen3_asr.config import LANGUAGE_CODES

    for code, canonical in LANGUAGE_CODES.items():
        if canonical.lower() == name.lower():
            return code
    return name


def _untimed_transcription_kwargs(req: TranscriptionRequest, model: str, *, translates: bool = False) -> dict:
    """``model_kwargs`` for a model that takes no conditioning text and emits
    no timestamps: the OpenAI ``prompt`` is dropped, granularities refused.
    Whisper's ``task`` is the caller's to read when the model ``translates``;
    otherwise anything but ``transcribe`` is refused, not silently ignored."""
    mk = _transcription_kwargs(req)
    mk.pop("initial_prompt", None)
    if req.timestamp_granularities:
        raise ValueError(f"{model} does not produce timestamps; leave timestamp_granularities unset")
    mk.pop("timestamps", None)
    if not translates:
        task = mk.pop("task", None)
        if task not in (None, "transcribe"):
            raise ValueError(f"{model} only transcribes; task must be 'transcribe'; got {task!r}")
    return mk


class _UntimedTranscriptionAdapter(OpenAIAdapter):
    """The FastConformer ASR models' adapters: plain text, which a
    SentencePiece vocabulary starts with a space."""

    # greedy, unconditioned windows: a retry could only swap a correct
    # transcript for a sampled one
    compression_ratio_threshold = None

    def stream_text(self, raw: str) -> str:
        # parse_transcript strips the text, so the deltas drop the leading
        # space too; lstrip keeps a growing stream growing
        return raw.lstrip()


class CanaryAdapter(_UntimedTranscriptionAdapter):
    """NVIDIA Canary: speech recognition, and translation to or from English.

    ``language`` is the speech's language (English by default); an
    ``extra_body`` ``target_language`` other than it asks for a translation.
    The decoder takes no free-text conditioning and emits no timestamps, so
    ``prompt`` is dropped and timestamp granularities are refused.
    """

    supports_transcriptions = True
    # Canary was trained on clips of up to 40 s; windows don't condition on
    # each other, so they run in parallel
    max_audio_seconds = 40.0
    long_form = "parallel"

    def transcription_to_request(self, req: TranscriptionRequest, audio_path: str) -> SubmitArgs:
        from mstar.model.canary.config import check_task

        mk = _untimed_transcription_kwargs(req, "Canary", translates=True)
        # Whisper's convention: translate means into English
        task = mk.pop("task", None)
        if task not in (None, "transcribe", "translate"):
            raise ValueError(f"task must be 'transcribe' or 'translate'; got {task!r}")
        source = mk.get("language") or "en"
        target = mk.get("target_language") or ("en" if task == "translate" else source)
        mk["language"], mk["target_language"] = check_task(source, target)
        if not isinstance(mk.get("punctuation", True), bool):
            raise ValueError(f"punctuation must be true or false; got {mk['punctuation']!r}")
        return SubmitArgs(
            text="",
            file_paths={"audio": [audio_path]},
            input_modalities=["audio", "text"],
            output_modalities=["text"],
            model_kwargs=mk,
        )


class CanaryQwenAdapter(_UntimedTranscriptionAdapter):
    """NVIDIA Canary-Qwen: English speech recognition by an LLM, under its
    trained instruction. OpenAI's ``prompt`` is context for the transcript
    (and carries the previous window in sequential long form), which this
    model was not trained to take, so it is dropped; no timestamps."""

    supports_transcriptions = True
    max_audio_seconds = 40.0
    long_form = "parallel"

    def transcription_to_request(self, req: TranscriptionRequest, audio_path: str) -> SubmitArgs:
        from mstar.model.utils import base_language

        mk = _untimed_transcription_kwargs(req, "Canary-Qwen")
        if base_language(mk.get("language") or "en") != "en":
            raise ValueError("Canary-Qwen transcribes English only")
        return SubmitArgs(
            text="",
            file_paths={"audio": [audio_path]},
            input_modalities=["audio", "text"],
            output_modalities=["text"],
            model_kwargs=mk,
        )


class TransducerAdapter(_UntimedTranscriptionAdapter):
    """NVIDIA Parakeet / Nemotron ASR: transducer speech recognition.

    Parakeet-TDT (``detects_language``) detects the language from the speech
    and refuses only a ``language`` it doesn't know; Nemotron takes it as its
    prompt (a locale such as ``en-US``, a bare code, or ``auto``), checked
    against the checkpoint's locales when the upload is preprocessed. Neither
    takes free-text conditioning or emits timestamps here.
    """

    supports_transcriptions = True
    # full attention over the clip; longer uploads run as independent windows
    max_audio_seconds = 600.0
    long_form = "parallel"

    def __init__(self, detects_language: bool = False):
        self.detects_language = detects_language

    def transcription_to_request(self, req: TranscriptionRequest, audio_path: str) -> SubmitArgs:
        mk = _untimed_transcription_kwargs(req, "this model")
        if self.detects_language:
            from mstar.model.parakeet.config import check_detected_language

            check_detected_language(mk.get("language"))
        return SubmitArgs(
            text="",
            file_paths={"audio": [audio_path]},
            input_modalities=["audio", "text"],
            output_modalities=["text"],
            model_kwargs=mk,
        )


class HiggsAudioAdapter(OpenAIAdapter):
    """Higgs-Audio v3 STT: an instruction-following LLM decoder, so the OpenAI
    ``prompt`` is the transcription instruction (the model has a default)."""

    supports_transcriptions = True

    def transcription_to_request(self, req: TranscriptionRequest, audio_path: str) -> SubmitArgs:
        mk = _transcription_kwargs(req)
        mk.pop("initial_prompt", None)
        return SubmitArgs(
            text=req.prompt or "",
            file_paths={"audio": [audio_path]},
            input_modalities=["audio", "text"],
            output_modalities=["text"],
            model_kwargs=mk,
        )


class TextChatAdapter(OpenAIAdapter):
    """Text-only LLMs: chat messages in, text out. The model's ``process_prompt``
    applies its own chat template; sampling fields reach it as model_kwargs."""

    supports_chat = True
    # set by a subclass whose output parser splits tool calls out of the reply
    serves_tool_calls = False

    def chat_to_request(self, req: ChatCompletionRequest, upload_dir: Path) -> SubmitArgs:
        if not self.serves_tool_calls:
            _refuse_tool_calls(req)
        text, _, modalities, parts = flatten_messages(req.messages, upload_dir, allow_remote=False)
        # an image or audio part was decoded to disk and then dropped: the model
        # answered as if it had seen it
        other = sorted(set(modalities) - {"text"})
        if other:
            raise ValueError(f"this model takes text only; the messages carry {other}")
        if not text:
            raise ValueError("messages carry no text content")
        for name in ("max_tokens", "max_completion_tokens"):
            value = getattr(req, name, None)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        # what the model cannot serve is refused, not dropped: off, a field passes
        if req.n not in (None, 1):
            raise ValueError(f"n={req.n}: this model returns one choice")
        self.stop_strings(req)
        mk = _passthrough(req)
        for name in ("logprobs", "top_logprobs", "logit_bias", "presence_penalty", "frequency_penalty",
                     "stop_token_ids"):
            if mk.get(name):
                raise ValueError(f"{name} is not supported; got {mk[name]!r}")
            mk.pop(name, None)
        if mk.pop("best_of", None) not in (None, 1):
            raise ValueError("best_of is not supported; this model returns one choice")
        if mk.pop("response_format", None) not in (None, {"type": "text"}):
            raise ValueError("response_format needs constrained decoding; only {'type': 'text'} is served")
        _apply_sampling(req, mk)
        # extra_body can override max_tokens with anything; a string reached the
        # conductor's min() and took its whole message batch down
        mk = normalize_generation_kwargs(mk)
        return SubmitArgs(
            text=text,
            input_modalities=["text"],
            output_modalities=["text"],
            model_kwargs=mk,
            prompt_parts=parts,
        )

    def stop_strings(self, req: ChatCompletionRequest) -> list[str]:
        """``stop``, as OpenAI takes it: a string or up to 4. The serving layer
        matches them on the decoded answer and ends the request at the first."""
        stops = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
        if len(stops) > 4:
            raise ValueError(f"stop takes at most 4 strings, got {len(stops)}")
        if "" in stops:
            raise ValueError("a stop string cannot be empty")
        return stops


class GlmChatAdapter(TextChatAdapter):
    """GLM-5.2 / GLM-5.3: the messages with their roles, and ``tools``, go to
    the checkpoint's chat template; the reply comes back with
    ``reasoning_content`` and ``tool_calls`` split out (see ``glm_chat``).
    Anthropic's ``/v1/messages`` and OpenAI's ``/v1/responses`` are translated
    onto this chat.

    ``tool_choice`` is ``"auto"`` (default), ``"none"``, which leaves the tools
    out of the prompt, ``"required"`` or a named function. With no grammar, a
    forced call is the prompt's: it ends in ``<tool_call>`` (and the name), so
    the reply opens inside the call. ``"required"`` over one tool names it;
    over several the model picks, and a name it makes up stays text. The call
    opens the reply, so a forced call has no reasoning: GLM-5.2's prompt ends
    as with thinking off, GLM-5.3's closes the empty ``<think>``, and an
    explicit ``enable_thinking`` with it is a 400.

    ``parallel_tool_calls: false`` ends the reply, and the request, when its
    first call ends.

    ``enable_thinking`` (GLM-5.2 only: GLM-5.3 always reasons, and turning it
    off is a 400), ``reasoning_effort`` and ``clear_thinking`` reach the
    template, flat in ``extra_body`` or under ``chat_template_kwargs``; other
    template keys are dropped, as vLLM drops the ones a template does not read.
    """

    supports_messages = True
    supports_responses = True
    serves_tool_calls = True

    def __init__(self, *, always_thinks: bool, reasoning_efforts: tuple[str, ...]):
        self.always_thinks = always_thinks
        # the template's reasoning_effort levels, least first; it reads any other as the last
        self.reasoning_efforts = reasoning_efforts

    def chat_to_request(self, req: ChatCompletionRequest, upload_dir: Path) -> SubmitArgs:
        # flatten_messages refuses a tool turn; the template takes each turn's role
        # from template_messages below, so the base checks read a tool result as user text
        checked = req.model_copy(update={"messages": [
            m.model_copy(update={"role": "user"}) if m.role == "tool" else m for m in req.messages
        ]})
        args = super().chat_to_request(checked, upload_dir)
        mk = args.model_kwargs
        tools, template_kwargs, forced = self._options(mk)
        thinking = template_kwargs.get("enable_thinking")
        if self.always_thinks and thinking is not None and not thinking:
            # the template has no switch, so off would still reason
            raise ValueError("this model always reasons; enable_thinking cannot turn it off")
        mk["messages"] = glm_chat.template_messages(req.messages)
        if tools:
            mk["tools"] = tools
        if forced is not None:
            # GLM-5.3's prompt ends inside <think>: an empty reasoning, as the template
            # renders a turn without one
            mk["reply_prefix"] = (glm_chat.THINK_END if self.always_thinks else "") + glm_chat.CALL_START + forced
        mk["chat_template_kwargs"] = template_kwargs
        return args

    def make_output_parser(self, req: ChatCompletionRequest):
        mk = _passthrough(req)
        single_call = mk.get("parallel_tool_calls") is False
        tools, template_kwargs, forced = self._options(mk)
        # truthiness, as the GLM-5.2 template reads it
        thinking = self.always_thinks or bool(template_kwargs.get("enable_thinking", True))
        params = glm_chat.tool_parameters(tools) if tools else None
        if forced:
            # the prompt opened a call to this function: a name the model goes on to write
            # (get_weather -> get_weather_hourly) is not another tool's call
            params = {forced: params[forced]}
        return glm_chat.GlmReplyParser(
            thinking=thinking, tools=params, forced_call=forced, single_call=single_call,
        )

    def _options(self, mk: dict) -> tuple[list | None, dict, str | None]:
        """Takes ``tools``, ``tool_choice`` and the template's keys out of ``mk``.
        The third is the call a forced ``tool_choice`` opens the reply with:
        the function's name, or ``""`` for the model to name one."""
        if "reply_prefix" in mk:
            raise ValueError("reply_prefix is not a request field; tool_choice sets it")
        # false ends the reply at its first call (the parser's single_call); read, not sent on
        parallel = mk.pop("parallel_tool_calls", None)
        if parallel is not None and not isinstance(parallel, bool):
            raise ValueError(f"parallel_tool_calls must be a boolean, got {parallel!r}")
        tools = mk.pop("tools", None) or None
        choice = mk.pop("tool_choice", None)
        names = list(glm_chat.tool_parameters(tools)) if tools is not None else []
        template_kwargs = mk.pop("chat_template_kwargs", None) or {}
        if not isinstance(template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be an object")
        template_kwargs = {k: v for k, v in template_kwargs.items() if k in glm_chat.TEMPLATE_KWARGS}
        for key in glm_chat.TEMPLATE_KWARGS:
            if key in mk:
                template_kwargs.setdefault(key, mk.pop(key))
        forced = None
        if choice == "none":
            tools = None
        elif choice == "required" or isinstance(choice, dict):
            if not names:
                raise ValueError(f"tool_choice {choice!r} needs tools")
            if choice == "required":
                forced = names[0] if len(names) == 1 else ""
            else:
                fn = choice.get("function")
                forced = fn.get("name") if choice.get("type") == "function" and isinstance(fn, dict) else None
                if forced not in names:
                    raise ValueError(f"tool_choice {choice!r} names no function in tools {names}")
            if template_kwargs.get("enable_thinking"):
                raise ValueError("a forced tool call opens the reply, which leaves no room to reason; "
                                 "leave enable_thinking unset, or use tool_choice 'auto'")
            if not self.always_thinks:
                # the prompt ends in <think></think>, as a turn that does not reason
                template_kwargs["enable_thinking"] = False
        elif choice not in (None, "auto"):
            raise ValueError(f"tool_choice takes 'auto', 'none', 'required' or "
                             f"{{'type': 'function', 'function': {{'name': ...}}}}; got {choice!r}")
        return tools, template_kwargs, forced


class OmnilingualASRAdapter(OpenAIAdapter):
    """Omnilingual ASR CTC: speech-to-text in 1,600+ languages, no language code.

    The model hears at most 40 s (the reference rejects longer clips) and is
    most accurate on shorter ones, so a clip of up to 20 s is transcribed
    whole, exactly as the reference does, and a longer upload is cut into the
    fewest windows of at most 16 s, of about equal length, each ending at the
    quietest 0.1 s within 5 s of its boundary. The windows run in parallel: a
    CTC model takes no conditioning text, so they are independent.
    ``window_seconds`` in extra_body (5 to 40) sets both lengths for one
    request. ``language``, ``prompt`` and ``temperature`` have no
    effect (argmax CTC decoding has nothing to condition or sample). There
    are no timestamps: each window with text is one segment spanning it, so
    ``srt`` / ``vtt`` get one cue and ``verbose_json`` one segment per window.
    """

    supports_transcriptions = True
    max_audio_seconds = 40.0
    # CER on FLEURS concatenations is lowest with clips up to 20 s heard whole
    # and longer ones cut into balanced windows of at most 16 s at a pause (40 s
    # windows more than double it)
    window_seconds = 16.0
    whole_seconds = 20.0
    balanced_windows = True
    window_search_seconds = 5.0
    window_frame_seconds = 0.1
    long_form = "parallel"
    # argmax decoding has no temperature to fall back to
    compression_ratio_threshold = None

    def transcription_to_request(self, req: TranscriptionRequest, audio_path: str) -> SubmitArgs:
        return SubmitArgs(
            text="",
            file_paths={"audio": [audio_path]},
            input_modalities=["audio"],
            output_modalities=["text"],
            model_kwargs=_transcription_kwargs(req),
        )


# Only models with an OpenAI-standard surface are registered. Action/world-model
# models (pi05, vjepa2) are deliberately absent → /v1/* 404s; use /generate.
ADAPTER_REGISTRY: dict[str, OpenAIAdapter] = {
    "bagel": BagelAdapter(),
    "chatterbox": ChatterboxAdapter(),
    "chatterbox_multilingual": ChatterboxAdapter(),
    "chatterbox_turbo": ChatterboxAdapter(),
    "fish_s2_pro": FishS2ProAdapter(),
    "cosyvoice3": CosyVoice3Adapter(),
    "qwen3_omni": Qwen3OmniAdapter(),
    "omnivoice": OmniVoiceAdapter(),
    "orpheus": OrpheusAdapter(),
    "qwen3_tts": Qwen3TTSAdapter(),
    "qwen3_tts_1p7b": Qwen3TTSAdapter(),
    "qwen3_tts_voicedesign": Qwen3TTSAdapter(),
    "qwen3_tts_base": Qwen3TTSAdapter(),
    "kokoro": KokoroAdapter(),
    "cosmos3": Cosmos3Adapter(),
    "cosmos3_droid": Cosmos3Adapter(),
    "cosmos3_edge": Cosmos3EdgeAdapter(),
    "cosmos3_edge_droid": Cosmos3EdgeAdapter(),
    "cosmos3_super": Cosmos3Adapter(),
    "cosmos3_super_i2v_4step": Cosmos3Adapter(),
    "cosmos3_super_t2i_4step": Cosmos3Adapter(),
    "wan22": Wan22Adapter(),
    "ltx2_5": LTX25Adapter(),
    "whisper_large": WhisperAdapter(),
    "whisper_large_v3_turbo": WhisperAdapter(),
    "qwen3_asr": Qwen3ASRAdapter(),
    "qwen3_asr_realtime": Qwen3ASRAdapter(),
    "canary": CanaryAdapter(),
    "canary_qwen": CanaryQwenAdapter(),
    "parakeet": TransducerAdapter(detects_language=True),
    "nemotron_asr": TransducerAdapter(),
    "higgs_audio": HiggsAudioAdapter(),
    "omnilingual_asr_ctc_300m": OmnilingualASRAdapter(),
    "omnilingual_asr_ctc_1b": OmnilingualASRAdapter(),
    "omnilingual_asr_ctc_3b": OmnilingualASRAdapter(),
    "omnilingual_asr_ctc_7b": OmnilingualASRAdapter(),
    "flux2_klein": DiffusionImageAdapter(),
    "flux2_klein_9b": DiffusionImageAdapter(),
    "z_image_turbo": DiffusionImageAdapter(),
    "glm52": GlmChatAdapter(always_thinks=False, reasoning_efforts=("high", "max")),
    "glm5_next": GlmChatAdapter(always_thinks=True, reasoning_efforts=("low", "high", "max")),
}

# One key per size; every size takes the same adapter.
ADAPTER_REGISTRY.update({
    f"qwen3_5_{size}b": Qwen3_5Adapter() for size in QWEN_3_5_DENSE_SIZES
})
# Qwen3.8-27B is the Qwen3.5 architecture and chat template family.
ADAPTER_REGISTRY["qwen3_8_27b"] = Qwen3_5Adapter()


def get_adapter(model_name: str) -> OpenAIAdapter | None:
    return ADAPTER_REGISTRY.get(model_name)

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

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from mstar.api_server import media_io
from mstar.model.multimodal import PromptPart
from mstar.model.qwen3_asr.constants import ASR_TEXT_TAG
from mstar.model.registry import qwen_3_5_dense_sizes as QWEN_3_5_DENSE_SIZES
from mstar.model.whisper.config import WORDS_MARKER

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
    supports_videos: bool = False   # POST /v1/videos/generations
    supports_realtime: bool = False  # /v1/realtime (bidirectional speech WebSocket)
    supports_transcriptions: bool = False  # POST /v1/audio/transcriptions
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

    def chat_to_request(self, req: ChatCompletionRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        # Output modalities vary by model: e.g. Qwen3-Omni speech output also
        # emits text, whereas BAGEL chat is text-only.
        raise NotImplementedError("chat is not supported by this model")

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


class BagelAdapter(OpenAIAdapter):
    """BAGEL: text chat (text out) + text-to-image / image editing.

    BAGEL's ``get_sampling_config`` reads the model config, so per-request
    ``temperature`` / ``top_p`` are not honored; ``max_output_tokens`` and
    ``seed`` are.
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

    Sampling is split across three stages: the Thinker (text) takes
    ``thinker_*`` keys, the Talker (speech) ``talker_*``, and the Talker's
    CodePredictor (residual codec groups) ``code_predictor_*``. Everything
    except temperature/top_p is not an OpenAI field (``talker_top_k``,
    ``talker_repetition_penalty``, ``code_predictor_top_k``, …) — pass those
    via ``extra_body``.
    """

    supports_chat = True
    supports_speech = True
    supports_realtime = True

    def _voice(self, req) -> str | None:
        audio_cfg = getattr(req, "audio", None) or {}
        if isinstance(audio_cfg, dict) and audio_cfg.get("voice"):
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
        if voice:
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


class Qwen3_5Adapter(OpenAIAdapter):
    """Qwen3.5: text, image and video chat (``image_url`` / ``video_url`` parts), text out.

    Chat only; the other ``/v1/*`` surfaces stay 404. Sampling keys:
    ``temperature``, ``top_p``, ``max_output_tokens`` and ``seed``, plus two
    non-OpenAI fields via ``extra_body``:

    * ``repetition_penalty`` — the sampler applies it over the prompt's tokens
      as well as the generated ones.
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
        if getattr(req, "voice", None) and not mk.get("instruct"):
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
    the residual-group ``subtalker_*`` sampling, ``max_new_tokens``.
    ``temperature`` / ``top_p`` / ``seed`` map onto the Talker sampler.

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
        if getattr(req, "voice", None):
            mk["voice"] = req.voice
        # OpenAI's field is ``instructions``; the model reads ``instruct``.
        instructions = mk.pop("instructions", None)
        if instructions:
            mk.setdefault("instruct", instructions)
        _apply_sampling(req, mk, temperature_key="temperature", top_p_key="top_p", max_tokens_key=None)
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
        if getattr(req, "voice", None):
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


def _resolve_ref_audio(ref: str, upload_dir: Path, decoder: str) -> str:
    """A reference voice clip from the request: a data URL, an http(s) URL (only
    when the server allows remote fetch), or bare base64 that ``decoder`` (the
    model's audio loader) can open. Returns its path. Every form gets the same
    ``check_ref_audio``, so a clip the data worker would fail on, or one longer
    than ``MAX_REF_AUDIO_SECONDS``, is a 400 here."""
    if not isinstance(ref, str):
        raise ValueError(f"ref_audio must be a data URL or base64 string, got {type(ref).__name__}")
    if ref.startswith("data:") or ref.split(":", 1)[0].lower() in ("http", "https"):
        path = media_io.resolve_media_ref(ref, upload_dir)[1]
        media_io.check_ref_audio(path, decoder)
        return path
    path = media_io.save_base64_audio(ref, upload_dir, decoder)
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
    The model's own knobs --
    ``exaggeration``, ``cfg_weight``, ``min_p``, ``repetition_penalty``,
    ``top_k``, ``n_cfm_timesteps``, ``watermark``, ``max_new_tokens`` and, for
    the multilingual checkpoint, ``language_id`` -- pass through ``extra_body``
    verbatim; ``temperature``/``top_p``/``seed`` are the standard fields.
    ``speed`` is not a Chatterbox control and is dropped.
    """

    supports_speech = True

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:
        mk = _passthrough(req)
        if getattr(req, "voice", None):
            mk.setdefault("voice", req.voice)
        _apply_sampling(req, mk, temperature_key="temperature", top_p_key="top_p", max_tokens_key=None)
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
    the reasoner's sampler, and ``extra_body`` knobs (``enable_thinking``,
    ``top_k``, ``repetition_penalty``, ``video_fps`` / ``video_num_frames``)
    pass through.
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
            mk.setdefault("enable_thinking", bool(template_kwargs["enable_thinking"]))
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
    first-class fields, and any other knobs pass through via
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

    @staticmethod
    def _size_kwargs(size: str | None, mk: dict) -> None:
        if not size:
            return
        try:
            width, height = (int(v) for v in size.lower().split("x"))
        except ValueError:
            raise ValueError(f"size must be 'WxH' (e.g. '1024x1024'); got {size!r}") from None
        mk.setdefault("width", width)
        mk.setdefault("height", height)

    def image_to_request(self, req: ImageGenerationRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        mk = _passthrough(req)
        self._size_kwargs(getattr(req, "size", None), mk)
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
        self._size_kwargs(mk.pop("size", None), mk)
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


# sampling knobs the engine reads as numbers; a string there fails every
# request batched with it, so they are checked before submission
_NUMERIC_KWARGS = {
    "temperature": float, "top_p": float, "min_p": float, "repetition_penalty": float,
    "top_k": int, "max_output_tokens": int, "seed": int,
}


def _check_numeric_kwargs(mk: dict) -> dict:
    """Coerce the known numeric ``model_kwargs`` to numbers, or raise ValueError."""
    for key, kind in _NUMERIC_KWARGS.items():
        value = mk.get(key)
        if value is None or (kind is int and isinstance(value, int) and not isinstance(value, bool)):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number; got {value!r}") from None
        if not math.isfinite(number) or (kind is int and not number.is_integer()):
            raise ValueError(f"{key} must be a finite {kind.__name__}; got {value!r}")
        mk[key] = kind(number)
    return mk


def _transcription_kwargs(req: TranscriptionRequest) -> dict:
    """Map the OpenAI transcription fields shared by every ASR model onto
    ``model_kwargs``. ``language`` / ``temperature`` / ``seed`` pass through
    under their own names; ``prompt`` becomes ``initial_prompt`` (the
    conditioning text, named as faster-whisper does — ``prompt`` itself is
    the request's text argument); a timestamped ``response_format`` or an
    explicit granularity asks the model for timestamps."""
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
    return _check_numeric_kwargs(mk)


class WhisperAdapter(OpenAIAdapter):
    """Whisper (large-v3, large-v3-turbo): speech-to-text.

    The transcript stream is Whisper's own token stream rendered as text:
    a leading ``<|xx|>`` language token when the language was detected or
    forced, and ``<|s.ss|>`` timestamp tokens around each segment when
    timestamps were requested. ``parse_transcript`` lifts those into
    :class:`Transcript`; ``stream_text`` hides them from streaming clients.
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

    def chat_to_request(self, req: ChatCompletionRequest, upload_dir: Path) -> SubmitArgs:
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
        if req.stop:
            raise ValueError("stop sequences are not supported; the reply ends at a stop token or max_tokens")
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
        _check_numeric_kwargs(mk)
        if mk.get("max_output_tokens") is not None and mk["max_output_tokens"] < 1:
            raise ValueError(f"max_output_tokens must be at least 1, got {mk['max_output_tokens']}")
        return SubmitArgs(
            text=text,
            input_modalities=["text"],
            output_modalities=["text"],
            model_kwargs=mk,
            prompt_parts=parts,
        )


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
    "glm52": TextChatAdapter(),
}

# One key per size; every size takes the same adapter.
ADAPTER_REGISTRY.update({
    f"qwen3_5_{size}b": Qwen3_5Adapter() for size in QWEN_3_5_DENSE_SIZES
})
# Qwen3.8-27B is the Qwen3.5 architecture and chat template family.
ADAPTER_REGISTRY["qwen3_8_27b"] = Qwen3_5Adapter()


def get_adapter(model_name: str) -> OpenAIAdapter | None:
    return ADAPTER_REGISTRY.get(model_name)

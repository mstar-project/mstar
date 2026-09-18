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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from mstar.api_server import media_io
from mstar.model.multimodal import PromptPart

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
    messages: list, upload_dir: Path, allow_remote: bool = True
) -> tuple[str | None, dict[str, list[str]], list[str], list[PromptPart]]:
    """Flatten OpenAI chat ``messages`` into (text, file_paths, input_modalities, parts).

    ``parts`` is the ordered sequence as written; the other three derive from
    it — text newline-joined, attachments persisted under ``upload_dir`` and
    grouped by modality, ``input_modalities`` the per-part modality sequence.
    So an attachment's position and a repeated modality both survive.
    (Multi-turn role structure is flattened — a v1 simplification; the models
    apply their own prompt formatting in ``process_prompt``.)
    """
    parts: list[PromptPart] = []
    file_paths: dict[str, list[str]] = {}

    def add_file(modality: str, path: str) -> None:
        paths = file_paths.setdefault(modality, [])
        parts.append(PromptPart(modality=modality, index=len(paths)))
        paths.append(path)

    def add_text(text: str) -> None:
        # Adjacent text parts were newline-joined before ordering was kept;
        # merge them here so only text an attachment separates gets its own
        # part, and the rendered prompt keeps the separator it used to have.
        if parts and parts[-1].modality == "text":
            parts[-1] = PromptPart(
                modality="text", text=f"{parts[-1].text}\n{text}"
            )
            return
        parts.append(PromptPart(modality="text", text=text))

    for msg in messages or []:
        # Messages may be pydantic ChatMessage objects or plain dicts.
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if content is None:
            continue
        if isinstance(content, str):
            if content:
                add_text(content)
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                if part.get("text"):
                    add_text(part["text"])
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

    text_parts = [p.text for p in parts if p.modality == "text"]
    text = "\n".join(text_parts) if text_parts else None
    return text, file_paths, [p.modality for p in parts], parts


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
    supports_videos: bool = False   # POST /v1/videos/generations
    supports_realtime: bool = False  # /v1/realtime (bidirectional speech WebSocket)
    supports_transcriptions: bool = False  # POST /v1/audio/transcriptions
    # Longest clip the model transcribes in one request; longer uploads are cut
    # into windows of this length by the transcription route (None: unlimited).
    max_audio_seconds: float | None = None
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
    # (gzip bytes ratio) is a repetition loop; it is decoded again at rising
    # temperatures (openai-whisper's fallback). None disables the check.
    compression_ratio_threshold: float | None = 2.4
    # Streaming transcription over /v1/realtime: the model must be able to
    # continue a hypothesis it is handed (see ``realtime_step_request``).
    supports_realtime_transcription: bool = False

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

    def stream_delta(self, text: str) -> str:
        """The client-facing part of one streamed text chunk. The default
        passes it through; a model with control tokens strips them."""
        return text

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


# Whisper-style control tokens: ``<|en|>`` (language), ``<|12.34|>`` (timestamp),
# and the task/format markers. ASR models render the ones that carry
# information (language, timestamps) into their text stream so the adapter can
# lift them out here; everything else is dropped.
_CONTROL_TOKEN = re.compile(r"<\|([^|<>]*)\|>")
_TIMESTAMP = re.compile(r"^\d+\.\d{2}$")
_LANGUAGE = re.compile(r"^[a-z]{2,3}$")
# Whisper's word timings follow the transcript after this marker, one
# ``<|start|> word <|end|>`` per word in the same timestamp vocabulary
_WORDS_MARKER = "<|startoflm|>"


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
    return mk


class WhisperAdapter(OpenAIAdapter):
    """Whisper (large-v3, large-v3-turbo): speech-to-text.

    The transcript stream is Whisper's own token stream rendered as text:
    a leading ``<|xx|>`` language token when the language was detected or
    forced, and ``<|s.ss|>`` timestamp tokens around each segment when
    timestamps were requested. ``parse_transcript`` lifts those into
    :class:`Transcript`; ``stream_delta`` hides them from streaming clients.
    """

    supports_transcriptions = True
    # Whisper hears one 30 s window; the route cuts longer uploads into them,
    # each starting where the previous window's last segment closed.
    max_audio_seconds = 30.0
    seeks_by_timestamps = True

    def transcription_to_request(self, req: TranscriptionRequest, audio_path: str) -> SubmitArgs:
        return SubmitArgs(
            # Whisper is conditioned by its forced token prompt, not free text;
            # ``prompt`` reaches the model as the ``initial_prompt`` kwarg
            # (``<|startofprev|>`` context).
            text="",
            file_paths={"audio": [audio_path]},
            input_modalities=["audio", "text"],
            output_modalities=["text"],
            model_kwargs=_transcription_kwargs(req),
        )

    def stream_delta(self, text: str) -> str:
        if _WORDS_MARKER in text:
            return ""  # the word timings arrive as one chunk after the transcript
        return _CONTROL_TOKEN.sub("", text)

    def parse_transcript(self, text: str, req: TranscriptionRequest) -> Transcript:
        text, _, timed_words = text.partition(_WORDS_MARKER)
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


# Only models with an OpenAI-standard surface are registered. Action/world-model
# models (pi05, vjepa2) are deliberately absent → /v1/* 404s; use /generate.
ADAPTER_REGISTRY: dict[str, OpenAIAdapter] = {
    "bagel": BagelAdapter(),
    "qwen3_omni": Qwen3OmniAdapter(),
    "orpheus": OrpheusAdapter(),
    "cosmos3": Cosmos3Adapter(),
    "cosmos3_droid": Cosmos3Adapter(),
    "cosmos3_super": Cosmos3Adapter(),
    "wan22": Wan22Adapter(),
    "whisper_large": WhisperAdapter(),
    "higgs_audio": HiggsAudioAdapter(),
}


def get_adapter(model_name: str) -> OpenAIAdapter | None:
    return ADAPTER_REGISTRY.get(model_name)

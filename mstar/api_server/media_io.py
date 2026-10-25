"""Media decode/encode helpers shared by the native and OpenAI-compatible APIs.

Two directions:

* **Inbound** — turn media referenced by an OpenAI-style request (``data:`` URLs,
  ``http(s)`` URLs, or base64 blobs) into files under the API server's
  ``upload_dir``, so ``model.load_image`` / ``load_audio`` / ``load_video`` can
  read them by path. This is the same contract ``/generate`` already uses for
  multipart uploads.
* **Outbound** — wrap raw model audio output (16-bit PCM, no container header)
  into a real audio container (WAV by default) and encode image bytes (PNG) as a
  ``data:`` URL for OpenAI chat image output.

Only stdlib + numpy are required. ``mp3`` / ``flac`` / ``ogg`` encoding is opt-in
and degrades to WAV when the optional ``soundfile`` backend is unavailable, so
the base install stays slim.
"""

from __future__ import annotations

import base64
import io
import math
import os
import threading
import time
import wave
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np

# MIME (top-level type or full type) -> file extension used when persisting.
_MIME_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/aac": ".aac",
    "audio/m4a": ".m4a",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
}

# Output audio container -> (extension, mime type) for OpenAI audio responses.
AUDIO_FORMAT_MIME: dict[str, str] = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/ogg",
}

# response_format -> soundfile (format, subtype); wav and pcm need no encoder
_SOUNDFILE_AUDIO: dict[str, tuple[str, str | None]] = {
    "mp3": ("MP3", None),
    "flac": ("FLAC", None),
    "opus": ("OGG", "OPUS"),
}


# libsndfile's Opus encoder takes only these rates (44.1 kHz fails)
_OPUS_RATES = (8000, 12000, 16000, 24000, 48000)


def audio_formats(stream: bool = False, sample_rate: int | None = None) -> tuple[str, ...]:
    """The ``response_format`` values this frontend can produce.

    A stream carries an open-ended WAV or bare PCM, so only those two; the
    compressed containers need the optional ``soundfile`` (the ``audio`` extra),
    and Opus also needs one of its native sample rates when ``sample_rate`` is given.
    """
    if stream:
        return ("wav", "pcm")
    try:
        import soundfile  # type: ignore  # noqa: F401
    except ImportError:
        return ("wav", "pcm")
    return ("wav", "pcm", *(
        f for f in _SOUNDFILE_AUDIO
        if f != "opus" or sample_rate is None or int(sample_rate) in _OPUS_RATES
    ))


def check_audio_format(fmt: str | None, stream: bool = False, sample_rate: int | None = None) -> str:
    """``fmt`` lower-cased (``wav`` when unset); ValueError when it cannot be produced."""
    fmt = (fmt or "wav").lower()
    allowed = audio_formats(stream, sample_rate)
    if fmt not in allowed:
        how = "streamed" if stream else "produced"
        raise ValueError(
            f"response_format {fmt!r} cannot be {how} by this server; use one of: {', '.join(allowed)}"
        )
    return fmt


def modality_from_mime(mime: str) -> str:
    """Map a MIME type to one of our modality strings (image/audio/video)."""
    top = (mime or "").split("/", 1)[0].lower()
    if top in ("image", "audio", "video"):
        return top
    return "unknown"


def _ext_for(mime: str, fallback: str = ".bin") -> str:
    mime = (mime or "").lower()
    if mime in _MIME_TO_EXT:
        return _MIME_TO_EXT[mime]
    # Fall back to the top-level type's most common extension.
    top = mime.split("/", 1)[0]
    return {"image": ".png", "audio": ".wav", "video": ".mp4"}.get(top, fallback)


# ---------------------------------------------------------------------------
# Inbound: persist request media into upload_dir, return (modality, path)
# ---------------------------------------------------------------------------

def _save_bytes(raw: bytes, mime: str, upload_dir: Path) -> tuple[str, str]:
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid4()}{_ext_for(mime)}"
    path.write_bytes(raw)
    return modality_from_mime(mime), str(path)


def _decode_base64_payload(payload: str) -> bytes:
    """Decode client-provided base64 after normalizing common wire variants."""
    # OpenAI-compatible clients may omit padding or wrap the payload at a
    # fixed column width. Normalize both forms before asking the decoder to
    # validate the alphabet, so malformed requests fail as client errors.
    cleaned = "".join(payload.split())
    padded = cleaned + "=" * (-len(cleaned) % 4)
    return base64.b64decode(padded, validate=True)


def save_data_url(data_url: str, upload_dir: Path) -> tuple[str, str]:
    """Persist a ``data:<mime>;base64,<payload>`` URL. Returns (modality, path)."""
    header, _, payload = data_url.partition(",")
    if not payload:
        raise ValueError("Malformed data URL: missing payload")
    mime = header[len("data:"):].split(";", 1)[0] or "application/octet-stream"
    raw = _decode_base64_payload(payload)
    return _save_bytes(raw, mime, upload_dir)


def save_base64(b64: str, fmt: str, modality_hint: str, upload_dir: Path) -> tuple[str, str]:
    """Persist a bare base64 blob with a known ``fmt`` (e.g. ``"wav"``)."""
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    raw = _decode_base64_payload(b64)
    # Sanitize the client-controlled fmt: alphanumerics only, so it cannot
    # inject path separators into the upload path.
    clean = "".join(c for c in fmt.lstrip(".") if c.isalnum())
    ext = "." + clean if clean else ".bin"
    path = upload_dir / f"{uuid4()}{ext}"
    path.write_bytes(raw)
    return modality_hint, str(path)


# Telephone band, the lowest rate speech is recorded or coded at. A clip below it
# is not a voice reference, and resampling it up to the probe's or the model's
# rate multiplies its decoded size by that ratio (a 64 KB WAV at 1 Hz is 2 GB at
# 16 kHz).
MIN_REF_AUDIO_SAMPLE_RATE = 8000
# Voice cloning needs seconds of speech. Twice the longest reference a model
# takes (Qwen3-TTS refuses over 30 s, Chatterbox keeps 30 s), so this never
# refuses a clip a model would use; without it a 66 MB FLAC probe-decodes to
# about 23 GB.
MAX_REF_AUDIO_SECONDS = 60.0
# The probe decode runs in the request thread pool (32 threads); this bounds
# how many decode at once.
REF_AUDIO_DECODE_SLOTS = 4
_REF_AUDIO_PROBE_RATE = 16000
_ref_audio_decodes = threading.BoundedSemaphore(REF_AUDIO_DECODE_SLOTS)


class RefAudioTooLongError(ValueError):
    """A clip that opens as audio but runs past ``MAX_REF_AUDIO_SECONDS``."""


def save_base64_audio(b64: str, upload_dir: Path, decoder: str) -> str | None:
    """Persist bare base64 that passes :func:`check_ref_audio`; return its path,
    or None (leaving nothing behind) when it is not base64 or not loadable audio.

    A path such as ``/etc/passwd`` is valid base64 too, so the alphabet cannot
    tell them apart; the decoder can. A clip that is audio but too long raises
    :class:`RefAudioTooLongError`.
    """
    try:
        _, path = save_base64(b64, "wav", "audio", upload_dir)
    except ValueError:
        return None
    try:
        check_ref_audio(path, decoder)
    except RefAudioTooLongError:
        raise
    except ValueError:
        return None
    return path


def check_ref_audio(path: str, decoder: str) -> None:
    """Open the reference clip at ``path`` the way the model's ``load_audio``
    will; on failure delete it and raise ValueError.

    ``decoder`` is the library that ``load_audio`` uses, ``"soundfile"`` or
    ``"torchcodec"``. torchcodec decodes the clip, not just its header: some
    FFmpeg builds parse a container header (MAT4, MAT5) they cannot decode. A
    clip whose header rate is below ``MIN_REF_AUDIO_SAMPLE_RATE``, or whose
    header duration is over ``MAX_REF_AUDIO_SECONDS``, is refused before any
    decode; the decode itself stops just past ``MAX_REF_AUDIO_SECONDS``, so a
    header that hides or understates the length cannot make it unbounded.
    """
    try:
        if decoder == "torchcodec":
            from torchcodec.decoders import AudioDecoder

            header = AudioDecoder(path).metadata
            _require_ref_audio_rate(header.sample_rate)
            _require_ref_audio_seconds(header.duration_seconds_from_header)
            with _ref_audio_decodes:
                # Mono at a fixed rate: a small compressed upload can declare many
                # channels at a high rate, and decoding it natively takes gigabytes.
                samples = AudioDecoder(
                    path, sample_rate=_REF_AUDIO_PROBE_RATE, num_channels=1
                ).get_samples_played_in_range(0.0, MAX_REF_AUDIO_SECONDS + 1.0)
            if samples.duration_seconds == 0:
                raise ValueError("reference audio has no samples in its first seconds")
            _require_ref_audio_seconds(samples.pts_seconds + samples.duration_seconds)
        else:
            import soundfile

            from mstar.model.media_limits import count_sound_file_seconds, sound_file_seconds

            info = soundfile.info(path)
            _require_ref_audio_rate(info.samplerate)
            seconds = sound_file_seconds(info.frames, info.samplerate)
            if seconds is None:
                # no sample count in the header (a FLAC written to a pipe)
                with _ref_audio_decodes:
                    seconds = count_sound_file_seconds(path, MAX_REF_AUDIO_SECONDS)
            _require_ref_audio_seconds(seconds)
    except RefAudioTooLongError:
        Path(path).unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 — not loadable here, so the model cannot load it either
        Path(path).unlink(missing_ok=True)
        raise ValueError(
            f"ref_audio must be audio the model can decode, at {MIN_REF_AUDIO_SAMPLE_RATE} Hz or more"
        ) from exc


def _require_ref_audio_rate(rate: int | None) -> None:
    if not rate or rate < MIN_REF_AUDIO_SAMPLE_RATE:
        raise ValueError(f"reference audio at {rate} Hz is below {MIN_REF_AUDIO_SAMPLE_RATE} Hz")


def _require_ref_audio_seconds(seconds: float | None) -> None:
    if seconds is not None and seconds > MAX_REF_AUDIO_SECONDS:
        raise RefAudioTooLongError(
            f"ref_audio must be {MAX_REF_AUDIO_SECONDS:g} s or shorter; voice cloning needs a few seconds of speech"
        )


def save_remote_url(url: str, upload_dir: Path, timeout: float = 30.0) -> tuple[str, str]:
    """Download an ``http(s)`` URL into ``upload_dir``. Returns (modality, path).

    Note: fetching arbitrary URLs has SSRF surface. Callers exposing this
    publicly should allowlist hosts or disable remote fetch (data-URL only).
    """
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (caller-gated)
        raw = resp.read()
        mime = resp.headers.get_content_type() if resp.headers else ""
    if not mime:
        # Infer from the URL path extension.
        suffix = Path(urlparse(url).path).suffix.lower()
        rev = {v: k for k, v in _MIME_TO_EXT.items()}
        mime = rev.get(suffix, "application/octet-stream")
    return _save_bytes(raw, mime, upload_dir)


def resolve_media_ref(ref: str, upload_dir: Path, *, allow_remote: bool = False) -> tuple[str, str]:
    """Resolve a media reference from a request: a data URL, or an http(s) URL
    when remote fetch is allowed. Returns ``(modality, path)``.

    A request cannot name a file on the server, not even one under
    ``upload_dir``: plain paths and other URL schemes are refused without
    accessing client-named server files, so a caller can neither read files nor
    probe for them.
    """
    if ref.startswith("data:"):
        return save_data_url(ref, upload_dir)
    if urlparse(ref).scheme.lower() in ("http", "https"):
        allow_remote = allow_remote or os.getenv("MSTAR_ALLOW_REMOTE", "0") == "1"
        if not allow_remote:
            raise ValueError("Remote media fetch is disabled on this server")
        return save_remote_url(ref, upload_dir)
    raise ValueError("server file paths are refused; send media as a data URL or a file upload")


# How long a submitted request's uploads outlive it: the data worker reads them
# from its queue after the submission returns. The Rust frontend waits 60 s too.
UPLOAD_CLEANUP_DELAY_S = 60.0


def remove_uploads(file_paths: dict[str, list[str]] | None, upload_dir: Path) -> None:
    """Delete a request's files under ``upload_dir`` now. A path elsewhere is
    left alone: no route puts one in ``file_paths`` today, and a file the
    server did not save is never the request's to delete."""
    root = Path(upload_dir).resolve()
    for paths in (file_paths or {}).values():
        for path in paths:
            p = Path(path)
            if not p.resolve().is_relative_to(root):
                continue
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def schedule_upload_cleanup(file_paths: dict[str, list[str]] | None, upload_dir: Path) -> None:
    """:func:`remove_uploads` after :data:`UPLOAD_CLEANUP_DELAY_S`, for a
    request that was submitted and may not have been read yet."""
    if not file_paths:
        return

    def _cleanup() -> None:
        time.sleep(UPLOAD_CLEANUP_DELAY_S)
        remove_uploads(file_paths, upload_dir)

    threading.Thread(target=_cleanup, daemon=True).start()


# ---------------------------------------------------------------------------
# Outbound: wrap raw model output for client surfaces
# ---------------------------------------------------------------------------

# torchaudio's resampling kernel holds (target / g) * (rate / g + 2 * width)
# floats, g = gcd(rate, target), width its filter's half-length (default
# lowpass_filter_width 6, rolloff 0.99): 3.0M for the 47,952 Hz pull-down rate
# to 16 kHz, but 1.15G for a 48,001 Hz header resampled to 24 kHz
MAX_RESAMPLE_KERNEL = 1 << 24
MAX_SAMPLE_RATE = 768_000


def resample_kernel_size(rate: int, target: int) -> int:
    """Floats in torchaudio's default sinc kernel resampling ``rate`` to ``target``."""
    g = math.gcd(rate, target)
    orig, new = rate // g, target // g
    width = math.ceil(6 * orig / (min(orig, new) * 0.99))
    return new * (orig + 2 * width)


def check_resample_rate(rate: int, target: int) -> None:
    """Refuse a file rate outside ``1..MAX_SAMPLE_RATE`` Hz or whose
    resampling to ``target`` would cost more than :data:`MAX_RESAMPLE_KERNEL`
    (every real rate, pull-down rates included, passes)."""
    if not 1 <= rate <= MAX_SAMPLE_RATE or resample_kernel_size(rate, target) > MAX_RESAMPLE_KERNEL:
        raise ValueError(f"unsupported sample rate {rate} Hz; resample the audio to a standard rate")


# Longest request audio any decoder here produces (transcription uploads, chat
# and /generate audio, reference clips): the transcription route's limit.
MAX_AUDIO_SECONDS = 30 * 60
_SOUNDFILE_BLOCK_SECONDS = 10
# libsndfile's MP3 decoder is seamless across reads only at whole 1152-sample
# MPEG frames; a read that splits a frame glitches the samples after it
_SOUNDFILE_BLOCK_ALIGN = 1152


class AudioRefusedError(ValueError):
    """Audio a decoder opened and refuses (too long, or an unsupported rate):
    callers re-raise it rather than fall back to another decoder."""


class AudioTooLongError(AudioRefusedError):
    """Audio that decodes past the limit, whatever its header declares."""

    status_code = 413

    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        super().__init__(f"audio is longer than the {max_seconds:g} s limit")


def read_soundfile_capped(path: str, max_seconds: float = MAX_AUDIO_SECONDS) -> tuple[np.ndarray, int]:
    """libsndfile decode of ``path``: float32 mono at the file's own rate.

    Reads in blocks and counts the frames it decodes, so a header that hides
    or understates the length cannot make the read unbounded; past
    ``max_seconds`` it raises :class:`AudioTooLongError`. A rate outside
    ``1..MAX_SAMPLE_RATE`` is refused (:class:`AudioRefusedError`) before
    any read. A container libsndfile cannot read raises soundfile's own error.
    """
    import soundfile as sf

    with sf.SoundFile(path) as f:
        rate = f.samplerate
        # the frame cap scales with the rate; the range alone here, the
        # kernel to the caller's target is checked after the decode
        try:
            check_resample_rate(rate, rate)
        except ValueError as exc:
            raise AudioRefusedError(str(exc)) from None
        limit = int(max_seconds * rate)
        block = max(1, _SOUNDFILE_BLOCK_SECONDS * rate // _SOUNDFILE_BLOCK_ALIGN) * _SOUNDFILE_BLOCK_ALIGN
        parts, frames = [], 0
        while True:
            data = f.read(min(block, limit + 1 - frames), dtype="float32", always_2d=True)
            if not len(data):
                break
            frames += len(data)
            if frames > limit:
                raise AudioTooLongError(max_seconds)
            # a mono block is its own mean, and numpy's reduction over the
            # length-1 axis costs ~10x the copy
            parts.append(data[:, 0] if data.shape[1] == 1 else data.mean(axis=1))
    return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), rate


def decode_torchcodec_capped(path: str, sample_rate: int, max_seconds: float = MAX_AUDIO_SECONDS):
    """FFmpeg (torchcodec) decode of ``path``: a float32 mono tensor at
    ``sample_rate``, the same samples ``get_all_samples()`` returns.

    The decode stops just past ``max_seconds`` of play time, and the audio is
    refused with :class:`AudioTooLongError` when it plays longer or holds
    more samples than that. The decoder resamples every frame to
    ``sample_rate``, so what is counted is decoded output, not the rate or
    duration the container declares.
    """
    return torchcodec_samples_capped(path, max_seconds, sample_rate=sample_rate, num_channels=1).data[0]


def torchcodec_samples_capped(path: str, max_seconds: float = MAX_AUDIO_SECONDS, **decoder_args):
    """``AudioDecoder(path, **decoder_args)``'s samples from the stream's first
    frame, as ``get_all_samples()`` returns them, decoding just past
    ``max_seconds`` of play time; refused with :class:`AudioTooLongError` when
    the audio plays longer or holds more samples than that."""
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(path, **decoder_args)
    # play time counts from the stream's first timestamp, which need not be 0
    start = max(0.0, getattr(decoder.metadata, "begin_stream_seconds_from_header", None) or 0.0)
    samples = decoder.get_samples_played_in_range(start, start + max_seconds + 1.0)
    rate = decoder_args.get("sample_rate") or samples.sample_rate
    played = samples.pts_seconds - start + samples.duration_seconds
    if played > max_seconds or samples.data.shape[-1] > max_seconds * rate:
        raise AudioTooLongError(max_seconds)
    return samples


def decode_audio(path: str, sample_rate: int = 16000, max_seconds: float = MAX_AUDIO_SECONDS):
    """Decode an audio file to a float32 mono numpy waveform at ``sample_rate``,
    refused with :class:`AudioTooLongError` past ``max_seconds``.

    libsndfile (``soundfile``) first — no FFmpeg needed for wav/flac/ogg/mp3 —
    then torchcodec for the rest (m4a, webm, ...). A file neither decoder
    reads raises ``ValueError``.
    """
    try:
        audio, sr = read_soundfile_capped(path, max_seconds)
    except AudioRefusedError:
        raise
    except Exception:  # noqa: BLE001 — not a libsndfile container
        try:
            samples = decode_torchcodec_capped(path, sample_rate, max_seconds)
        except RuntimeError as err:
            # FFmpeg refused it: the upload's fault, and FFmpeg's message (paths,
            # internals) stays out of the response
            raise ValueError("could not decode the audio upload") from err
        return samples.numpy().astype(np.float32)
    check_resample_rate(sr, sample_rate)
    if sr != sample_rate:
        try:
            import torch
            import torchaudio

            audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, sample_rate).numpy()
        except ImportError:
            idx = np.linspace(0, len(audio) - 1, int(round(len(audio) * sample_rate / sr)))
            audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return audio


def window_stop(start: int, total: int, step: int, balanced: bool) -> int:
    """Nominal end of the window starting at ``start``: ``step`` samples on,
    or, ``balanced``, an equal share of what is left over the fewest windows
    of at most ``step`` (so the last window is never a sliver)."""
    if not balanced:
        return min(total, start + step)
    remaining = total - start
    windows = -(-remaining // step)
    return start + -(-remaining // windows)


def split_windows(
    audio, window_seconds: float, sample_rate: int = 16000,
    search_seconds: float = 0.0, frame_seconds: float = 0.02, balanced: bool = False,
) -> list:
    """Cut a waveform into consecutive windows of at most ``window_seconds``;
    concatenating them gives the input back sample for sample.

    With ``search_seconds`` a cut is moved back from its nominal boundary to
    the end of the quietest ``frame_seconds`` frame within that span (the
    last such frame on a tie, so silence cuts at the boundary): the cut lands
    in a pause rather than on a word when there is one to find. ``balanced``
    places the nominal boundaries as :func:`window_stop` does.
    """
    audio = np.asarray(audio)
    step = int(round(window_seconds * sample_rate))
    search = int(round(search_seconds * sample_rate))
    frame = max(1, int(round(frame_seconds * sample_rate)))
    pieces, start = [], 0
    while start < len(audio):
        stop = window_stop(start, len(audio), step, balanced)
        if stop < len(audio) and search > 0:
            num_frames = (min(search, stop - start - frame)) // frame
            if num_frames > 0:
                span = audio[stop - num_frames * frame:stop].astype(np.float32)
                energy = np.square(span.reshape(num_frames, frame)).mean(axis=1)
                quietest = num_frames - 1 - int(np.argmin(energy[::-1]))
                stop = stop - num_frames * frame + (quietest + 1) * frame
        pieces.append(audio[start:stop])
        start = stop
    return pieces


class StreamResampler:
    """Resample a stream that arrives in pieces of any size (``push``), with
    the same output however it was split. A Kaiser-windowed sinc lowpass at
    the lower Nyquist, applied polyphase: output ``k`` is centered on input
    time ``k * rate_in / rate_out``, so timing is kept exactly. The passband
    is flat to 0.1 dB up to 0.9 of the lower Nyquist and 3 dB down at 0.94
    (7.2 and 7.5 kHz at 24 to 16 kHz), close to what a 16 kHz recording
    holds; anything above that Nyquist, which would fold back, is down 46 dB
    or more, and 100 dB from 1.06 of it on. Each output waits for half a
    kernel of input past its center (72 samples, 3 ms, at 24 to 16 kHz); ``flush``
    emits the rest as if the input went on as
    silence, ``ceil(n_in * rate_out / rate_in)`` samples in all, and starts
    over. The kernel has ``2 * half_width * max(up, down) + 1`` taps for the
    reduced ratio ``up/down``, so callers bound the rates."""

    def __init__(self, rate_in: int, rate_out: int, half_width: int = 48, beta: float = 10.0, rolloff: float = 0.95):
        g = math.gcd(rate_in, rate_out)
        self.up, self.down = rate_out // g, rate_in // g
        self.half = half_width * max(self.up, self.down)        # kernel half length, upsampled samples
        n = np.arange(-self.half, self.half + 1)
        cutoff = rolloff * 0.5 / max(self.up, self.down)          # cycles per upsampled sample
        self.kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(n.size, beta) * self.up
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float64)
        self._base = 0      # input index of _buf[0]
        self._n_in = 0      # input samples pushed
        self._k_out = 0     # output samples emitted

    def push(self, x: np.ndarray) -> np.ndarray:
        self._buf = np.concatenate([self._buf, np.asarray(x, dtype=np.float64)])
        self._n_in += int(np.size(x))
        # output k is final once every input within half a kernel of it is in
        return self._emit(-(-(self._n_in * self.up - self.half) // self.down))

    def held(self, extra: int = 0) -> int:
        """Output samples ``flush`` would still emit, after ``extra`` more
        input samples."""
        return -(-((self._n_in + extra) * self.up) // self.down) - self._k_out

    def flush(self) -> np.ndarray:
        out = self._emit(-(-(self._n_in * self.up) // self.down))
        self.reset()
        return out

    def _emit(self, k_end: int) -> np.ndarray:
        up, down, half = self.up, self.down, self.half
        if k_end <= self._k_out or not self._buf.size:
            n, self._k_out = max(0, k_end - self._k_out), max(self._k_out, k_end)
            return np.zeros(n, dtype=np.float32)
        span = half // up + 1
        taps = np.arange(-span, span + 1)[None, :]
        out = []
        for k0 in range(self._k_out, k_end, 4096):              # bounds the gather to 4096 rows
            t = np.arange(k0, min(k0 + 4096, k_end))[:, None] * down   # centers, upsampled
            ns = t // up + taps                                  # candidate input indices
            idx = t - ns * up + half
            ok = (idx >= 0) & (idx <= 2 * half) & (ns >= 0) & (ns < self._n_in)
            vals = self._buf[np.clip(ns - self._base, 0, self._buf.size - 1)]
            out.append((np.where(ok, vals * self.kernel[np.clip(idx, 0, 2 * half)], 0.0)).sum(axis=1))
        self._k_out = k_end
        keep = max(self._base, -(-(k_end * down - half) // up))  # first input the next output needs
        self._buf = self._buf[keep - self._base:]
        self._base = keep
        return np.concatenate(out).astype(np.float32)


def write_wav(audio, path: str, sample_rate: int = 16000) -> str:
    """Float waveform -> 16-bit PCM WAV file at ``path``. The inverse of
    reading PCM16 as ``int16 / 32768``, so 16-bit audio comes back sample for
    sample."""
    scaled = np.rint(np.asarray(audio, dtype=np.float64) * 32768.0)
    pcm = np.clip(scaled, -32768, 32767).astype("<i2").tobytes()
    Path(path).write_bytes(pcm16_to_wav_bytes(pcm, sample_rate))
    return path


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap raw little-endian 16-bit PCM (the model's audio output) into a WAV blob."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return buf.getvalue()


def pcm16_to_container(pcm: bytes, sample_rate: int, fmt: str = "wav") -> tuple[bytes, str]:
    """Encode raw 16-bit PCM into ``fmt``. Returns ``(bytes, mime_type)``.

    ``wav`` and ``pcm`` use the stdlib (the bytes are already PCM_16); the
    compressed formats use the optional ``soundfile``. A format this server
    cannot produce raises ValueError (see :func:`check_audio_format`) rather
    than coming back as WAV under another name.
    """
    fmt = check_audio_format(fmt, sample_rate=sample_rate)
    if fmt == "wav":
        return pcm16_to_wav_bytes(pcm, sample_rate), AUDIO_FORMAT_MIME["wav"]
    if fmt == "pcm":
        return pcm, AUDIO_FORMAT_MIME["pcm"]

    import soundfile as sf  # type: ignore

    sf_format, subtype = _SOUNDFILE_AUDIO[fmt]
    audio = np.frombuffer(pcm, dtype="<i2")
    buf = io.BytesIO()
    sf.write(buf, audio, int(sample_rate), format=sf_format, subtype=subtype)
    return buf.getvalue(), AUDIO_FORMAT_MIME[fmt]


def mux_mp4_with_pcm16(
    video_mp4: bytes, pcm: bytes, sample_rate: int, num_channels: int = 2,
    video_fps: float | None = None,
) -> bytes:
    """Mux an H.264 mp4 and raw interleaved 16-bit PCM into one mp4 with an AAC
    audio track (stream-copies the video). ``video_fps`` is the true frame rate
    of the content: when it differs from the container's rate the video
    timestamps are rescaled (no re-encode) so playback runs at the real rate
    and stays in sync with the audio track. Requires the ``ffmpeg`` binary;
    raises on failure — callers fall back to the video-only mp4."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        vpath = Path(tmp) / "v.mp4"
        apath = Path(tmp) / "a.pcm"
        opath = Path(tmp) / "out.mp4"
        vpath.write_bytes(video_mp4)
        apath.write_bytes(pcm)
        itsscale = None
        if video_fps:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(vpath)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            try:
                num, den = probe.stdout.strip().split("/")
                container_fps = float(num) / float(den)
                if container_fps > 0 and abs(container_fps - video_fps) > 1e-3:
                    itsscale = container_fps / float(video_fps)
            except (ValueError, ZeroDivisionError):
                pass
        cmd = ["ffmpeg", "-y", "-v", "error"]
        if itsscale is not None:
            cmd += ["-itsscale", f"{itsscale:.6f}"]
        cmd += [
            "-i", str(vpath),
            "-f", "s16le", "-ar", str(int(sample_rate)), "-ac", str(int(num_channels)), "-i", str(apath),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(opath),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed: {proc.stderr.strip()[:500]}")
        return opath.read_bytes()


def wav_stream_header(sample_rate: int, num_channels: int = 1, bits: int = 16) -> bytes:
    """A 44-byte WAV header with streaming (unknown-length) size fields.

    Used to stream TTS audio over a single HTTP response: emit this header, then
    16-bit PCM frames as they arrive. The 0xFFFFFFFF placeholders signal an
    open-ended stream, which players and the OpenAI client's
    ``stream_to_file`` handle.
    """
    import struct

    byte_rate = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


def png_to_data_url(png_bytes: bytes) -> str:
    """Encode PNG image bytes (the model's image output) as a data URL."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

"""Decode and check HTTP responses from an M* server. Runs on CPU; every check
returns a Verdict instead of raising."""
import base64
import dataclasses
import functools
import hashlib
import io
import json
import re
import wave

import numpy as np

SPECIAL_TOKEN_MARKERS = ("<|", "|>")
VIDEO_CONTAINERS = ("mp4", "mov", "webm", "matroska", "avi", "mpegts")


@dataclasses.dataclass(frozen=True)
class Expect:
    """What a case's output must look like. ``kind`` is text, image, video,
    frames, audio or float32; ``modality`` is the ``/generate`` response key."""
    kind: str
    modality: str | None = None
    contains_any: tuple[str, ...] = ()  # text: one of these as a whole word; "not red" passes "red"
    color: str | None = None            # image: "red" or "blue" dominates
    size: tuple[int, int] | None = None  # image, video, frames: (width, height)
    frames: int | None = None            # video, frames: exact frame count
    min_frames: int = 1                  # video, frames
    min_seconds: float = 0.0             # audio
    multiple_of: int | None = None       # float32: element count is a multiple of this
    count: int | None = None             # float32: exact element count
    transcript: str | None = None        # text: reference transcript
    min_word_recall: float = 0.8         # text: share of transcript words required


@dataclasses.dataclass
class Verdict:
    passed: bool
    detail: str
    sha256: str | None = None   # of the decoded output, compared across checkouts
    warnings: list[str] = dataclasses.field(default_factory=list)
    artifact: tuple[str, bytes] | None = None  # (suffix, bytes) saved next to the report


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _image_suffix(raw: bytes) -> str:
    if raw[:3] == b"\xff\xd8\xff":
        return ".jpg"
    return ".webp" if raw[8:12] == b"WEBP" else ".png"


def _keeps(suffix):
    """Attach the checked payload to the verdict so the runner can save it."""
    def decorate(check):
        @functools.wraps(check)
        def checked(raw, expect):
            verdict = check(raw, expect)
            verdict.artifact = (suffix(raw) if callable(suffix) else suffix, raw)
            return verdict
        return checked
    return decorate


def _words(text: str) -> list[str]:
    """Lowercase words; a decimal ("4.5", "1,000") is one word, so it does not match "4"."""
    return re.findall(r"\d+(?:[.,]\d+)+|[^\W_]+", text.lower())


def check_text(text: str, expect: Expect) -> Verdict:
    warnings = [f"special token in output: {text!r}"] if any(m in text for m in SPECIAL_TOKEN_MARKERS) else []
    words = set(_words(text))
    artifact = (".txt", text.encode())
    ok = bool(text.strip()) and (not expect.contains_any or any(a.lower() in words for a in expect.contains_any))
    detail = repr(text[:80])
    if expect.transcript:
        reference, heard = _words(expect.transcript), set(_words(text))
        recall = sum(w in heard for w in reference) / len(reference)
        detail += f", {recall:.0%} of reference words"
        ok = ok and recall >= expect.min_word_recall
    return Verdict(ok, detail, _digest(text.encode()), warnings, artifact)


@_keeps(_image_suffix)
def check_image(raw: bytes, expect: Expect) -> Verdict:
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(raw))
        fmt = image.format
        pixels = np.asarray(image.convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        return Verdict(False, f"not a decodable image: {exc}")
    h, w = pixels.shape[:2]
    r, g, b = (float(x) for x in pixels.reshape(-1, 3).mean(0))
    detail = f"{fmt} {w}x{h}, mean RGB ({r:.0f}, {g:.0f}, {b:.0f})"
    digest = _digest(pixels.tobytes())
    if fmt not in ("PNG", "JPEG", "WEBP"):
        return Verdict(False, detail + ": unexpected image format", digest)
    if expect.size and (w, h) != tuple(expect.size):
        return Verdict(False, detail + f": expected {expect.size[0]}x{expect.size[1]}", digest)
    if pixels.std() <= 5:
        return Verdict(False, detail + ": blank image", digest)
    if expect.color == "red" and not (r > g + 40 and r > b + 40):
        return Verdict(False, detail + ": red does not dominate", digest)
    if expect.color == "blue" and not (b > r + 40 and b > g + 20):
        return Verdict(False, detail + ": blue does not dominate", digest)
    return Verdict(True, detail, digest)


@_keeps(".mp4")
def check_video(raw: bytes, expect: Expect) -> Verdict:
    import av

    try:
        with av.open(io.BytesIO(raw)) as container:
            fmt = container.format.name
            stream = container.streams.video[0]
            decoded = list(container.decode(stream))
            frames = [f.to_ndarray(format="rgb24") for f in decoded]
            timing = [stream.average_rate, stream.time_base, [f.pts for f in decoded]]
    except Exception as exc:  # noqa: BLE001
        return Verdict(False, f"not a decodable video: {exc}")
    # PyAV also "decodes" a still image as a one-frame video; require a video container.
    if not any(name in fmt for name in VIDEO_CONTAINERS):
        return Verdict(False, f"{fmt} is not a video container")
    if not frames:
        return Verdict(False, "video has no frames")
    stack = np.stack(frames)
    detail = f"{len(frames)} frames, {stack.shape[2]}x{stack.shape[1]}, {float(timing[0] or 0):g} fps"
    digest = _digest(f"{stack.shape}:{timing}:".encode() + stack.tobytes())
    return _check_frames(stack, expect, detail, digest)


def _check_frames(stack: np.ndarray, expect: Expect, detail: str, digest: str) -> Verdict:
    n, h, w = stack.shape[:3]
    if expect.frames is not None and n != expect.frames:
        return Verdict(False, detail + f": expected {expect.frames} frames", digest)
    if n < expect.min_frames:
        return Verdict(False, detail + f": expected at least {expect.min_frames} frames", digest)
    if expect.size and (w, h) != tuple(expect.size):
        return Verdict(False, detail + f": expected {expect.size[0]}x{expect.size[1]}", digest)
    if stack.std() <= 5:
        return Verdict(False, detail + ": blank", digest)
    return Verdict(True, detail, digest)


@_keeps(".wav")
def check_audio(raw: bytes, expect: Expect) -> Verdict:
    try:
        with wave.open(io.BytesIO(raw)) as w:
            rate, channels, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            frames = w.readframes(n)
    except Exception as exc:  # noqa: BLE001
        return Verdict(False, f"not a WAV file: {exc}")
    if width != 2:
        return Verdict(False, f"{width * 8}-bit WAV; expected 16-bit PCM")
    if not rate or not channels:
        return Verdict(False, f"invalid WAV header: {rate} Hz, {channels} channels")
    # Count the samples that arrived, not the frames the header declares.
    arrived = len(frames) // (width * channels)
    samples = np.frombuffer(frames[:arrived * width * channels], dtype="<i2")
    seconds = arrived / rate
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0
    detail = f"{seconds:.2f} s at {rate} Hz, {channels} ch, rms {rms:.0f}"
    digest = _digest(f"{rate}:{channels}:{width}:".encode() + frames)
    if arrived < n:
        return Verdict(False, detail + f": header declares {n} frames, {arrived} arrived", digest)
    if seconds < max(expect.min_seconds, 1e-9):
        return Verdict(False, detail + f": expected at least {expect.min_seconds} s", digest)
    if rms < 50:
        return Verdict(False, detail + ": silent", digest)
    return Verdict(True, detail, digest)


@_keeps(".f32")
def check_float32(raw: bytes, expect: Expect) -> Verdict:
    if len(raw) % 4:
        return Verdict(False, f"{len(raw)} bytes is not a whole number of float32 values")
    values = np.frombuffer(raw, dtype="<f4")
    detail = f"{values.size} float32 values"
    digest = _digest(raw)
    if not values.size:
        return Verdict(False, "no values", digest)
    if not np.isfinite(values).all():
        return Verdict(False, detail + ": contains NaN or inf", digest)
    if expect.count is not None and values.size != expect.count:
        return Verdict(False, detail + f": expected exactly {expect.count}", digest)
    if expect.multiple_of and values.size % expect.multiple_of:
        return Verdict(False, detail + f": expected a multiple of {expect.multiple_of}", digest)
    return Verdict(True, detail, digest)


_CHECKS = {"image": check_image, "video": check_video, "audio": check_audio, "float32": check_float32}


def check_generate_response(body: bytes, expect: Expect) -> Verdict:
    """Check the chunks under ``expect.modality`` as ``expect.kind``."""
    try:
        outputs = json.loads(body)["outputs"]
    except Exception as exc:  # noqa: BLE001
        return Verdict(False, f"not a /generate response: {exc}")
    if expect.modality not in outputs:
        return Verdict(False, f"no {expect.modality!r} output; got {sorted(outputs)}")
    raw = b"".join(base64.b64decode(c["data"]) for c in outputs[expect.modality])
    if expect.kind == "text":
        return check_text(raw.decode("utf-8", errors="replace"), expect)
    return _CHECKS[expect.kind](raw, expect)


def check_frame_stream(body: bytes, expect: Expect) -> Verdict:
    """Streamed NDJSON of raw RGB chunks with height, width and frame_count metadata."""
    frames, fps = [], None
    try:
        for line in body.decode().splitlines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk["modality"] == "error":
                return Verdict(False, f"stream error: {base64.b64decode(chunk['data']).decode(errors='replace')[:200]}")
            if chunk["modality"] != expect.modality:
                continue
            meta = chunk["metadata"]
            h, w, n = int(meta["height"]), int(meta["width"]), int(meta.get("frame_count", 1))
            fps = meta.get("fps")
            data = np.frombuffer(base64.b64decode(chunk["data"]), dtype=np.uint8)
            if data.size != n * h * w * 3:
                return Verdict(False, f"chunk holds {data.size} bytes; metadata says {n}x{h}x{w}x3")
            frames.append(data.reshape(n, h, w, 3))
    except Exception as exc:  # noqa: BLE001
        return Verdict(False, f"not a {expect.modality} stream: {exc}")
    if not frames:
        return Verdict(False, f"no {expect.modality!r} chunks in the stream")
    stack = np.concatenate(frames)
    detail = f"{stack.shape[0]} frames in {len(frames)} chunks, {stack.shape[2]}x{stack.shape[1]}"
    digest = _digest(f"{stack.shape}:{fps}:".encode() + stack.tobytes())
    return _check_frames(stack, expect, detail, digest)


def check_transcription_response(body: bytes, expect: Expect) -> Verdict:
    try:
        text = json.loads(body)["text"]
    except Exception as exc:  # noqa: BLE001
        return Verdict(False, f"not a transcription response: {exc}")
    return check_text(text, expect)


def check_chat_response(body: bytes, expect: Expect, stream: bool) -> Verdict:
    """An OpenAI chat completion, streamed (SSE) or not."""
    try:
        if stream:
            lines = body.decode().splitlines()
            if "data: [DONE]" not in lines:
                return Verdict(False, "stream did not end with [DONE]")
            events = [json.loads(line[6:]) for line in lines if line.startswith("data: ") and line != "data: [DONE]"]
            if any("error" in e for e in events):
                return Verdict(False, f"stream reported an error: {events}")
            text = "".join(e["choices"][0].get("delta", {}).get("content") or "" for e in events if e.get("choices"))
        else:
            message = json.loads(body)["choices"][0]["message"]
            if expect.kind == "audio":
                audio = (message.get("audio") or {}).get("data")
                if not audio:
                    return Verdict(False, "chat response carries no message.audio")
                return check_audio(base64.b64decode(audio), expect)
            text = message["content"]
    except Exception as exc:  # noqa: BLE001
        return Verdict(False, f"not a chat completion: {exc}")
    return check_text(text, expect)


def check_speech_response(body: bytes, expect: Expect) -> Verdict:
    return check_audio(body, expect)


def differ(a: Verdict, b: Verdict) -> bool:
    return a.sha256 is not None and b.sha256 is not None and a.sha256 != b.sha256

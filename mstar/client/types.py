"""Typed results and streaming events for the mstar Python SDK."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AudioBuffer:
    """Decoded audio output: little-endian 16-bit PCM samples plus the sample rate.

    The server emits audio as headerless int16 PCM; this wraps it with the rate
    so callers can save a real WAV or get a numpy array without caring about the
    on-the-wire encoding.
    """

    pcm: bytes
    sample_rate: int = 24000

    def to_numpy(self):
        """int16 samples as a numpy array (divide by 32768 for float in [-1, 1])."""
        import numpy as np

        return np.frombuffer(self.pcm, dtype="<i2")

    def wav_bytes(self) -> bytes:
        from mstar.client.media import pcm16_to_wav_bytes

        return pcm16_to_wav_bytes(self.pcm, self.sample_rate)

    def to_wav(self, path) -> str:
        with open(path, "wb") as f:
            f.write(self.wav_bytes())
        return str(path)

    def __len__(self) -> int:
        return len(self.pcm) // 2  # int16 -> 2 bytes per sample


# --- streaming events (yielded by MStarClient.stream / generate(stream=True)) ---

@dataclass
class TextChunk:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ImageChunk:
    data: bytes  # PNG-encoded image bytes
    metadata: dict = field(default_factory=dict)

    def save(self, path) -> str:
        with open(path, "wb") as f:
            f.write(self.data)
        return str(path)


@dataclass
class AudioChunk:
    pcm: bytes  # raw little-endian 16-bit PCM
    sample_rate: int = 24000
    metadata: dict = field(default_factory=dict)

    def to_numpy(self):
        import numpy as np

        return np.frombuffer(self.pcm, dtype="<i2")


@dataclass
class VideoFrameChunk:
    """A contiguous batch of raw RGB24 frames from a native stream.

    The wire payload is deliberately not an encoded video container. Metadata
    supplies the shape and timing needed to interpret it without copying.
    """

    data: bytes
    metadata: dict
    width: int = field(init=False)
    height: int = field(init=False)
    fps: float = field(init=False)
    pixel_format: str = field(init=False)
    frame_index: int = field(init=False)
    frame_count: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise ValueError(
                f"video_frame data must be immutable bytes; got {type(self.data).__name__}"
            )
        if not isinstance(self.metadata, dict):
            raise ValueError(
                f"video_frame metadata must be a dict; got {type(self.metadata).__name__}"
            )
        required = (
            "width", "height", "fps", "pixel_format", "frame_index", "frame_count",
        )
        missing = [name for name in required if name not in self.metadata]
        if missing:
            raise ValueError(
                "video_frame metadata is missing required field(s): "
                + ", ".join(missing)
            )

        width = self.metadata["width"]
        height = self.metadata["height"]
        frame_index = self.metadata["frame_index"]
        frame_count = self.metadata["frame_count"]
        for name, value, minimum in (
            ("width", width, 1),
            ("height", height, 1),
            ("frame_index", frame_index, 0),
            ("frame_count", frame_count, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f"video_frame metadata {name!r} must be an int >= {minimum}; "
                    f"got {value!r}"
                )

        fps = self.metadata["fps"]
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError(
                f"video_frame metadata 'fps' must be a finite positive number; got {fps!r}"
            )
        pixel_format = self.metadata["pixel_format"]
        if pixel_format != "rgb24":
            raise ValueError(
                "video_frame metadata 'pixel_format' must be 'rgb24'; "
                f"got {pixel_format!r}"
            )

        expected = frame_count * height * width * 3
        if len(self.data) != expected:
            raise ValueError(
                "video_frame payload length does not match its metadata: "
                f"expected {expected} bytes, got {len(self.data)}"
            )

        self.width = width
        self.height = height
        self.fps = float(fps)
        self.pixel_format = pixel_format
        self.frame_index = frame_index
        self.frame_count = frame_count

    def to_numpy(self):
        """Return a zero-copy, read-only ``[T, H, W, 3]`` uint8 view."""
        import numpy as np

        return np.frombuffer(self.data, dtype=np.uint8).reshape(
            self.frame_count, self.height, self.width, 3
        )


# A streaming iteration yields one of these per output chunk.
StreamEvent = TextChunk | ImageChunk | AudioChunk | VideoFrameChunk


@dataclass
class GenerateResult:
    """Aggregated, decoded output of a non-streaming request."""

    request_id: str | None = None
    text: str | None = None
    images: list[bytes] = field(default_factory=list)  # PNG bytes, in arrival order
    audio: AudioBuffer | None = None
    raw: list[dict] = field(default_factory=list)  # decoded chunks: {modality, bytes, metadata}

    def save_image(self, path, index: int = 0) -> str:
        if index >= len(self.images):
            raise IndexError(f"No image at index {index} (have {len(self.images)})")
        with open(path, "wb") as f:
            f.write(self.images[index])
        return str(path)

    def save_audio(self, path) -> str:
        if self.audio is None:
            raise RuntimeError("Result has no audio output")
        return self.audio.to_wav(path)

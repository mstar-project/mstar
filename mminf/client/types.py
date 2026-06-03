"""Typed results and streaming events for the mminf Python SDK."""

from __future__ import annotations

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
        from mminf.client.media import pcm16_to_wav_bytes

        return pcm16_to_wav_bytes(self.pcm, self.sample_rate)

    def to_wav(self, path) -> str:
        with open(path, "wb") as f:
            f.write(self.wav_bytes())
        return str(path)

    def __len__(self) -> int:
        return len(self.pcm) // 2  # int16 -> 2 bytes per sample


# --- streaming events (yielded by MMInfClient.stream / generate(stream=True)) ---

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


# A streaming iteration yields one of these per output chunk.
StreamEvent = TextChunk | ImageChunk | AudioChunk


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

"""Audio and video decoding that works with or without torchcodec.

torchcodec needs FFmpeg's shared libraries and its PyPI wheels link CUDA, so it
won't load on ROCm or on hosts without FFmpeg. Then PyAV (which ships its own
FFmpeg) decodes instead, and soundfile reads plain PCM/FLAC audio and the few
formats FFmpeg can't. The choice is made once per process.

Outputs match torchcodec's: float32 [channels, samples] audio, uint8
[frames, 3, H, W] RGB video, and video metadata under torchcodec's
VideoStreamMetadata field names.
"""

from __future__ import annotations

import functools
import io
import logging
import os
import wave
from collections.abc import Callable, Sequence
from dataclasses import asdict
from fractions import Fraction
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)

# A path, or the encoded bytes of a file.
MediaSource = str | os.PathLike | bytes


class AudioSamples(NamedTuple):
    """Float32 ``[channels, samples]`` in [-1, 1] at ``sample_rate``."""
    data: torch.Tensor
    sample_rate: int


class AudioHeader(NamedTuple):
    """What an audio stream's header says, None where it says nothing."""
    sample_rate: int | None
    duration: float | None


class VideoFrames(NamedTuple):
    """Uint8 ``[frames, 3, H, W]`` RGB frames and the stream's metadata."""
    data: torch.Tensor
    metadata: dict


# (height, width, count) of frames a video decode is about to keep
FramesHook = Callable[[int, int, int], None]


def decode_audio(
    source: MediaSource, sample_rate: int | None = None, num_channels: int | None = None,
    max_seconds: float | None = None,
) -> AudioSamples:
    """All samples of the first audio stream, or those of its first
    ``max_seconds``, resampled and mixed down if asked."""
    return _decoders().audio.decode(source, sample_rate, num_channels, max_seconds)


def audio_header(source: MediaSource) -> AudioHeader:
    """The first audio stream's sample rate and duration as its header gives
    them, decoding nothing. Raises if the audio can't be opened."""
    return _decoders().audio.header(source)


def audio_duration(source: MediaSource) -> float | None:
    """Duration in seconds from the header, or None. Raises if the audio can't be opened."""
    return _decoders().audio.duration(source)


def video_metadata(source: MediaSource) -> dict:
    """Metadata of the first video stream. Decodes at most one frame."""
    return _decoders().video.metadata(source)


def decode_video(
    source: MediaSource, indices: Sequence[int] | None = None, device: str | torch.device = "cpu",
    on_frames: FramesHook | None = None,
) -> VideoFrames:
    """Frames at ``indices`` in presentation order (all of them if None), on ``device``.

    ``on_frames(height, width, count)`` is told about frames before they are
    kept and may raise to stop the decode: one at a time as they decode, or,
    for a torchcodec decode of ``indices`` (one batched call), all of them up
    front at the stream's size.
    """
    return _decoders().video.decode(source, indices, device, on_frames)


def sample_video(
    source: MediaSource, choose: Callable[[dict], Sequence[int]], device: str | torch.device = "cpu",
    on_frames: FramesHook | None = None,
) -> VideoFrames:
    """Decode the frames ``choose(metadata)`` returns, telling ``on_frames`` as
    ``decode_video`` does. torchcodec opens the file only once."""
    return _decoders().video.sample(source, choose, device, on_frames)


def encode_video(frames: torch.Tensor, frame_rate: float, path: str | os.PathLike) -> None:
    """Write uint8 ``[frames, 3, H, W]`` RGB as an H.264 mp4 at ``path``."""
    _decoders().video.encode(frames, frame_rate, path)


def uses_torchcodec() -> bool:
    """Whether torchcodec decodes in this process (else PyAV and libsndfile do)."""
    return isinstance(_decoders().video, _TorchcodecVideo)


def video_metadata_dict(metadata_obj) -> dict:
    # torchcodec>=0.9 reports pixel_aspect_ratio as a Fraction, which the wire codec rejects.
    return {k: float(v) if isinstance(v, Fraction) else v for k, v in asdict(metadata_obj).items()}


def _torchcodec_metadata(decoder) -> dict:
    metadata = video_metadata_dict(decoder.metadata)
    # older torchcodec has these as properties, so asdict skips them
    for key in ("num_frames", "average_fps", "duration_seconds"):
        if key not in metadata:
            metadata[key] = getattr(decoder.metadata, key, None)
    return metadata


def encode_mp4_pyav(frames: torch.Tensor, fps: float, crf: int = 18, preset: str = "ultrafast") -> bytes:
    """H.264 mp4 bytes from uint8 frames ``[T, H, W, 3]`` through PyAV (its
    wheel bundles FFmpeg + libx264), matching the torchcodec encoder's
    CRF / preset / threading settings. Odd frame sizes are edge-padded to
    even dimensions for yuv420p."""
    import av
    import numpy as np

    _, h, w, _ = frames.shape
    arr = frames.numpy()
    if (w % 2) or (h % 2):
        arr = np.pad(arr, ((0, 0), (0, h % 2), (0, w % 2), (0, 0)), mode="edge")
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        stream.width, stream.height = arr.shape[2], arr.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": preset, "threads": "0"}
        for frame in arr:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Choosing the decoders
# ---------------------------------------------------------------------------


class _Decoders(NamedTuple):
    audio: _AudioDecoder
    video: _VideoDecoder


@functools.cache
def _decoders() -> _Decoders:
    reason = _torchcodec_unusable()
    if reason is None:
        return _Decoders(_TorchcodecAudio(), _TorchcodecVideo())
    logger.info(
        "torchcodec can't decode here (%s), using PyAV and soundfile",
        reason,
    )
    return _Decoders(_FallbackAudio(), _PyAVVideo())


def _torchcodec_unusable() -> str | None:
    """Why torchcodec can't decode here, or None if it can. It imports fine
    without FFmpeg and only fails on use, so decode a tiny WAV too."""
    try:
        from torchcodec.decoders import AudioDecoder

        AudioDecoder(_tiny_wav()).get_all_samples()
    except Exception as exc:  # noqa: BLE001 — missing, CUDA-only build or no FFmpeg
        first_line = (str(exc).strip().splitlines() or [""])[0][:200]
        return f"{type(exc).__name__}: {first_line}"
    return None


def _tiny_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes(64))
    return buf.getvalue()


def _readable(source: MediaSource):
    """``source`` as soundfile and PyAV open it."""
    if isinstance(source, bytes | bytearray | memoryview):
        return io.BytesIO(source)
    return os.fspath(source)


def _torchcodec_source(source: MediaSource):
    return source if isinstance(source, bytes) else os.fspath(source)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


class _AudioDecoder:
    name: str

    def decode(
        self, source: MediaSource, sample_rate: int | None, num_channels: int | None,
        max_seconds: float | None = None,
    ) -> AudioSamples:
        raise NotImplementedError

    def duration(self, source: MediaSource) -> float | None:
        raise NotImplementedError

    def header(self, source: MediaSource) -> AudioHeader:
        raise NotImplementedError


class _TorchcodecAudio(_AudioDecoder):
    name = "torchcodec"

    def decode(self, source, sample_rate, num_channels, max_seconds=None):
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(
            _torchcodec_source(source), sample_rate=sample_rate, num_channels=num_channels,
        )
        samples = (
            decoder.get_all_samples() if max_seconds is None
            else decoder.get_samples_played_in_range(0.0, max_seconds)
        )
        return AudioSamples(samples.data, int(samples.sample_rate))

    def duration(self, source):
        return self.header(source).duration

    def header(self, source):
        from torchcodec.decoders import AudioDecoder

        meta = AudioDecoder(_torchcodec_source(source)).metadata
        seconds = getattr(meta, "duration_seconds_from_header", None)
        if seconds is None:
            seconds = getattr(meta, "duration_seconds", None)
        rate = getattr(meta, "sample_rate", None)
        return AudioHeader(int(rate) if rate else None, float(seconds) if seconds is not None else None)


class _SoundfileAudio(_AudioDecoder):
    """libsndfile: WAV, FLAC, OGG/Opus, MP3 and the other formats it lists."""
    name = "soundfile"

    def decode(self, source, sample_rate, num_channels, max_seconds=None):
        import soundfile as sf

        frames = -1
        if max_seconds is not None:
            frames = int(max_seconds * sf.info(_readable(source)).samplerate)
        data, rate = sf.read(_readable(source), frames=frames, dtype="float32", always_2d=True)
        audio = _mix_channels(torch.from_numpy(data).T, num_channels)
        if sample_rate is not None and sample_rate != rate:
            if audio.shape[-1]:
                import torchaudio

                from mstar.api_server.media_io import check_resample_rate

                # torchaudio's sinc kernel grows with the rates' coprime parts
                check_resample_rate(rate, sample_rate)
                audio = torchaudio.functional.resample(audio, rate, sample_rate)
            rate = sample_rate
        return AudioSamples(audio.contiguous(), int(rate))

    def duration(self, source):
        return self.header(source).duration

    def header(self, source):
        import soundfile as sf

        info = sf.info(_readable(source))
        rate = info.samplerate or None
        # a count of 0, or libsndfile's 2**63 - 1 for a stream that never
        # wrote one (a FLAC written to a pipe), gives no length
        known = rate and 0 < info.frames < 1 << 62
        return AudioHeader(rate, info.frames / rate if known else None)


# "Nc" is FFmpeg's layout for N channels in no particular order
_LAYOUTS = {1: "mono", 2: "stereo"}


class _PyAVAudio(_AudioDecoder):
    """FFmpeg via PyAV. Resampling and mixing go through swresample, like torchcodec."""
    name = "pyav"

    def decode(self, source, sample_rate, num_channels, max_seconds=None):
        import av
        import numpy as np

        chunks = []
        with av.open(_readable(source)) as container:
            if not container.streams.audio:
                raise ValueError("no audio stream")
            stream = container.streams.audio[0]
            rate = sample_rate or stream.rate
            channels = num_channels or stream.channels
            limit = None if max_seconds is None else int(max_seconds * rate)
            layout = None if num_channels is None else _LAYOUTS.get(num_channels, f"{num_channels}c")
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
            decoded = 0
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray())
                    decoded += chunks[-1].shape[1]
                if limit is not None and decoded >= limit:
                    break
            else:
                chunks.extend(out.to_ndarray() for out in resampler.resample(None))
        if not chunks:
            return AudioSamples(torch.zeros(channels, 0), int(rate))
        audio = np.concatenate(chunks, axis=1)
        if limit is not None and audio.shape[1] > limit:
            audio = np.ascontiguousarray(audio[:, :limit])
        return AudioSamples(torch.from_numpy(audio), int(rate))

    def duration(self, source):
        import av

        with av.open(_readable(source)) as container:
            if not container.streams.audio:
                raise ValueError("no audio stream")
            stream = container.streams.audio[0]
            # FFmpeg can find an audio stream in bytes that aren't audio, so check a frame decodes
            if next(container.decode(stream), None) is None:
                raise ValueError("no decodable audio frame")
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
            return container.duration / av.time_base if container.duration is not None else None

    def header(self, source):
        import av

        with av.open(_readable(source)) as container:
            if not container.streams.audio:
                raise ValueError("no audio stream")
            stream = container.streams.audio[0]
            if stream.duration is not None and stream.time_base is not None:
                seconds = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                seconds = container.duration / av.time_base
            else:
                seconds = None
            return AudioHeader(stream.rate or None, seconds)


class _FallbackAudio(_AudioDecoder):
    """PyAV first, since it resamples and mixes like torchcodec. soundfile reads
    PCM/FLAC that needs no conversion (same samples, 0.4 ms instead of 13 ms
    for a 3.5 s clip) and the formats FFmpeg can't read."""
    name = "pyav+soundfile"

    # formats where soundfile and FFmpeg give identical samples
    _AS_IS_FORMATS = frozenset({"WAV", "WAVEX", "W64", "RF64", "FLAC", "AIFF", "CAF", "AU"})
    _AS_IS_SUBTYPES = frozenset({"PCM_S8", "PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE", "ULAW", "ALAW"})

    def __init__(self):
        self._pyav = _PyAVAudio()
        self._soundfile = _SoundfileAudio()

    def decode(self, source, sample_rate, num_channels, max_seconds=None):
        if self._as_is(source, sample_rate, num_channels):
            return self._soundfile.decode(source, sample_rate, num_channels, max_seconds)
        try:
            return self._pyav.decode(source, sample_rate, num_channels, max_seconds)
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 — FFmpeg can't read it, or no PyAV
            first = exc
        try:
            return self._soundfile.decode(source, sample_rate, num_channels, max_seconds)
        except Exception as exc:  # noqa: BLE001 — neither works, report both
            raise ValueError(f"cannot decode audio: PyAV: {first}; soundfile: {exc}") from first

    def _as_is(self, source, sample_rate, num_channels) -> bool:
        """True if soundfile can read ``source`` with no resampling or mixing."""
        try:
            import soundfile as sf

            info = sf.info(_readable(source))
        except Exception:  # noqa: BLE001 — not a libsndfile container, or no soundfile
            return False
        return (
            info.format in self._AS_IS_FORMATS and info.subtype in self._AS_IS_SUBTYPES
            and sample_rate in (None, info.samplerate) and num_channels in (None, info.channels)
        )

    def duration(self, source):
        try:
            return self._pyav.duration(source)
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 — as in decode
            first = exc
        try:
            return self._soundfile.duration(source)
        except Exception as exc:  # noqa: BLE001 — as in decode
            raise ValueError(f"cannot read audio: PyAV: {first}; soundfile: {exc}") from first

    def header(self, source):
        # libsndfile first: its headers are exact for the formats it reads, some
        # of which FFmpeg misreads (MAT5 as AMR-NB with a guessed duration)
        try:
            return self._soundfile.header(source)
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 — not a libsndfile container
            first = exc
        try:
            return self._pyav.header(source)
        except Exception as exc:  # noqa: BLE001 — neither opens it, report both
            raise ValueError(f"cannot read audio: soundfile: {first}; PyAV: {exc}") from first


def _mix_channels(audio: torch.Tensor, num_channels: int | None) -> torch.Tensor:
    """Mix ``[C, N]`` down to mono by averaging, or copy a mono track out to ``num_channels``."""
    have = audio.shape[0]
    if num_channels is None or num_channels == have:
        return audio
    if num_channels == 1:
        return audio.mean(dim=0, keepdim=True)
    if have == 1:
        return audio.expand(num_channels, -1)
    raise ValueError(f"cannot mix {have} audio channels to {num_channels}")


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


class _VideoDecoder:
    name: str

    def metadata(self, source: MediaSource) -> dict:
        raise NotImplementedError

    def decode(
        self, source: MediaSource, indices: Sequence[int] | None, device, on_frames: FramesHook | None = None,
    ) -> VideoFrames:
        raise NotImplementedError

    def sample(
        self, source: MediaSource, choose: Callable[[dict], Sequence[int]], device,
        on_frames: FramesHook | None = None,
    ) -> VideoFrames:
        return self.decode(source, choose(self.metadata(source)), device, on_frames)

    def encode(self, frames: torch.Tensor, frame_rate: float, path) -> None:
        raise NotImplementedError


class _TorchcodecVideo(_VideoDecoder):
    name = "torchcodec"

    def metadata(self, source):
        from torchcodec.decoders import VideoDecoder

        return _torchcodec_metadata(VideoDecoder(_torchcodec_source(source)))

    def decode(self, source, indices, device, on_frames=None):
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(_torchcodec_source(source), device=device)
        return VideoFrames(self._frames(decoder, indices, on_frames), _torchcodec_metadata(decoder))

    def sample(self, source, choose, device, on_frames=None):
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(_torchcodec_source(source), device=device)
        metadata = _torchcodec_metadata(decoder)
        return VideoFrames(self._frames(decoder, choose(metadata), on_frames), metadata)

    @staticmethod
    def _frames(decoder, indices: Sequence[int] | None, on_frames: FramesHook | None = None) -> torch.Tensor:
        if indices is None:
            frames = []
            for frame in decoder:
                if on_frames is not None:
                    on_frames(*frame.shape[-2:], 1)
                frames.append(frame)
            return torch.stack(frames)
        if on_frames is not None and len(indices):
            # the sampled frames come back from one call, so they are told up
            # front at the stream's size
            height, width = decoder.metadata.height, decoder.metadata.width
            if not (height and width):
                height, width = decoder[indices[0]].shape[-2:]
            on_frames(height, width, len(indices))
        if hasattr(decoder, "get_frames_at"):
            return decoder.get_frames_at(indices=list(indices)).data
        return torch.stack([decoder[i] for i in indices])  # older torchcodec: no batched sampled decode

    def encode(self, frames, frame_rate, path):
        from torchcodec.encoders import VideoEncoder

        VideoEncoder(frames=frames, frame_rate=frame_rate).to_file(os.fspath(path))


class _PyAVVideo(_VideoDecoder):
    """FFmpeg via PyAV. num_frames and the stream bounds come from a pass over
    the packets, like torchcodec's default exact mode. Frames are numbered in
    presentation order and rotated upright, as torchcodec does."""
    name = "pyav"

    def metadata(self, source):
        return self._read(source, indices=(), device="cpu").metadata

    def decode(self, source, indices, device, on_frames=None):
        return self._read(source, indices, device, on_frames)

    def encode(self, frames, frame_rate, path):
        data = encode_mp4_pyav(frames.permute(0, 2, 3, 1).contiguous().cpu(), frame_rate)
        with open(path, "wb") as f:
            f.write(data)

    def _read(
        self, source, indices: Sequence[int] | None, device, on_frames: FramesHook | None = None,
    ) -> VideoFrames:
        import av

        if indices is not None and len(set(indices)) != len(indices):
            # torchcodec returns repeated indices as repeated frames
            unique = sorted({int(i) for i in indices})
            frames = self._read(source, unique, device, on_frames)
            pick = torch.tensor([unique.index(int(i)) for i in indices], dtype=torch.long)
            return VideoFrames(frames.data[pick.to(frames.data.device)], frames.metadata)
        wanted = None if indices is None else {int(i): pos for pos, i in enumerate(indices)}
        # always decode frame 0, PyAV only exposes the rotation on frames
        last = None if wanted is None else max(wanted, default=0)
        kept: list[torch.Tensor | None] = [None] * len(wanted) if wanted is not None else []
        scan = _PacketScan()
        decoded = 0
        rotation = None
        with av.open(_readable(source)) as container:
            if not container.streams.video:
                raise ValueError("no video stream")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for packet in container.demux(stream):
                # feed every packet, the decoder drops discarded frames itself.
                # PyAV's last packet is empty and flushes the decoder.
                scan.add(packet)
                if last is not None and decoded > last:
                    continue  # no more frames needed, keep counting packets
                for frame in packet.decode():
                    if rotation is None:
                        rotation = float(frame.rotation)
                    keep = wanted is None or decoded in wanted
                    if keep and on_frames is not None:
                        on_frames(frame.height, frame.width, 1)
                    if wanted is None:
                        kept.append(_frame_tensor(frame))
                    elif keep:
                        kept[wanted[decoded]] = _frame_tensor(frame)
                    decoded += 1
            metadata = _pyav_metadata(stream, scan, rotation)
        if wanted is not None:
            missing = [i for i, pos in wanted.items() if kept[pos] is None]
            if missing:
                raise IndexError(f"frame indices {missing[:8]} out of range for a {decoded}-frame video")
        if not kept:
            if indices is None:
                raise ValueError("no video frames decoded")
            dims = (0, 3, metadata["height"] or 0, metadata["width"] or 0)
            return VideoFrames(torch.empty(dims, dtype=torch.uint8, device=device), metadata)
        return VideoFrames(torch.stack(kept).to(device), metadata)


class _PacketScan:
    """Packet count and pts range, as torchcodec's scan records them."""

    def __init__(self):
        self.count = 0
        self.begin: int | None = None
        self.end: int | None = None

    def add(self, packet) -> None:
        """Count ``packet`` unless it's PyAV's empty flush packet or marked discard."""
        pts = packet.pts if packet.pts is not None else packet.dts
        if packet.size == 0 or packet.is_discard or pts is None:
            return
        self.count += 1
        self.begin = pts if self.begin is None else min(self.begin, pts)
        end = pts + (packet.duration or 0)
        self.end = end if self.end is None else max(self.end, end)


def _frame_tensor(frame) -> torch.Tensor:
    image = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
    quarter_turns = round((frame.rotation or 0) / 90) % 4
    if quarter_turns:
        # display matrix angles are counter-clockwise
        image = torch.rot90(image, quarter_turns, dims=(1, 2))
    return image.contiguous()


def _seconds(value, time_base) -> float | None:
    return float(value * time_base) if value is not None and time_base is not None else None


def _pyav_metadata(stream, scan: _PacketScan, rotation: float | None) -> dict:
    """Metadata for ``stream`` under torchcodec's VideoStreamMetadata keys,
    with the computed fields worked out the way torchcodec does after a scan."""
    ctx = stream.codec_context
    tb = stream.time_base
    # torchcodec uses r_frame_rate, which PyAV calls base_rate
    fps_from_header = float(stream.base_rate) if stream.base_rate else None
    begin = _seconds(scan.begin, tb) if scan.count else 0.0
    end = _seconds(scan.end, tb) if scan.count else 0.0
    duration = end - begin
    sar = stream.sample_aspect_ratio
    width, height = ctx.width or None, ctx.height or None
    if rotation is not None and round(rotation / 90) % 2:
        width, height = height, width
    return {
        "duration_seconds_from_header": _seconds(stream.duration, tb) if stream.duration else None,
        "begin_stream_seconds_from_header": _seconds(stream.start_time, tb),
        "bit_rate": float(ctx.bit_rate) if ctx.bit_rate is not None else None,
        "codec": getattr(ctx.codec, "canonical_name", None) or ctx.name,
        "stream_index": stream.index,
        "duration_seconds": duration,
        "begin_stream_seconds": begin,
        "begin_stream_seconds_from_content": begin,
        "end_stream_seconds_from_content": end,
        "width": width,
        "height": height,
        "num_frames_from_header": stream.frames or None,
        "num_frames_from_content": scan.count,
        "average_fps_from_header": fps_from_header,
        "pixel_aspect_ratio": float(sar) if sar else None,
        "rotation": rotation if rotation else None,
        # PyAV doesn't expose FFmpeg's names for these
        "color_primaries": None,
        "color_space": None,
        "color_transfer_characteristic": None,
        "pixel_format": ctx.format.name if ctx.format is not None else None,
        "end_stream_seconds": end,
        "num_frames": scan.count,
        "average_fps": scan.count / duration if duration else fps_from_header,
    }

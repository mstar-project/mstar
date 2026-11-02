"""Decoded-size caps for request media.

A compressed image or video can be tiny on the wire and enormous once decoded
(a flat PNG compresses about 1000:1, a still video far more), so the loaders
check what a file will decode to, not how big it is. A refusal is a
``ValueError``, which the data worker answers with a 400.

The pixel and audio caps hold for a whole request: while the data worker loads
a request's files inside :class:`RequestMediaBudget`, every check also counts
against what the request's earlier files used.
"""
import os
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from mstar.utils import cpus


def video_decode_threads() -> int:
    """FFmpeg threads for one video decode. torchcodec decodes single-threaded
    by default, which dominates a long clip's request latency; frame threading
    is bit-identical. Bounded by 8 and by the CPUs the process's plan leaves
    unpinned, as ``CpuPlan.png_threads`` is. The bound is per decode:
    concurrent decodes each start this many threads. Not tied to torch's
    intra-op pool: FFmpeg's threads are its own, and that pool is kept small
    for fork-join ops a decode doesn't run."""
    return min(8, cpus.plan().left)


# PIL's DecompressionBombWarning threshold (about 9459 x 9459 RGB)
DEFAULT_MAX_IMAGE_PIXELS = 89_478_485
# about 39 s of 720p30, or 2.6 min of 640x360 at 30 fps, decoded at once: a
# full decode of every short (up to 2 min) Video-MME clip fits
DEFAULT_MAX_VIDEO_PIXELS = 1 << 30
# the least a decoded frame counts against the video cap
MIN_FRAME_PIXELS = 64 * 64
DEFAULT_MAX_MEDIA_FILES = 16
# twice the transcription endpoint's 30 min upload cap, so one such upload fits
# whatever its decoded length rounds to
DEFAULT_MAX_AUDIO_SECONDS = 60 * 60
# torchvision decodes these (GIF is read with Pillow); anything else is a 400
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
# a loader that decodes with Pillow also reads the other image types the API accepts
PILLOW_IMAGE_FORMATS = IMAGE_FORMATS + ("BMP", "TIFF")


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}")
    return value


def max_image_pixels() -> int:
    """``MSTAR_MAX_IMAGE_PIXELS``: pixels a request's images may decode to,
    summed over them."""
    return _positive_int_env("MSTAR_MAX_IMAGE_PIXELS", DEFAULT_MAX_IMAGE_PIXELS)


def max_video_pixels() -> int:
    """``MSTAR_MAX_VIDEO_PIXELS``: pixels a request's videos may decode to,
    summed over the frames the loader decodes from each."""
    return _positive_int_env("MSTAR_MAX_VIDEO_PIXELS", DEFAULT_MAX_VIDEO_PIXELS)


def max_media_files() -> int:
    """``MSTAR_MAX_MEDIA_FILES``: images, audio clips and videos one request
    may attach, together."""
    return _positive_int_env("MSTAR_MAX_MEDIA_FILES", DEFAULT_MAX_MEDIA_FILES)


def max_audio_seconds() -> int:
    """``MSTAR_MAX_AUDIO_SECONDS``: seconds a request's audio clips may decode
    to, summed over them."""
    return _positive_int_env("MSTAR_MAX_AUDIO_SECONDS", DEFAULT_MAX_AUDIO_SECONDS)


def check_media_limits() -> None:
    """Read the limits once at startup: a bad value then stops the server,
    instead of turning every media request into a 400 that blames the client.

    Also lifts Pillow's own bomb check to the image limit, which it would
    otherwise apply first (at about 179M pixels) whatever the limit is.
    """
    from PIL import Image

    limit = max_image_pixels()
    max_video_pixels()
    max_media_files()
    max_audio_seconds()
    if Image.MAX_IMAGE_PIXELS is not None and limit > Image.MAX_IMAGE_PIXELS:
        Image.MAX_IMAGE_PIXELS = limit


_request_budget: ContextVar["RequestMediaBudget | None"] = ContextVar("mstar_request_media_budget", default=None)


class RequestMediaBudget:
    """Holds one request's media to the caps together, not file by file, so
    many files in one request cannot multiply them; refuses more than
    ``max_media_files()`` files before any is decoded.

    The data worker loads each file inside :meth:`file`; the image, video and
    audio checks charge the request through it."""

    # kind: what the refusal calls its files, their unit, the env var
    _LIMITS = {
        "image": ("images", "pixels", "MSTAR_MAX_IMAGE_PIXELS"),
        "video": ("videos", "pixels", "MSTAR_MAX_VIDEO_PIXELS"),
        "audio": ("audio clips", "seconds", "MSTAR_MAX_AUDIO_SECONDS"),
    }

    def __init__(self, num_files: int) -> None:
        limit = max_media_files()
        if num_files > limit:
            raise ValueError(
                f"the request attaches {num_files} media files; the limit is {limit} (MSTAR_MAX_MEDIA_FILES)"
            )
        self.limits = {"image": max_image_pixels(), "video": max_video_pixels(), "audio": max_audio_seconds()}
        self.used = dict.fromkeys(self.limits, 0)
        self._file = dict.fromkeys(self.limits, 0)

    @contextmanager
    def file(self):
        """Load one file. It is charged the most any single check counted for
        it, so a header estimate and the decode after it are not added up."""
        token = _request_budget.set(self)
        try:
            yield
        finally:
            _request_budget.reset(token)
            for kind, amount in self._file.items():
                self.used[kind] += amount
                self._file[kind] = 0

    def charge(self, kind: str, amount: float) -> None:
        self._file[kind] = max(self._file[kind], amount)
        if self.used[kind] + self._file[kind] > self.limits[kind]:
            files, unit, env = self._LIMITS[kind]
            raise ValueError(
                f"the request's {files} decode to more than {self.limits[kind]} {unit} together; "
                f"send fewer or smaller files (the limit is {env})"
            )


def _charge_request(kind: str, amount: float) -> None:
    budget = _request_budget.get()
    if budget is not None:
        budget.charge(kind, amount)


def check_file_counts(counts: dict[str, int], model_limits: dict[str, int]) -> None:
    """Refuse more files of a modality than the model takes
    (:meth:`mstar.model.base.Model.max_input_files`), before any is decoded."""
    for modality, limit in model_limits.items():
        count = counts.get(modality, 0)
        if count > limit:
            raise ValueError(
                f"the request attaches {count} {modality} files; this model takes at most {limit}"
            )


# libsndfile reports 2**63 - 1 frames for a FLAC whose STREAMINFO has no sample
# count (a stream written to a pipe); anything this large is "unknown"
UNKNOWN_FRAMES = 1 << 62
# frames per read while counting a file whose header gives no length
_COUNT_BLOCK_FRAMES = 4096
# bits a sample of each libsndfile subtype takes in the file; the file-size
# bound below uses the smallest (4, an ADPCM nibble) for any it does not list
_SUBTYPE_BITS = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
    "ULAW": 8,
    "ALAW": 8,
}
_MIN_SAMPLE_BITS = 4


def probe_audio_seconds(filepath: str) -> float | None:
    """The duration an audio file's header declares, or None when it declares
    none (a sample count of 0 or libsndfile's "unknown" value) or neither
    soundfile nor torchcodec can read it."""
    try:
        import soundfile

        info = soundfile.info(filepath)
        if 0 < info.frames < UNKNOWN_FRAMES and info.samplerate:
            return info.frames / info.samplerate
    except Exception:  # noqa: BLE001, S110 -- not a libsndfile container: try torchcodec
        pass
    try:
        from torchcodec.decoders import AudioDecoder

        seconds = AudioDecoder(filepath).metadata.duration_seconds_from_header
        if seconds:
            return float(seconds)
    except Exception:  # noqa: BLE001, S110 -- unreadable: the model's loader reports it
        pass
    return None


def _count_audio_seconds(filepath: str) -> None:
    """Charge the request a libsndfile file's length counted as it decodes, for
    a header that gives none; refuses as soon as the count passes the limit,
    so the decode stops there.

    Reads fixed blocks and charges the frames counted so far after each. A
    header-less FLAC cannot seek past its last block, so libsndfile raises on
    the read that returns it; that block is charged whole, which can overstate
    the length by under one block (0.26 s at 16 kHz). A file libsndfile reads
    as empty (a WAV whose data chunk length is 0) is charged from its size at
    its subtype's sample width (4 bits for one not in the table), an upper
    bound, since a loader that reads to the end of the file decodes what
    libsndfile does not count.
    """
    try:
        import soundfile
    except ImportError:
        return
    try:
        with soundfile.SoundFile(filepath) as sound:
            channels = max(sound.channels, 1)
            rate = sound.samplerate
            bits = _SUBTYPE_BITS.get(sound.subtype, _MIN_SAMPLE_BITS)
            frames = 0
            while True:
                try:
                    block = sound.read(_COUNT_BLOCK_FRAMES, dtype="int16", always_2d=True)
                except RuntimeError:  # soundfile.LibsndfileError
                    # the end of a header-less FLAC (or damage the loader will report)
                    frames += _COUNT_BLOCK_FRAMES
                    _charge_request("audio", frames / rate)
                    return
                frames += len(block)
                _charge_request("audio", frames / rate)
                if len(block) < _COUNT_BLOCK_FRAMES:
                    break
            if frames == 0:
                _charge_request("audio", os.path.getsize(filepath) * 8 / (bits * channels) / rate)
    except (RuntimeError, OSError):  # soundfile.LibsndfileError, a file it cannot open
        return  # not a libsndfile container: the loader's decode is charged instead


def check_audio_duration(filepath: str) -> None:
    """Charge the request what an audio file's header declares, before the
    model's loader decodes it; a file whose header declares nothing is counted
    as it decodes (:func:`_count_audio_seconds`)."""
    seconds = probe_audio_seconds(filepath)
    if seconds is not None:
        _charge_request("audio", seconds)
    else:
        _count_audio_seconds(filepath)


def charge_decoded_audio(audio: torch.Tensor, sample_rate: int) -> None:
    """Charge the request the samples a loader returned, as seconds at
    ``sample_rate``; a file whose header understated its length or could not be
    read is counted here."""
    _charge_request("audio", audio.numel() / sample_rate)


def sound_file_seconds(frames: int, rate: int) -> float | None:
    """The length a libsndfile header declares, or None when it declares none
    (a sample count of 0 or libsndfile's "unknown" value)."""
    return frames / rate if 0 < frames < UNKNOWN_FRAMES and rate else None


def _sound_file_blocks(sound, max_seconds: float, dtype: str):
    """Blocks of an open ``soundfile.SoundFile`` until its end or just past
    ``max_seconds``, for a header that gives no sample count.

    A header-less FLAC cannot seek past its last block, so libsndfile raises on
    the read that reaches it; that read ends the clip, which loses its last
    FLAC frame (4096 samples from libsndfile's own encoder).
    """
    limit = max_seconds * sound.samplerate
    frames = 0
    while frames <= limit:
        try:
            block = sound.read(_COUNT_BLOCK_FRAMES, dtype=dtype, always_2d=True)
        except RuntimeError:  # soundfile.LibsndfileError
            return
        if len(block):
            yield block
        frames += len(block)
        if len(block) < _COUNT_BLOCK_FRAMES:
            return


def count_sound_file_seconds(filepath: str, max_seconds: float) -> float:
    """The length of a libsndfile file whose header gives none, counted as it
    decodes and keeping no samples; stops just past ``max_seconds``, so the
    caller's limit check refuses a longer clip without decoding all of it."""
    import soundfile

    with soundfile.SoundFile(filepath) as sound:
        frames = sum(len(block) for block in _sound_file_blocks(sound, max_seconds, "int16"))
        return frames / sound.samplerate


def read_sound_file(filepath: str, max_seconds: float):
    """``soundfile.read(filepath, dtype="float32", always_2d=True)`` for a
    header that gives no sample count, where ``soundfile.read`` would size its
    buffer from the header (libsndfile's 2**63 - 1 frames) and fail; decodes
    block by block and stops just past ``max_seconds``, so the caller's length
    check refuses a longer clip."""
    import numpy as np
    import soundfile

    with soundfile.SoundFile(filepath) as sound:
        blocks = list(_sound_file_blocks(sound, max_seconds, "float32"))
        empty = np.zeros((0, sound.channels), dtype=np.float32)
        return (np.concatenate(blocks) if blocks else empty), sound.samplerate


def check_image_size(filepath: str, formats: tuple[str, ...] = IMAGE_FORMATS) -> tuple[str, bool]:
    """Refuse an image whose header promises more than ``max_image_pixels()``
    or that is not in ``formats`` (the ones the caller's decoder reads);
    returns Pillow's name for the format and whether the file holds more than
    one frame.

    Reads the header only. The decoders allocate what the header declares, so
    it cannot understate the decode.
    """
    fmt, animated, _, _ = _check_image_header(filepath, formats)
    return fmt, animated


def _check_image_header(filepath: str, formats: tuple[str, ...]) -> tuple[str, bool, str, bool]:
    """:func:`check_image_size`, plus Pillow's mode and whether the file
    carries transparency (an alpha channel or a tRNS or GIF transparent color)."""
    import warnings

    from PIL import Image, UnidentifiedImageError

    limit = max_image_pixels()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(filepath, formats=formats) as image:
                width, height = image.size
                fmt = image.format or ""
                animated = bool(getattr(image, "is_animated", False))
                mode = image.mode
                transparent = bool(image.has_transparency_data)
    except Image.DecompressionBombError:
        raise ValueError(f"image is too large to decode; the limit is {limit} pixels") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        # the decoder's message can name the server path; the client needs only this
        raise ValueError("the image is not in a format this server can read") from None
    if width * height > limit:
        raise ValueError(f"image is {width}x{height} ({width * height} pixels); the limit is {limit} pixels")
    _charge_request("image", width * height)
    return fmt, animated, mode, transparent


def _on_white(image: torch.Tensor) -> torch.Tensor:
    """uint8 ``[4, H, W]`` RGBA composited on white, as ``[3, H, W]``."""
    rgb, alpha = image[:3].to(torch.int32), image[3:].to(torch.int32)
    return ((rgb * alpha + 255 * (255 - alpha) + 127) // 255).to(torch.uint8)


def read_image(filepath: str, *, rgb: bool = False) -> torch.Tensor:
    """A request image as uint8 ``[C, H, W]``, refused first if it is too large.

    ``rgb`` forces three channels, with any transparency composited on white
    as the Qwen-VL reference preprocessing does for RGBA (dropping the alpha
    would turn a transparent background black); otherwise the file's own
    channels are kept, as ``torchvision.io.decode_image`` does. A GIF or an
    animated WebP is read with Pillow, as its first frame in RGB: torchvision's
    GIF decoder is not safe on untrusted files before torchvision 0.28, and its
    WebP decoder refuses animation.
    """
    import torchvision

    fmt, animated, pil_mode, transparent = _check_image_header(filepath, IMAGE_FORMATS)
    # torchvision ignores a tRNS color on a gray or RGB PNG; Pillow applies it
    trns_color = rgb and transparent and pil_mode in ("L", "RGB")
    if fmt == "GIF" or (fmt == "WEBP" and animated) or trns_color:
        import numpy as np
        from PIL import Image

        with Image.open(filepath) as image:
            array = np.array(image.convert("RGBA"))
        return _on_white(torch.from_numpy(array).permute(2, 0, 1))
    if not rgb:
        mode = torchvision.io.ImageReadMode.UNCHANGED
    else:
        # only a file with transparency is read with alpha: a CMYK JPEG's
        # four channels are not RGBA
        mode = torchvision.io.ImageReadMode.RGB_ALPHA if transparent else torchvision.io.ImageReadMode.RGB
    image = torchvision.io.decode_image(filepath, mode=mode)
    if image.dtype == torch.uint16:
        # a 16-bit PNG: its high byte, as Pillow reads a 16-bit RGB PNG
        image = (image.to(torch.int32) >> 8).to(torch.uint8)
    if rgb and image.shape[0] == 4:
        image = _on_white(image)
    return image


def probe_video(filepath: str) -> tuple[int, float, int, int] | None:
    """``(num_frames, fps, height, width)`` from the container header, or None
    when neither torchcodec nor PyAV can read it."""
    try:
        from torchcodec.decoders import VideoDecoder

        # "exact" would scan every packet to count frames; the header is enough here
        meta = VideoDecoder(filepath, seek_mode="approximate").metadata
        if meta.num_frames and meta.average_fps and meta.height and meta.width:
            return int(meta.num_frames), float(meta.average_fps), int(meta.height), int(meta.width)
    except Exception:  # noqa: BLE001, S110 -- missing or unable to read it: try PyAV
        pass
    try:
        import av

        with av.open(filepath) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate or stream.base_rate
            fps = float(rate) if rate else 0.0
            frames = stream.frames or (
                int(float(stream.duration * stream.time_base) * fps) if stream.duration and fps else 0
            )
            ctx = stream.codec_context
            if frames and fps and ctx.height and ctx.width:
                return int(frames), fps, int(ctx.height), int(ctx.width)
    except Exception:  # noqa: BLE001, S110 -- unreadable: the loader's own decoder reports it
        pass
    return None


class VideoBudget:
    """Counts decoded frames against ``max_video_pixels()``, so a loader stops
    before it holds more than the cap whatever the container claims.

    A frame counts as at least ``MIN_FRAME_PIXELS``: each decoded frame costs
    about 2 KB of tensor and frame objects whatever its size, so a clip of
    millions of 2x2 frames would otherwise pass the cap at hundreds of GB."""

    def __init__(self) -> None:
        self.limit = max_video_pixels()
        self.pixels = 0

    def add(self, height: int, width: int, frames: int = 1) -> None:
        self.pixels += max(int(height) * int(width), MIN_FRAME_PIXELS) * int(frames)
        if self.pixels > self.limit:
            raise ValueError(
                f"video decodes to more than {self.limit} pixels; send a shorter "
                "or smaller clip (the limit is MSTAR_MAX_VIDEO_PIXELS)"
            )
        _charge_request("video", self.pixels)

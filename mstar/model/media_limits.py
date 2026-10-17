"""Decoded-size caps for request media.

A compressed image or video can be tiny on the wire and enormous once decoded
(a flat PNG compresses about 1000:1, a still video far more), so the loaders
check what a file will decode to, not how big it is. A refusal is a
``ValueError``, which the data worker answers with a 400.
"""
import os

import torch

# PIL's DecompressionBombWarning threshold (about 9459 x 9459 RGB)
DEFAULT_MAX_IMAGE_PIXELS = 89_478_485
# about 39 s of 720p30, or 2.6 min of 640x360 at 30 fps, decoded at once: a
# full decode of every short (up to 2 min) Video-MME clip fits
DEFAULT_MAX_VIDEO_PIXELS = 1 << 30


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
    """``MSTAR_MAX_IMAGE_PIXELS``: pixels one request image may decode to."""
    return _positive_int_env("MSTAR_MAX_IMAGE_PIXELS", DEFAULT_MAX_IMAGE_PIXELS)


def max_video_pixels() -> int:
    """``MSTAR_MAX_VIDEO_PIXELS``: pixels one request video may decode to,
    summed over the frames a loader decodes."""
    return _positive_int_env("MSTAR_MAX_VIDEO_PIXELS", DEFAULT_MAX_VIDEO_PIXELS)


def check_media_limits() -> None:
    """Read both limits once at startup: a bad value then stops the server,
    instead of turning every media request into a 400 that blames the client."""
    max_image_pixels()
    max_video_pixels()


def check_image_size(filepath: str) -> str:
    """Refuse an image whose header promises more than ``max_image_pixels()``;
    returns Pillow's name for the format.

    Reads the header only. The decoders allocate what the header declares, so
    it cannot understate the decode.
    """
    import warnings

    from PIL import Image, UnidentifiedImageError

    limit = max_image_pixels()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(filepath) as image:
                width, height = image.size
                fmt = image.format or ""
    except Image.DecompressionBombError:
        raise ValueError(f"image is too large to decode; the limit is {limit} pixels") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        # the decoder's message can name the server path; the client needs only this
        raise ValueError("the image is not in a format this server can read") from None
    if width * height > limit:
        raise ValueError(f"image is {width}x{height} ({width * height} pixels); the limit is {limit} pixels")
    return fmt


def read_image(filepath: str, *, rgb: bool = False) -> torch.Tensor:
    """A request image as uint8 ``[C, H, W]``, refused first if it is too large.

    ``rgb`` forces three channels; otherwise the file's own channels are kept,
    as ``torchvision.io.decode_image`` does. A GIF is read with Pillow, as its
    first frame in RGB: torchvision's GIF decoder is not safe on untrusted
    files before torchvision 0.28.
    """
    import torchvision

    if check_image_size(filepath) == "GIF":
        import numpy as np
        from PIL import Image

        with Image.open(filepath) as image:
            array = np.array(image.convert("RGB"))
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mode = torchvision.io.ImageReadMode.RGB if rgb else torchvision.io.ImageReadMode.UNCHANGED
    return torchvision.io.decode_image(filepath, mode=mode)


def probe_video(filepath: str) -> tuple[int, float, int, int] | None:
    """``(num_frames, fps, height, width)`` from the container header, or None
    when neither torchcodec nor PyAV can read it."""
    try:
        from torchcodec.decoders import VideoDecoder

        meta = VideoDecoder(filepath).metadata
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


def read_video_whole_budgeted(ele: dict) -> tuple[torch.Tensor, dict, float]:
    """Drop-in for qwen-omni-utils' torchvision reader, which ``fetch_video``
    falls back to when torchcodec or decord raises and which decodes the whole
    clip. Same frames (``torchvision.io.read_video`` decodes with PyAV too) and
    the same sampling, but every frame is counted as it is decoded: a header
    can understate the clip, so checking it first would not bound this."""
    import av
    from qwen_omni_utils.v2_5.vision_process import smart_nframes

    path = ele["video"]
    if path.startswith("file://"):
        path = path[len("file://"):]
    start, end = ele.get("video_start", 0.0), ele.get("video_end")
    budget = VideoBudget()
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        for frame in container.decode(stream):
            if frame.time is not None and frame.time < start:
                continue
            if end is not None and frame.time is not None and frame.time > end:
                break
            budget.add(frame.height, frame.width)
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
    if not frames:
        raise ValueError("no video frames could be decoded")
    total = len(frames)
    nframes = smart_nframes(ele, total_frames=total, video_fps=fps)
    idx = torch.linspace(0, total - 1, nframes).round().long()
    video = torch.stack([frames[i] for i in idx.tolist()])
    metadata = dict(fps=fps, frames_indices=idx, total_num_frames=total, video_backend="torchvision")
    return video, metadata, nframes / max(total, 1e-6) * fps


class VideoBudget:
    """Counts decoded frames against ``max_video_pixels()``, so a loader stops
    before it holds more than the cap whatever the container claims."""

    def __init__(self) -> None:
        self.limit = max_video_pixels()
        self.pixels = 0

    def add(self, height: int, width: int, frames: int = 1) -> None:
        self.pixels += int(height) * int(width) * int(frames)
        if self.pixels > self.limit:
            raise ValueError(
                f"video decodes to more than {self.limit} pixels; send a shorter "
                "or smaller clip (the limit is MSTAR_MAX_VIDEO_PIXELS)"
            )

"""Decoded-size caps for request media.

A compressed image or video can be tiny on the wire and enormous once decoded
(a flat PNG compresses about 1000:1, a still video far more), so the loaders
check what a file will decode to, not how big it is. A refusal is a
``ValueError``, which the data worker answers with a 400.
"""
import os

# PIL's DecompressionBombWarning threshold (about 9459 x 9459 RGB)
DEFAULT_MAX_IMAGE_PIXELS = 89_478_485
# about 39 s of 720p30, or 17 s of 1080p30, decoded at once
DEFAULT_MAX_VIDEO_PIXELS = 1 << 30


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}")
    return value


def max_image_pixels() -> int:
    """``MSTAR_MAX_IMAGE_PIXELS``: pixels one request image may decode to,
    counting every frame of an animated GIF."""
    return _positive_int_env("MSTAR_MAX_IMAGE_PIXELS", DEFAULT_MAX_IMAGE_PIXELS)


def max_video_pixels() -> int:
    """``MSTAR_MAX_VIDEO_PIXELS``: pixels one request video may decode to,
    summed over the frames a loader keeps."""
    return _positive_int_env("MSTAR_MAX_VIDEO_PIXELS", DEFAULT_MAX_VIDEO_PIXELS)


def check_image_size(filepath: str) -> None:
    """Refuse an image whose header promises more than ``max_image_pixels()``.

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
                # torchvision decodes every frame of a GIF and the first of the rest
                frames = getattr(image, "n_frames", 1) if image.format == "GIF" else 1
    except Image.DecompressionBombError:
        raise ValueError(f"image is too large to decode; the limit is {limit} pixels") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        # the decoder's message can name the server path; the client needs only this
        raise ValueError("the image is not in a format this server can read") from None
    pixels = width * height * frames
    if pixels > limit:
        what = f"{width}x{height}" + (f" x {frames} frames" if frames > 1 else "")
        raise ValueError(f"image is {what} ({pixels} pixels); the limit is {limit} pixels")


class VideoBudget:
    """Counts decoded frames against ``max_video_pixels()``, so a loader stops
    before it holds more than the cap whatever the container claims."""

    def __init__(self) -> None:
        self.limit = max_video_pixels()
        self.pixels = 0

    def add(self, height: int, width: int) -> None:
        self.pixels += int(height) * int(width)
        if self.pixels > self.limit:
            raise ValueError(
                f"video decodes to more than {self.limit} pixels; send a shorter "
                "or smaller clip (the limit is MSTAR_MAX_VIDEO_PIXELS)"
            )

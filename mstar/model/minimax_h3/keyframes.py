"""FL2VA keyframes onto the canvas (``fit_keyframes``: ``MiniMaxH3ResizeStep``, with the
released model's rounding) and into the prompt (``presentation``: a ``"<Picture i>: "``
label and a vision block per keyframe, then the prompt). Vision blocks are tagged video
for the DiT's AdaLN.
"""

from __future__ import annotations

from typing import NamedTuple

from PIL import Image

from mstar.model.minimax_h3.config import TEXT_TAG, VIDEO_TAG, resolve_canvas_size

ANCHORS = ("first", "last")


class Keyframes(NamedTuple):
    height: int
    width: int
    images: list[Image.Image]   # on the canvas, in packed order
    anchors: tuple[str, ...]     # "first" / "last", positional with ``images``


def fit_keyframes(
    first: Image.Image | None, last: Image.Image | None, height: int | None = None, width: int | None = None,
) -> Keyframes:
    keyframes = [image for image in (first, last) if image is not None]
    anchors = tuple(anchor for anchor, image in zip(ANCHORS, (first, last), strict=True) if image is not None)
    if not keyframes:
        raise ValueError("an FL2VA request needs a first or a last keyframe")
    if (height is None) != (width is None):
        raise ValueError("height and width have to be passed together, or neither of them")
    if height is None:
        height, width = resolve_canvas_size(*keyframes[0].size)
    fitted = []
    for index, image in enumerate(keyframes):
        image = image.convert("RGB")
        if image.size == (width, height):
            fitted.append(image)
        elif index == 0:
            fitted.append(image.resize((width, height), Image.Resampling.LANCZOS))
        else:
            # not VaeImageProcessor's "crop" mode: that floors and centres differently, which
            # moves the conditioning latents off the released model's by a pixel on about
            # half of aspect ratios
            scale = max(width / image.size[0], height / image.size[1])
            size = (max(width, round(image.size[0] * scale)), max(height, round(image.size[1] * scale)))
            left, top = max(0, (size[0] - width) // 2), max(0, (size[1] - height) // 2)
            fitted.append(image.resize(size, Image.Resampling.LANCZOS).crop((left, top, left + width, top + height)))
    return Keyframes(height, width, fitted, anchors)


class Presentation(NamedTuple):
    token_ids: list[int]
    tags: list[int]       # per token: TEXT_TAG, or VIDEO_TAG for a vision block


def vision_spans(tokenizer, image_tokens: list[int]) -> list[tuple[int, int]]:
    """``(start, length)`` of each keyframe's vision block in ``presentation``'s rows."""
    spans, start = [], 0
    for index, count in enumerate(image_tokens):
        start += len(tokenizer.encode(f"<Picture {index + 1}>: ", add_special_tokens=False).ids)
        spans.append((start, count + 2))
        start += count + 2
    return spans


def presentation(tokenizer, prompt: str, image_tokens: list[int]) -> Presentation:
    """``tokenizer`` is the checkpoint's ``tokenizers.Tokenizer``; ``image_tokens`` is each
    keyframe's merged token count (``grid_t * grid_h * grid_w / merge^2``)."""
    start, pad, end = (tokenizer.token_to_id(t) for t in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"))
    if None in (start, pad, end):
        raise ValueError("the tokenizer has no Qwen vision tokens")
    token_ids: list[int] = []
    tags: list[int] = []
    for index, count in enumerate(image_tokens):
        label = tokenizer.encode(f"<Picture {index + 1}>: ", add_special_tokens=False).ids
        vision = [start] + [pad] * count + [end]
        token_ids += label + vision
        tags += [TEXT_TAG] * len(label) + [VIDEO_TAG] * len(vision)
    text = tokenizer.encode(prompt, add_special_tokens=False).ids
    return Presentation(token_ids + text, tags + [TEXT_TAG] * len(text))

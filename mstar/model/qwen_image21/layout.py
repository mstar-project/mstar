"""One request's prompt as the DiT runs it: the prefix spans, the target, and their KV layout.

The prompt (system turn dropped) is text₀, img₁, text₁, …, imgₙ, textₙ. Each span is one
committing step of the dit: a text span reads its rows of the Qwen3-VL hidden states, an image
span its VAE latents (each of the image's Qwen3-VL slots stands for 2x2 latents, whose hidden
states the DiT never reads). A text token takes the next position on all three rope axes; an image
holds the frame axis at the position its text reached and moves it on by ``max(h, w)``. The
target sits at the position its label's prefix ends at, so under CFG ``uncond``'s differs.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from mstar.model.base import LabelLayout, Span
from mstar.model.qwen_image21.components.transformer import TOKENS_PER_SLOT
from mstar.model.qwen_image21.config import MAIN, PREFIX_IMAGE_WALK, PREFIX_TEXT_WALK, UNCOND

# The request tensor the keyed text spans' ids are read from (the spans before the last image).
PREFIX_IDS = "prefix_ids"


class PrefixSpan(NamedTuple):
    """One prefix step. ``image`` is -1 for text; ``rows`` starts the span in the text embeds (text)
    or the concatenated reference latents (image)."""

    image: int
    rows: int
    length: int
    position: int
    height: int = 0
    width: int = 0

    @property
    def is_text(self) -> bool:
        return self.image < 0

    @property
    def advance(self) -> int:
        return self.length if self.is_text else max(self.height, self.width)


class RequestLayout(NamedTuple):
    spans: tuple[PrefixSpan, ...]
    # ``uncond``'s tail (its rows in the text embeds and length), when the request runs CFG
    uncond_tail: tuple[int, int] | None
    target: tuple[int, int]  # latent (h, w)

    @property
    def images(self) -> int:
        return sum(not span.is_text for span in self.spans)

    def frames(self) -> tuple[int, ...]:
        """The target's frame position per label: ``main``, then ``uncond`` under CFG."""
        tail = self.spans[-1]
        out = (tail.position + tail.length,)
        if self.uncond_tail is not None:
            out += (tail.position + self.uncond_tail[1],)
        return out

    def keyed_spans(self) -> int:
        """Every span through the last image: an edit repeats its images, not its prompt, and the
        last image's step must always run, since it forks ``uncond`` under CFG."""
        return len(self.spans) - 1

    def to_metadata(self) -> dict:
        return {
            "spans": [list(span) for span in self.spans],
            "uncond_tail": None if self.uncond_tail is None else list(self.uncond_tail),
            "target": list(self.target),
        }

    @classmethod
    def from_metadata(cls, meta: dict) -> "RequestLayout":
        tail = meta["uncond_tail"]
        return cls(
            spans=tuple(PrefixSpan(*span) for span in meta["spans"]),
            uncond_tail=None if tail is None else (int(tail[0]), int(tail[1])),
            target=(int(meta["target"][0]), int(meta["target"][1])),
        )

    def kv_layout(self, output_resolution: int) -> dict[str, LabelLayout]:
        """``main`` writes every span and denoises over them; ``uncond`` holds ``main``'s spans
        through the last image (forked after it, or nothing for text-to-image) and its own tail."""
        spans = []
        for span in self.spans:
            if span.is_text:
                spans.append(Span("ids", PREFIX_TEXT_WALK, span.length, span.length, PREFIX_IDS))
            else:
                spans.append(Span(
                    "digest", PREFIX_IMAGE_WALK, span.length, span.advance, ("image", span.image),
                    params=(output_resolution,),
                ))
        scratch = self.target[0] * self.target[1]
        layouts = {MAIN: LabelLayout(spans=tuple(spans), scratch=scratch, keyed_spans=self.keyed_spans())}
        if self.uncond_tail is not None:
            tail = self.uncond_tail[1]
            layouts[UNCOND] = LabelLayout(
                spans=(*spans[:-1], Span("ids", PREFIX_TEXT_WALK, tail, tail, PREFIX_IDS)), scratch=scratch,
            )
        return layouts


def prompt_spans(
    ids: torch.Tensor, image_token_id: int, grids: list[tuple[int, int]], target: tuple[int, int],
    uncond_ids: torch.Tensor | None = None,
) -> RequestLayout:
    """``ids [L]`` the prompt past the system turn, with each image's run of image pads; ``grids`` each
    image's latent ``(h, w)``; ``uncond_ids`` the negative prompt's, whose image runs are the same.
    Their hidden states sit end to end in the text embeds."""
    is_image = (ids == image_token_id).tolist()
    spans: list[PrefixSpan] = []
    position = latent_row = image = 0
    start = 0
    while start < len(is_image):
        end = start
        while end < len(is_image) and is_image[end] == is_image[start]:
            end += 1
        if is_image[start]:
            if image >= len(grids):
                raise ValueError("more image-pad runs than images")
            h, w = grids[image]
            if (end - start) * TOKENS_PER_SLOT != h * w:
                raise ValueError(
                    f"image {image} has {end - start} Qwen3-VL slots for a {h}x{w} latent grid; each slot is 2x2 "
                    "latents, so the image was resized off the 32-px grid"
                )
            spans.append(PrefixSpan(image, latent_row, h * w, position, h, w))
            latent_row, position, image = latent_row + h * w, position + max(h, w), image + 1
        else:
            spans.append(PrefixSpan(-1, start, end - start, position))
            position += end - start
        start = end
    if image != len(grids):
        raise ValueError(f"{len(grids)} images but {image} image-pad runs")
    if not spans or not spans[-1].is_text:
        raise ValueError("the prompt must end on text (the user turn's close and the assistant header)")
    uncond_tail = None
    if uncond_ids is not None:
        if not torch.equal(uncond_ids[:spans[-1].rows], ids[:spans[-1].rows]):
            raise ValueError("the negative prompt's template must match the prompt's through the last image")
        uncond_tail = (len(is_image) + spans[-1].rows, int(uncond_ids.numel()) - spans[-1].rows)
    return RequestLayout(tuple(spans), uncond_tail, target)

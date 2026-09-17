"""Prompt-side plumbing for the Cosmos3-Edge reasoner: media preprocessing,
placeholder expansion and the 3D mRoPE positions of a VLM prompt.

Pure tensor / string helpers with no model state, shared by
``Cosmos3Model.process_prompt`` (data worker) and the reasoner submodule, and
matching the Hugging Face ``Cosmos3EdgeProcessor`` / ``Cosmos3EdgeModel``
byte-for-byte:

* ``preprocess_image`` / ``preprocess_video``: bicubic antialiased resize to
  multiples of ``patch_size * merge_size`` inside ``[min_pixels, max_pixels]``
  (on 8-bit pixels, like the reference), ``(x/255 - mean) / std``, then
  block-major 2x2 patchify so consecutive patches form the merger's blocks.
* ``expand_placeholders``: one ``<|image_pad|>`` per merged block for images;
  videos become one timestamped ``<t seconds><|vision_start|>...<|vision_end|>``
  span per frame.
* ``mrope_position_ids``: text tokens advance all three axes together; a
  vision span puts ``t`` on the temporal axis and the merged ``(h, w)`` grid on
  the spatial ones, all offset by the current position, and advances the
  cursor by ``max(h, w)`` merged patches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvF

from mstar.model.cosmos3.config import Cosmos3MediaProcessorConfig, Cosmos3ReasonerConfig

IMAGE = "image"
VIDEO = "video"


@dataclass(frozen=True)
class MediaGrid:
    """The patch grid of one preprocessed image (``t == 1``) or video, plus a
    video's per-frame timestamps in seconds."""

    t: int
    h: int
    w: int
    timestamps: tuple[float, ...] = ()

    def tokens(self, merge_size: int) -> int:
        """LLM tokens this media occupies (one per merged block)."""
        return self.t * self.h * self.w // (merge_size * merge_size)

    def tokens_per_frame(self, merge_size: int) -> int:
        return self.h * self.w // (merge_size * merge_size)

    @property
    def thw(self) -> tuple[int, int, int]:
        return (self.t, self.h, self.w)


def smart_resize(
    num_frames: int, height: int, width: int, temporal_factor: int, factor: int,
    min_pixels: int, max_pixels: int,
) -> tuple[int, int]:
    """Target (height, width): both multiples of ``factor``, the (temporal
    patch count x) pixel count inside ``[min_pixels, max_pixels]``, the aspect
    ratio kept as closely as possible."""
    if num_frames < temporal_factor:
        raise ValueError(f"num_frames={num_frames} must be >= temporal_factor={temporal_factor}")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _to_uint8(frames: torch.Tensor) -> torch.Tensor:
    """``[..., C, H, W]`` pixels to 8-bit: the data worker hands floats in
    [0, 1] (decoded 8-bit / 255), which round back exactly; 8-bit passes
    through. The reference resizes 8-bit pixels, so the resize must too."""
    if frames.dtype == torch.uint8:
        return frames
    return frames.mul(255.0).round().clamp_(0, 255).to(torch.uint8)


def _resize_normalize(frames: torch.Tensor, height: int, width: int, cfg: Cosmos3MediaProcessorConfig) -> torch.Tensor:
    """``[B, C, H, W]`` 8-bit -> resized, normalized fp32 ``[B, C, height, width]``."""
    frames = tvF.resize(frames, [height, width], interpolation=InterpolationMode.BICUBIC, antialias=True)
    frames = frames.to(torch.float32)
    mean = torch.tensor(cfg.image_mean, dtype=torch.float32, device=frames.device).view(1, -1, 1, 1) * 255.0
    std = torch.tensor(cfg.image_std, dtype=torch.float32, device=frames.device).view(1, -1, 1, 1) * 255.0
    return (frames - mean) / std


def patchify(frames: torch.Tensor, patch_size: int, merge_size: int) -> torch.Tensor:
    """``[T, C, H, W]`` -> ``[T * gh * gw, patch_size**2 * C]`` patches in
    time-major, block-major order (the ``merge_size x merge_size`` patches of
    one merger block are consecutive), pixel values ordered ``(ph, pw, C)``
    inside a patch."""
    t, c, h, w = frames.shape
    gh, gw = h // patch_size, w // patch_size
    x = frames.reshape(t, c, gh // merge_size, merge_size, patch_size, gw // merge_size, merge_size, patch_size)
    x = x.permute(0, 2, 5, 3, 6, 4, 7, 1)  # t, bh, bw, mh, mw, ph, pw, c
    return x.reshape(t * gh * gw, patch_size * patch_size * c)


def preprocess_image(image: torch.Tensor, cfg: Cosmos3MediaProcessorConfig) -> tuple[torch.Tensor, MediaGrid]:
    """One ``[C, H, W]`` image (8-bit or [0, 1] float) -> packed patches
    ``[gh * gw, patch_size**2 * C]`` and its grid."""
    if image.ndim != 3:
        raise ValueError(f"expected a [C, H, W] image, got {tuple(image.shape)}")
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    elif image.shape[0] == 4:
        image = image[:3]
    factor = cfg.patch_size * cfg.merge_size
    h, w = smart_resize(
        cfg.temporal_patch_size, image.shape[-2], image.shape[-1], cfg.temporal_patch_size, factor,
        cfg.min_pixels, cfg.max_pixels,
    )
    frames = _resize_normalize(_to_uint8(image).unsqueeze(0), h, w, cfg)
    return patchify(frames, cfg.patch_size, cfg.merge_size), MediaGrid(1, h // cfg.patch_size, w // cfg.patch_size)


def sample_frame_indices(
    total_frames: int, source_fps: float | None, cfg: Cosmos3MediaProcessorConfig,
    num_frames: int | None = None, fps: float | None = None,
) -> list[int]:
    """Uniform frame sampling: ``fps`` frames per second of source video
    (default the processor's), clamped to ``[min_frames, max_frames]`` and to
    the clip, or exactly ``num_frames``. Indices are linspace-rounded over the
    whole clip, like the reference."""
    if num_frames is not None and fps is not None:
        raise ValueError("num_frames and fps are mutually exclusive")
    if num_frames is None:
        rate = cfg.fps if fps is None else fps
        source_fps = source_fps or 24.0
        num_frames = int(total_frames / source_fps * rate)
        num_frames = min(max(num_frames, cfg.min_frames), cfg.max_frames, total_frames)
    num_frames = max(1, min(int(num_frames), total_frames))
    return torch.linspace(0, total_frames - 1, num_frames).round().long().tolist()


def preprocess_video(
    video: torch.Tensor, cfg: Cosmos3MediaProcessorConfig, source_fps: float | None,
    num_frames: int | None = None, fps: float | None = None,
) -> tuple[torch.Tensor, MediaGrid]:
    """A decoded ``[T, C, H, W]`` clip -> packed patches ``[T' * gh * gw, ...]``
    over the sampled frames and its grid with per-frame timestamps."""
    if video.ndim != 4:
        raise ValueError(f"expected a [T, C, H, W] video, got {tuple(video.shape)}")
    if video.shape[1] == 4:
        video = video[:, :3]
    indices = sample_frame_indices(video.shape[0], source_fps, cfg, num_frames=num_frames, fps=fps)
    frames = video[indices]
    factor = cfg.patch_size * cfg.merge_size
    h, w = smart_resize(
        frames.shape[0], frames.shape[-2], frames.shape[-1], cfg.temporal_patch_size, factor,
        cfg.min_pixels, cfg.max_pixels,
    )
    frames = _resize_normalize(_to_uint8(frames), h, w, cfg)
    timestamps = tuple(i / (source_fps or 24.0) for i in indices)
    grid = MediaGrid(frames.shape[0], h // cfg.patch_size, w // cfg.patch_size, timestamps)
    return patchify(frames, cfg.patch_size, cfg.merge_size), grid


def expand_placeholders(
    text: str, tokenizer, cfg: Cosmos3ReasonerConfig,
    image_grids: list[MediaGrid], video_grids: list[MediaGrid],
) -> str:
    """Replace the chat template's single-token media placeholders with the
    tokens the encoder will fill: ``<|image_pad|>`` x merged blocks per image;
    ``<|vision_start|><|video_pad|><|vision_end|>`` -> one
    ``<t seconds><|vision_start|>{pads}<|vision_end|>`` span per frame."""
    merge = cfg.vision.spatial_merge_size
    image_pad, video_pad, vs, ve = (
        tokenizer.convert_ids_to_tokens(i)
        for i in (cfg.image_token_id, cfg.video_token_id, cfg.vision_start_token_id, cfg.vision_end_token_id)
    )
    video_wrapper = vs + video_pad + ve
    out: list[str] = []
    i = 0
    images = iter(image_grids)
    videos = iter(video_grids)
    while i < len(text):
        if text.startswith(video_wrapper, i):
            grid = next(videos, None)
            if grid is None:
                raise ValueError("prompt has more video placeholders than video inputs")
            per_frame = grid.tokens_per_frame(merge)
            timestamps = grid.timestamps or tuple(range(grid.t))
            out.append("".join(
                f"<{ts:.1f} seconds>{vs}{video_pad * per_frame}{ve}" for ts in timestamps
            ))
            i += len(video_wrapper)
        elif text.startswith(image_pad, i):
            grid = next(images, None)
            if grid is None:
                raise ValueError("prompt has more image placeholders than image inputs")
            out.append(image_pad * grid.tokens(merge))
            i += len(image_pad)
        else:
            j = min(
                (k for k in (text.find(image_pad, i), text.find(video_wrapper, i)) if k != -1),
                default=len(text),
            )
            out.append(text[i:j])
            i = j
    if next(images, None) is not None or next(videos, None) is not None:
        raise ValueError("prompt has fewer media placeholders than media inputs")
    return "".join(out)


def mrope_position_ids(
    input_ids: torch.Tensor, cfg: Cosmos3ReasonerConfig,
    image_grids: list[MediaGrid], video_grids: list[MediaGrid],
) -> tuple[torch.Tensor, int]:
    """3D mRoPE ids ``[3, N]`` (temporal, height, width) of a rendered prompt
    and the position the first generated token takes.

    Text runs put the same increasing ids on all three axes. Each run of
    ``<|image_pad|>`` / ``<|video_pad|>`` tokens is one frame of its media
    (videos are rendered one span per frame): temporal = the cursor, height /
    width = the cursor plus the merged-grid coordinates; the cursor then
    advances by ``max(h, w)`` merged patches. Decoding continues at
    ``max(position) + 1`` on all axes."""
    ids = input_ids.tolist()
    merge = cfg.vision.spatial_merge_size
    frames: dict[int, list[tuple[int, int]]] = {
        cfg.image_token_id: [(g.h // merge, g.w // merge) for g in image_grids],
        cfg.video_token_id: [(g.h // merge, g.w // merge) for g in video_grids for _ in range(g.t)],
    }
    cursors = {cfg.image_token_id: 0, cfg.video_token_id: 0}
    pos = torch.empty(3, len(ids), dtype=torch.long)
    cur = 0
    i = 0
    n = len(ids)
    while i < n:
        tok = ids[i]
        if tok in frames:
            j = i
            while j < n and ids[j] == tok:
                j += 1
            k = cursors[tok]
            if k >= len(frames[tok]):
                raise ValueError("prompt has more media placeholder runs than media frames")
            h, w = frames[tok][k]
            cursors[tok] += 1
            if j - i != h * w:
                raise ValueError(f"placeholder run of {j - i} tokens does not match a {h}x{w} merged grid")
            hh = torch.arange(h).view(h, 1).expand(h, w).reshape(-1)
            ww = torch.arange(w).view(1, w).expand(h, w).reshape(-1)
            pos[0, i:j] = cur
            pos[1, i:j] = cur + hh
            pos[2, i:j] = cur + ww
            cur += max(h, w)
            i = j
        else:
            j = i
            while j < n and ids[j] not in frames:
                j += 1
            pos[:, i:j] = torch.arange(cur, cur + (j - i)).view(1, -1)
            cur += j - i
            i = j
    for tok, k in cursors.items():
        if k != len(frames[tok]):
            raise ValueError("prompt has fewer media placeholder runs than media frames")
    next_pos = int(pos.max().item()) + 1 if n else 0
    return pos, next_pos


def render_chat(
    tokenizer, parts, cfg: Cosmos3ReasonerConfig, enable_thinking: bool | None = None,
    system_prompt: str | None = None,
) -> str:
    """Render ordered prompt parts (text / image / video, as written) through
    the checkpoint's chat template with the generation prompt appended.
    Media parts become the template's single-token placeholders, which
    ``expand_placeholders`` then grows."""
    content: list[dict] = []
    for part in parts:
        if part.modality == "text":
            if part.text:
                content.append({"type": "text", "text": part.text})
        elif part.modality in (IMAGE, VIDEO):
            content.append({"type": part.modality})
        else:
            raise ValueError(f"the Cosmos3 reasoner has no encoder for {part.modality!r} inputs")
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = bool(enable_thinking)
    elif not cfg.enable_thinking:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)

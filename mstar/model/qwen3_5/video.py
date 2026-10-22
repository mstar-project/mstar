"""Qwen3.5 video input: frame sampling, timestamps and the prompt layout.

Mirrors transformers' ``Qwen3VLProcessor`` / ``Qwen3VLVideoProcessor``, which
Qwen3.5 checkpoints name. A video renders as its chat-template triple with the
pad replaced, per temporal group of ``temporal_patch_size`` frames, by
``<{t:.1f} seconds><|vision_start|><|video_pad|>*n<|vision_end|>``; the outer
sentinels stay. Only the pads are vision tokens: timestamps and every sentinel
take text positions, and each group is placed in MRoPE like a one-frame image.

Kept free of engine imports so the parts that must match transformers can be
tested without Triton.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import NamedTuple

import torch

from mstar.model.multimodal import TEXT, PromptPart

VIDEO = "video"


class SampledVideo(NamedTuple):
    """What ``load_video`` hands ``process_prompt``: the sampled frames only."""

    # (frames, C, H, W) uint8 RGB, as transformers' torchcodec reader returns
    frames: torch.Tensor
    # `transformers.video_utils.VideoMetadata` fields, wire-safe
    metadata: dict


class ProcessedVideo(NamedTuple):
    """One video as the vision walk takes it."""

    # (t * h * w, C * temporal_patch_size * patch_size ** 2), as the processor packs it
    pixel_values: torch.Tensor
    # (3,): t temporal groups of an h x w patch grid
    grid: torch.Tensor
    # seconds per temporal group, written into the prompt before each group
    timestamps: list[float]


def ffmpeg_unavailable() -> str | None:
    """Why torchcodec can't decode video in this process, or None if it can."""
    try:
        import torchcodec
        import torchcodec.decoders  # noqa: F401
    except (ImportError, RuntimeError, OSError) as exc:
        # torchcodec 0.9 and earlier raise here when FFmpeg is missing
        return f"it failed to load: {exc}"
    # 0.16 imports without FFmpeg and fails only at decode (0.8 has no
    # ffmpeg_major_version, and has already raised above)
    if getattr(torchcodec, "ffmpeg_major_version", 0) is None:
        return "it found no FFmpeg shared libraries"
    return None


def sample_video(path: str, video_processor) -> SampledVideo:
    """Decode the frames the checkpoint's video processor would sample.

    The same call transformers makes for a video path: torchcodec on the host
    (``seek_mode="exact"``), indices from the processor's own
    ``sample_frames`` (``fps`` per second, clamped to ``[min_frames,
    max_frames]``), and only those frames decoded. Not on the GPU: NVDEC's
    colour conversion need not match FFmpeg's, and there is no fallback
    decoder for the same reason.
    """
    # transformers' loader fetches URLs itself; the API server has already
    # resolved every attachment to a local file
    if not os.path.isfile(path):
        raise ValueError(f"Qwen3.5 video input {path!r} is not a local file")
    try:
        from transformers.video_utils import load_video
    except ImportError as exc:
        raise ValueError(f"Qwen3.5 video input needs transformers: {exc}") from exc
    if reason := ffmpeg_unavailable():
        raise ValueError(
            "Qwen3.5 video input needs torchcodec with FFmpeg, the decoder "
            f"transformers uses; {reason}"
        )

    from mstar.model.base import video_metadata_dict
    from mstar.model.media_limits import VideoBudget

    budget = VideoBudget()

    def sampler(metadata, **kwargs):
        indices = video_processor.sample_frames(
            metadata, num_frames=video_processor.num_frames, fps=video_processor.fps, **kwargs,
        )
        # transformers samples from the header before decoding, so the cap
        # holds before any frame is in memory
        if not (metadata.height and metadata.width):
            raise ValueError(f"Qwen3.5 video {path!r} has no frame size in its header")
        budget.add(metadata.height, metadata.width, len(indices))
        return indices

    try:
        frames, metadata = load_video(path, backend="torchcodec", sample_indices_fn=sampler)
    except RuntimeError as exc:
        # torchcodec's error for a corrupt or unsupported file: the client's input, a 400
        raise ValueError(f"Qwen3.5 could not decode video {path!r}: {exc}") from exc
    return SampledVideo(frames, video_metadata_dict(metadata))


def group_timestamps(
    frames_indices: Sequence[int], fps: float | None, temporal_patch_size: int,
) -> list[float]:
    """Seconds per temporal group: the mean of its first and last frame's time.

    ``Qwen3VLProcessor._calculate_timestamps``, including its padding of an
    uneven tail with the last index (the patchifier repeats the last frame
    the same way) and its 24 fps fallback.
    """
    fps = 24 if fps is None else fps
    # plain ints: transformers samples numpy, and these become prompt metadata
    indices = [int(i) for i in frames_indices]
    if pad := -len(indices) % temporal_patch_size:
        indices += [indices[-1]] * pad
    times = [i / fps for i in indices]
    return [
        (times[i] + times[i + temporal_patch_size - 1]) / 2
        for i in range(0, len(times), temporal_patch_size)
    ]


def expand_video_pads(
    text: str, pad: str, start: str, end: str, timestamps: Sequence[Sequence[float]],
) -> str:
    """Replace the nth ``pad`` in ``text`` with the nth video's groups.

    One pad per group: `find_media_spans` reads each group back as a span and
    the vision walk inserts its embeddings, as for an image.
    """
    pieces = text.split(pad)
    if len(pieces) != len(timestamps) + 1:
        raise ValueError(
            f"the rendered prompt has {len(pieces) - 1} video placeholders "
            f"for {len(timestamps)} videos"
        )
    out = [pieces[0]]
    for times, after in zip(timestamps, pieces[1:], strict=True):
        out += [f"<{t:.1f} seconds>{start}{pad}{end}" for t in times]
        out.append(after)
    return "".join(out)


def expand_video_parts(
    parts: list[PromptPart], groups: Sequence[int],
) -> list[PromptPart]:
    """``parts`` with each video as text, group, text, ..., group, text.

    The prefill plan's view of `expand_video_pads`: the text around each group
    is the outer sentinel and the timestamps, so a video always has text on
    both sides, and `prefill_plan` merges it with the user's text next to it.
    Video parts are renumbered per group, as `find_media_spans` numbers spans.
    A video with no entry in ``groups`` expands to nothing.
    """
    out: list[PromptPart] = []
    group = 0
    for part in parts:
        if part.modality != VIDEO:
            out.append(part)
            continue
        n = groups[part.index] if part.index < len(groups) else 0
        for g in range(n):
            out.append(PromptPart(modality=TEXT))
            out.append(PromptPart(modality=VIDEO, index=group + g))
        out.append(PromptPart(modality=TEXT))
        group += n
    return out


def group_owners(groups: Sequence[int]) -> list[int]:
    """The video each group (numbered across all videos) belongs to."""
    return [v for v, n in enumerate(groups) for _ in range(n)]

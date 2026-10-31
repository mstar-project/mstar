"""Sampled-frame video decode, split across the clip's keyframe intervals.

Decoding a frame means decoding every frame from the keyframe before it, so a
sample spread over a whole clip costs a decode of most of the clip on one
decoder. Each keyframe interval (GOP) decodes independently, so the sampled
frames are grouped by GOP and the groups decoded on separate decoders in
threads (FFmpeg releases the GIL). Every frame is decoded from the same
keyframe as on one decoder, so the frames are bit-identical.
"""
from __future__ import annotations

import bisect
import functools
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import torch

from mstar.model.media_limits import video_decode_threads
from mstar.utils.cpus import usable_cpus

logger = logging.getLogger(__name__)


@functools.cache
def _max_decoders() -> int:
    """The shared decode pool's size, so also the most groups one request's
    frames split into. Each decoder runs ``video_decode_threads()`` FFmpeg
    threads, so the pool fits the CPUs this process may use; and at most 6,
    as a larger pool measured within noise on V-JEPA2's benchmark."""
    return max(1, min(6, usable_cpus() // video_decode_threads()))


_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _decode_pool() -> ThreadPoolExecutor:
    """One pool for the process's concurrent preprocesses, so they share the decoder cap."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(_max_decoders(), thread_name_prefix="video_decode")
        return _pool


@functools.cache
def _warn_no_key_frames() -> None:
    # called while the first failure is handled, so exc_info logs its cause
    logger.warning(
        "could not read a video's keyframe index from torchcodec, so its sampled frames decode on "
        "one decoder, not split by keyframe interval (warned once; each failure is logged at debug level)",
        exc_info=True,
    )


Transform = Callable[[torch.Tensor], torch.Tensor]


def _identity(frames: torch.Tensor) -> torch.Tensor:
    return frames


def _decode(source, indices: list[int], transform: Transform, decoder=None) -> torch.Tensor:
    if decoder is None:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(source, num_ffmpeg_threads=video_decode_threads())
    return transform(decoder.get_frames_at(indices=indices).data)


def decode_frames_at(
    source: str | torch.Tensor, indices: list[int], transform: Transform | None = None,
) -> torch.Tensor:
    """The frames at ``indices``, in that order: ``[len(indices), C, H, W]``
    uint8, or what ``transform`` makes of them.

    ``source``: the video as torchcodec's ``VideoDecoder`` opens it, a path or
    the file's bytes as a uint8 tensor; every decoder reads the bytes without
    the file. ``transform``: a frame-by-frame map of a ``[n, C, H, W]`` uint8
    batch (e.g. convert and resize), applied to each group on its decoding
    thread, so it overlaps the other groups' decode.
    """
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(source, num_ffmpeg_threads=video_decode_threads())
    keys: list[int] = []
    if _max_decoders() > 1:
        try:
            # private in torchcodec (0.5 to 0.16 at least): any failure means one decoder
            keys = sorted(torch.as_tensor(decoder._get_key_frame_indices()).tolist())
        except Exception:  # noqa: BLE001
            logger.debug("keyframe index unavailable: the sampled frames decode on one decoder", exc_info=True)
            _warn_no_key_frames()
    order = sorted(set(indices))
    by_gop: dict[int, list[int]] = {}
    for i in order:
        by_gop.setdefault(bisect.bisect_right(keys, i), []).append(i)
    gops = list(by_gop.values())
    transform = transform or _identity
    if len(gops) < 2:
        return transform(decoder.get_frames_at(indices=indices).data)
    workers = min(len(gops), _max_decoders())
    # contiguous runs of GOPs, balanced by frame count
    groups: list[list[int]] = [[] for _ in range(workers)]
    target, done = len(order) / workers, 0
    for gop in gops:
        groups[min(int((done + len(gop) / 2) / target), workers - 1)].extend(gop)
        done += len(gop)
    groups = [grp for grp in groups if grp]
    # the first group decodes on the decoder already open, each other group opens its own
    decoders = [decoder] + [None] * (len(groups) - 1)
    decoded = list(_decode_pool().map(lambda grp, dec: _decode(source, grp, transform, dec), groups, decoders))
    where = {i: (frames, r) for grp, frames in zip(groups, decoded, strict=True) for r, i in enumerate(grp)}
    out = torch.empty((len(indices), *decoded[0].shape[1:]), dtype=decoded[0].dtype)
    for k, i in enumerate(indices):
        frames, r = where[i]
        out[k] = frames[r]
    return out

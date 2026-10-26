"""Bounded video readers for qwen-omni-utils' ``fetch_video``."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import torch

from mstar.model.media_limits import VideoBudget

# the readers that decode only the frames fetch_video samples
SAMPLED_READERS = ("decord", "torchcodec")

_header_charged: ContextVar[bool] = ContextVar("qwen3_omni_video_header_charged", default=False)


@contextmanager
def header_charged(charged: bool):
    """Within this, the sampled readers may run only if ``charged``: the
    loader counted the frames they will decode, from the clip's header."""
    token = _header_charged.set(charged)
    try:
        yield
    finally:
        _header_charged.reset(token)


def _after_header_charge(reader):
    @wraps(reader)
    def read(ele: dict):
        if not _header_charged.get():
            # fetch_video catches this and decodes with its torchvision reader,
            # which install_budgeted_readers made read_video_whole_budgeted
            raise ValueError("no header frame count to check the sampled frames against")
        return reader(ele)

    read.mstar_after_header_charge = True
    return read


def install_budgeted_readers(backends: dict) -> None:
    """Bound every reader in qwen-omni-utils' ``VIDEO_READER_BACKENDS``.

    The whole-clip one becomes :func:`read_video_whole_budgeted`. The sampled
    ones decode their frames with no count of their own, so they run only
    inside ``header_charged(True)``; a clip whose header gives no frame count
    (an MKV or WebM without one) goes to the whole-clip reader instead."""
    backends["torchvision"] = read_video_whole_budgeted
    for name in SAMPLED_READERS:
        reader = backends.get(name)
        if reader is not None and not getattr(reader, "mstar_after_header_charge", False):
            backends[name] = _after_header_charge(reader)


def read_video_whole_budgeted(ele: dict) -> tuple[torch.Tensor, dict, float]:
    """Drop-in for qwen-omni-utils' torchvision reader, which ``fetch_video``
    falls back to when torchcodec or decord raises and which decodes the whole
    clip. Same frames (``torchvision.io.read_video`` decodes with PyAV too) and
    the same sampling, but every frame is counted as it is decoded: a header
    can understate the clip, so checking it first would not bound this.

    The original no longer runs at all from torchvision 0.27, which removed
    ``torchvision.io.read_video``."""
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

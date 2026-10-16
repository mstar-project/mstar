"""Video decode with PyAV, for where torchcodec cannot load.

torchcodec's PyPI wheels are CUDA builds and need system FFmpeg libraries, so
on ROCm, and on hosts without FFmpeg, importing or running it fails. PyAV
ships its own FFmpeg.
"""
import torch


def decode_video_pyav(filepath: str) -> tuple[torch.Tensor, dict]:
    """All frames as ``[N, C, H, W]`` uint8, with torchcodec's metadata keys."""
    import av

    frames = []
    with av.open(filepath) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        fps = float(rate) if rate else None
        for frame in container.decode(stream):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
    if not frames:
        raise ValueError(f"no video frames decoded from {filepath}")
    video = torch.stack(frames)
    metadata = {
        "num_frames": len(frames), "average_fps": fps,
        "duration_seconds": (len(frames) / fps) if fps else None,
        "height": int(video.shape[-2]), "width": int(video.shape[-1]),
    }
    return video, metadata

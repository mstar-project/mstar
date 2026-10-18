"""Build the latent stream the decoder benchmark replays.

Encode real video with the Wan VAE encoder so the latents have the
statistics of real content, which matters for the quality (PSNR)
comparisons. Timing does not depend on latent values.

Output: ``{"raw": [1, z, 1 + 3*N, h, w] fp32, "source": str, "repo": str}`` where
``raw`` is the VAE's own latent space (``latent_dist.mode()``, unnormalized).

  python prepare_latents.py --video <clip> --chunks 170 --out latents_wan21.pt
  python prepare_latents.py --random --chunks 170 --out latents_wan21.pt
"""

import argparse
import subprocess
import sys

import numpy as np
import torch
from common import CHUNK_LATENTS, WAN21_REPO, WAN22_REPO, latent_stats, load_vae


def read_frames(path: str, width: int, height: int, max_frames: int) -> np.ndarray:
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-frames:v", str(max_frames),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    buf = subprocess.run(cmd, check=True, capture_output=True).stdout
    frames = np.frombuffer(buf, np.uint8).reshape(-1, height, width, 3)
    if len(frames) == 0:
        sys.exit(f"no frames decoded from {path}")
    return frames


def ping_pong(frames: np.ndarray, n: int) -> np.ndarray:
    """Extend to ``n`` frames by playing forward/backward, so there is no cut."""
    if len(frames) == 1:
        return np.repeat(frames, n, axis=0)
    cycle = np.concatenate([frames, frames[-2:0:-1]])
    reps = -(-n // len(cycle))
    return np.concatenate([cycle] * reps)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--vae", choices=["wan21", "wan22"], default="wan21")
    ap.add_argument("--chunks", type=int, default=170, help="steady chunks after the first latent")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = WAN21_REPO if args.vae == "wan21" else WAN22_REPO
    n_lat = 1 + CHUNK_LATENTS * args.chunks
    n_frames = 4 * (n_lat - 1) + 1

    if args.random:
        from common import hf_snapshot
        from diffusers import AutoencoderKLWan

        cfg = AutoencoderKLWan.load_config(str(hf_snapshot(repo) / "vae"))
        vae = AutoencoderKLWan.from_config(cfg)
        mean, std = latent_stats(vae, "cpu")
        s = vae.config.scale_factor_spatial
        g = torch.Generator().manual_seed(0)
        noise = torch.randn(1, vae.config.z_dim, n_lat, args.height // s, args.width // s, generator=g)
        raw = noise * std + mean
        source = "random N(latents_mean, latents_std)"
    else:
        frames = ping_pong(read_frames(args.video, args.width, args.height, n_frames), n_frames)
        vae = load_vae(repo, torch.float32, args.device, keep_encoder=True)
        video = torch.from_numpy(frames).permute(3, 0, 1, 2)[None].float().div(127.5).sub(1.0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            # Encode in independent pieces of 1 + 4k frames to bound memory; the
            # seams only perturb latent statistics slightly, which is irrelevant here.
            piece = 1 + 4 * 30
            outs = []
            start = 0
            while start < n_frames:
                end = min(start + piece, n_frames)
                clip = video[:, :, start:end].to(args.device)
                outs.append(vae.encode(clip).latent_dist.mode().float().cpu())
                start = end
            raw = torch.cat(outs, dim=2)[:, :, :n_lat]
        source = f"{args.video} (ping-pong to {n_frames} frames, {args.height}x{args.width})"

    assert raw.shape[2] == n_lat, (raw.shape, n_lat)
    torch.save({"raw": raw.contiguous(), "source": source, "repo": repo}, args.out)
    print(f"wrote {args.out}: raw latents {tuple(raw.shape)} from {source}")


if __name__ == "__main__":
    main()

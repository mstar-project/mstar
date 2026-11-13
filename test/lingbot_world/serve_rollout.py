#!/usr/bin/env python3
"""Serve LingBot-World-V2 1.3B through mstar and stream one rollout, end to end.

Launches ``mstar/api_server/entrypoint.py`` on ``configs/lingbot_world_1_3b.yaml``
(API server, conductor and GPU worker), sends the example's image, prompt and
camera trajectory through ``MStarClient``, and collects the ``video_frame``
chunks: one per rollout chunk, 13 frames then 16 each. The request runs twice
under different ids with the same seed; identical bytes the second time show
the first rollout left no state behind. With ``--oracle-dir`` the frames are
also compared against the reference video (quantized the way the dit node
quantizes); the server decodes the JPEG with torchvision rather than PIL, so
the conditioning pixels, and with them the rollout, may differ slightly.
``--reference-numerics`` serves a copy of the config with the reference's
numerics (``REFERENCE_MODEL_KWARGS``), which need flash-attn. The default
numerics don't match the oracle bit for bit.

    python test/lingbot_world/serve_rollout.py --example <repo>/examples/03 \
        --oracle-dir /tmp/lingbot/oracle --out /tmp/lingbot/serve --reference-numerics
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mstar.client import MStarClient, VideoFrameChunk  # noqa: E402

PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant "
    "snow-capped mountains under a bright blue sky with drifting white clouds \u2014 gentle ripples "
    "reflect the tree and sky, creating a tranquil, meditative atmosphere."
)


@contextlib.contextmanager
def serve(out: Path, port: int, reference_numerics: bool = False) -> Iterator[MStarClient]:
    """A client of the API server on ``configs/lingbot_world_1_3b.yaml``, or on a copy of it with
    ``REFERENCE_MODEL_KWARGS`` under ``reference_numerics``, once the server answers. The server logs to
    ``out / "server.log"`` and stops on exit."""
    workdir = Path(tempfile.mkdtemp(prefix="lingbot-serve-"))
    config = REPO / "configs/lingbot_world_1_3b.yaml"
    if reference_numerics:
        from mstar.model.lingbot_world.lingbot_world_model import REFERENCE_MODEL_KWARGS

        served = yaml.safe_load(config.read_text())
        served["model_kwargs"] = {**served.get("model_kwargs", {}), **REFERENCE_MODEL_KWARGS}
        config = workdir / "lingbot.yaml"
        config.write_text(yaml.safe_dump(served))
    log = open(out / "server.log", "w")
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "mstar/api_server/entrypoint.py"), "--config", str(config),
         "--port", str(port), "--host", "127.0.0.1", "--socket-path-prefix", str(workdir / "sock"),
         "--upload-dir", str(workdir / "uploads"), "--tensor-comm-protocol", "SHM", "--log-level", "INFO",
         "--timeout", "600"],
        stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
    )
    client = MStarClient(f"http://127.0.0.1:{port}")
    try:
        deadline = time.time() + 900
        while not client.health():
            if proc.poll() is not None or time.time() > deadline:
                raise SystemExit(f"server did not come up (exit {proc.poll()}); see {out / 'server.log'}")
            time.sleep(2)
        yield client
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def rollout(
    client: MStarClient, example: Path, request_id: str, num_frames: int, seed: int = 42,
) -> tuple[list[bytes], list[dict], list[float]]:
    chunks, metas, arrivals = [], [], []
    t0 = time.perf_counter()
    for event in client.stream(
        text=PROMPT, images=str(example / "image.jpg"), input_modalities=("text", "image"),
        output_modalities=("video_frame",), request_id=request_id, seed=seed, num_frames=num_frames,
        camera_poses=np.load(example / "poses.npy").tolist(),
        intrinsics=np.load(example / "intrinsics.npy")[0].tolist(),
    ):
        if not isinstance(event, VideoFrameChunk):
            raise RuntimeError(f"unexpected stream event {type(event).__name__}")
        chunks.append(event.data)
        metas.append(event.metadata)
        arrivals.append(time.perf_counter() - t0)
        print(f"  {request_id} chunk {len(chunks) - 1}  bytes={len(event.data)}  "
              f"t={arrivals[-1]:.2f}s  {event.metadata}")
    return chunks, metas, arrivals


def chunk_frames(chunks: list[bytes], metas: list[dict]) -> list[np.ndarray]:
    """Each chunk's frames, ``[F, H, W, 3]`` uint8."""
    h, w = metas[0].get("height"), metas[0].get("width")
    return [np.frombuffer(c, dtype=np.uint8).reshape(-1, h, w, 3) for c in chunks]


def oracle_frames(oracle_dir: Path) -> np.ndarray:
    """The reference video, ``[F, H, W, 3]`` uint8, quantized the way the dit node quantizes."""
    import torch

    ref = torch.load(oracle_dir / "video.pt")
    return (ref / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 3, 0).numpy()


def vs_oracle(video: np.ndarray, ref: np.ndarray) -> dict:
    mse = None
    if video.shape == ref.shape:
        mse = float(np.mean((video.astype(np.float64) - ref.astype(np.float64)) ** 2))
    return {
        "shape_equal": video.shape == ref.shape,
        "bit_exact": bool(video.shape == ref.shape and np.array_equal(video, ref)),
        "psnr_db": None if mse is None else (math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--oracle-dir", type=Path)
    ap.add_argument("--num-frames", type=int, default=93)
    ap.add_argument("--port", type=int, default=18431)
    ap.add_argument("--reference-numerics", action="store_true",
                    help="serve REFERENCE_MODEL_KWARGS: FA2 attention and an eager DiT; needs flash-attn")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    with serve(args.out, args.port, args.reference_numerics) as client:
        first, metas, arrivals = rollout(client, args.example, "lingbot-a", args.num_frames)
        second, _, arrivals_b = rollout(client, args.example, "lingbot-b", args.num_frames)
        report["chunks"] = len(first)
        report["chunk_metadata"] = metas
        report["arrival_s"] = {"a": arrivals, "b": arrivals_b}
        report["repeat_identical"] = first == second
        video = np.concatenate(chunk_frames(first, metas))
        report["frames"] = int(video.shape[0])
        np.save(args.out / "served_frames.npy", video)
        if args.oracle_dir is not None:
            report["vs_oracle"] = vs_oracle(video, oracle_frames(args.oracle_dir))
    (args.out / "serve.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "chunk_metadata"}, indent=2, default=str))
    if not report.get("repeat_identical") or report.get("vs_oracle", {}).get("bit_exact") is False:
        print("served rollout FAILED: the two runs differ or the frames differ from the oracle")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

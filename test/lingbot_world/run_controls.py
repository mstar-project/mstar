#!/usr/bin/env python3
"""Negative controls and timing for the LingBot-World parity run.

The parity run reports bit-exact; these show the comparison can fail and that
the paths it covers matter. The rollouts are served with the reference's
numerics (``serve_rollout.py --reference-numerics``), so each compares the
frames a chunk emits:

  repeat          a seed-42 rollout after a warm-up one: every chunk's frames
                  equal the oracle's
  seed            seed 43 instead of 42: every chunk's frames must differ
  decode variants the oracle latents through the streamed decoder in process,
                  with channels-last fp32 and with bf16 (#425's perf switches)
  timing          each chunk's arrival at the client

Also writes a contact sheet of reference vs mstar frames and an mp4 of the
served rollout.

    python test/lingbot_world/run_controls.py --oracle-dir /tmp/lingbot/oracle \
        --example <repo>/examples/03 --out /tmp/lingbot/controls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from run_parity import DEVICE, frame_stats  # noqa: E402
from serve_rollout import chunk_frames, oracle_frames, rollout, serve, vs_oracle  # noqa: E402

from mstar.model.components import wan_vae  # noqa: E402
from mstar.model.lingbot_world.lingbot_world_model import REFERENCE_MODEL_KWARGS, LingBotWorldModel  # noqa: E402


def per_chunk(chunks: list[np.ndarray], ref: np.ndarray) -> list[dict]:
    """Each chunk's frames against the same frames of ``ref``."""
    ends = np.cumsum([len(c) for c in chunks])
    return [vs_oracle(c, ref[end - len(c):end]) for c, end in zip(chunks, ends, strict=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", required=True)
    ap.add_argument("--example", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18431)
    args = ap.parse_args()
    oracle, example, out = Path(args.oracle_dir), Path(args.example), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((oracle / "metadata.json").read_text())
    ref_latents = torch.load(oracle / "latents.pt")
    ref_video = torch.load(oracle / "video.pt")
    ref_frames = oracle_frames(oracle)
    num_frames = meta["frame_num_requested"]
    report = {}

    with serve(out, args.port, reference_numerics=True) as client:
        rollout(client, example, "warmup", num_frames)
        chunks, metas, arrivals = rollout(client, example, "seed42", num_frames)
        chunks43, metas43, _ = rollout(client, example, "seed43", num_frames, seed=43)
    served = chunk_frames(chunks, metas)
    report["repeat_bit_exact"] = all(c["bit_exact"] for c in per_chunk(served, ref_frames))
    report["chunk_arrival_s"] = arrivals
    report["seed43_vs_oracle"] = per_chunk(chunk_frames(chunks43, metas43), ref_frames)
    mstar_video = np.concatenate(served)

    model = LingBotWorldModel(**REFERENCE_MODEL_KWARGS)
    cfg = model.config
    cs = cfg.chunk_size
    lat_f = ref_latents.shape[1]
    sub = model.get_submodule("dit", device="cuda").to(DEVICE)

    def decode_all(vae_dtype_fp32: bool, channels_last: bool):
        sub._decode_dtype = torch.float32 if vae_dtype_fp32 else torch.bfloat16
        vae = model._load_vae().to(DEVICE)
        if not vae_dtype_fp32:
            vae = vae.to(torch.bfloat16)
        sub.vae, sub._decoder = vae, wan_vae.FastDecoder(vae, channels_last=channels_last, compile=False)
        feat_cache = wan_vae.new_feat_cache(vae)
        times, outs = [], []
        with torch.inference_mode():
            for k in range(lat_f // cs):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                chunk = ref_latents[:, k * cs:(k + 1) * cs].to(DEVICE)
                outs.append(sub._decode(chunk, feat_cache, clip_start=k == 0).float().cpu())
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
        return frame_stats(torch.cat(outs, dim=2)[0], ref_video) | {"chunk_ms": times}

    report["decode_fp32_contiguous"] = decode_all(True, False)
    report["decode_fp32_channels_last"] = decode_all(True, True)
    report["decode_bf16_channels_last"] = decode_all(False, True)
    sub._decode_dtype = torch.float32

    (out / "controls.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))

    # contact sheet: reference row over mstar row at a few frames
    from PIL import Image

    picks = [0, 30, 60, len(ref_frames) - 1]
    rows = [np.concatenate([v[f] for f in picks], axis=1) for v in (ref_frames, mstar_video)]
    Image.fromarray(np.concatenate(rows, axis=0)).resize((4 * 416, 2 * 232)).save(out / "frames_ref_vs_mstar.png")
    import av

    container = av.open(str(out / "mstar_rollout.mp4"), mode="w")
    stream = container.add_stream("libx264", rate=cfg.video_fps)
    stream.height, stream.width = mstar_video.shape[1], mstar_video.shape[2]
    stream.pix_fmt = "yuv420p"
    for frame in mstar_video:
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


if __name__ == "__main__":
    main()

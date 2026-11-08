#!/usr/bin/env python3
"""Negative controls and timing for the LingBot-World parity run.

The parity harness reports bit-exact; these show the comparison can fail and
that the paths it covers matter:

  seed            seed 43 instead of 42: every chunk must diverge
  no_window       local_attn_size=-1 (keep all K/V): chunks before the first
                  eviction stay bit-exact, chunks after it must diverge
  decode variants the oracle latents through the streamed decoder with
                  channels-last fp32 and with bf16 (#425's perf switches)
  timing          per chunk: the five DiT forwards vs the fused decode (eager)

Uses the oracle's own context and condition (teacher-forced inputs), so only
the DiT/decoder side varies. Also writes a contact sheet of reference vs mstar
frames and an mp4 of the mstar rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from run_parity import DEVICE, drive, frame_stats, fwd_info, same, stats  # noqa: E402

from mstar.model.components import wan_vae  # noqa: E402
from mstar.model.lingbot_world import submodules  # noqa: E402
from mstar.model.lingbot_world.lingbot_world_model import REFERENCE_MODEL_KWARGS, LingBotWorldModel  # noqa: E402
from mstar.model.lingbot_world.submodules import first_frame_mask  # noqa: E402


def timed_rollout(sub, text, image, cond, poses, intr, md, rid, seed=42):
    events = []
    def mark(tag):
        events.append([tag, torch.cuda.Event(enable_timing=True)])
        events[-1][1].record()

    pre = sub.dit.register_forward_pre_hook(lambda m, a: mark("s"))
    post = sub.dit.register_forward_hook(lambda m, a, o: mark("e"))
    info = fwd_info("rollout", md, rid)
    info.random_seed = seed
    inputs = {"text_embeds_pos": [text], "image_inputs": [image], "camera_poses": [poses],
              "intrinsics": [intr], "clock": []}
    latents, pixels, chunk_ms, dit_ms = [], [], [], []
    encode = submodules.encode_condition_chunk
    submodules.encode_condition_chunk = lambda vae, image, cache, first, n, cfg: cond[:, first:first + n]
    for _ in range(md["num_chunks"]):
        events.clear()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        drive(sub, "rollout", info, inputs)
        end.record()
        torch.cuda.synchronize()
        chunk_ms.append(start.elapsed_time(end))
        dit_ms.append(sum(s[1].elapsed_time(e[1]) for s, e in zip(events[0::2], events[1::2], strict=True)))
        st = sub.request_state(rid)
        latents.append(st["chunk_latents"].cpu())
        pixels.append(st["chunk_pixels"].cpu())
    submodules.encode_condition_chunk = encode
    pre.remove()
    post.remove()
    sub.cleanup_request(rid)
    return latents, pixels, chunk_ms, dit_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", required=True)
    ap.add_argument("--example", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    oracle, out = Path(args.oracle_dir), Path(args.out)
    meta = json.loads((oracle / "metadata.json").read_text())
    ref_in = torch.load(oracle / "inputs.pt")
    ref_latents = torch.load(oracle / "latents.pt")
    ref_video = torch.load(oracle / "video.pt")

    model = LingBotWorldModel(**REFERENCE_MODEL_KWARGS)
    cfg = model.config
    cs = cfg.chunk_size
    tensors = model.process_prompt(
        ref_in["prompt"], ["text", "image"], ["video_frame"], tensors={"image_inputs": [ref_in["image_rgb"]]},
        camera_poses=np.load(Path(args.example) / "poses.npy"),
        intrinsics=np.load(Path(args.example) / "intrinsics.npy"), num_frames=meta["frame_num_requested"],
    )
    poses, intr = tensors["camera_poses"][0], tensors["intrinsics"][0]
    lat_f = poses.shape[0]
    _, _, lat_h, lat_w = ref_in["vae_latent"].shape
    md = {"is_prefill": False, "guidance_scale": 1.0, "num_latent_frames": lat_f, "num_chunks": lat_f // cs}
    n_tok = ref_in["context"].shape[0]
    text = torch.zeros(1, cfg.text_len, cfg.text_dim, dtype=ref_in["context"].dtype)
    text[0, :n_tok] = ref_in["context"]
    text = text.to(DEVICE)
    cond = torch.cat([first_frame_mask(lat_f, lat_h, lat_w, "cpu"), ref_in["vae_latent"]]).to(DEVICE)
    image = tensors["image_inputs"][0].to(DEVICE)

    sub = model.get_submodule("dit", device="cuda").to(DEVICE)
    sub.keep_chunk_outputs = True
    report = {}

    # warm-up pass (kernel selection, allocator), then the timed reference-config pass
    timed_rollout(sub, text, image, cond, poses, intr, md, rid=10)
    lat, pix, chunk_ms, dit_ms = timed_rollout(sub, text, image, cond, poses, intr, md, rid=11)
    report["repeat_bit_exact"] = all(same(x, ref_latents[:, k * cs:(k + 1) * cs]) for k, x in enumerate(lat))
    report["timing_ms"] = [{"chunk": k, "total": c, "dit_5_forwards": d, "decode_and_rest": c - d}
                           for k, (c, d) in enumerate(zip(chunk_ms, dit_ms, strict=True))]
    mstar_video = torch.cat(pix, dim=2)[0]

    lat43, _, _, _ = timed_rollout(sub, text, image, cond, poses, intr, md, rid=12, seed=43)
    report["seed43_vs_oracle"] = [stats(x, ref_latents[:, k * cs:(k + 1) * cs]) for k, x in enumerate(lat43)]

    cfg.local_attn_size = -1
    lat_full, _, _, _ = timed_rollout(sub, text, image, cond, poses, intr, md, rid=13)
    cfg.local_attn_size = 18
    report["no_window_vs_oracle"] = [stats(x, ref_latents[:, k * cs:(k + 1) * cs]) for k, x in enumerate(lat_full)]

    def decode_all(vae_dtype_fp32: bool, channels_last: bool):
        sub._decode_dtype = torch.float32 if vae_dtype_fp32 else torch.bfloat16
        vae = model._load_vae().to(DEVICE)
        if not vae_dtype_fp32:
            vae = vae.to(torch.bfloat16)
        sub.vae, sub._decoder = vae, wan_vae.FastDecoder(vae, channels_last=channels_last, compile=False)
        st = sub.request_state(20)
        st.add("feat_cache", wan_vae.new_feat_cache(vae))
        times, outs = [], []
        with torch.inference_mode():
            for k in range(lat_f // cs):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                chunk = ref_latents[:, k * cs:(k + 1) * cs].to(DEVICE)
                outs.append(sub._decode(chunk, st, clip_start=k == 0).float().cpu())
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
        sub.cleanup_request(20)
        return frame_stats(torch.cat(outs, dim=2)[0], ref_video) | {"chunk_ms": times}

    report["decode_fp32_contiguous"] = decode_all(True, False)
    report["decode_fp32_channels_last"] = decode_all(True, True)
    report["decode_bf16_channels_last"] = decode_all(False, True)
    sub._decode_dtype = torch.float32

    (out / "controls.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))

    # contact sheet: reference row over mstar row at a few frames
    from PIL import Image

    def to_img(v, f):
        return ((v[:, f].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().numpy()

    picks = [0, 30, 60, ref_video.shape[1] - 1]
    rows = [np.concatenate([to_img(v, f) for f in picks], axis=1) for v in (ref_video, mstar_video)]
    Image.fromarray(np.concatenate(rows, axis=0)).resize((4 * 416, 2 * 232)).save(out / "frames_ref_vs_mstar.png")
    import av

    container = av.open(str(out / "mstar_rollout.mp4"), mode="w")
    stream = container.add_stream("libx264", rate=cfg.video_fps)
    stream.height, stream.width = mstar_video.shape[2], mstar_video.shape[3]
    stream.pix_fmt = "yuv420p"
    for f in range(mstar_video.shape[1]):
        for packet in stream.encode(av.VideoFrame.from_ndarray(to_img(mstar_video, f), format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


if __name__ == "__main__":
    main()

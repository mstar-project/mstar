#!/usr/bin/env python3
"""Compare mstar's LingBot-World-V2 1.3B against the reference oracle.

Drives the mstar submodules the way the engine does (prepare_inputs ->
preprocess -> forward, one rollout iteration per chunk) on the oracle's
inputs, and compares:

  inputs        tokenizer ids, the resized conditioning image, each chunk's
                Plucker rays
  text_encoder  mstar's UMT5 (wan22's submodule, Wan2.2 diffusers weights)
                vs the reference T5 context
  condition     the dit's chunk-by-chunk diffusers Wan2.1 VAE encode vs the
                reference's whole-clip ``y``
  dit (teacher-forced)  the mstar DiT fed the reference's own context and y:
                every DiT call's input and output, every chunk's x0, and the
                streamed warm-cache decode vs the reference's whole-clip decode
  decode        mstar's per-chunk warm-cache decode of the oracle latents
  end-to-end    mstar's own text embeds and condition through the whole rollout

    python test/lingbot_world/run_parity.py --oracle-dir /tmp/lingbot/oracle \
        --example <repo>/examples/03 --out /tmp/lingbot/parity
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.model.lingbot_world import submodules  # noqa: E402
from mstar.model.lingbot_world.components.camera import plucker_embedding  # noqa: E402
from mstar.model.lingbot_world.lingbot_world_model import REFERENCE_MODEL_KWARGS, LingBotWorldModel  # noqa: E402
from mstar.model.lingbot_world.submodules import ROLLOUT_LOOP_NAME, first_frame_mask  # noqa: E402
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402

DEVICE = torch.device("cuda")


def same(ours: torch.Tensor, ref: torch.Tensor) -> bool:
    """Bit-exact: same shape, same dtype, same values. Nothing is cast first."""
    return ours.shape == ref.shape and ours.dtype == ref.dtype and torch.equal(ours.cpu(), ref.cpu())


def stats(ours: torch.Tensor, ref: torch.Tensor, peak: float | None = None) -> dict:
    ours, ref = ours.detach(), ref.detach()
    a, b = ours.double().cpu(), ref.double().cpu()
    if a.shape != b.shape:
        return {"bit_exact": False, "shape_mismatch": [list(a.shape), list(b.shape)]}
    diff = (a - b).abs()
    mse = float((diff**2).mean())
    peak = float(b.max() - b.min()) if peak is None else peak
    return {
        "bit_exact": same(ours, ref),
        "dtypes": [str(ours.dtype), str(ref.dtype)],
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "psnr_db": math.inf if mse == 0 else 10 * math.log10(peak**2 / mse),
    }


def frame_stats(ours: torch.Tensor, ref: torch.Tensor) -> dict:
    """Frames in [-1, 1]: PSNR on the [0, 1] scale, plus 8-bit agreement."""
    s = stats((ours.float() + 1) / 2, (ref.float() + 1) / 2, peak=1.0)
    s["bit_exact"] = same(ours, ref)
    s["dtypes"] = [str(ours.dtype), str(ref.dtype)]
    q = lambda x: (x.float().cpu() / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)  # noqa: E731
    s["uint8_equal_fraction"] = float((q(ours) == q(ref)).float().mean())
    return s


def mismatches(report, path: str = "") -> list[str]:
    """Every point of ``report`` that differs from the reference: every flag
    must be true, except that the rollout stops on its last chunk only."""
    if path == ".check_stop_flags":
        return [] if report == [False] * (len(report) - 1) + [True] else [path]
    if isinstance(report, bool):
        return [] if report else [path]
    if isinstance(report, dict):
        return [p for key, value in report.items() for p in mismatches(value, f"{path}.{key}")]
    if isinstance(report, list):
        return [p for i, value in enumerate(report) for p in mismatches(value, f"{path}[{i}]")]
    return []


def fwd_info(walk: str, md: dict, rid: int) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=f"parity-{rid}", graph_walk=walk, fwd_index=0, random_seed=42, max_tokens=0,
        step_metadata=md, rid_handle=rid,
    )


def drive(sub, walk: str, info: CurrentForwardPassInfo, inputs: dict) -> dict:
    node_inputs = sub.prepare_inputs(graph_walk=walk, fwd_info=info, inputs=inputs)
    eng = ModelInputsFromEngine(request_ids=[info.rid_handle], per_request_info={info.rid_handle: info})
    pre = sub.preprocess(graph_walk=walk, engine_inputs=eng, inputs=[node_inputs])
    return sub.forward(graph_walk=walk, engine_inputs=eng, **pre)


def rollout(sub, text_embeds, image, camera_poses, intrinsics, md: dict, rid: int, condition=None):
    """One full rollout through the dit node; per chunk: x0 latents, float
    frames, emitted uint8 frames, wall time. Also every DiT call (x, t, flow).
    ``condition`` (``[20, lat_f, h, w]``) stands in for the dit's own encode."""
    calls: list[dict] = []
    handle = sub.dit.register_forward_hook(
        lambda m, a, o: calls.append({"x": a[0].detach().cpu().clone(), "t": a[1].detach().cpu().clone(),
                                      "flow": o.detach().cpu().clone()})
    )
    info = fwd_info("rollout", md, rid)
    inputs = {"text_embeds_pos": [text_embeds], "image_inputs": [image],
              "camera_poses": [camera_poses], "intrinsics": [intrinsics], "clock": []}
    latents, pixels, conditions, frames, times = [], [], [], [], []
    stops = []
    encode = submodules.encode_condition_chunk
    if condition is not None:
        submodules.encode_condition_chunk = (
            lambda vae, image, cache, first, n, cfg: condition[:, first:first + n].to(image.device))
    try:
        for k in range(md["num_chunks"]):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = drive(sub, "rollout", info, inputs)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            st = sub.request_state(rid)
            latents.append(st["chunk_latents"].cpu())
            pixels.append(st["chunk_pixels"].cpu())
            conditions.append(st["chunk_condition"].cpu())
            frames.append(out["video_output"][0].cpu())
            info.dynamic_loop_iter_counts[ROLLOUT_LOOP_NAME] = k
            stops.append(bool(sub.check_stop(info.request_id, info, out)))
            info.dynamic_loop_iter_counts[ROLLOUT_LOOP_NAME] = k + 1
            inputs["clock"] = out["clock"]
    finally:
        submodules.encode_condition_chunk = encode
    overshoot = sub.prepare_inputs(graph_walk="rollout", fwd_info=info, inputs=inputs)
    handle.remove()
    sub.cleanup_request(rid)
    return {"latents": latents, "pixels": pixels, "conditions": conditions, "frames": frames, "times": times,
            "calls": calls, "stop_flags": stops, "overshoot_vetoed": overshoot is None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", required=True)
    ap.add_argument("--example", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    oracle = Path(args.oracle_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((oracle / "metadata.json").read_text())
    ref_in = torch.load(oracle / "inputs.pt")
    ref_calls = torch.load(oracle / "calls.pt")
    ref_latents = torch.load(oracle / "latents.pt")
    ref_video = torch.load(oracle / "video.pt")
    report: dict = {"oracle": meta, "float32_matmul_precision": torch.get_float32_matmul_precision()}

    model = LingBotWorldModel(**REFERENCE_MODEL_KWARGS)
    cfg = model.config
    cs = cfg.chunk_size
    poses = np.load(Path(args.example) / "poses.npy")
    intr = np.load(Path(args.example) / "intrinsics.npy")
    tensors = model.process_prompt(
        ref_in["prompt"], ["text", "image"], ["video_frame"],
        tensors={"image_inputs": [ref_in["image_rgb"]]},
        camera_poses=poses, intrinsics=intr, num_frames=meta["frame_num_requested"],
    )
    lat_f = int(tensors["camera_poses"][0].shape[0])
    md = {"is_prefill": False, "guidance_scale": 1.0, "num_latent_frames": lat_f, "num_chunks": lat_f // cs}
    image = tensors["image_inputs"][0]
    lat_h, lat_w = image.shape[1] // 8, image.shape[2] // 8

    # ---- inputs -----------------------------------------------------------
    ours_ids = tensors["text_inputs"][0]
    report["tokenizer_ids_equal"] = same(ours_ids, ref_in["token_ids"])
    report["image"] = stats(image, ref_in["vae_input"][:, 0])
    report["vae_input_tail_is_zero"] = bool((ref_in["vae_input"][:, 1:] == 0).all())
    report["geometry"] = {"latent_frames": lat_f, "lat_h": lat_h, "lat_w": lat_w,
                          "height": lat_h * 8, "width": lat_w * 8, "chunks": lat_f // cs}
    # per chunk, as the dit builds them; the reference builds the whole trajectory's at once
    report["plucker_chunks"] = [
        stats(plucker_embedding(tensors["camera_poses"][0][k * cs:(k + 1) * cs].float().to(DEVICE),
                                tensors["intrinsics"][0].float(), lat_h * 8, lat_w * 8, lat_h, lat_w
                                ).to(torch.bfloat16), ref)
        for k, ref in enumerate(ref_in["plucker_chunks"])
    ]

    # ---- text encoder -----------------------------------------------------
    text_sub = model.get_submodule("text_encoder", device="cuda").to(DEVICE)
    text_out = drive(text_sub, "encode_text", fwd_info("encode_text", md, 1), {"text_inputs": tensors["text_inputs"]})
    ours_text = text_out["text_embeds_pos"][0]
    n_tok = ref_in["context"].shape[0]
    report["text_embeds"] = stats(ours_text[0, :n_tok], ref_in["context"])
    report["text_embeds_tail_zero"] = bool((ours_text[0, n_tok:] == 0).all())
    del text_sub, model._submodule_cache["text_encoder"]
    torch.cuda.empty_cache()

    # ---- condition, chunk by chunk ------------------------------------------
    dit_sub = model.get_submodule("dit", device="cuda").to(DEVICE)
    dit_sub.keep_chunk_outputs = True
    cache = submodules.EncoderStream(dit_sub.encoder_vae)
    cond = torch.cat([
        submodules.encode_condition_chunk(dit_sub.encoder_vae, image.to(DEVICE), cache, k * cs, cs, cfg)
        for k in range(lat_f // cs)
    ], dim=1)
    ref_mask = first_frame_mask(lat_f, lat_h, lat_w, "cpu")
    report["vae_encode"] = stats(cond[4:], ref_in["vae_latent"])
    report["mask_equal"] = same(cond[:4].cpu(), ref_mask)

    # ---- dit, teacher-forced ----------------------------------------------
    ref_text = torch.zeros(1, cfg.text_len, cfg.text_dim, dtype=ref_in["context"].dtype)
    ref_text[0, :n_tok] = ref_in["context"]
    ref_cond = torch.cat([ref_mask, ref_in["vae_latent"]])
    tf = rollout(dit_sub, ref_text.to(DEVICE), image.to(DEVICE), tensors["camera_poses"][0],
                 tensors["intrinsics"][0], md, rid=3, condition=ref_cond.to(DEVICE))
    report["dit_calls"] = {"ours": len(tf["calls"]), "ref": len(ref_calls)}
    report["dit_calls_bit_exact"] = {
        "x_in": all(same(a["x"], b["x"]) for a, b in zip(tf["calls"], ref_calls, strict=True)),
        "t": all(same(a["t"], b["t"]) for a, b in zip(tf["calls"], ref_calls, strict=True)),
        "flow": all(same(a["flow"], b["flow"]) for a, b in zip(tf["calls"], ref_calls, strict=True)),
    }
    pairs = enumerate(zip(tf["calls"], ref_calls, strict=True))
    first_bad = next((i for i, (a, b) in pairs if not same(a["flow"], b["flow"])), None)
    report["dit_first_divergent_call"] = first_bad
    if first_bad is not None:
        report["dit_first_divergent_flow"] = stats(tf["calls"][first_bad]["flow"], ref_calls[first_bad]["flow"])
    report["teacher_forced_chunks"] = [
        {"chunk": k, "latent": stats(x, ref_latents[:, k * cs:(k + 1) * cs])} for k, x in enumerate(tf["latents"])
    ]
    tf_video = torch.cat(tf["pixels"], dim=2)[0]
    report["teacher_forced_video"] = frame_stats(tf_video, ref_video)
    report["teacher_forced_frames_per_chunk"] = [p.shape[2] for p in tf["pixels"]]
    report["check_stop_flags"] = tf["stop_flags"]
    report["overshoot_vetoed"] = tf["overshoot_vetoed"]

    # ---- decode only ------------------------------------------------------
    st = dit_sub.request_state(99)
    from mstar.model.components import wan_vae
    st.add("feat_cache", wan_vae.new_feat_cache(dit_sub.vae))
    with torch.inference_mode():
        dec = [dit_sub._decode(ref_latents[:, k * cs:(k + 1) * cs].to(DEVICE), st, clip_start=k == 0).cpu()
               for k in range(lat_f // cs)]
    dit_sub.cleanup_request(99)
    report["decode_streamed_vs_reference"] = frame_stats(torch.cat(dec, dim=2)[0], ref_video)

    # ---- end to end -------------------------------------------------------
    e2e = rollout(dit_sub, ours_text.to(DEVICE), image.to(DEVICE), tensors["camera_poses"][0],
                  tensors["intrinsics"][0], md, rid=4)
    report["e2e_condition"] = stats(torch.cat(e2e["conditions"], dim=1), ref_cond)
    report["e2e_chunks"] = []
    for k, (x, p) in enumerate(zip(e2e["latents"], e2e["pixels"], strict=True)):
        lo = 0 if k == 0 else 1 + (k * cs - 1) * 4
        hi = 1 + ((k + 1) * cs - 1) * 4
        report["e2e_chunks"].append({
            "chunk": k, "latent": stats(x, ref_latents[:, k * cs:(k + 1) * cs]),
            "frames": frame_stats(p[0], ref_video[:, lo:hi]),
        })
    report["e2e_video"] = frame_stats(torch.cat(e2e["pixels"], dim=2)[0], ref_video)
    report["chunk_seconds_e2e"] = e2e["times"]
    report["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30

    torch.save({"e2e_frames": e2e["frames"], "tf_frames": tf["frames"]}, out_dir / "mstar_frames.pt")
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    if bad := mismatches(report):
        raise SystemExit(f"parity FAILED at {len(bad)} point(s): {bad[:8]}")


if __name__ == "__main__":
    main()

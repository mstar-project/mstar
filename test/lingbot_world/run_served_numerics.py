"""LingBot-World's served numerics, driven by hand: the dit node with its engine
resources built and stepped the way the engine steps them (declare_step ->
admit -> plan -> forward -> commit), no server.

Records each chunk's x0 latents and emitted uint8 frames for a solo rollout and
for two rollouts interleaved chunk by chunk, so a change to where the rolling
K/V lives can be held to ``torch.equal`` against main's served path. The
interleaved run's first session must also equal its solo run, which is what
shows one session's state never reaches another.

    python test/lingbot_world/run_served_numerics.py --out /tmp/lingbot/served_main.pt
    python test/lingbot_world/run_served_numerics.py --out /tmp/lingbot/served_new.pt \
        --compare-to /tmp/lingbot/served_main.pt

The trajectory is synthetic (a slow forward dolly with a little yaw); the
conditioning image defaults to one of Cosmos3-Edge's assets.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from huggingface_hub import hf_hub_download  # noqa: E402
from run_parity import DEVICE, bind_resources, drive  # noqa: E402

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.engine.resources import StepRunner  # noqa: E402
from mstar.model.lingbot_world.lingbot_world_model import LingBotWorldModel  # noqa: E402
from mstar.model.lingbot_world.submodules import ROLLOUT_LOOP_NAME, ROLLOUT_WALK  # noqa: E402

PROMPT = "A quiet suburban street on a clear afternoon, the camera driving slowly forward."


def trajectory(num_frames: int, yaw_per_frame: float, step: float) -> np.ndarray:
    """OpenCV camera-to-world poses: forward along +z with a constant yaw rate."""
    poses = np.tile(np.eye(4), (num_frames, 1, 1))
    for t in range(num_frames):
        a = yaw_per_frame * t
        poses[t, :3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        poses[t, :3, 3] = [step * t * math.sin(a), 0.0, step * t * math.cos(a)]
    return poses


class Session:
    def __init__(self, rid: int, seed: int, tensors: dict, text_embeds: torch.Tensor, cs: int):
        lat_f = int(tensors["camera_poses"][0].shape[0])
        self.rid = rid
        self.num_chunks = lat_f // cs
        md = {"is_prefill": False, "guidance_scale": 1.0, "num_latent_frames": lat_f, "num_chunks": self.num_chunks}
        self.info = CurrentForwardPassInfo(
            request_id=f"served-{rid}", graph_walk=ROLLOUT_WALK, fwd_index=0, random_seed=seed, max_tokens=0,
            step_metadata=md, rid_handle=rid,
        )
        self.inputs = {"text_embeds_pos": [text_embeds], "image_inputs": [tensors["image_inputs"][0].to(DEVICE)],
                       "camera_poses": tensors["camera_poses"], "intrinsics": tensors["intrinsics"], "clock": []}
        self.latents: list[torch.Tensor] = []
        self.frames: list[torch.Tensor] = []
        self.times: list[float] = []

    @property
    def done(self) -> bool:
        return len(self.latents) >= self.num_chunks


def step_chunk(sub, runner: StepRunner, s: Session) -> None:
    """One rollout iteration of ``s``, as the engine runs a batch of one."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = drive(sub, ROLLOUT_WALK, s.info, s.inputs, runner)
    torch.cuda.synchronize()
    s.times.append(time.perf_counter() - t0)
    st = sub.request_state(s.rid)
    s.latents.append(st["chunk_latents"].cpu())
    s.frames.append(out["video_output"][0].cpu())
    s.info.dynamic_loop_iter_counts[ROLLOUT_LOOP_NAME] = len(s.latents)
    s.inputs["clock"] = out["clock"]


def finish(sub, runner: StepRunner, s: Session) -> None:
    runner.remove_request(s.rid)
    sub.cleanup_request(s.rid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-to")
    ap.add_argument("--chunks", type=int, default=8, help="per session; eviction starts at chunk 4")
    ap.add_argument("--image")
    ap.add_argument("--resource-overrides", default="{}", help="JSON: resource key -> apply_yaml_overrides kwargs")
    ap.add_argument("--solo-only", action="store_true", help="skip the interleaved pair (a one-slot deployment)")
    args = ap.parse_args()
    import json

    image_path = args.image or hf_hub_download("nvidia/Cosmos3-Edge", "assets/edge_action_id_av_0_output.png")
    model = LingBotWorldModel()
    cfg = model.config
    cs = cfg.chunk_size
    num_frames = (args.chunks * cs - 1) * 4 + 1
    image = model.load_image(image_path, "cpu").data
    intrinsics = [480.0, 480.0, 416.0, 240.0]

    def prompt_tensors(yaw: float) -> dict:
        return model.process_prompt(
            PROMPT, ["text", "image"], ["video_frame"], tensors={"image_inputs": [image]},
            camera_poses=trajectory(num_frames, yaw, 0.02), intrinsics=intrinsics, num_frames=num_frames,
        )

    tensors_a, tensors_b = prompt_tensors(0.002), prompt_tensors(-0.004)

    text_sub = model.get_submodule("text_encoder", device="cuda").to(DEVICE)
    info = CurrentForwardPassInfo(request_id="text", graph_walk="encode_text", fwd_index=0, random_seed=0,
                                  max_tokens=0, step_metadata={"guidance_scale": 1.0}, rid_handle=0)
    text_out = drive(text_sub, "encode_text", info, {"text_inputs": tensors_a["text_inputs"]})
    text_embeds = text_out["text_embeds_pos"][0].to(DEVICE)
    del text_sub, model._submodule_cache["text_encoder"]
    torch.cuda.empty_cache()

    sub = model.get_submodule("dit", device="cuda").to(DEVICE)
    sub.keep_chunk_outputs = True
    runner = bind_resources(model, sub, json.loads(args.resource_overrides))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()

    # solo A
    solo = Session(1, 42, tensors_a, text_embeds, cs)
    runner.ingest_request(solo.rid)
    while not solo.done:
        step_chunk(sub, runner, solo)
    finish(sub, runner, solo)
    solo_peak = torch.cuda.max_memory_allocated() - base_mem
    solo_peak_total = torch.cuda.max_memory_allocated()
    if args.solo_only:
        torch.save({"solo": {"latents": solo.latents, "frames": solo.frames, "times": solo.times},
                    "memory_gib": {"before_rollouts": base_mem / 2**30, "solo_peak_total": solo_peak_total / 2**30}},
                   args.out)
        print(f"memory: {base_mem / 2**30:.2f} GiB held before any rollout, "
              f"solo peak {solo_peak_total / 2**30:.2f} GiB")
        print("chunk seconds (solo):", " ".join(f"{t:.3f}" for t in solo.times))
        return

    # A and B interleaved, chunk by chunk; A again under a fresh handle
    a, b = Session(2, 42, tensors_a, text_embeds, cs), Session(3, 7, tensors_b, text_embeds, cs)
    for s in (a, b):
        runner.ingest_request(s.rid)
    while not (a.done and b.done):
        for s in (a, b):
            if not s.done:
                step_chunk(sub, runner, s)
    for s in (a, b):
        finish(sub, runner, s)
    pair_peak = torch.cuda.max_memory_allocated() - base_mem

    result = {
        "solo": {"latents": solo.latents, "frames": solo.frames, "times": solo.times},
        "a": {"latents": a.latents, "frames": a.frames},
        "b": {"latents": b.latents, "frames": b.frames},
        "geometry": tuple(tensors_a["image_inputs"][0].shape),
        "peak_extra_gib": {"solo": solo_peak / 2**30, "pair": pair_peak / 2**30},
        "memory_gib": {"before_rollouts": base_mem / 2**30, "solo_peak_total": solo_peak_total / 2**30,
                       "pair_peak_total": (pair_peak + base_mem) / 2**30},
    }
    torch.save(result, args.out)

    same = lambda x, y: all(torch.equal(p, q) for p, q in zip(x, y, strict=True))  # noqa: E731
    print(f"geometry {result['geometry']}, {solo.num_chunks} chunks per session")
    print("chunk seconds (solo):", " ".join(f"{t:.3f}" for t in solo.times))
    print(f"memory: {base_mem / 2**30:.2f} GiB held before any rollout; peak {solo_peak_total / 2**30:.2f} GiB "
          f"with one rollout, {(pair_peak + base_mem) / 2**30:.2f} GiB with two")
    print("interleaved A == solo A (latents, frames):", same(a.latents, solo.latents), same(a.frames, solo.frames))
    if args.compare_to:
        ref = torch.load(args.compare_to)
        for name in ("solo", "a", "b"):
            per_chunk = [torch.equal(x, y) for x, y in zip(result[name]["latents"], ref[name]["latents"], strict=True)]
            frames_ok = same(result[name]["frames"], ref[name]["frames"])
            print(f"{name}: latents bit-exact per chunk {per_chunk}, frames bit-exact {frames_ok}")
            if not all(per_chunk):
                k = per_chunk.index(False)
                d = (result[name]["latents"][k].double() - ref[name]["latents"][k].double()).abs().max().item()
                print(f"   first divergent chunk {k}: max |diff| {d:.3e}")


if __name__ == "__main__":
    main()

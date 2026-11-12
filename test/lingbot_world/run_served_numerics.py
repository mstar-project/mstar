"""LingBot-World's served numerics, driven by hand: the dit node over every
resource its declared state builds, stepped the way the engine steps it
(declare_step -> admit -> plan -> forward -> commit), no server.

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
from run_parity import DEVICE, drive  # noqa: E402

from mstar.communication.tensors import LocalTransferEngine  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.engine.resources import KVReqConfig, StepRunner  # noqa: E402
from mstar.engine.resources.base import EngineResourceInfo, build_resource  # noqa: E402
from mstar.engine.resources.kv.transfer import TransferEngineInfo  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.engine.state.adapters import StateSetup, kv_layouts, node_handles  # noqa: E402
from mstar.engine.state.state_op import bind_state_ops  # noqa: E402
from mstar.model.base import ProcessPromptOutput, deployment_setup, serving_resources  # noqa: E402
from mstar.model.lingbot_world.lingbot_world_model import REFERENCE_MODEL_KWARGS, LingBotWorldModel  # noqa: E402
from mstar.model.lingbot_world.submodules import (  # noqa: E402
    DECODER_CACHE,
    DIT_KV,
    ENCODER_CACHE,
    ROLLOUT_LOOP_NAME,
    ROLLOUT_WALK,
)
from mstar.model.submodule_base import ModelInputsFromEngine, engine_step  # noqa: E402

PROMPT = "A quiet suburban street on a clear afternoon, the camera driving slowly forward."


def trajectory(num_frames: int, yaw_per_frame: float, step: float) -> np.ndarray:
    """OpenCV camera-to-world poses: forward along +z with a constant yaw rate."""
    poses = np.tile(np.eye(4), (num_frames, 1, 1))
    for t in range(num_frames):
        a = yaw_per_frame * t
        poses[t, :3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        poses[t, :3, 3] = [step * t * math.sin(a), 0.0, step * t * math.cos(a)]
    return poses


def bind(model: LingBotWorldModel, sub, overrides: dict) -> tuple[StepRunner, StateSetup]:
    """Build every resource ``model`` declares as the engine does for a deployment whose ``resources:`` block is
    ``overrides``, and bind ``sub`` to its handles. Returns the runner that steps the resources and the setup that
    holds the draws' state."""
    config = {"resources": overrides}
    setup = deployment_setup(model, config)
    specs = {spec.resource_key: spec for spec in serving_resources(model, config, setup)}
    transfer = TransferEngineInfo("local", "local", LocalTransferEngine("local"))
    resources = {
        key: build_resource(spec, EngineResourceInfo(
            device=DEVICE, transfer_engine_info=transfer, kv_dtype=torch.bfloat16, needs_remote_transfer=False,
            nodes=frozenset(spec.nodes), dependencies={dep: specs[dep] for dep in spec.depends_on()},
        ))
        for key, spec in specs.items()
    }
    for key, spec in specs.items():
        resources[key].bind_dependencies({dep: resources[dep] for dep in spec.depends_on()})
    # as the worker sets it from the dit's writes, so admission counts a rollout's chunks
    resources[DIT_KV].local_loops = {ROLLOUT_LOOP_NAME}
    handles = node_handles(model, "dit", resources, setup=setup)
    bind_state_ops(handles)
    sub.bind_node_state(handles)
    return StepRunner(resources), setup


class Session:
    def __init__(self, rid: int, seed: int, prompt: ProcessPromptOutput, text_embeds: torch.Tensor, cs: int):
        tensors = prompt.new_input_tensors
        lat_f = int(tensors["camera_poses"][0].shape[0])
        self.rid = rid
        self.num_chunks = lat_f // cs
        self.state_layout = prompt.metadata["state_layout"]
        md = {"is_prefill": False, "guidance_scale": 1.0, "num_latent_frames": lat_f, "num_chunks": self.num_chunks}
        self.info = CurrentForwardPassInfo(
            request_id=f"served-{rid}", graph_walk=ROLLOUT_WALK, fwd_index=0, random_seed=seed, max_tokens=0,
            step_metadata=md, rid_handle=rid,
        )
        # the engine counts the loop's iterations; the DiT reads its chunk from the count
        self.info.dynamic_loop_iter_counts[ROLLOUT_LOOP_NAME] = 0
        self.inputs = {"text_embeds_pos": [text_embeds], "image_inputs": [tensors["image_inputs"][0].to(DEVICE)],
                       "camera_poses": tensors["camera_poses"], "intrinsics": tensors["intrinsics"], "clock": []}
        self.latents: list[torch.Tensor] = []
        self.frames: list[torch.Tensor] = []
        self.times: list[float] = []

    @property
    def done(self) -> bool:
        return len(self.latents) >= self.num_chunks

    def open(self, model: LingBotWorldModel, runner: StepRunner, setup: StateSetup) -> None:
        """Open the rollout with the pool config the conductor builds from its layout, and admit it through the
        engine's queue."""
        rows = {label: layout.row()
                for label, layout in kv_layouts(model.declare_state(), self.state_layout, {})[DIT_KV].items()}
        request = KVReqConfig(needed_labels=list(rows))
        request.apply_conductor_config(kv_layout=rows, loop_caps={ROLLOUT_LOOP_NAME: self.num_chunks})
        # the worker finds no rebuild for the dit, so the pool guarantees a rollout its footprint
        request.rebuildable = False
        runner.ingest_request(self.rid, {DIT_KV: request})
        setup.host_fields.open(self.rid)
        admitted = runner.admit_retrieve(self.rid, "dit", ROLLOUT_WALK)
        if not (admitted.ok and admitted.ready):
            raise RuntimeError(f"rollout {self.rid} was not admitted: {admitted}")


def step_chunk(sub, runner: StepRunner, s: Session) -> None:
    """One rollout iteration of ``s``, as the engine runs a batch of one."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    inputs = [sub.prepare_inputs(graph_walk=ROLLOUT_WALK, fwd_info=s.info, inputs=s.inputs)]
    step = engine_step(sub, graph_walk=ROLLOUT_WALK, request_ids=[s.rid], inputs=inputs)
    step.set_ctx(StepContext(request_ids=(s.rid,), graph_walk=ROLLOUT_WALK, slot=0, capture=False))
    admitted = runner.admit(step)
    if not admitted.ok:
        raise RuntimeError(f"chunk {len(s.latents)} of rollout {s.rid} was refused: {admitted.reason}")
    runner.plan(step)
    step.begin_calls()
    eng = ModelInputsFromEngine(request_ids=[s.rid], per_request_info={s.rid: s.info})
    out = sub.forward(graph_walk=ROLLOUT_WALK, engine_inputs=eng,
                      **sub.preprocess(graph_walk=ROLLOUT_WALK, engine_inputs=eng, inputs=inputs))
    step.check_calls()
    runner.commit(step)
    step.commit_calls()
    step.end_calls()
    torch.cuda.synchronize()
    s.times.append(time.perf_counter() - t0)
    s.latents.append(out["chunk_latents"][0].cpu())
    s.frames.append(out["video_output"][0].cpu())
    s.info.dynamic_loop_iter_counts[ROLLOUT_LOOP_NAME] = len(s.latents)
    # the rollout loop's loop-back edges, which the engine feeds to the next chunk
    s.inputs.update({name: out[name] for name in ("clock", ENCODER_CACHE, DECODER_CACHE)})


def finish(runner: StepRunner, setup: StateSetup, s: Session) -> None:
    runner.remove_request(s.rid)
    setup.host_fields.remove_request(s.rid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-to")
    ap.add_argument("--chunks", type=int, default=8, help="per session; eviction starts at chunk 4")
    ap.add_argument("--image")
    ap.add_argument("--resource-overrides", default="{}",
                    help='JSON: the resources block of a YAML, as {"dit_kv": {"max_num_pages": 215}}')
    ap.add_argument("--solo-only", action="store_true", help="skip the interleaved pair (a one-slot deployment)")
    ap.add_argument("--reference-numerics", action="store_true",
                    help="REFERENCE_MODEL_KWARGS: FA2 attention and an eager DiT; needs flash-attn")
    args = ap.parse_args()
    import json

    image_path = args.image or hf_hub_download("nvidia/Cosmos3-Edge", "assets/edge_action_id_av_0_output.png")
    model = LingBotWorldModel(keep_chunk_outputs=True, **(REFERENCE_MODEL_KWARGS if args.reference_numerics else {}))
    cfg = model.config
    cs = cfg.chunk_size
    num_frames = (args.chunks * cs - 1) * 4 + 1
    image = model.load_image(image_path, "cpu").data
    intrinsics = [480.0, 480.0, 416.0, 240.0]

    def prompt(yaw: float) -> ProcessPromptOutput:
        return model.process_prompt(
            PROMPT, ["text", "image"], ["video_frame"], tensors={"image_inputs": [image]},
            camera_poses=trajectory(num_frames, yaw, 0.02), intrinsics=intrinsics, num_frames=num_frames,
        )

    prompt_a, prompt_b = prompt(0.002), prompt(-0.004)

    text_sub = model.get_submodule("text_encoder", device="cuda").to(DEVICE)
    info = CurrentForwardPassInfo(request_id="text", graph_walk="encode_text", fwd_index=0, random_seed=0,
                                  max_tokens=0, step_metadata={"guidance_scale": 1.0}, rid_handle=0)
    text_out = drive(text_sub, "encode_text", info, {"text_inputs": prompt_a.new_input_tensors["text_inputs"]})
    text_embeds = text_out["text_embeds_pos"][0].to(DEVICE)
    del text_sub, model._submodule_cache["text_encoder"]
    torch.cuda.empty_cache()

    sub = model.get_submodule("dit", device="cuda").to(DEVICE)
    runner, setup = bind(model, sub, json.loads(args.resource_overrides))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()

    # solo A
    solo = Session(1, 42, prompt_a, text_embeds, cs)
    solo.open(model, runner, setup)
    while not solo.done:
        step_chunk(sub, runner, solo)
    finish(runner, setup, solo)
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
    a, b = Session(2, 42, prompt_a, text_embeds, cs), Session(3, 7, prompt_b, text_embeds, cs)
    for s in (a, b):
        s.open(model, runner, setup)
    while not (a.done and b.done):
        for s in (a, b):
            if not s.done:
                step_chunk(sub, runner, s)
    for s in (a, b):
        finish(runner, setup, s)
    pair_peak = torch.cuda.max_memory_allocated() - base_mem

    result = {
        "solo": {"latents": solo.latents, "frames": solo.frames, "times": solo.times},
        "a": {"latents": a.latents, "frames": a.frames},
        "b": {"latents": b.latents, "frames": b.frames},
        "geometry": tuple(prompt_a.new_input_tensors["image_inputs"][0].shape),
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

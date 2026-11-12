#!/usr/bin/env python3
"""Kernel-time breakdown of one LTX-2.5 DiT step at a request shape, outside the server.

    python benchmark/ltx2_5/profile_step.py [--attention sdpa|flashinfer] [--compile]
    torchrun --nproc_per_node 2 benchmark/ltx2_5/profile_step.py --parallel tp|sp [...]

Groups CUDA kernel time into GEMM / attention / everything else, and prints the
top kernels. ``--attention flashinfer`` runs the denoise step over the engine's
resources, planned by hand (as the runner would), so the kernels are the served ones;
sequence parallelism (``--parallel sp``) needs it.
"""
import argparse
import collections
import dataclasses
import os
import time

import torch
import torch.distributed as dist

from mstar.model.components.diffusion.denoise_loop import StepKey
from mstar.model.ltx2_5.config import LTX25_REPO, SNAPSHOT_PATTERNS, LTX25Config
from mstar.model.ltx2_5.weight_loader import build_transformer
from mstar.utils.hf_snapshot import resolve_snapshot_dir


def denoise_step(config, dit, shape, batch, device, joint, attention):
    """The dit's denoise submodule over its own resources and state (the model's declarations, sharded over ``joint``
    as the engine shards them), with one step at ``shape`` planned by hand as the runner would: the served kernels, the
    sequence shard's split and exchanges included. Returns the submodule and the step to open around each call."""
    from mstar.engine.resources import SeqShardSpec, StepContext, StepRunner
    from mstar.engine.resources.base import EngineResourceInfo, build_resource
    from mstar.engine.resources.spec import resolve_spec_dependencies
    from mstar.engine.state.adapters import node_handles, node_resources, state_setup
    from mstar.model.ltx2_5 import submodules as S
    from mstar.model.ltx2_5.ltx2_5_model import LTX25Model
    from mstar.model.submodule_base import NodeInputs, engine_step

    model = LTX25Model(skip_weight_loading=True)
    model.set_config(config)
    # every attention resource; the sequence shard isn't one
    on_sdpa = [spec.resource_key for spec in node_resources(model) if not isinstance(spec, SeqShardSpec)]
    sdpa = {key: {"backend": "sdpa"} for key in on_sdpa} if attention == "sdpa" else {}
    setup, _ = state_setup({"resources": sdpa}, model.declare_state())
    specs = resolve_spec_dependencies(node_resources(model, setup=setup))
    info = EngineResourceInfo(device=device, kv_dtype=torch.bfloat16, joint_comm_group=joint)
    res = {
        key: build_resource(spec, dataclasses.replace(info, dependencies={k: specs[k] for k in spec.depends_on()}))
        for key, spec in specs.items()
    }
    for key, spec in specs.items():
        res[key].bind_dependencies({dep: res[dep] for dep in spec.depends_on()})
    sub = S.LTXDenoiseSubmodule(dit, config, loop_name="x")
    sub.bind_node_resources(res)
    sub.bind_node_state(node_handles(model, "dit", res, setup=setup))
    rids = list(range(batch))
    step = engine_step(sub, "g", rids, [NodeInputs(resource_step_info=StepKey(shape))] * batch)
    step.set_ctx(StepContext(request_ids=rids, graph_walk="g", slot=0, capture=False))
    StepRunner(res).plan(step)
    return sub, step


def classify(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("gemm", "cutlass", "sm90_xmma", "nvjet", "cublas", "matmul")):
        return "gemm"
    if "nccl" in n:
        return "collective"
    if any(k in n for k in ("flash", "fmha", "attention", "prefill", "attn")):
        return "attention"
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attention", choices=("sdpa", "flashinfer"), default="flashinfer")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--parallel", choices=("tp", "sp"), help="under torchrun: shard the DiT across the ranks")
    args = p.parse_args()
    if args.parallel == "sp" and args.attention != "flashinfer":
        p.error("--parallel sp splits the step through the engine's resources: --attention flashinfer")
    from mstar.distributed.communication import CommGroup, JointGroups

    joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial(), instance=CommGroup.trivial())
    world_size, rank = 1, 0
    if args.parallel:
        dist.init_process_group("nccl")
        world_size, rank = dist.get_world_size(), dist.get_rank()
        torch.cuda.set_device(rank)
        group = CommGroup(rank, rank, list(range(world_size)))
        group.device_group, group.initialized = dist.group.WORLD, True
        trivial = CommGroup.trivial()
        joint = (JointGroups(tp_group=group, sp_group=trivial, instance=group) if args.parallel == "tp"
                 else JointGroups(tp_group=trivial, sp_group=group, instance=group))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    snapshot = resolve_snapshot_dir(LTX25_REPO, allow_patterns=SNAPSHOT_PATTERNS)
    config = LTX25Config.from_snapshot(snapshot)
    from mstar.model.ltx2_5.submodules import shape_from_metadata

    shape = shape_from_metadata(config, {"height": 544, "width": 960, "num_frames": 121, "fps": 24.0})
    dit = build_transformer(config, snapshot, device, comm_group=joint.tp_group)
    if args.compile:
        dit.forward = torch.compile(dit.forward, dynamic=False)
    b = args.batch
    t = config.transformer
    video, audio, text, audio_text = (
        torch.randn(b, shape.video_tokens, t.in_channels, device=device, dtype=torch.bfloat16),
        torch.randn(b, shape.audio_frames, t.audio_in_channels, device=device, dtype=torch.bfloat16),
        torch.randn(b, shape.text_len, t.cross_attention_dim, device=device, dtype=torch.bfloat16),
        torch.randn(b, shape.text_len, t.audio_cross_attention_dim, device=device, dtype=torch.bfloat16),
    )
    timestep = torch.full((b,), 900.0, device=device)
    sub, opened = denoise_step(config, dit, shape, b, device, joint, args.attention)
    sigma = torch.full((b, 1, 1), 0.9, device=device)

    def step():
        opened.begin_calls()
        try:
            sub.denoise(None, StepKey(shape), video, timestep, sigma, sigma * 0.5, audio, text, audio_text)
        finally:
            opened.end_calls()

    with torch.inference_mode():
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        wall = 1000 * (time.perf_counter() - t0) / 5
        # every rank runs the profiled step: under TP / SP its collectives need them all
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
    if args.parallel:
        dist.destroy_process_group()
    if rank:
        return
    print(f"step wall {wall:.1f} ms  (attention={args.attention} compile={args.compile} batch={b} "
          f"parallel={args.parallel or 'none'} x{world_size})")
    groups = collections.Counter()
    kernels = collections.Counter()
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            groups[classify(evt.name)] += evt.device_time
            kernels[evt.name[:90]] += evt.device_time
    total = sum(groups.values())
    for g, us in groups.most_common():
        print(f"  {g:10s} {us / 1000:8.1f} ms  {100 * us / total:5.1f}%")
    for name, us in kernels.most_common(12):
        print(f"    {us / 1000:7.1f} ms  {name}")


if __name__ == "__main__":
    main()

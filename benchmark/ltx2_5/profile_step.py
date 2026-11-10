#!/usr/bin/env python3
"""Kernel-time breakdown of one LTX-2.5 DiT step at a request shape, outside the server.

    python benchmark/ltx2_5/profile_step.py [--attention sdpa|flashinfer] [--compile]
    torchrun --nproc_per_node 2 benchmark/ltx2_5/profile_step.py --parallel tp|sp [...]

Groups CUDA kernel time into GEMM / attention / everything else, and prints the
top kernels. ``--attention flashinfer`` plans the engine's ragged resources by hand
(as the runner would) so the kernels are the served ones.
"""
import argparse
import collections
import os
import time

import torch
import torch.distributed as dist

from mstar.model.ltx2_5.components.transformer import build_rope
from mstar.model.ltx2_5.config import LTX25_REPO, SNAPSHOT_PATTERNS, LTX25Config
from mstar.model.ltx2_5.weight_loader import build_transformer
from mstar.utils.hf_snapshot import resolve_snapshot_dir


def open_attention(dit, config, shape, batch, device, attention, comm_group=None, sp_group=None):
    """Build, bind and plan what the engine would for one step of ``dit``, so the kernels are the served ones. Returns
    the step to open around each call, or None on SDPA with no exchange, where nothing opens."""
    from mstar.distributed.communication import CommGroup, JointGroups
    from mstar.engine.resources.base import EngineResourceInfo, build_resource
    from mstar.engine.resources.step import StepContext
    from mstar.engine.state.adapters import node_handles, node_resources, state_setup, state_steps
    from mstar.model.ltx2_5 import submodules as S
    from mstar.model.ltx2_5.ltx2_5_model import LTX25Model
    from mstar.model.submodule_base import NodeInputs

    model = LTX25Model(skip_weight_loading=True)
    model.set_config(config)
    on_sdpa = [spec.resource_key for spec in node_resources(model)] if attention == "sdpa" else []
    setup, _ = state_setup({"resources": {key: {"backend": "sdpa"} for key in on_sdpa}}, model.declare_state())
    joint = JointGroups(tp_group=comm_group or CommGroup.trivial(), sp_group=sp_group or CommGroup.trivial())
    info = EngineResourceInfo(device=device, kv_dtype=torch.bfloat16, joint_comm_group=joint)
    resources = {
        spec.resource_key: build_resource(spec, info)
        for spec in node_resources(model, setup=setup)
    }
    handles = node_handles(model, "dit", resources, sp_group=sp_group, setup=setup)
    for module in dit.modules():
        if hasattr(module, "bind_state"):
            module.bind_state(handles)
    rows = list(range(batch))
    sub = S.LTXDenoiseSubmodule(torch.nn.Linear(1, 1), config, loop_name="x")
    step = state_steps(sub.declare_step("g", rows, [NodeInputs(resource_step_info=shape)] * batch), handles)
    if step is None:
        return None
    ctx = StepContext(request_ids=rows, graph_walk="g", slot=0, capture=False)
    step.set_ctx(ctx)
    for key, declared in step.steps.items():
        resources[key].plan(declared, ctx)
    return step


def run_step(step, dit, *args):
    """One call of ``dit``, with its handles open around it as the engine opens them."""
    if step is None:
        return dit(*args)
    step.begin_calls()
    try:
        return dit(*args)
    finally:
        step.end_calls()


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
    comm_group = sp_group = None
    world_size, rank = 1, 0
    if args.parallel:
        from mstar.distributed.communication import CommGroup

        dist.init_process_group("nccl")
        world_size, rank = dist.get_world_size(), dist.get_rank()
        torch.cuda.set_device(rank)
        group = CommGroup(rank, rank, list(range(world_size)))
        group.device_group, group.initialized = dist.group.WORLD, True
        comm_group, sp_group = (group, None) if args.parallel == "tp" else (None, group)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    snapshot = resolve_snapshot_dir(LTX25_REPO, allow_patterns=SNAPSHOT_PATTERNS)
    config = LTX25Config.from_snapshot(snapshot)
    from mstar.model.ltx2_5.submodules import shape_from_metadata

    shape = shape_from_metadata(config, {"height": 544, "width": 960, "num_frames": 121, "fps": 24.0})
    dit = build_transformer(config, snapshot, device, comm_group=comm_group, sp_group=sp_group)
    if args.compile:
        dit.forward = torch.compile(dit.forward, dynamic=False)
    rope = build_rope(config.transformer, shape.frames, shape.height, shape.width, shape.fps, shape.audio_frames,
                      device)
    step = open_attention(dit, config, shape, args.batch, device, args.attention, comm_group, sp_group)
    b = args.batch
    t = config.transformer
    inputs = (
        torch.randn(b, shape.video_tokens, t.in_channels, device=device, dtype=torch.bfloat16),
        torch.randn(b, shape.audio_frames, t.audio_in_channels, device=device, dtype=torch.bfloat16),
        torch.randn(b, shape.text_len, t.cross_attention_dim, device=device, dtype=torch.bfloat16),
        torch.randn(b, shape.text_len, t.audio_cross_attention_dim, device=device, dtype=torch.bfloat16),
        torch.full((b,), 900.0, device=device),
    )
    with torch.inference_mode():
        for _ in range(3):
            run_step(step, dit, *inputs, rope)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            run_step(step, dit, *inputs, rope)
        torch.cuda.synchronize()
        wall = 1000 * (time.perf_counter() - t0) / 5
        # every rank runs the profiled step: under TP / SP its collectives need them all
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            run_step(step, dit, *inputs, rope)
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

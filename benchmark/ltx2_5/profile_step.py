#!/usr/bin/env python3
"""Kernel-time breakdown of one LTX-2.5 DiT step at a request shape, outside the server.

    python benchmark/ltx2_5/profile_step.py [--attention sdpa|flashinfer] [--compile]

Groups CUDA kernel time into GEMM / attention / everything else, and prints the
top kernels. ``--attention flashinfer`` plans the engine's ragged resources by hand
(as the runner would) so the kernels are the served ones.
"""
import argparse
import collections
import time

import torch

from mstar.model.components.diffusion.attention import sdpa_attention
from mstar.model.ltx2_5.components.transformer import LTX2Attends, build_rope
from mstar.model.ltx2_5.config import LTX25_REPO, SNAPSHOT_PATTERNS, LTX25Config
from mstar.model.ltx2_5.weight_loader import build_transformer
from mstar.utils.hf_snapshot import resolve_snapshot_dir


def ragged_attends(config, shape, batch, device):
    from mstar.engine.resources import (
        RaggedAttentionConfig,
        RaggedAttentionSpec,
        RaggedCrossAttentionSpec,
        StepContext,
    )
    from mstar.engine.resources.base import EngineResourceInfo, build_resource
    from mstar.model.ltx2_5 import submodules as S
    from mstar.model.submodule_base import NodeInputs

    t = config.transformer
    res = {}
    video_heads = (t.num_attention_heads, t.attention_head_dim)
    audio_heads = (t.audio_num_attention_heads, t.audio_attention_head_dim)
    for spec_cls, key, (heads, hd) in ((RaggedAttentionSpec, S.VIDEO_ATTN, video_heads),
                                       (RaggedAttentionSpec, S.AUDIO_ATTN, audio_heads),
                                       (RaggedCrossAttentionSpec, S.VIDEO_XATTN, video_heads),
                                       (RaggedCrossAttentionSpec, S.AUDIO_XATTN, audio_heads)):
        spec = spec_cls(resource_key=key, nodes={"dit"}, config=RaggedAttentionConfig(
            num_qo_heads=heads, num_kv_heads=heads, head_dim=hd, dtype=torch.bfloat16))
        res[key] = build_resource(spec, EngineResourceInfo(device=device, kv_dtype=torch.bfloat16))
    sub = S.LTXDenoiseSubmodule(torch.nn.Linear(1, 1), config, loop_name="x", use_ragged_attention=True)
    sub.node_resources = res
    step = sub.declare_step("g", list(range(batch)), [NodeInputs(resource_step_info=shape)] * batch)
    for key, r in res.items():
        r.plan(step.steps[key], StepContext(request_ids=list(range(batch)), graph_walk="g", slot=0, capture=False))
    return sub.attends()


def classify(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("gemm", "cutlass", "sm90_xmma", "nvjet", "cublas", "matmul")):
        return "gemm"
    if any(k in n for k in ("flash", "fmha", "attention", "prefill", "attn")):
        return "attention"
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attention", choices=("sdpa", "flashinfer"), default="flashinfer")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()
    device = torch.device("cuda:0")
    snapshot = resolve_snapshot_dir(LTX25_REPO, allow_patterns=SNAPSHOT_PATTERNS)
    config = LTX25Config.from_snapshot(snapshot)
    from mstar.model.ltx2_5.submodules import shape_from_metadata

    shape = shape_from_metadata(config, {"height": 544, "width": 960, "num_frames": 121, "fps": 24.0})
    dit = build_transformer(config, snapshot, device)
    if args.compile:
        dit.forward = torch.compile(dit.forward, dynamic=False)
    rope = build_rope(config.transformer, shape.frames, shape.height, shape.width, shape.fps, shape.audio_frames,
                      device)
    attends = (LTX2Attends(*([sdpa_attention] * 6)) if args.attention == "sdpa"
               else ragged_attends(config, shape, args.batch, device))
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
            dit(*inputs, rope, attends)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            dit(*inputs, rope, attends)
        torch.cuda.synchronize()
        print(f"step wall {1000 * (time.perf_counter() - t0) / 5:.1f} ms  (attention={args.attention} "
              f"compile={args.compile} batch={b})")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            dit(*inputs, rope, attends)
            torch.cuda.synchronize()
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

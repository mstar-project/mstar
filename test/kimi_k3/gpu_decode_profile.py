"""Kimi K3.1 decode on one GPU, compiled glue off and on: what a decode step launches and where.

The release cut (``truncate.py``: the first 4 layers, 3 KDA + 1 MLA, 1 dense MLP + 3 latent MoE) through
the real Engine, decode captured at the given batch sizes. Per batch size and mode:
- the captured decode step's time (CUDA events over replays);
- an eager decode step under torch.profiler, each kernel attributed by its launching op's enclosing
  range: a layer's attention (KDA / MLA), its MLP (MoE / dense) or the glue between them (the AttnRes
  reads, the norms: the layer minus both), and the step outside the layers (embedding, output read,
  head, sampler);
- the full model's launches and kernel time per step extrapolated from those per-layer-type figures
  (69 KDA, 24 MLA, 92 MoE, 1 dense, 93 layers of glue, the rest once), with the top kernels by launches.

No all-reduce on one GPU, and the cut's shapes are TP1's (96 heads, every expert's full width): the
launch counts carry to a TP8 rank (the same ops), the times only roughly.

    python test/kimi_k3/gpu_decode_profile.py --ckpt <mxfp4 cut> --batch 1 8 32 64 --out prof.json
"""
from __future__ import annotations

import argparse
import collections
import json
import time

import torch

FULL = {"kda": 69, "mla": 24, "moe": 92, "dense": 1, "glue": 93, "rest": 1}


DEVICE = "cuda:0"
# which MoE branch runs on the side stream (shared_experts_overlap: routed / shared / off), as --overlap sets it
OVERLAP = "routed"


# mode -> (compile_glue, compile_reads): nothing compiled, every region, the MLP blocks only (the reads eager)
MODES = {"off": (False, False), "on": (True, True), "mlp": (True, False)}


def build(ckpt: str, mode: str, capture: bool, sizes: tuple[int, ...]):
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv import manager as kv_mod
    from mstar.engine.resources.kv.transfer import TransferEngineInfo
    from mstar.model.kimi_k3.config import SAMPLER
    from mstar.model.kimi_k3.kimi_k3_model import KimiK31Model
    from mstar.model.kimi_k3.tests import engine_helpers as H
    from mstar.quantization.scope import load_submodule

    kv_mod.KVTransferManager = H._StubTransfer
    model = KimiK31Model(ckpt, strict_config=False, kda_max_requests=max(sizes), tokenizer_mode="byte",
                         decode_capture_batch_sizes=sizes, compile_glue=MODES[mode][0], compile_reads=MODES[mode][1],
                         shared_experts_overlap=OVERLAP)
    dtype = torch.bfloat16 if DEVICE != "cpu" else torch.float32
    if DEVICE == "cpu":
        model.kda_conv_dtype = torch.float32
    sub = load_submodule(model, "LLM", {}, DEVICE, autocast_dtype=dtype)
    if not capture:
        sub.get_accelerator_graph_configs = lambda *a, **k: []
    engine = Engine(None, autocast_dtype=dtype)
    engine.load_model(
        {"LLM": sub}, model.get_node_resources(), parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device(DEVICE),
        transfer_engine_info=TransferEngineInfo(my_entity_id="e0", my_session_id="s",
                                                transfer_engine=LocalTransferEngine("localhost")),
        kv_cache_type=dtype)
    greedy = H.GreedySampler()
    engine._resources[SAMPLER] = engine._submodules["LLM"].resources[SAMPLER] = greedy
    sub.node_resources[SAMPLER] = greedy
    t = time.time()
    engine.warmup()
    return engine, sub, time.time() - t


def start(engine, bs: int, vocab: int, prompt_len: int = 32, tag: str = ""):
    """``bs`` requests prefilled; returns their next inputs."""
    from mstar.model.kimi_k3.tests import engine_helpers as H

    nxt = {}
    for r in range(bs):
        rid = f"{tag}r{r}"
        H.add_request(engine, rid)
        hi = min(vocab, 160000)
        ids = torch.randint(min(1000, hi // 4), hi, (prompt_len,), generator=torch.Generator().manual_seed(r))
        nxt.update(H.run_step(engine, "prefill", {rid: ids.to(DEVICE)}))
    return {rid: tok.reshape(1) for rid, tok in nxt.items()}


def decode(engine, inputs):
    from mstar.model.kimi_k3.tests import engine_helpers as H

    return {rid: tok.reshape(1) for rid, tok in H.run_step(engine, "decode", inputs).items()}


def finish(engine, inputs):
    for rid in inputs:
        engine.remove_request(rid)


def step_ms(engine, inputs, steps: int = 20) -> tuple[float, dict]:
    if DEVICE == "cpu":
        t = time.time()
        for _ in range(3):
            inputs = decode(engine, inputs)
        return (time.time() - t) * 1e3 / 3, inputs
    for _ in range(3):
        inputs = decode(engine, inputs)
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(steps):
        inputs = decode(engine, inputs)
    t1.record()
    t1.synchronize()
    return t0.elapsed_time(t1) / steps, inputs


def _ranges(sub):
    """Forward hooks opening a profiler range around each layer, its attention and its MLP."""
    from torch.profiler import record_function

    handles, open_ = [], {}
    for i, layer in enumerate(sub.language_model.model.layers):
        attn = "kda" if layer.is_linear_attention else "mla"
        mlp = "moe" if hasattr(layer, "block_sparse_moe") else "dense"
        spans = ((layer, f"k31:layer:{i}"), (layer.self_attn, f"k31:{attn}:{i}"), (layer.ffn, f"k31:{mlp}:{i}"))
        for mod, name in spans:
            def pre(m, args, name=name):
                open_[name] = record_function(name)
                open_[name].__enter__()

            def post(m, args, out, name=name):
                open_.pop(name).__exit__(None, None, None)

            handles += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
    return handles


def attribute(prof) -> dict:
    """``part -> kernel name -> [launches, us]`` for one profiled step: each kernel under the closest
    ``k31:*`` range of its launching op ("glue" = a layer's own kernels outside its attention and MLP;
    "rest" = outside every layer)."""
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    for ev in prof.events():
        # the CPU dry run (--cpu) counts the leaf aten ops in place of kernels
        launched = ev.kernels if DEVICE != "cpu" else ([ev] if ev.name.startswith("aten::") and not ev.cpu_children
                                                         else [])
        if not launched:
            continue
        part = "rest"
        p = ev
        while p is not None:
            if p.name.startswith("k31:"):
                kind = p.name.split(":")[1]
                part = "glue" if kind == "layer" else kind
                break
            p = p.cpu_parent
        for k in launched:
            cell = out[part][k.name]
            cell[0] += 1
            cell[1] += k.duration if hasattr(k, "duration") else k.time_range.elapsed_us()
    return out


def profile_eager(engine, sub, inputs) -> tuple[dict, dict]:
    from torch.profiler import ProfilerActivity, profile

    handles = _ranges(sub)
    try:
        inputs = decode(engine, inputs)
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE != "cpu" else [])
        with profile(activities=activities) as prof:
            inputs = decode(engine, inputs)
    finally:
        for h in handles:
            h.remove()
    parts = attribute(prof)
    from torch.autograd import DeviceType

    seen = sum(1 for ev in prof.events() if ev.device_type == DeviceType.CUDA)
    got = sum(v[0] for kernels in parts.values() for v in kernels.values())
    if DEVICE != "cpu" and seen != got:
        print(f"  note: {seen - got} of {seen} kernels had no launching op to attribute", flush=True)
    return parts, inputs


def layer_types(sub) -> collections.Counter:
    c = collections.Counter()
    for layer in sub.language_model.model.layers:
        c["kda" if layer.is_linear_attention else "mla"] += 1
        c["moe" if hasattr(layer, "block_sparse_moe") else "dense"] += 1
        c["glue"] += 1
    c["rest"] = 1
    return c


def extrapolate(parts: dict, have: collections.Counter) -> dict:
    """The full model's step: each part's per-instance kernels times its count in the release."""
    full = collections.defaultdict(lambda: [0.0, 0.0])
    per_part = {}
    for part, kernels in parts.items():
        n = have.get(part, 0)
        if not n:
            continue
        scale = FULL[part] / n
        launches = sum(v[0] for v in kernels.values())
        us = sum(v[1] for v in kernels.values())
        per_part[part] = {"per_instance_launches": launches / n, "per_instance_us": us / n,
                          "full_launches": launches * scale, "full_us": us * scale}
        for name, (cnt, t) in kernels.items():
            full[name][0] += cnt * scale
            full[name][1] += t * scale
    top = sorted(full.items(), key=lambda kv: -kv[1][0])[:15]
    return {"parts": per_part, "total_launches": sum(v[0] for v in full.values()),
            "total_kernel_us": sum(v[1] for v in full.values()),
            "top_by_launches": [(name[:90], round(c), round(t)) for name, (c, t) in top]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--modes", nargs="+", default=["off", "on"], choices=sorted(MODES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true", help="dry run on the CPU (a tiny release): the flow, no kernels")
    ap.add_argument("--overlap", choices=("routed", "shared", "off"), default="routed",
                    help="which MoE branch runs on the side stream (shared_experts_overlap)")
    args = ap.parse_args()
    global DEVICE, OVERLAP
    OVERLAP = args.overlap
    if args.cpu:
        DEVICE = "cpu"
        from mstar.model.kimi_k3.tests import engine_helpers as H

        class _Patch:
            def setitem(self, d, k, v):
                d[k] = v

            def setattr(self, o, n, v):
                setattr(o, n, v)

        H.install_cpu_stubs(_Patch())
    sizes = tuple(sorted(set(args.batch)))
    report = {}
    for mode in args.modes:
        row = report[mode] = {}
        engine, sub, warm = build(args.ckpt, mode, capture=True, sizes=sizes)
        row["warmup_s"] = round(warm, 1)
        print(json.dumps({"mode": mode, "warmup_s": row["warmup_s"]}), flush=True)
        for bs in sizes:
            inputs = start(engine, bs, sub.config.vocab_size, tag=f"c{bs}")
            ms, inputs = step_ms(engine, inputs)
            row[f"captured_ms_bs{bs}"] = round(ms, 3)
            print(json.dumps({"mode": mode, "bs": bs, "captured_step_ms": round(ms, 3)}), flush=True)
            finish(engine, inputs)
        del engine, sub
        if DEVICE != "cpu":
            torch.cuda.empty_cache()
        engine, sub, _ = build(args.ckpt, mode, capture=False, sizes=sizes)
        have = layer_types(sub)
        for bs in (1, max(sizes)):
            inputs = start(engine, bs, sub.config.vocab_size, tag=f"e{bs}")
            parts, inputs = profile_eager(engine, sub, inputs)
            ex = extrapolate(parts, have)
            row[f"eager_bs{bs}"] = ex
            cut = sum(v[0] for k in parts.values() for v in k.values())
            summary = {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in ex["parts"].items()}
            print(json.dumps({"mode": mode, "bs": bs, "cut_launches": cut,
                              "full_model_launches": round(ex["total_launches"]),
                              "full_model_kernel_ms": round(ex["total_kernel_us"] / 1e3, 2), "parts": summary}),
                  flush=True)
            for name, cnt, us in ex["top_by_launches"][:10]:
                print(f"  [{mode} bs{bs}] {cnt:7d} launches {us / 1e3:8.2f} ms  {name}", flush=True)
            finish(engine, inputs)
        del engine, sub
        if DEVICE != "cpu":
            torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

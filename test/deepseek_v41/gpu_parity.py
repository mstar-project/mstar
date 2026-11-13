"""DeepSeek-V4.1 on GPUs against its reference's own kernels. Not a CI test: it needs a GPU, the
checkpoint repo's ``inference/`` code and, for the kernels, ``tilelang`` (the reference's pin).

    python test/deepseek_v41/gpu_parity.py --reference <snapshot dir> --mode kernels --out k.json
    python test/deepseek_v41/gpu_parity.py --reference <snapshot dir> --mode tp1 --out tp1.json
    python test/deepseek_v41/gpu_parity.py --reference <snapshot dir> --mode tp2 --out tp2.json

``kernels``: ``inference/kernel.py`` against the torch stand-ins the CPU parity tests run the
reference on (``_testing.reference_kernels``), on random and on tie inputs: the roundings bit for
bit, the GEMMs and the attention in units of the result's bf16 spacing.
``tp1``: the reduced random-weight model. In fp32, M* through the ``Engine`` on the GPU against
the reference on the stand-ins (the CPU gate, on CUDA). In bf16, the reference on its own kernels
against itself on the stand-ins and against M*, next to the reference's own spread when only its
attention kernel is swapped for an exact softmax.
``tp2``: two ranks (NCCL on cuda:0 and cuda:1; gloo with ``--engine-device cpu``) against one.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, ".")

import torch
import torch.multiprocessing as mp

PROMPT, STEPS = 45, 14
TP2 = dict(moe_intermediate_size=64, o_lora_rank=32)


def _real_kernels(reference: str) -> types.ModuleType:
    import importlib.util

    spec = importlib.util.spec_from_file_location("dsv41_reference_kernel", Path(reference) / "inference" / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spacing(x: torch.Tensor) -> torch.Tensor:
    """The gap between adjacent bf16 values at each element of ``x``."""
    return torch.ldexp(torch.ones_like(x.float()), torch.frexp(x.float().abs().clamp_min(1e-30))[1] - 8)


def _tie_inputs(device) -> torch.Tensor:
    """Rows whose blocks hold exact midpoints of the E2M1 and E4M3 grids at a power-of-two block
    maximum, all-zero blocks, and maxima on and next to powers of two."""
    fp4_mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=device)
    fp8_mid = torch.tensor([1.0625, 1.1875, 1.3125, 1.4375, 17.0, 19.0, 232.0], device=device)
    rows = []
    for mids, top in ((fp4_mid, 6.0), (fp8_mid, 448.0)):
        for shift in (-9, -3, 0, 4):
            block = torch.cat([mids, -mids, torch.tensor([top, -top, 0.0, top / 2], device=device)])
            block = block.repeat(4)[:32] * 2.0 ** shift
            rows.append(block.repeat(4))
    rows.append(torch.zeros(128, device=device))
    for top in (1.0, 1.0 + 2.0 ** -7, 1.0 - 2.0 ** -8, 448.0, 6.0, 3.0e-5, 2.0e4):
        row = torch.linspace(-1, 1, 128, device=device) * top
        row[5] = top
        rows.append(row)
    return torch.stack(rows).to(torch.bfloat16)


def mode_kernels(args) -> dict:
    from mstar.model.deepseek_v41 import _testing as T

    device = torch.device(args.engine_device)
    real, fake = _real_kernels(args.reference), T.reference_kernels()
    torch.set_default_dtype(torch.bfloat16)
    g = torch.Generator(device=device).manual_seed(0)
    rows: dict[str, dict] = {}

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, device=device) * scale).to(torch.bfloat16)

    def bits(name: str, got, want) -> None:
        got = [t.contiguous().view(torch.uint8) if t.dtype.itemsize == 1 else t for t in got]
        want = [t.contiguous().view(torch.uint8) if t.dtype.itemsize == 1 else t for t in want]
        rows[name] = {"equal": all(torch.equal(a, b) for a, b in zip(got, want, strict=True)),
                      "differing": int(sum((a != b).sum() for a, b in zip(got, want, strict=True)))}

    def close(name: str, got, want) -> None:
        err = (got.float() - want.float()).abs() / _spacing(want)
        rows[name] = {"max_ulp": err.max().item(), "mean_ulp": err.mean().item(),
                      "finite": bool(torch.isfinite(got.float()).all())}

    spread = torch.exp(torch.linspace(-12, 8, 96, device=device))[:, None]
    inputs = {"random": (randn(96, 640) * spread).to(torch.bfloat16), "ties": _tie_inputs(device)}
    for label, x in inputs.items():
        for block in (32, 128):
            bits(f"act_quant/{label}/b{block}/pow2", real.act_quant(x.clone(), block, "ue8m0", T.E8M0),
                 fake.act_quant(x.clone(), block, "ue8m0", T.E8M0))
            bits(f"act_quant/{label}/b{block}/pow2/inplace",
                 [real.act_quant(x.clone(), block, "ue8m0", T.E8M0, True)],
                 [fake.act_quant(x.clone(), block, "ue8m0", T.E8M0, True)])
        bits(f"fp4_act_quant/{label}/b32/e8m0", real.fp4_act_quant(x.clone(), 32), fake.fp4_act_quant(x.clone(), 32))
        bits(f"fp4_act_quant/{label}/b32/e8m0/inplace", [real.fp4_act_quant(x.clone(), 32, True)],
             [fake.fp4_act_quant(x.clone(), 32, True)])
        bits(f"fp4_act_quant/{label}/b16/e4m3/inplace",
             [real.fp4_act_quant(x.clone(), 16, True, scale_dtype=T.FP8)],
             [fake.fp4_act_quant(x.clone(), 16, True, scale_dtype=T.FP8)])

    # the GEMMs as Linear.forward calls them: activations quantized per 32, E8M0 scales
    m, k, n = 67, 640, 96
    a, a_s = fake.act_quant(randn(m, k), 32, "ue8m0", T.E8M0)
    w8 = T._fp8_pair(torch.randn(n, k, generator=g, device=device) * k ** -0.5, 32, 32)
    w4 = T._fp4_pair(torch.randn(n, k, generator=g, device=device) * k ** -0.5, 32)
    close("fp8_gemm", real.fp8_gemm(a, a_s, w8["weight"], w8["scale"], T.E8M0, 32),
          fake.fp8_gemm(a, a_s, w8["weight"], w8["scale"], T.E8M0, 32))
    b4 = w4["weight"].view(T.FP4X2)
    close("fp4_gemm", real.fp4_gemm(a, a_s, b4, w4["scale"], T.E8M0, 32),
          fake.fp4_gemm(a, a_s, b4, w4["scale"], T.E8M0, 32))

    for heads, dim, keys, picked in ((16, 64, 300, 136), (64, 512, 2000, 640), (4, 32, 40, 16)):
        q, kv = randn(2, 9, heads, dim), randn(2, keys, dim)
        sink = torch.randn(heads, generator=g, device=device)
        idx = torch.randint(0, keys, (2, 9, picked), generator=g, device=device, dtype=torch.int32)
        idx[:, :, -picked // 5:] = -1
        idx[0, 0] = -1  # a row that attends nothing
        got = real.sparse_attn(q, kv, sink, idx, dim ** -0.5)
        close(f"sparse_attn/h{heads}d{dim}k{picked}", got, fake.sparse_attn(q, kv, sink, idx, dim ** -0.5))

    hc = 4
    mixes = torch.randn(2, 33, (2 + hc) * hc, generator=g, device=device)
    scale = 1 + 0.2 * torch.randn(3, generator=g, device=device)
    base = torch.randn((2 + hc) * hc, generator=g, device=device)
    for name, got, want in zip(("pre", "post", "comb"), real.hc_split_sinkhorn(mixes, scale, base, hc, 20, 1e-6),
                               fake.hc_split_sinkhorn(mixes, scale, base, hc, 20, 1e-6), strict=True):
        rows[f"hc_split_sinkhorn/{name}"] = {"max_abs": (got.float() - want.float()).abs().max().item()}
    return rows


def _reference_run(ref, snapshot, config, tokenizer, ids, prompt, dtype, device) -> torch.Tensor:
    from mstar.model.deepseek_v41 import _testing as T

    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    try:
        with tempfile.TemporaryDirectory() as work:
            model = T.reference_model(ref, snapshot, config, tokenizer, work)
        ids = ids.to(device)
        out = [model(ids[None, :prompt], 0)[1][0]]
        for i in range(prompt, ids.numel()):
            out.append(model(ids[None, i:i + 1], i)[1][0])
        return torch.stack(out).float().cpu()
    finally:
        torch.set_default_dtype(default)
        torch.set_default_device("cpu")


def _exact_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
    rows = kv[batch, topk_idxs.clamp_min(0).long()].float()
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), rows) * softmax_scale
    scores = scores.masked_fill(~(topk_idxs != -1)[:, :, None, :], -torch.inf)
    sink = attn_sink.float()[None, None, :, None].expand(*scores.shape[:3], 1)
    prob = torch.cat([scores, sink], dim=-1).softmax(-1)[..., :-1]
    return torch.einsum("bmhk,bmkd->bmhd", prob, rows).to(q.dtype)


def _engine_run(snapshot, ids, prompt, dtype, device, **build) -> torch.Tensor:
    from mstar.engine.resources.kv import manager as kv_manager
    from mstar.model.deepseek_v41 import _testing as T

    kv_manager.KVTransferManager = T.StubTransfer  # one engine, nothing to transfer
    engine, sampler, _ = T.build_engine(snapshot, dtype, device=device, **build)
    return T.run_request(engine, sampler, ids.to(device), prompt).float().cpu()


def mode_tp1(args) -> dict:
    from mstar.model.deepseek_v41 import _testing as T
    from mstar.model.deepseek_v41.config import DeepseekV41Config

    device = args.engine_device
    rows: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        config, tokenizer = T.write_snapshot(tmp, DeepseekV41Config.reduced(), seed=0)
        ids = T.random_ids(config, PROMPT + STEPS, seed=1)
        standin = T.load_reference(args.reference)
        want32 = _reference_run(standin, tmp, config, tokenizer, ids, PROMPT, torch.float32, device)
        got32 = _engine_run(tmp, ids, PROMPT, torch.float32, device)
        rows["fp32/mstar-vs-reference(stand-ins)"] = {
            "gap": T.logit_gap(got32, want32), "argmax_equal": bool(torch.equal(got32.argmax(-1), want32.argmax(-1)))}
        if not args.no_kernels:
            real = T.load_reference(args.reference, _real_kernels(args.reference))
            want16 = _reference_run(real, tmp, config, tokenizer, ids, PROMPT, torch.bfloat16, device)
            same16 = _reference_run(standin, tmp, config, tokenizer, ids, PROMPT, torch.bfloat16, device)
            real.model.sparse_attn = _exact_sparse_attn
            other16 = _reference_run(real, tmp, config, tokenizer, ids, PROMPT, torch.bfloat16, device)
            got16 = _engine_run(tmp, ids, PROMPT, torch.bfloat16, device)
            runs = (("reference(stand-ins)", same16), ("reference(exact softmax)", other16), ("mstar", got16))
            for name, got in runs:
                rows[f"bf16/{name}-vs-reference(kernels)"] = {
                    "gap": T.logit_gap(got, want16),
                    "argmax_equal_steps": int((got.argmax(-1) == want16.argmax(-1)).sum()), "steps": got.shape[0]}
        torch.save({"fp32": got32, "ids": ids}, args.out + ".pt")
    return rows


def _tp2_rank(rank: int, args, port: int, snapshot: str) -> None:
    from mstar.distributed.communication import CommGroup, WorkerParallelGroups
    from mstar.model.deepseek_v41 import _testing as T
    from mstar.model.deepseek_v41.config import DeepseekV41Config

    config = DeepseekV41Config.reduced(**TP2)
    ids = T.random_ids(config, PROMPT + STEPS, seed=1)
    if not args.engine_device.startswith("cuda"):
        group, groups = T.gloo_tp_rank(rank, 2, f"{snapshot}.store")
        try:
            torch.save(_engine_run(snapshot, ids, PROMPT, torch.float32, "cpu", tp_group=group, parallel_groups=groups),
                       f"{args.out}.rank{rank}")
        finally:
            T.close_gloo_tp_rank(group)
        return
    group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
    groups = WorkerParallelGroups(num_workers=2, global_rank=rank, any_parallelism=True,
                                  world_parallel_groups=[(0, 1)], dist_timeout_s=600)
    groups.add("LLM", group, "tp")
    groups.node_to_instance_comm["LLM"] = group
    groups.init_dist(init_method=f"tcp://127.0.0.1:{port}", device=torch.device("cuda", rank))
    import os
    import traceback

    code = 0
    try:
        out = {dtype: _engine_run(snapshot, ids, PROMPT, getattr(torch, dtype), f"cuda:{rank}", tp_group=group,
                                  parallel_groups=groups) for dtype in ("float32", "bfloat16")}
        torch.save(out, f"{args.out}.rank{rank}")
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # no NCCL teardown: ranks that have saved their results can hang in it


def mode_tp2(args) -> dict:
    from mstar.model.deepseek_v41 import _testing as T
    from mstar.model.deepseek_v41.config import DeepseekV41Config

    cuda = args.engine_device.startswith("cuda")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    rows: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = str(Path(tmp) / "snap")
        config, _ = T.write_snapshot(snapshot, DeepseekV41Config.reduced(**TP2), seed=0)
        ids = T.random_ids(config, PROMPT + STEPS, seed=1)
        mp.spawn(_tp2_rank, args=(args, port, snapshot), nprocs=2)
        r0, r1 = (torch.load(f"{args.out}.rank{rank}") for rank in (0, 1))
        if not cuda:
            r0, r1 = {"float32": r0}, {"float32": r1}
        for dtype, got in r0.items():
            one = _engine_run(snapshot, ids, PROMPT, getattr(torch, dtype), "cuda:0" if cuda else "cpu")
            rows[f"{dtype}/tp2-vs-tp1"] = {
                "ranks_equal": bool(torch.equal(got, r1[dtype])), "gap": T.logit_gap(got, one),
                "argmax_equal_steps": int((got.argmax(-1) == one.argmax(-1)).sum()), "steps": got.shape[0]}
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="a snapshot directory holding inference/")
    ap.add_argument("--mode", choices=("kernels", "tp1", "tp2"), required=True)
    ap.add_argument("--engine-device", default="cuda:0")
    ap.add_argument("--no-kernels", action="store_true", help="tp1 without tilelang: the fp32 row only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = {"kernels": mode_kernels, "tp1": mode_tp1, "tp2": mode_tp2}[args.mode](args)
    meta = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": args.engine_device,
            "gpu": torch.cuda.get_device_name(0) if args.engine_device.startswith("cuda") else None}
    Path(args.out).write_text(json.dumps({"mode": args.mode, "meta": meta, "rows": rows}, indent=1))
    for name, row in rows.items():
        print(f"{name}: {json.dumps(row)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

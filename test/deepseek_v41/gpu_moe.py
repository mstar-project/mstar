"""DeepSeek-V4.1's routed experts on the grouped GEMM kernel against the torch dispatch, on a GPU.
Not a CI test: it needs a GPU with Triton (and, for one ``engine`` row, the checkpoint repo's
``inference/`` code).

    python test/deepseek_v41/gpu_moe.py --mode parity --out parity.json
    python test/deepseek_v41/gpu_moe.py --mode engine --reference <snapshot dir> --out engine.json
    python test/deepseek_v41/gpu_moe.py --mode bench --shapes tp8 --out bench.json

Shapes: ``reduced`` (the reduced model's block), ``full`` (hidden 5120, 384 experts, top 6,
2304 intermediate dims: one rank), ``tp8`` (a TP8 rank's 288-wide slice of every expert) and
``ep8`` (a TP8 rank's 48 whole experts). Experts are random MXFP4 in the checkpoint's packing;
inputs are unit-RMS rows whose dims' gains lie e-folds apart, so their 32-wide blocks round on
scales binades apart.

``parity``: the block on the kernel (``moe_kernel: auto``) against the block on the dispatch
(``torch``), in units of the result's bf16 spacing; the routed sum alone; on the experts a
step hit most, each GEMM alone against ``fp32_linear`` (and both against the exact sum) and
the rounded activation the down GEMM reads against the dispatch's; with
``--reference-kernels``, both GEMMs against the reference's own ``fp4_gemm`` kernel; in fp32
what the tensor-core dot keeps of activations on and off the FP8 grid; two controls that must
fail (the routing weight on the down GEMM's output, no rounding before it); and the block
recorded in a CUDA graph and replayed on other rows.
``engine``: the reduced model through the ``Engine``, kernel against dispatch in fp32 and bf16,
and in fp32 against the reference on its torch stand-ins.
``bench``: ms a call (median, p10, p90, a synchronize around each) of the dispatch, the grouped
path, its launches and its torch steps alone, the shared expert, and as a speed reference only
(other numerics) ``fused_experts_fp8`` on E4M3 experts of the same shape.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch

PROMPT, STEPS = 45, 14
BOUND = 2e-5  # the fp32 gate of the reference parity, in units of the result's spread
TOKENS = (1, 8, 32, 64, 256, 1024, 8192)
# shape -> (ranks of the node group, its expert split); the block is rank 0's
SHAPES = {"reduced": (1, "tp"), "full": (1, "tp"), "tp8": (8, "tp"), "ep8": (8, "ep")}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


class Rows:
    """Results by name: printed as they land and written to ``--out`` each time, so a run cut
    short keeps what it finished."""

    def __init__(self, args, mode: str) -> None:
        self.out, self.rows, self.started = args.out, {}, time.monotonic()
        self.budget = args.seconds
        device = torch.device(args.device)
        self.meta = {"mode": mode, "device": args.device, "torch": torch.__version__, "cuda": torch.version.cuda,
                     "loadavg": os.getloadavg(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            import triton

            self.meta["triton"] = triton.__version__
        except ImportError:
            pass
        if device.type == "cuda":
            self.meta["gpu"] = torch.cuda.get_device_name(device)
            self.meta["capability"] = torch.cuda.get_device_capability(device)
            query = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            self.meta["driver"] = subprocess.run(
                query, capture_output=True, text=True, check=False).stdout.split("\n")[0]
        print(f"meta: {json.dumps(self.meta)}", flush=True)

    def emit(self, name: str, row: dict) -> None:
        row = {**row, "at_s": round(time.monotonic() - self.started, 1)}
        self.rows[name] = row
        print(f"{name}: {json.dumps(row)}", flush=True)
        Path(self.out).write_text(json.dumps({"meta": self.meta, "rows": self.rows}, indent=1))

    def left(self) -> float:
        """Seconds of the run's budget left."""
        return self.budget - (time.monotonic() - self.started)


def _spacing(x: torch.Tensor) -> torch.Tensor:
    """The gap between adjacent bf16 values at each element of ``x``."""
    return torch.ldexp(torch.ones_like(x.float()), torch.frexp(x.float().abs().clamp_min(1e-30))[1] - 8)


def _compare(got: torch.Tensor, want: torch.Tensor) -> dict:
    """``got`` against ``want``: bit equality, and the error in ``want``'s bf16 spacings and in
    its spread (``gap``, the logit-gap unit, is the largest error; ``rms`` the root mean square)."""
    err = (got.float() - want.float()).abs()
    ulp = err / _spacing(want)
    spread = want.float().std()
    return {"equal": bool(torch.equal(got, want)), "differing": int((got != want).sum()), "of": want.numel(),
            "max_ulp": ulp.max().item(), "mean_ulp": ulp.mean().item(), "gap": (err.max() / spread).item(),
            "rms": (err.square().mean().sqrt() / spread).item(), "finite": bool(torch.isfinite(got.float()).all())}


def _inputs(tokens: int, hidden: int, dtype, device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    gains = torch.exp(torch.randn(hidden, generator=g, device=device))
    x = torch.randn(tokens, hidden, generator=g, device=device) * gains
    return (2 * x * torch.rsqrt(x.square().mean(-1, keepdim=True))).to(dtype)


def _block(shape: str, dtype, device, kernel: str = "auto", like=None):
    """One MoE block of ``shape`` on ``device`` as rank 0 of its group (its partial output: no
    other rank runs), with random weights in the checkpoint's formats, or ``like``'s."""
    from mstar.distributed.communication import CommGroup
    from mstar.model.components.moe_parallel import MoeParallelConfig
    from mstar.model.deepseek_v41 import _testing as T
    from mstar.model.deepseek_v41.components.moe import DeepseekV41MoE
    from mstar.model.deepseek_v41.config import DeepseekV41Config
    from mstar.model.deepseek_v41.deepseek_v41_model import _cast_floating
    from mstar.utils.mx_round import dequant_block_fp8

    tp, parallel = SHAPES[shape]
    config = DeepseekV41Config.reduced(moe_kernel=kernel) if shape == "reduced" else DeepseekV41Config(
        moe_kernel=kernel)
    group = CommGroup(my_global_rank=0, my_group_rank=0, group_members=list(range(tp)))
    with torch.device(device):
        ffn = _cast_floating(DeepseekV41MoE(config, group, MoeParallelConfig(parallel=parallel)), dtype)
    ffn.comm_group.all_reduce = lambda tensor: tensor
    if like is not None:
        for p, q in zip(ffn.parameters(), like.parameters(), strict=True):
            p.data = q.data if p.dtype == q.dtype else q.data.to(p.dtype)
    else:
        g = torch.Generator(device=device).manual_seed(0)

        def randn(*shape_, gain):
            return torch.randn(*shape_, generator=g, device=device) * (gain * shape_[-1] ** -0.5)

        def fp4(packed, scale, gain):
            experts, rows, cols = packed.shape[0], packed.shape[1], packed.shape[2] * 2
            step = max(1, (1 << 26) // (rows * cols))  # a chunk of experts: 256 MB of fp32
            for start in range(0, experts, step):
                pair = T._fp4_pair(randn(min(step, experts - start), rows, cols, gain=gain), 32)
                packed.data[start:start + step] = pair["weight"].view(torch.uint8)
                scale.data[start:start + step] = pair["scale"].view(torch.uint8)

        def fp8(param, gain):
            pair = T._fp8_pair(randn(*param.shape, gain=gain), 32, 32)
            param.data = dequant_block_fp8(pair["weight"], pair["scale"], param.dtype)

        fp4(ffn.experts.gate_up_proj_packed, ffn.experts.gate_up_proj_scale, 2.0)
        fp4(ffn.experts.down_proj_packed, ffn.experts.down_proj_scale, 1.0)
        fp8(ffn.shared.gate_up_proj, 2.0)
        fp8(ffn.shared.down_proj, 1.0)
        ffn.gate.weight.data = randn(*ffn.gate.weight.shape, gain=2.0).to(ffn.gate.weight.dtype)
        ffn.gate.e_score_correction_bias.data = 0.1 * torch.randn(config.n_routed_experts, generator=g, device=device)
    ffn.process_weights_after_loading(torch.device(device))
    if kernel == "auto" and not ffn.capturable:
        raise SystemExit(f"the grouped GEMM kernel does not run on {device}")
    return ffn.eval()


def _slots(ffn, weights: torch.Tensor, ids: torch.Tensor):
    """The router's ``(weights, ids)`` as the grouped path hands them to the kernel: a token's
    experts by ascending id, another rank's as the id past this rank's last."""
    ids, order = ids.sort(dim=1)
    weights = weights.gather(1, order)
    if ffn.expert_parallel:
        local = ids - ffn.expert_start
        ids = torch.where((local >= 0) & (local < ffn.local_experts), local, ffn.local_experts)
    return weights, ids


def _routed(ffn, tokens: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, control: str | None = None,
            seen: dict | None = None) -> torch.Tensor:
    """``DeepseekV41MoE._grouped`` (one pass) step by step, keeping its intermediates in ``seen``,
    or with one of its steps changed: ``control`` "weight-on-down-output" scales the down GEMM's
    output by the routing weight instead of the activation before its rounding, "no-rounding"
    feeds the down GEMM the activation unrounded."""
    import torch.nn.functional as F

    from mstar.utils.fused_moe.w4a16 import grouped_experts_mxfp4

    exp = ffn.experts
    weights, ids = _slots(ffn, weights, ids)

    def between(gate_up: torch.Tensor) -> torch.Tensor:
        if control == "weight-on-down-output":
            act = ffn._activation(gate_up)
        elif control == "no-rounding":
            gate, up = gate_up.float().chunk(2, dim=-1)
            limit = ffn.swiglu_limit
            act = (weights.reshape(-1, 1) * (F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit))).to(gate_up.dtype)
        else:
            act = ffn._activation(gate_up, weights.reshape(-1, 1))
        if seen is not None:
            seen.update(gate_up=gate_up, act=act, ids=ids, weights=weights)
        return act

    slots = grouped_experts_mxfp4(tokens, exp.gate_up_proj_packed, exp.down_proj_packed, exp.gate_up_proj_scale,
                                  exp.down_proj_scale, ids, between, skip_invalid=ffn.expert_parallel)
    if seen is not None:
        seen["slots"] = slots
    wide = slots.float()
    if control == "weight-on-down-output":
        wide = (wide * weights[:, :, None]).to(tokens.dtype).float()
    total = wide[:, 0]
    for slot in range(1, wide.shape[1]):
        total = total + wide[:, slot]
    return total


def _gemm_rows(rows: Rows, name: str, ffn, tokens: torch.Tensor, seen: dict, experts: int) -> None:
    """On the ``experts`` most-hit experts, step by step against the dispatch's arithmetic for
    the same rows: each GEMM alone against ``fp32_linear`` on the dequantized expert (and both
    against the exact sum rounded once), and the rounded activation the down GEMM reads."""
    from mstar.model.deepseek_v41.components.linear import fp32_linear

    ids, top_k, dtype = seen["ids"].flatten(), seen["ids"].shape[1], tokens.dtype
    counts = torch.bincount(ids, minlength=ffn.local_experts + 1)[:ffn.local_experts]
    slots_out, weights = seen["slots"].reshape(ids.numel(), -1), seen["weights"].reshape(-1, 1)
    parts = {"gate-up": [], "down-input": [], "down": []}
    for expert in counts.topk(min(experts, counts.numel()))[1].tolist():
        slot = (ids == expert).nonzero().flatten()
        if not slot.numel():
            continue
        gate_up, down = ffn._expert_weights(expert, dtype)
        a = tokens[slot // top_k]
        torch_gate_up = fp32_linear(a, gate_up)
        parts["gate-up"].append((seen["gate_up"][slot], torch_gate_up, a.double() @ gate_up.double().T))
        act = seen["act"][slot]
        parts["down-input"].append((act, ffn._activation(torch_gate_up, weights[slot]), None))
        parts["down"].append((slots_out[slot], fp32_linear(act, down), act.double() @ down.double().T))
    for part, triples in parts.items():
        got, want = (torch.cat([t[i] for t in triples]) for i in (0, 1))
        row = {**_compare(got, want), "rows": got.shape[0], "experts": len(triples)}
        if part == "down-input":
            row["rows_differing"] = int((got != want).any(-1).sum())
            rows.emit(f"{name}/{part}(rounded-activation)-vs-dispatch", row)
            continue
        exact = torch.cat([t[2] for t in triples])
        once = exact.float().to(dtype)
        row.update(kernel_is_exact_sum=(got == once).float().mean().item(),
                   torch_is_exact_sum=(want == once).float().mean().item())
        if dtype == torch.float32:
            # in units of the summed magnitudes' fp32 spacing: a sum's own may be tiny where it cancels
            row["kernel_max_rel_err"] = ((got.double() - exact).abs() / exact.abs().max()).max().item()
            row["torch_max_rel_err"] = ((want.double() - exact).abs() / exact.abs().max()).max().item()
        rows.emit(f"{name}/{part}-gemm-vs-fp32_linear", row)


def _reference_rows(rows: Rows, name: str, real, ffn, x: torch.Tensor, seen: dict) -> None:
    """Each GEMM of the most-hit expert as the checkpoint's reference computes it, its own
    ``act_quant`` and ``fp4_gemm`` kernels on the expert's bytes (bf16 out): the grouped
    kernel's rows and ``fp32_linear``'s against it."""
    from mstar.model.deepseek_v41 import _testing as T
    from mstar.model.deepseek_v41.components.linear import fp32_linear

    ids, top_k = seen["ids"].flatten(), seen["ids"].shape[1]
    expert = int(torch.bincount(ids, minlength=ffn.local_experts + 1)[:ffn.local_experts].argmax())
    slot = (ids == expert).nonzero().flatten()
    exp, weights = ffn.experts, ffn._expert_weights(expert, x.dtype)
    gemms = (
        ("gate-up", x[slot // top_k], exp.gate_up_proj_packed, exp.gate_up_proj_scale, weights[0],
         seen["gate_up"][slot]),
        ("down", seen["act"][slot], exp.down_proj_packed, exp.down_proj_scale, weights[1],
         seen["slots"].reshape(ids.numel(), -1)[slot]),
    )
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        for gemm, a, packed, scale, dense, got in gemms:
            q, a_scale = real.act_quant(a.clone(), ffn.block, "ue8m0", T.E8M0)
            want = real.fp4_gemm(q, a_scale, packed[expert].view(T.FP4X2).contiguous(),
                                 scale[expert].view(T.E8M0).contiguous(), T.E8M0, ffn.block)
            rounded = (q.float().unflatten(-1, (-1, ffn.block)) * a_scale.float().unsqueeze(-1)).flatten(-2)
            rows.emit(f"{name}/{gemm}-gemm-vs-reference-kernel", {
                "rows": got.shape[0], "kernel": _compare(got, want),
                "fp32_linear": _compare(fp32_linear(rounded.to(x.dtype), dense), want)})
    except Exception as error:  # noqa: BLE001 - a shape the reference's kernel refuses is a result
        rows.emit(f"{name}/gemm-vs-reference-kernel", {"error": f"{type(error).__name__}: {str(error)[-300:]}"})
    finally:
        torch.set_default_dtype(default)


def _dot_precision(rows: Rows, name: str, ffn, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> None:
    """In fp32, the gate/up GEMM's error against the exact sum, in units of the summed
    magnitudes: activations on the FP8 grid (4 significant bits) against raw fp32 ones. A
    tensor-core dot that keeps fewer bits than fp32 shows on the second only."""
    from mstar.utils.mx_round import round_fp8

    row = {}
    for label, a in (("fp8_grid", round_fp8(x, ffn.block)), ("raw_fp32", x)):
        seen: dict = {}
        _routed(ffn, a, weights, ids, seen=seen)
        flat = seen["ids"].flatten()
        expert = int(torch.bincount(flat, minlength=ffn.local_experts + 1)[:ffn.local_experts].argmax())
        slot = (flat == expert).nonzero().flatten()
        weight = ffn._expert_weights(expert, torch.float32)[0].double()
        mine = a[slot // seen["ids"].shape[1]].double()
        err = (seen["gate_up"][slot].double() - mine @ weight.T).abs() / (mine.abs() @ weight.abs().T)
        row[f"{label}_max_rel_err"] = err.max().item()
    rows.emit(f"{name}/fp32-dot-precision", row)


def _capture(rows: Rows, name: str, ffn, tokens: int, dtype, device) -> None:
    """The block recorded in a CUDA graph on one step's rows and replayed on others (other
    routing): each replay against the eager block."""
    xs = [_inputs(tokens, ffn.hidden_size, dtype, device, seed) for seed in (21, 22, 23)]
    try:
        static = xs[0].clone()
        ffn(static)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = ffn(static)
        equal = []
        for x in xs[1:]:
            static.copy_(x)
            graph.replay()
            equal.append(bool(torch.equal(out, ffn(x))))
        rows.emit(f"{name}/captured-replay-vs-eager", {"equal": equal, "tokens": tokens})
    except Exception as error:  # noqa: BLE001 - a capture that fails is the result
        import traceback

        frames = [f for f in traceback.extract_tb(error.__traceback__) if "/mstar/" in f.filename]
        rows.emit(f"{name}/captured-replay-vs-eager", {
            "error": f"{type(error).__name__}: {str(error)[-300:]}",
            "raised_under": [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in frames[-3:]]})


def _fp8_scale_row(rows: Rows, device) -> None:
    """``quant_fp8``'s scales on this device against the product spelled with an fp32 tensor:
    block maxima around every power-of-two edge over 40 binades, and random rows."""
    from mstar.utils.mx_round import pow2_ceil, quant_fp8

    g = torch.Generator(device=device).manual_seed(4)
    edge = 448.0 * torch.exp2(torch.arange(-20.0, 20.0, device=device))
    amax = torch.cat([edge, torch.nextafter(edge, edge * 0), torch.nextafter(edge, edge * 2)])
    x = torch.rand(4096, amax.numel(), 32, generator=g, device=device) * amax[None, :, None]
    x[0, :, 5] = amax
    x = torch.cat([x.flatten(0, 1), torch.zeros(1, 32, device=device)])
    want = pow2_ceil(x.abs().amax(-1).clamp_min(1e-4) * torch.tensor(1 / 448.0, dtype=torch.float32, device=device))
    rows.emit("fp8-scale-is-amax-times-fp32-reciprocal", {
        "equal": bool(torch.equal(quant_fp8(x, 32)[1][:, 0], want)), "blocks": x.shape[0]})


@torch.inference_mode()
def mode_parity(args, rows: Rows) -> None:
    from mstar.utils.mx_round import round_fp8

    device = torch.device(args.device)
    _fp8_scale_row(rows, device)
    real = None
    if args.reference_kernels:
        sys.path.insert(0, str(Path(__file__).parent))
        import gpu_parity

        real = gpu_parity._real_kernels(args.reference)
    for shape in args.shapes:
        first = None
        for dtype_name in args.dtypes:
            dtype = DTYPES[dtype_name]
            kernel = _block(shape, dtype, device, "auto", like=first)
            first = kernel if first is None else first
            dispatch = _block(shape, dtype, device, "torch", like=kernel)
            for tokens in args.tokens or (8, 512):
                name = f"{shape}/{dtype_name}/t{tokens}"
                x = _inputs(tokens, kernel.hidden_size, dtype, device, seed=tokens)
                rows.emit(f"{name}/block", _compare(kernel(x), dispatch(x)))
                weights, ids, _ = kernel.gate(x)
                rounded = round_fp8(x, kernel.block)
                want = dispatch._dispatch(rounded, weights, ids)
                got = kernel._grouped(rounded, weights, ids)
                seen: dict = {}
                replica = _routed(kernel, rounded, weights, ids, seen=seen)
                routed = _compare(got, want)
                rows.emit(f"{name}/routed-sum", {**routed, "replica_is_the_path": bool(torch.equal(replica, got))})
                for control in ("weight-on-down-output", "no-rounding"):
                    off = _compare(_routed(kernel, rounded, weights, ids, control=control), want)
                    rows.emit(f"{name}/control/{control}", {
                        "gap": off["gap"], "rms": off["rms"], "path_gap": routed["gap"], "path_rms": routed["rms"],
                        "fails": off["gap"] > 10 * max(BOUND, routed["gap"])})
                _gemm_rows(rows, name, kernel, rounded, seen, args.experts)
                if real is not None and dtype == torch.bfloat16:
                    _reference_rows(rows, name, real, kernel, x, seen)
                if dtype == torch.float32:
                    _dot_precision(rows, name, kernel, x, weights, ids)
            if device.type == "cuda":
                _capture(rows, f"{shape}/{dtype_name}", kernel, 8, dtype, device)
            del dispatch
        del first, kernel
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _engine_run(snapshot, ids, dtype, device, **build):
    from mstar.engine.resources.kv import manager as kv_manager
    from mstar.model.deepseek_v41 import _testing as T

    kv_manager.KVTransferManager = T.StubTransfer  # one engine, nothing to transfer
    engine, sampler, model = T.build_engine(snapshot, dtype, device=device, **build)
    layers = model.get_submodule("LLM").language_model.model.layers
    return T.run_request(engine, sampler, ids.to(device), PROMPT).float().cpu(), all(
        layer.ffn.capturable for layer in layers)


def mode_engine(args, rows: Rows) -> None:
    from mstar.model.deepseek_v41 import _testing as T
    from mstar.model.deepseek_v41.config import DeepseekV41Config

    def gap(got, want) -> dict:
        return {"gap": T.logit_gap(got, want), "equal": bool(torch.equal(got, want)),
                "argmax_equal_steps": int((got.argmax(-1) == want.argmax(-1)).sum()), "steps": got.shape[0]}

    for widths, knobs in (("reduced", {}), ("reduced-wide", T.KERNEL_WIDTHS)):
        with tempfile.TemporaryDirectory() as tmp:
            config, tokenizer = T.write_snapshot(tmp, DeepseekV41Config.reduced(**knobs), seed=0)
            ids = T.random_ids(config, PROMPT + STEPS, seed=1)
            for dtype_name in args.dtypes:
                dtype = DTYPES[dtype_name]
                got, on_kernel = _engine_run(tmp, ids, dtype, args.device)
                want, _ = _engine_run(tmp, ids, dtype, args.device, moe_kernel="torch")
                row = {**gap(got, want), "every_layer_capturable": on_kernel}
                if dtype == torch.float32:
                    row["within_bound"] = row["gap"] <= BOUND
                rows.emit(f"{widths}/{dtype_name}/kernel-vs-dispatch", row)
                if dtype == torch.float32 and args.reference and widths == "reduced":
                    sys.path.insert(0, str(Path(__file__).parent))
                    import gpu_parity

                    reference = gpu_parity._reference_run(
                        T.load_reference(args.reference), tmp, config, tokenizer, ids, PROMPT, dtype, args.device)
                    for label, logits in (("kernel", got), ("dispatch", want)):
                        row = gap(logits, reference)
                        rows.emit(f"{widths}/fp32/{label}-vs-reference(stand-ins)", {
                            **row, "within_bound": row["gap"] <= BOUND})


def _time(fn, sync, calls: int, seconds: float, warm: int = 2) -> dict:
    """``fn``'s wall time a call in ms after ``warm`` calls: up to ``calls`` of them, at least 3,
    stopping once ``seconds`` have passed."""
    for _ in range(warm):
        fn()
    sync()
    ms, stop = [], time.perf_counter() + seconds
    while len(ms) < calls and (len(ms) < 3 or time.perf_counter() < stop):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        ms.append((time.perf_counter() - start) * 1e3)
    ms = torch.tensor(ms, dtype=torch.float64)
    return {"ms": ms.median().item(), "p10": ms.quantile(0.1).item(), "p90": ms.quantile(0.9).item(), "calls": len(ms)}


def _peak_mib(fn, device) -> float:
    """MiB ``fn`` allocates at its peak, over what was held before it."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    held = torch.cuda.memory_allocated(device)
    fn()
    torch.cuda.synchronize(device)
    return (torch.cuda.max_memory_allocated(device) - held) / 2 ** 20


def _fp8_experts(ffn, device):
    """E4M3 experts of ``ffn``'s shape in ``fused_experts_fp8``'s layout (32 x 32 scale blocks),
    random: a speed reference, not ``ffn``'s values."""
    g = torch.Generator(device=device).manual_seed(5)
    experts, hidden, inter = ffn.local_experts, ffn.hidden_size, ffn.expert_inter

    def weight(rows, cols):
        out = torch.empty(experts, rows, cols, dtype=torch.float8_e4m3fn, device=device)
        step = max(1, (1 << 26) // (rows * cols))
        for start in range(0, experts, step):
            n = min(step, experts - start)
            out[start:start + n] = (64 * torch.randn(n, rows, cols, generator=g, device=device)).to(out.dtype)
        return out, torch.full((experts, rows // 32, cols // 32), cols ** -0.5 / 64, device=device)

    return (*weight(2 * inter, hidden), *weight(hidden, inter))


def _sum_slots(slots: torch.Tensor) -> torch.Tensor:
    wide = slots.float()
    total = wide[:, 0]
    for slot in range(1, wide.shape[1]):
        total = total + wide[:, slot]
    return total


def _bench_step(args, rows: Rows, name: str, kernel, fp8, x: torch.Tensor, sync) -> tuple:
    """Every item of one step's rows but the dispatch; the step's rounded rows and routing."""
    from mstar.model.deepseek_v41.components import moe
    from mstar.utils.fused_moe import w4a16
    from mstar.utils.fused_moe.align import moe_align_block_size
    from mstar.utils.mx_round import round_fp8

    def timed(item: str, fn) -> None:
        if rows.left() < 5:
            rows.emit(f"{name}/{item}", {"skipped": "out of the run's seconds"})
        else:
            rows.emit(f"{name}/{item}", _time(fn, sync, args.calls, seconds=2.0))

    exp, hidden, top_k, dtype, device = kernel.experts, kernel.hidden_size, kernel.gate.top_k, x.dtype, x.device
    weights, ids, _ = kernel.gate(x)
    rounded = round_fp8(x, kernel.block)
    timed("grouped", lambda: kernel._grouped(rounded, weights, ids))
    timed("block(grouped)", lambda: kernel(x))
    # one pass of the grouped path, step by step
    part = slice(0, moe.GROUPED_ROWS)
    a, (w, i) = rounded[part], _slots(kernel, weights[part], ids[part])
    n, inter = a.shape[0], kernel.expert_inter
    config = w4a16.moe_config(n, kernel.local_experts, 2 * inter, hidden, top_k, 32)
    alignment = moe_align_block_size(i, config["BLOCK_SIZE_M"], kernel.local_experts)
    gate_up = torch.zeros(n * top_k, 2 * inter, dtype=dtype, device=device)
    slots = torch.zeros(n, top_k, hidden, dtype=dtype, device=device)
    compute = w4a16._compute_type(dtype)

    def gemm(src, packed, scale, dst, k) -> None:
        w4a16._launch(src, packed, scale, None, dst, src, *alignment, i, False, k, config, compute,
                      w4a16.KIND_E2M1, 32)

    gemm(a, exp.gate_up_proj_packed, exp.gate_up_proj_scale, gate_up, top_k)
    act = kernel._activation(gate_up, w.reshape(-1, 1))
    rows.emit(f"{name}/pass", {"rows": n, "passes": -(-x.shape[0] // moe.GROUPED_ROWS), "tiles": config})
    timed("pass/route-sort", lambda: _slots(kernel, weights[part], ids[part]))
    timed("pass/align", lambda: moe_align_block_size(i, config["BLOCK_SIZE_M"], kernel.local_experts))
    timed("pass/gate-up-gemm", lambda: gemm(a, exp.gate_up_proj_packed, exp.gate_up_proj_scale, gate_up, top_k))
    timed("pass/activation", lambda: kernel._activation(gate_up, w.reshape(-1, 1)))
    timed("pass/down-gemm", lambda: gemm(act, exp.down_proj_packed, exp.down_proj_scale, slots.view(-1, hidden), 1))
    timed("pass/slot-sum", lambda: _sum_slots(slots))
    timed("round-input", lambda: round_fp8(x, kernel.block))
    timed("router", lambda: kernel.gate(x))
    timed("shared-expert", lambda: kernel._expert(rounded, kernel.shared.gate_up_proj, kernel.shared.down_proj))
    if fp8 is not None:
        from mstar.utils.fused_moe import fused_experts_fp8

        w1, s1, w2, s2 = fp8
        timed("fused_experts_fp8(reference)", lambda: fused_experts_fp8(
            x, w1, w2, s1, s2, weights, ids, block_size=(32, 32), swiglu_limit=10.0))
    if device.type == "cuda":
        rows.emit(f"{name}/peak-mib", {"grouped": _peak_mib(lambda: kernel._grouped(rounded, weights, ids), device)})
    return rounded, weights, ids


def _bench_dispatch(args, rows: Rows, name: str, dispatch, step: tuple, sync) -> None:
    if rows.left() < 5:
        rows.emit(f"{name}/dispatch", {"skipped": "out of the run's seconds"})
        return
    row = _time(lambda: dispatch._dispatch(*step), sync, args.calls, seconds=args.dispatch_seconds, warm=1)
    device = step[0].device
    if device.type == "cuda":
        row["peak_mib"] = _peak_mib(lambda: dispatch._dispatch(*step), device)
    rows.emit(f"{name}/dispatch", row)


@torch.inference_mode()
def mode_bench(args, rows: Rows) -> None:
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    dtype = DTYPES[args.dtypes[-1]]

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize(device)

    for shape in args.shapes:
        kernel = _block(shape, dtype, device, "auto")
        dispatch = _block(shape, dtype, device, "torch", like=kernel)
        fp8 = None
        if cuda and args.fp8 and dtype == torch.bfloat16 and not kernel.expert_parallel:
            fp8 = _fp8_experts(kernel, device)
        steps = {}
        for tokens in args.tokens or TOKENS:
            x = _inputs(tokens, kernel.hidden_size, dtype, device, seed=tokens)
            steps[f"{shape}/t{tokens}"] = _bench_step(args, rows, f"{shape}/t{tokens}", kernel, fp8, x, sync)
        # the dispatch last, small steps first: it takes seconds a call at the checkpoint's shapes
        for name, step in steps.items():
            _bench_dispatch(args, rows, name, dispatch, step, sync)
        del kernel, dispatch, fp8, steps
        if cuda:
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    def names(known):
        return lambda text: [n if n in known else ap.error(f"{n!r} is not one of {sorted(known)}")
                             for n in text.split(",")]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("parity", "engine", "bench"), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shapes", type=names(SHAPES), default=["reduced", "full", "tp8", "ep8"])
    ap.add_argument("--dtypes", type=names(DTYPES), default=["fp32", "bf16"], help="bench times the last")
    ap.add_argument("--tokens", type=lambda text: [int(n) for n in text.split(",")], default=None,
                    help="rows a step: parity 8,512; bench 1,8,32,64,256,1024,8192")
    ap.add_argument("--experts", type=int, default=48, help="parity: experts whose GEMMs are compared one by one")
    ap.add_argument("--calls", type=int, default=50, help="bench: timed calls an item")
    ap.add_argument("--dispatch-seconds", type=float, default=3.0,
                    help="bench: the dispatch stops after these seconds (3 calls at least)")
    ap.add_argument("--no-fp8", dest="fp8", action="store_false", help="bench without fused_experts_fp8")
    ap.add_argument("--seconds", type=float, default=1e9, help="bench: skip what is left after these seconds")
    ap.add_argument("--reference", default=os.environ.get("DSV41_REFERENCE_DIR"),
                    help="a snapshot directory holding inference/: engine's reference row, parity's kernels")
    ap.add_argument("--reference-kernels", action="store_true",
                    help="parity, bf16: each GEMM against the reference's own kernel too (needs tilelang)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = Rows(args, args.mode)
    {"parity": mode_parity, "engine": mode_engine, "bench": mode_bench}[args.mode](args, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

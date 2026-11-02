"""The Ulysses exchange inside one model block, at a sequence-parallel rank's shapes.

Random weights at the checkpoint's sizes, under ``torchrun`` with one process per SP
rank. Each round runs the block through each exchange on the same inputs, back to
back: ``general``, the batch exchange (``CommGroup.all_to_all``: q, k and v gathered
into contiguous tensors, the output sent back through its own buffer), and ``packed``,
``ulysses_attention`` itself. It checks the two give the same output bit for bit, then
times the exchanges alone. Attention runs where the server runs it: LTX-2.5 on the
engine's ragged resources, Cosmos3 on the dense FA3 path of an eager video step (a
frozen text prefix per guidance branch, as ``DenseAttentionManager`` gathers it).

    torchrun --nproc-per-node 2 -m benchmark.sequence_parallel.exchange_bench --model ltx2_5 --compile
    torchrun --nproc-per-node 4 -m benchmark.sequence_parallel.exchange_bench --model cosmos3 \\
        --shape 832x480x189 --compile --out /tmp/cosmos3_sp4.json

Block times are the mean of ``--iters`` back-to-back calls after ``--warm``, the slowest
rank's, once per round; the arms swap order every round.

``--probe`` (one process) asks each FlashInfer ragged backend instead whether its kernel
reads q, k and v views of one buffer exactly as contiguous copies, the condition for
adding it to ``STRIDED_QKV_BACKENDS``:

    python -m benchmark.sequence_parallel.exchange_bench --probe fa2,fa3,cudnn,cutlass

``--attention ROWSxHEADSxKV_HEADS`` (one process) times the attention kernels alone on
such views against contiguous copies, alternately, at one rank's post-exchange shapes:
what reading the views costs the kernel itself.

    python -m benchmark.sequence_parallel.exchange_bench --attention 37721x14x14 --kernels fa3,fa2,dense
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import json
import os
import statistics
import time
from collections.abc import Callable
from types import ModuleType

import torch
import torch.distributed as dist

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.sequence_parallel import sp_seq_split, ulysses_attention


def general_ulysses_attention(sp_group, q, k, v, run_attention, seq_sizes, seq_dim=0, head_dim=1):
    """The batch exchange, for any layout: the reference arm."""
    q, k, v = sp_group.all_to_all((q, k, v), scatter_dim=head_dim, gather_dim=seq_dim, gather_sizes=seq_sizes)
    out = run_attention(q=q, k=k, v=v)
    return sp_group.all_to_all(out, scatter_dim=seq_dim, gather_dim=head_dim, scatter_sizes=seq_sizes)


ARMS = {"general": general_ulysses_attention, "packed": ulysses_attention}


@dataclasses.dataclass
class Case:
    """One block on this rank's inputs, and what the arms swap."""

    call: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]]
    patched: list[ModuleType]          # modules whose ``ulysses_attention`` the arm replaces
    qkv: tuple[torch.Size, torch.Size]  # this rank's q and k/v shapes, for the exchanges alone
    layout: tuple[int, int]            # (seq_dim, head_dim)
    sizes: list[int]
    notes: dict


@contextlib.contextmanager
def exchange(arm: str, modules: list[ModuleType]):
    saved = [m.ulysses_attention for m in modules]
    for m in modules:
        m.ulysses_attention = ARMS[arm]
    try:
        yield
    finally:
        for m, fn in zip(modules, saved, strict=True):
            m.ulysses_attention = fn


def randomize(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0, 0.02)


def ltx2_5_case(group: CommGroup, args) -> Case:
    """The 1-block DiT (its embeddings, the block, the output head), as ``LTX2DiT`` runs
    under SP: the video tokens sharded, the video self-attention exchanged."""
    from benchmark.ltx2_5.profile_step import ragged_attends
    from mstar.model.components.diffusion.compile_utils import compile_transformer_forward
    from mstar.model.ltx2_5.components import layers
    from mstar.model.ltx2_5.components.transformer import LTX2DiT, build_rope
    from mstar.model.ltx2_5.config import LTX25Config
    from mstar.model.ltx2_5.submodules import shape_from_metadata

    width, height, frames = args.shape or (960, 544, 121)
    config = LTX25Config()
    shape = shape_from_metadata(config, {"height": height, "width": width, "num_frames": frames, "fps": 24.0})
    cfg = dataclasses.replace(config.transformer, num_layers=args.layers)
    config = dataclasses.replace(config, transformer=cfg)
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.device(device):
        dit = LTX2DiT(cfg, sp_group=group).to(torch.bfloat16)
    randomize(dit)
    if args.compile:
        compile_transformer_forward(dit)
    rope = build_rope(cfg, shape.frames, shape.height, shape.width, shape.fps, shape.audio_frames, device)
    attends = ragged_attends(config, shape, args.batch, device, 1, group)
    b = args.batch
    gen = torch.Generator(device).manual_seed(0)

    def randn(*size):
        return torch.randn(*size, device=device, dtype=torch.bfloat16, generator=gen)

    inputs = (
        randn(b, shape.video_tokens, cfg.in_channels), randn(b, shape.audio_frames, cfg.audio_in_channels),
        randn(b, shape.text_len, cfg.cross_attention_dim), randn(b, shape.text_len, cfg.audio_cross_attention_dim),
        torch.full((b,), 900.0, device=device),
    )
    sizes = sp_seq_split(shape.video_tokens, group.world_size)
    heads, head_dim = cfg.num_attention_heads, cfg.attention_head_dim
    local = torch.Size((b, sizes[group.rank], heads, head_dim))
    return Case(
        call=lambda: dit(*inputs, rope, attends), patched=[layers], qkv=(local, local), layout=(1, 2),
        sizes=sizes, notes={"video_tokens": shape.video_tokens, "audio_tokens": shape.audio_frames,
                            "text_tokens": shape.text_len, "layers": args.layers},
    )


class DenseGenAttend:
    """``DenseAttentionManager.run`` for an eager denoise step, one segment per guidance
    branch: each branch's frozen text prefix, then its fresh tokens, one varlen call."""

    def __init__(self, heads: int, kv_heads: int, head_dim: int, prefix: int, q_lens: list[int], device):
        from mstar.engine.resources.attn.dense import _fa3_unavailable_reason
        from mstar.engine.resources.attn.ragged.base import dense_or_flashinfer
        from mstar.engine.resources.attn.ragged.dense_wrappers import DensePrefixedPrefillWrapper
        from mstar.engine.resources.attn.ragged.wrappers import FlashInferPrefixedPrefillWrapper

        cls = dense_or_flashinfer(device, DensePrefixedPrefillWrapper, FlashInferPrefixedPrefillWrapper, "bench")
        extra = {} if cls is DensePrefixedPrefillWrapper else {
            "workspace_buffer": torch.empty(256 << 20, dtype=torch.uint8, device=device)}
        self.kernel = "fa3" if _fa3_unavailable_reason(device) is None else "flashinfer"
        self.wrapper = cls(num_qo_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim, device=device, **extra)
        cu_q, cu_k = [0], [0]
        for n in q_lens:
            cu_q.append(cu_q[-1] + n)
            cu_k.append(cu_k[-1] + prefix + n)
        self.wrapper.plan(torch.tensor(cu_q, dtype=torch.int32), torch.tensor(cu_k, dtype=torch.int32))
        gen = torch.Generator(device).manual_seed(1)
        self.prefixes = [
            tuple(torch.randn(prefix, kv_heads, head_dim, device=device, dtype=torch.bfloat16, generator=gen)
                  for _ in range(2))
            for _ in q_lens
        ]
        self.q_lens = q_lens

    @torch.compiler.disable
    def __call__(self, q, k, v):
        k_parts, v_parts, offset = [], [], 0
        for (k_pref, v_pref), n in zip(self.prefixes, self.q_lens, strict=True):
            k_parts += [k_pref, k[offset:offset + n]]
            v_parts += [v_pref, v[offset:offset + n]]
            offset += n
        return self.wrapper.run(q, torch.cat(k_parts), torch.cat(v_parts))


def cosmos3_case(group: CommGroup, args) -> Case:
    """One generation layer (``Cosmos3MoTDecoderLayer.forward_gen``) of Cosmos3-Nano over a
    video step's packed ``[cond | uncond]`` tokens, sharded as ``_sp_run_gen_layers`` does."""
    from mstar.model.cosmos3.components import transformer
    from mstar.model.cosmos3.components.transformer import Cosmos3MoTDecoderLayer, norm_class_for
    from mstar.model.cosmos3.config import Cosmos3Config
    from mstar.model.cosmos3.submodules import denoise_compile_kwargs, frame_tokens, latent_frames

    config = Cosmos3Config()
    width, height, frames = args.shape or (832, 480, 189)
    branch = frame_tokens(height, width, config) * latent_frames(frames, config)
    q_lens = [branch] * args.branches
    rows = sum(q_lens)
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.device(device):
        layer = Cosmos3MoTDecoderLayer(
            hidden_size=config.hidden_size, head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads, num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size, attention_bias=config.attention_bias,
            rms_norm_eps=config.rms_norm_eps, sp_group=group, hidden_act=config.hidden_act,
            qk_norm_for_text=config.qk_norm_for_text, use_und_k_norm_for_gen=config.use_und_k_norm_for_gen,
            norm_cls=norm_class_for(config),
        ).to(torch.bfloat16)
    randomize(layer)
    world = group.world_size
    heads, kv_heads, head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    attend = DenseGenAttend(heads // world, kv_heads // world, head_dim, args.prefix, q_lens, device)
    forward = layer.forward_gen
    if args.compile:
        forward = torch.compile(forward, **denoise_compile_kwargs(args.cosmos3_compile_mode))
    sizes = sp_seq_split(rows, world)
    gen = torch.Generator(device).manual_seed(0)
    n = sizes[group.rank]
    x = torch.randn(n, config.hidden_size, device=device, dtype=torch.bfloat16, generator=gen)
    cos, sin = (torch.rand(n, head_dim, device=device, generator=gen).mul_(2).sub_(1).to(torch.bfloat16)
                for _ in range(2))
    return Case(
        call=lambda: forward(x, cos, sin, attend, sizes), patched=[transformer],
        qkv=(torch.Size((n, heads, head_dim)), torch.Size((n, kv_heads, head_dim))), layout=(0, 1),
        sizes=sizes, notes={"rows": rows, "branches": args.branches, "prefix": args.prefix,
                            "attention_kernel": attend.kernel},
    )


CASES = {"ltx2_5": ltx2_5_case, "cosmos3": cosmos3_case}


def rank_max(value: float) -> float:
    t = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def time_ms(fn: Callable[[], object], iters: int, warm: int) -> float:
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
    return rank_max(start.elapsed_time(end) / iters)


def outputs(value) -> list[torch.Tensor]:
    return [value] if isinstance(value, torch.Tensor) else [t for t in value if isinstance(t, torch.Tensor)]


def top_kernels(fn: Callable[[], object], n: int = 10) -> list[tuple[str, float]]:
    """Device ms per call of the ``n`` heaviest kernels, over 3 calls; every rank runs
    them so the collectives pair up."""
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
    rows = sorted(prof.key_averages(), key=lambda e: e.device_time_total, reverse=True)
    return [(e.key[:100], round(e.device_time_total / 3e3, 3)) for e in rows[:n]]


def flashinfer_version() -> str | None:
    try:
        import flashinfer
    except Exception:  # noqa: BLE001
        return None
    return flashinfer.__version__


def probe(backends: list[str], rows: int = 9431, heads: int = 14, head_dim: int = 128) -> dict[str, str]:
    """Each backend's raw kernel on q/k/v views of one fused buffer against contiguous
    copies, bypassing the wrapper's gate; both also against FA2 on the copies, so a
    backend that reads the views wrong tells from one that only rounds differently."""
    from mstar.engine.resources.attn.ragged.wrappers import FlashInferRaggedPrefillWrapper

    device = torch.device("cuda")
    gen = torch.Generator(device).manual_seed(0)
    q, k, v = torch.randn(rows, 3 * heads, head_dim, device=device, dtype=torch.bfloat16,
                          generator=gen).split(heads, dim=1)
    workspace = torch.empty(512 << 20, dtype=torch.uint8, device=device)
    layout = torch.tensor([0, rows // 3, rows], dtype=torch.int32)
    reference = FlashInferRaggedPrefillWrapper(workspace, heads, heads, head_dim, backend="fa2", device=device)
    reference.plan(layout)
    want = reference.attn_wrapper.run(q.contiguous(), k.contiguous(), v.contiguous()).float()

    def off(t):
        return f"{(t.float() - want).abs().max().item():.4g}"

    found = {}
    for backend in backends:
        try:
            wrapper = FlashInferRaggedPrefillWrapper(workspace, heads, heads, head_dim, backend=backend, device=device)
            wrapper.plan(layout)
            raw = wrapper.attn_wrapper
            views = raw.run(q, k, v)
            dense = raw.run(q.contiguous(), k.contiguous(), v.contiguous())
            again = raw.run(q.contiguous(), k.contiguous(), v.contiguous())
            torch.cuda.synchronize()
            same = "views exact" if torch.equal(views, dense) else "views differ"
            control = "" if torch.equal(dense, again) else ", copies not repeatable"
            found[backend] = (f"{raw._backend}: {same}{control}; max abs off FA2: copies {off(dense)}, "
                              f"views {off(views)}")
        except Exception as exc:  # noqa: BLE001 -- a backend this GPU or build lacks is reported
            found[backend] = f"error: {type(exc).__name__}: {str(exc)[:200]}"
        print(f"{backend:10s} {found[backend]}", flush=True)
    return found


def attention_alone(shape: tuple[int, int, int], kernels: list[str], rounds: int, iters: int, warm: int,
                    head_dim: int = 128) -> dict[str, dict[str, list[float]]]:
    """Each kernel's ms on q/k/v views of one fused buffer and on contiguous copies, the
    two timed alternately (A B, B A, ...); ``dense`` is the FA3 varlen wrapper."""
    from mstar.engine.resources.attn.ragged.dense_wrappers import DenseRaggedPrefillWrapper
    from mstar.engine.resources.attn.ragged.wrappers import FlashInferRaggedPrefillWrapper

    rows, heads, kv_heads = shape
    device = torch.device("cuda")
    gen = torch.Generator(device).manual_seed(0)
    views = torch.randn(rows, heads + 2 * kv_heads, head_dim, device=device, dtype=torch.bfloat16,
                        generator=gen).split([heads, kv_heads, kv_heads], dim=1)
    copies = [t.contiguous() for t in views]
    workspace = torch.empty(512 << 20, dtype=torch.uint8, device=device)
    layout = torch.tensor([0, rows], dtype=torch.int32)

    def timed(fn) -> float:
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return round(start.elapsed_time(end) / iters, 4)

    found = {}
    for kernel in kernels:
        if kernel == "dense":
            run = DenseRaggedPrefillWrapper(heads, kv_heads, head_dim, device=device)
            run.plan(layout)
            call = run.run
        else:
            wrapper = FlashInferRaggedPrefillWrapper(
                workspace, heads, kv_heads, head_dim, backend=kernel, device=device,
            )
            wrapper.plan(layout)
            call = wrapper.attn_wrapper.run
        times = {"views": [], "copies": []}
        for r in range(rounds):
            for name in (("views", "copies") if r % 2 == 0 else ("copies", "views")):
                qkv = views if name == "views" else copies
                times[name].append(timed(lambda qkv=qkv, call=call: call(*qkv)))
        found[kernel] = times
        print(f"{kernel:6s} {shape}: views {statistics.mean(times['views']):8.4f} ms {times['views']}  "
              f"copies {statistics.mean(times['copies']):8.4f} ms {times['copies']}", flush=True)
    return found


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", default=None, help="comma-separated FlashInfer backends to probe; see the docstring")
    p.add_argument("--attention", default=None, type=lambda s: tuple(int(x) for x in s.split("x")),
                   help="ROWSxHEADSxKV_HEADS: time the attention kernels alone; see the docstring")
    p.add_argument("--kernels", default="fa3,fa2,dense", help="with --attention: FlashInfer backends, or dense")
    p.add_argument("--model", choices=sorted(CASES))
    p.add_argument("--shape", type=lambda s: tuple(int(x) for x in s.split("x")), default=None,
                   help="WIDTHxHEIGHTxFRAMES of the request (default: the model's)")
    p.add_argument("--batch", type=int, default=1, help="LTX-2.5: requests in the step")
    p.add_argument("--layers", type=int, default=1, help="LTX-2.5: DiT blocks")
    p.add_argument("--branches", type=int, default=2, help="Cosmos3: guidance branches packed in the step")
    p.add_argument("--prefix", type=int, default=512, help="Cosmos3: text prefix tokens per branch")
    p.add_argument("--cosmos3-compile-mode", default="deterministic")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warm", type=int, default=20)
    p.add_argument("--profile", action="store_true", help="print each arm's heaviest kernels")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.attention:
        found = attention_alone(args.attention, args.kernels.split(","), args.rounds, args.iters, args.warm)
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"device": torch.cuda.get_device_name(), "flashinfer": flashinfer_version(),
                           "shape": args.attention, "ms": found}, f, indent=1)
        return
    if args.probe:
        found = probe(args.probe.split(","))
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"device": torch.cuda.get_device_name(), "flashinfer": flashinfer_version(),
                           "probe": found}, f, indent=1)
        return
    if args.model is None:
        p.error("--model is required without --probe")

    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    group = CommGroup(rank, rank, list(range(world)))
    group.device_group, group.initialized = dist.group.WORLD, True
    torch.manual_seed(0)
    case = CASES[args.model](group, args)

    report = {
        "model": args.model, "world": world, "args": vars(args) | {"shape": args.shape},
        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "flashinfer": flashinfer_version(),
        "notes": case.notes, "sizes": case.sizes, "block_ms": {arm: [] for arm in ARMS},
    }
    t0 = time.time()
    results = {}
    for arm in ARMS:  # compiles each arm's graphs before anything is timed
        with exchange(arm, case.patched), torch.no_grad():
            results[arm] = [t.clone() for t in outputs(case.call())]
    torch.cuda.synchronize()
    report["first_calls_s"] = round(time.time() - t0, 1)
    diffs = [rank_max((a.float() - b.float()).abs().max().item() if a.numel() else 0.0)
             for a, b in zip(results["general"], results["packed"], strict=True)]
    report["max_abs_diff"] = max(diffs)
    report["bit_identical"] = bool(rank_max(0.0 if all(
        torch.equal(a, b) for a, b in zip(results["general"], results["packed"], strict=True)) else 1.0) == 0.0)
    for r in range(args.rounds):  # A B, B A, ...: a clock that drifts within a round favours neither arm
        for arm in (list(ARMS) if r % 2 == 0 else list(ARMS)[::-1]):
            with exchange(arm, case.patched):
                report["block_ms"][arm].append(round(time_ms(case.call, args.iters, args.warm), 3))

    seq_dim, head_dim = case.layout
    q_shape, kv_shape = case.qkv
    gen = torch.Generator("cuda").manual_seed(2 + rank)
    q = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16, generator=gen)
    k, v = (torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16, generator=gen) for _ in range(2))

    def fresh(q, k, v):  # an attention's output: a new contiguous tensor
        return torch.empty(q.shape, device=q.device, dtype=q.dtype)

    report["exchange_ms"] = {arm: [] for arm in ARMS}
    for r in range(args.rounds):
        for arm in (list(ARMS) if r % 2 == 0 else list(ARMS)[::-1]):
            call = functools.partial(ARMS[arm], group, q, k, v, fresh, case.sizes, seq_dim, head_dim)
            report["exchange_ms"][arm].append(round(time_ms(call, args.iters * 5, args.warm), 3))
    if args.profile:
        report["top_kernels"] = {}
        for arm in ARMS:
            with exchange(arm, case.patched):
                report["top_kernels"][arm] = top_kernels(case.call)
    if rank == 0:
        for arm in ARMS:
            blocks, ex = report["block_ms"][arm], report["exchange_ms"][arm]
            print(f"{args.model} {arm:8s} block {statistics.mean(blocks):8.3f} ms {blocks}  "
                  f"exchanges alone {statistics.mean(ex):7.3f} ms {ex}", flush=True)
        print(f"bit identical: {report['bit_identical']} (max abs diff {report['max_abs_diff']})", flush=True)
        for arm, kernels in report.get("top_kernels", {}).items():
            print(f"top kernels, {arm}:")
            for name, ms in kernels:
                print(f"  {ms:8.3f} ms  {name}")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

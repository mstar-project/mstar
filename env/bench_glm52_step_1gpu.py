#!/usr/bin/env python3
"""One-GPU GLM-5.2 MTP decode step at the REAL TP8 per-rank shapes.

The 753B config needs 8 GPUs and ~35 minutes to reach its first decode step,
so every trunk-graph kernel experiment costs half an hour. But TP8 shards the
model: each rank runs 8 attention heads, a 256-per-rank-intermediate MoE and a
19360-row lm_head — shapes that fit on ONE H200 as long as you keep the layer
COUNT down instead of the layer WIDTH. This script builds exactly those
per-rank dims with tp_size=1, captures the same piecewise graphs the engine
captures (``mtp_trunk``, ``mtp_draft``, ``mtp_sync``, ``mtp_draft_phase``,
via ``build_piecewise_runners`` — the real ``PiecewiseCudaGraphRunner``), and
times decode steps through the captured path.

Every kernel therefore runs at the shape it runs at in production. What is
NOT here, and what the printed table repeats so no number leaves without it:

- **no TP collectives** — o_proj's RowParallel all-reduce, the MoE block's
  all-reduce and the lm_head all-gather are all identity at tp_size=1. A TP8
  rank pays ~76 all-reduces per decode step on top of what this measures.
- **fewer layers** — ``--layers N`` (default 8) instead of 78. The trunk is
  the only phase that scales with N; the MTP draft plane is ONE layer in
  production too, so it does not. Both scalings are printed.
- **random weights** — timing only. Acceptance is ~0, which does NOT move the
  step time: the sync pass and draft chain are padded to k+1 rows per request
  by construction (``mtp_sync_padded_layout``), so a step costs the same
  whether 0 or k drafts are accepted.
- **vocab is the per-rank shard** (19360, not 154880), so the lm_head GEMM
  matches a rank but the verify argmax reads 8x fewer columns than the
  post-all-gather argmax a real rank runs.

Usage (one GPU, minutes):

    python env/bench_glm52_step_1gpu.py --layers 8 --steps 30

Output: per-step ms (captured and eager), the per-phase GPU|host split from
the engine's own ``_MtpStepTimer``, a torch.profiler kernel summary of 3
captured steps via ``env/kernel_trace_summary.py``, and a 78-layer estimate.
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

# Indexed on purpose: PiecewiseCudaGraphRunner.warmup_and_capture calls
# torch.cuda.set_device(self.device), which rejects a bare "cuda". With
# CUDA_VISIBLE_DEVICES set, cuda:0 IS the requested physical GPU.
DEVICE = torch.device("cuda:0")

# --- production per-rank geometry at TP8 (configs/glm52_tp8_mtp_fast.yaml) ---
TP = 8
FULL_LAYERS = 78
HIDDEN = 6144
FULL_VOCAB = 154880
FULL_HEADS = 64
FULL_INTERMEDIATE = 12288      # dense-layer MLP
FULL_MOE_INTERMEDIATE = 2048   # per routed expert


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, default=8,
                   help="trunk layers to build (production: 78). Memory is "
                        "~1.3 GB/MoE layer at these dims. Default 8.")
    p.add_argument("--dense", type=int, default=1,
                   help="how many of --layers are DENSE MLP layers "
                        "(production: 3 of 78). Default 1.")
    p.add_argument("--k", type=int, default=3,
                   help="mtp_num_draft_tokens; k+1 rows per trunk step "
                        "(production default 3). Default 3.")
    p.add_argument("--bs", type=int, default=1, help="batch size. Default 1.")
    p.add_argument("--ctx", type=int, default=1024,
                   help="prompt length, i.e. the KV context the decode steps "
                        "attend over. Must stay under index_topk=2048.")
    p.add_argument("--steps", type=int, default=30, help="timed decode steps.")
    p.add_argument("--warmup", type=int, default=5, help="untimed decode steps.")
    p.add_argument("--trace-steps", type=int, default=3,
                   help="captured steps inside the torch.profiler window.")
    p.add_argument("--eager-steps", type=int, default=5,
                   help="timed UNCAPTURED steps for comparison (0 to skip).")
    p.add_argument("--pages", type=int, default=128,
                   help="KV pages (page_size 128) in the pool.")
    p.add_argument("--out-dir", default=os.environ.get("TMPDIR") or "/tmp",
                   help="where the chrome trace lands. Default $TMPDIR.")
    p.add_argument("--top", type=int, default=15,
                   help="rows in the kernel-summary tables.")
    p.add_argument("--no-compile", action="store_true",
                   help="MSTAR_GLM52_GRAPH_COMPILE=0 — capture the eager "
                        "forward instead of the torch.compile'd one.")
    p.add_argument("--capture-prefill", action="store_true",
                   help="also capture the MTP prefill trunk (30 buckets in "
                        "production; off here — it does not touch the decode "
                        "step this benches, and capture is slow).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------- config ----

def indexer_offset_for(num_layers: int) -> int:
    """``index_skip_topk_offset`` that keeps the MTP layer FULL.

    ``Glm52MTPModule.__init__`` refuses to construct unless layer index
    ``num_hidden_layers`` is FULL under the IndexShare formula
    ``max(idx - offset + 1, 0) % freq == 0`` (freq=4). Production's offset=3
    works because 78 - 3 + 1 = 76 is a multiple of 4; an arbitrary --layers is
    not, so the offset moves instead. This is free: with dsa_long_context off
    the indexer NEVER runs (``_forward_absorbed`` passes dsa_ctx=None), so the
    offset only decides which dormant layers carry indexer weights — memory,
    not kernels.
    """
    return ((num_layers + 1) % 4) or 4


def build_config(args):
    from mstar.model.glm52.config import Glm52ModelConfig
    from mstar.model.glm52.quantization import Fp8BlockQuantConfig

    vocab = FULL_VOCAB // TP  # 19360 — the per-rank shard
    cfg = Glm52ModelConfig(
        # per-rank shard dims
        vocab_size=vocab,
        hidden_size=HIDDEN,
        num_hidden_layers=args.layers,
        num_attention_heads=FULL_HEADS // TP,           # 8
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        mla_absorb=True,                                 # production backend
        first_k_dense_replace=min(args.dense, args.layers),
        intermediate_size=FULL_INTERMEDIATE // TP,       # 1536
        moe_intermediate_size=FULL_MOE_INTERMEDIATE // TP,  # 256
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=8,
        # DSA indexer: dormant (dsa_long_context stays False) but its geometry
        # decides the MTP layer's FULL/SHARED position — see the helper.
        index_skip_topk_offset=indexer_offset_for(args.layers),
        index_topk=2048,
        dsa_long_context=False,
        # MTP + fp8, exactly as configs/glm52_tp8_mtp_fast.yaml asks for
        mtp_num_draft_tokens=args.k,
        moe_fp8_resident=True,
        moe_quant_kernel="triton",
        quantization_config=Fp8BlockQuantConfig(weight_block_size=(128, 128)),
        max_seq_len=2048,
        # token ids must live inside the SHARDED vocab
        eos_token_ids=(vocab - 4, vocab - 3, vocab - 2),
        pad_token_id=vocab - 1,
    )
    # One prefill bucket, one capture batch size: the full cross product is
    # 30 prefill captures, minutes of warmup for a graph this bench never
    # replays (decode is the subject).
    bucket = 1 << max(5, (args.ctx - 1).bit_length())
    cfg.prefill_token_buckets = [bucket]
    cfg.prefill_capture_batch_sizes = [args.bs]
    return cfg


# ----------------------------------------------------------------- model ----

def _fill_fp8_(param: torch.Tensor, chunk: int = 16, scale: float = 0.5) -> None:
    """Random e4m3 bytes in the uint8 container, generated per expert chunk.

    Materialising randn for all 256 experts at once would be a 3 GB fp32
    temporary; 16 at a time is ~200 MB. Going through the fp8 cast (rather
    than random BYTES) keeps the values finite — 0x7F/0xFF are e4m3 NaN.
    """
    for s in range(0, param.shape[0], chunk):
        e = min(s + chunk, param.shape[0])
        t = torch.randn(param[s:e].shape, device=param.device, dtype=torch.float32)
        param[s:e].copy_(t.mul_(scale).to(torch.float8_e4m3fn).view(torch.uint8))


def build_model(cfg, seed: int):
    """Production construction order (Glm52Model._create_submodule) with the
    checkpoint read replaced by random fill: meta -> bf16 -> to_empty ->
    fill -> restore fp32 params -> process_weights_after_loading."""
    from mstar.model.components.quantization import process_weights_after_loading
    from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
    from mstar.model.glm52.weight_loader import restore_fp32_params

    torch.manual_seed(seed)
    with torch.device("meta"):
        model = Glm52ForCausalLM(cfg)
    model = model.to(torch.bfloat16)      # free on meta; to_empty then allocates bf16
    model.to_empty(device=DEVICE)
    restore_fp32_params(model)            # router bias + block scales back to fp32

    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dtype == torch.uint8:            # fp8 expert bytes
                _fill_fp8_(p.data)
            elif name.endswith("_scale_inv"):     # fp32 block scales
                p.data.uniform_(0.004, 0.012)
            elif "norm" in name:
                p.data.fill_(1.0)
            elif p.dtype.is_floating_point:
                p.data.normal_(0, 0.02)
        for name, b in model.named_buffers():
            # w_kc/w_vc/fused_qkv_a_proj_weight are None until
            # process_weights_after_loading builds them; anything else that
            # survived to_empty is uninitialised memory.
            if b is not None and b.is_floating_point():
                b.zero_()
    process_weights_after_loading(model, DEVICE)
    model.eval()
    return model


# ------------------------------------------------------------------- kv ----

def make_kv(cfg, bs: int, pages: int, page_size: int = 128):
    """The mla_absorb cache the engine builds for GLM-5.2 (kv_cache_engine.py):
    a 4-D latent pool [layers, pages, page_size, 576], ONE extra layer plane
    for the MTP module, num_qo_heads = the per-rank head count."""
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.engine.cache_manager import WorkspaceBufferManager, create_cache_manager
    from mstar.engine.kv_store import (
        KVCacheConfig,
        PagedAllocationManager,
        TransferEngineInfo,
    )

    num_layers = cfg.num_hidden_layers + 1
    kv_cfg = KVCacheConfig(
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=cfg.cache_latent_dim,             # 512 + 64 = 576
        max_seq_len=cfg.max_seq_len,
        max_num_pages=pages,
        page_size=page_size,
        num_qo_heads=cfg.num_attention_heads,
        attention_backend="mla_absorb",
        softmax_scale=cfg.qk_head_dim ** -0.5,
        mla_ckv_dim=cfg.kv_lora_rank,
    )
    kv_cfg.shard(1)  # tp_world_size=1; mirrors KVCacheEngine's own call
    kv_cache = torch.zeros(
        num_layers, pages, page_size, kv_cfg.head_dim,
        dtype=torch.bfloat16, device=DEVICE,
    ).contiguous()
    alloc = PagedAllocationManager(
        config=kv_cfg, kv_cache=kv_cache,
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="glm52_bench", my_session_id="glm52_bench",
            transfer_engine=LocalTransferEngine("localhost"),
        ),
    )
    rids = [f"b{i}" for i in range(bs)]
    for rid in rids:
        alloc.add_request(rid, ["main"])
    buffers = WorkspaceBufferManager(
        int(os.environ.get("MSTAR_WORKSPACE_BUFFER_MB", "512")) * 1024 * 1024,
        device=DEVICE,
    )
    cm = create_cache_manager(
        request_ids=rids, active_labels_per_request={r: "main" for r in rids},
        kv_cache=kv_cache, alloc_manager=alloc, buffer_manager=buffers,
        kv_cache_config=kv_cfg, device=DEVICE,
    )
    return cm, alloc, buffers, kv_cfg, rids, kv_cache


def make_sampler(cfg, rids):
    from mstar.utils.sampling import Sampler

    s = Sampler(device=DEVICE)
    for rid in rids:
        s.add_request(rid)
        s.set_config(rid, vocab_size=cfg.vocab_size, temperature=0.0,
                     top_k=0, top_p=1.0, repetition_penalty=1.0)
    return s


def make_fwd_info(cfg, rid, max_tokens):
    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.utils.sampling import MultiSamplingConfig, SamplingConfig

    return CurrentForwardPassInfo(
        request_id=rid, graph_walk="decode", requires_cfg=False, fwd_index=0,
        random_seed=0, max_tokens=max_tokens,
        sampling_config={"LLM": MultiSamplingConfig(main=SamplingConfig(
            vocab_size=cfg.vocab_size, ignore_eos=True, temperature=0.0,
            repetition_penalty=1.0))},
        dynamic_loop_iter_counts={},
    )


# ------------------------------------------------------------------ step ----

def drain_phase_timer(sub) -> list[tuple[str, float, float]]:
    """Pull the last step's marks out of the submodule's own ``_MtpStepTimer``.

    ``MSTAR_GLM52_MTP_STEP_TIMING=1`` (set before the submodule is built) makes
    it record a CUDA event + host timestamp at every phase boundary. Its
    ``report()`` logs them at the START of the next step; draining here gets
    the same numbers as data and leaves report() a no-op.
    """
    timer = getattr(sub, "_mtp_timer", None)
    marks = getattr(timer, "_pending", None) if timer is not None else None
    if not marks:
        return []
    timer._pending = None
    marks[-1][1].synchronize()
    out = []
    for (_n0, e0, h0), (n1, e1, h1) in zip(marks, marks[1:]):
        out.append((n1, e0.elapsed_time(e1), (h1 - h0) * 1e3))
    return out


def decode_step(sub, ei, infos, rids, nxt):
    ars = [sub.prepare_inputs("decode", infos[rid], {"text_inputs": [nxt[rid]]})
           for rid in rids]
    packed = sub.preprocess("decode", ei, ars)
    with torch.no_grad():
        res = sub.forward_batched("decode", ei, **packed)
    for rid in rids:
        sub.postprocess(rid, infos[rid], res[rid])
        nxt[rid] = res[rid]["text_inputs"][0]
    return res


def run_arm(cfg, model, args, *, use_graphs: bool, steps: int, trace_path=None):
    """Prefill once, then warmup + `steps` timed decode steps on a fresh cache.

    ``use_graphs`` is the only difference between the two arms — exactly as in
    test_glm52_mtp_piecewise_gpu.py::_drive.
    """
    from mstar.engine.cuda_graph_runner import build_piecewise_runners
    from mstar.model.glm52.submodules import (
        MTP_DRAFT_BUNDLE,
        MTP_DRAFT_PHASE_LABEL,
        MTP_TRUNK_LABEL,
        Glm52LLMSubmodule,
    )
    from mstar.model.submodule_base import ModelInputsFromEngine

    sub = Glm52LLMSubmodule(model, cfg)
    # One capture bucket instead of production's [1, 2, 4, 8, 16]: the other
    # four cannot change what a bs=N step costs, and each costs capture time
    # and a FlashInfer workspace.
    sub.MTP_CAPTURE_BATCH_SIZES = [args.bs]
    cm, alloc, buffers, kv_cfg, rids, kv_cache = make_kv(cfg, args.bs, args.pages)
    sampler = make_sampler(cfg, rids)
    runners = {}
    out = {"capture_s": 0.0, "labels": []}
    try:
        if use_graphs:
            t0 = time.perf_counter()
            runners = build_piecewise_runners(
                sub, DEVICE, torch.bfloat16, tp_world_size=1,
                kv_cache_config=kv_cfg, alloc_manager=alloc,
                buffer_manager=buffers,
            )
            out["capture_s"] = time.perf_counter() - t0
            out["labels"] = sorted(runners)
            if MTP_TRUNK_LABEL not in runners:
                raise RuntimeError(
                    "the mtp_trunk graph did not capture — every number below "
                    "would be the eager path wearing a captured costume")
            if args.k >= 1 and MTP_DRAFT_PHASE_LABEL not in runners:
                raise RuntimeError("the mtp_draft_phase graph did not capture")

        ei = ModelInputsFromEngine(
            request_ids=rids, per_request_info={}, cache_manager=cm,
            sampler=sampler, piecewise_runners=runners,
        )
        infos = {rid: make_fwd_info(cfg, rid, 1 << 20) for rid in rids}
        prompts = [torch.arange(args.ctx, dtype=torch.long, device=DEVICE) + 3 + 5 * i
                   for i in range(args.bs)]

        ars = [sub.prepare_inputs("prefill", infos[rid], {"text_inputs": [p]})
               for rid, p in zip(rids, prompts, strict=True)]
        packed = sub.preprocess("prefill", ei, ars)
        t0 = time.perf_counter()
        with torch.no_grad():
            res = sub.forward_batched("prefill", ei, **packed)
        torch.cuda.synchronize()
        out["prefill_ms"] = (time.perf_counter() - t0) * 1e3
        nxt = {}
        for rid in rids:
            sub.postprocess(rid, infos[rid], res[rid])
            # Feed decode step 1 the prefill's [emitted, k drafts] bundle, which
            # is what the conductor's prefill-drafts edge delivers in production
            # (MSTAR_GLM52_MTP_PREFILL_DRAFTS, default on). postprocess alone
            # leaves text_inputs = the single emitted token, and a 1-row decode
            # step has no captured bucket — it would run the trunk EAGER once
            # and log a scary warning inside a benchmark.
            bundle = res[rid].get(MTP_DRAFT_BUNDLE)
            nxt[rid] = (bundle[0] if bundle is not None
                        else res[rid]["text_inputs"][0])

        for _ in range(args.warmup):
            decode_step(sub, ei, infos, rids, nxt)
        torch.cuda.synchronize()
        drain_phase_timer(sub)

        ev_ms, wall_ms = [], []
        phases: dict[str, list[tuple[float, float]]] = {}
        beg, end = (torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True))
        for _ in range(steps):
            t0 = time.perf_counter()
            beg.record()
            decode_step(sub, ei, infos, rids, nxt)
            end.record()
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - t0) * 1e3)
            ev_ms.append(beg.elapsed_time(end))
            for name, gpu, host in drain_phase_timer(sub):
                phases.setdefault(name, []).append((gpu, host))
        out["ev_ms"], out["wall_ms"] = ev_ms, wall_ms
        out["phases"] = {n: (statistics.mean(g for g, _ in v),
                             statistics.mean(h for _, h in v))
                         for n, v in phases.items()}
        out["phase_order"] = list(phases)
        out["accept"] = (sub._mtp_stat_emitted / max(sub._mtp_stat_steps, 1))
        out["fused_moe"] = sub._moe_resolved_fused()

        if trace_path is not None:
            from mstar.utils.profiler import StepKernelTrace

            os.environ["MSTAR_PROFILE_STEPS"] = f"0:{args.trace_steps}"
            os.environ["MSTAR_PROFILE_DIR"] = str(trace_path.parent)
            trace = StepKernelTrace(trace_path.stem.replace("step-trace-", ""),
                                    device=DEVICE)
            for _ in range(args.trace_steps):
                trace.before_execute()
                decode_step(sub, ei, infos, rids, nxt)
                trace.after_execute()
            os.environ.pop("MSTAR_PROFILE_STEPS", None)
        out["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        return out
    finally:
        alloc.cleanup()
        for rid in rids:
            sampler.remove_request(rid)
        del runners, cm, alloc, buffers, kv_cache, sub
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- report ----

def p50(xs):
    return statistics.median(xs)


def summarise_trace(trace_json: Path, top: int) -> list[str]:
    """Delegate to env/kernel_trace_summary.py — the summary logic lives there."""
    script = Path(__file__).with_name("kernel_trace_summary.py")
    if not script.exists():
        return [f"(kernel_trace_summary.py not found at {script})"]
    r = subprocess.run(
        [sys.executable, str(script), str(trace_json), "--top", str(top),
         "--launches"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return [f"(kernel_trace_summary.py failed: {r.stderr.strip()[:400]})"]
    return r.stdout.splitlines()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device visible", file=sys.stderr)
        return 2
    if args.dense > args.layers:
        print("--dense cannot exceed --layers", file=sys.stderr)
        return 2

    # Read at Glm52LLMSubmodule.__init__ / get_piecewise_cuda_graph_configs, so
    # they must be set before anything is constructed.
    os.environ["MSTAR_GLM52_MTP_STEP_TIMING"] = "1"
    os.environ["MSTAR_GLM52_MTP_CAPTURE_PREFILL"] = (
        "1" if args.capture_prefill else "0")
    if args.no_compile:
        os.environ["MSTAR_GLM52_GRAPH_COMPILE"] = "0"

    torch.cuda.set_device(DEVICE)
    name = torch.cuda.get_device_name(DEVICE)
    cap = torch.cuda.get_device_capability(DEVICE)

    cfg = build_config(args)
    from mstar.engine.cache_manager import _mla_kernel_available

    mla_kernel = _mla_kernel_available(cfg.kv_lora_rank, cfg.qk_rope_head_dim, cap[0])

    print(f"# GLM-5.2 MTP decode step, TP8 per-rank shapes on one {name} (sm{cap[0]}{cap[1]})")
    print(f"# layers {args.layers} ({cfg.num_dense_layers} dense + "
          f"{args.layers - cfg.num_dense_layers} MoE) + 1 MTP plane   "
          f"k={args.k}  bs={args.bs}  ctx={args.ctx}")
    print(f"# hidden {cfg.hidden_size}  heads {cfg.num_attention_heads}  "
          f"q_lora {cfg.q_lora_rank}  kv_lora {cfg.kv_lora_rank}  "
          f"nope/rope/v {cfg.qk_nope_head_dim}/{cfg.qk_rope_head_dim}/{cfg.v_head_dim}")
    print(f"# experts {cfg.n_routed_experts} top-{cfg.num_experts_per_tok} + "
          f"{cfg.n_shared_experts} shared, moe_inter {cfg.moe_intermediate_size}, "
          f"dense_inter {cfg.intermediate_size}, vocab {cfg.vocab_size}")
    print(f"# fp8 block {cfg.quantization_config.weight_block_size} "
          f"kernel={cfg.moe_quant_kernel}  mla_absorb={cfg.mla_absorb} "
          f"(flashinfer MLA kernel: {mla_kernel})  "
          f"graph_compile={os.environ.get('MSTAR_GLM52_GRAPH_COMPILE', '1')}")
    sys.stdout.flush()

    t0 = time.perf_counter()
    model = build_model(cfg, args.seed)
    build_s = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters())
    print(f"# model built in {build_s:.1f} s, {params / 1e9:.2f} B params, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB resident")
    sys.stdout.flush()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_json = out_dir / "step-trace-glm52bench.json"

    cap_arm = run_arm(cfg, model, args, use_graphs=True, steps=args.steps,
                      trace_path=trace_json)
    if not cap_arm["fused_moe"]:
        print("!! routed experts did NOT resolve to fused_experts_fp8 — the "
              "reference dispatch is 10x+ slower and is not what production "
              "runs", file=sys.stderr)
    eager_arm = None
    if args.eager_steps > 0:
        # This lane greps `running EAGER` as a correctness tripwire. The arm
        # below has no runners BY DESIGN, so it logs exactly that warning —
        # say so before it appears, or the benchmark's own log looks like a
        # failed capture.
        print("\n-- eager comparison arm follows; its 'running EAGER' warnings "
              "are the point, not a failure --", flush=True)
        eager_arm = run_arm(cfg, model, args, use_graphs=False,
                            steps=args.eager_steps)

    # ------------------------------------------------------------- tables --
    ev, wall = cap_arm["ev_ms"], cap_arm["wall_ms"]
    print()
    print(f"captured graphs : {', '.join(cap_arm['labels'])} "
          f"(captured in {cap_arm['capture_s']:.1f} s)")
    print(f"fused fp8 MoE   : {cap_arm['fused_moe']}   "
          f"accepted tokens/step {cap_arm['accept']:.2f} (random weights; "
          f"step cost is acceptance-independent — padded rows)")
    print(f"peak memory     : {cap_arm['peak_gib']:.1f} GiB")
    print()
    print("== per decode step, captured ==")
    print(f"  gpu events : mean {statistics.mean(ev):7.3f} ms   "
          f"p50 {p50(ev):7.3f}   min {min(ev):7.3f}   max {max(ev):7.3f}   "
          f"n={len(ev)}")
    print(f"  host wall  : mean {statistics.mean(wall):7.3f} ms   "
          f"p50 {p50(wall):7.3f}")
    if eager_arm:
        eev, ewall = eager_arm["ev_ms"], eager_arm["wall_ms"]
        print("== per decode step, EAGER (no captured graphs) ==")
        print(f"  gpu events : mean {statistics.mean(eev):7.3f} ms   "
              f"p50 {p50(eev):7.3f}   n={len(eev)}")
        print(f"  host wall  : mean {statistics.mean(ewall):7.3f} ms   "
              f"p50 {p50(ewall):7.3f}")
        print(f"  capture speedup: {statistics.mean(eev) / statistics.mean(ev):.2f}x")
        et = eager_arm["phases"].get("trunk")
        if et:
            print(f"  eager trunk phase: {et[0]:.3f} ms GPU / {et[1]:.3f} ms host")

    print()
    print("== phase split (engine _MtpStepTimer, mean over the timed steps) ==")
    print(f"  {'phase':<14}{'GPU ms':>9}{'host ms':>10}")
    trunk_gpu = 0.0
    for nm in cap_arm["phase_order"]:
        gpu, host = cap_arm["phases"][nm]
        if nm == "trunk":
            trunk_gpu = gpu
        print(f"  {nm:<14}{gpu:9.3f}{host:10.3f}")
    tot_gpu = sum(g for g, _ in cap_arm["phases"].values())
    print(f"  {'TOTAL':<14}{tot_gpu:9.3f}"
          f"{sum(h for _, h in cap_arm['phases'].values()):10.3f}")

    # ------------------------------------------------------- 78-layer est --
    scale = FULL_LAYERS / args.layers
    mean_ev = statistics.mean(ev)
    naive = mean_ev * scale
    # Anchor on the MEASURED step time, not on the timer's own total: only the
    # trunk grows with layer count.
    phase_aware = trunk_gpu * scale + max(mean_ev - trunk_gpu, 0.0)
    print()
    print(f"== 78-layer ESTIMATES (this run is {args.layers} layers; "
          f"x{scale:.2f}) — ESTIMATES, not measurements ==")
    print(f"  naive whole-step x{scale:.2f}       : {naive:8.2f} ms/step")
    print(f"  trunk-scaled, rest constant  : {phase_aware:8.2f} ms/step"
          "   <- the defensible one: the MTP draft plane is ONE layer in")
    print("                                              "
          "     production too, so only the trunk scales.")
    print(f"  implied tok/s at 1 tok/step  : {1000 / phase_aware:8.1f}   "
          f"(x accepted tokens/step; production measures ~2 at k=3)")
    print("  both ignore the ~76 TP8 all-reduces/step this 1-GPU run cannot see,")
    print("  and the trunk scaling charges embed+lm_head 78/N times instead of once.")
    print(f"  trunk measured: {trunk_gpu:.3f} ms GPU over {args.layers} layers. Two runs at")
    print("  different --layers give the MARGINAL per-layer cost (slope), which extrapolates")
    print("  without the fixed embed/lm_head/norm term this ratio double-counts.")

    # ------------------------------------------------------------- kernels --
    print()
    if trace_json.exists():
        lines = summarise_trace(trace_json, args.top)
        head = next((ln for ln in lines if ln.startswith("window")), "")
        n_ev = None
        if "GPU events" in head:
            try:
                n_ev = int(head.split(",")[1].strip().split()[0])
            except (IndexError, ValueError):
                n_ev = None
        print(f"== kernels, torch.profiler over {args.trace_steps} captured "
              f"steps ({trace_json}) ==")
        if n_ev is not None:
            print(f"  GPU events per step: {n_ev / args.trace_steps:.0f} "
                  f"({n_ev} over {args.trace_steps} steps)")
        for ln in lines:
            print("  " + ln)
    else:
        print(f"(no trace written at {trace_json})")

    print()
    print("== NOT production ==")
    print("  * tp_size=1: o_proj / MoE all-reduces and the lm_head all-gather are absent")
    print(f"  * {args.layers} layers, not 78; {cfg.num_dense_layers} dense of "
          f"{args.layers} vs production's 3 of 78")
    print(f"  * vocab is the per-rank shard ({cfg.vocab_size}); a real rank "
          f"argmaxes {FULL_VOCAB} columns after the all-gather")
    print(f"  * random weights (acceptance {cap_arm['accept']:.2f}/step); "
          "padded rows keep step cost acceptance-independent")
    print(f"  * index_skip_topk_offset={cfg.index_skip_topk_offset} (production 3) "
          "so layer N is FULL; the indexer never runs with dsa_long_context off")
    print(f"  * capture batch sizes [{args.bs}], not [1, 2, 4, 8, 16]; "
          f"prefill capture {'on' if args.capture_prefill else 'off'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

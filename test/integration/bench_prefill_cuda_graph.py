"""Latency benchmark: Qwen3-Omni Thinker prefill_text — eager vs CUDA graph.

For each (bs, num_tokens) bucket captured by the Thinker prefill_text graph,
times the eager ``forward_batched`` path against the CUDA-graph ``runner.run``
path on identical synthetic inputs. Reports median, p10, p90 wall-clock per
call plus the speedup ratio.

Plan §6.3 expectation: 2–5× speedup on prefill (less than decode because
prefill has more compute per launch; the ratio shrinks as num_tokens grows
and per-bucket compute dominates Python launch overhead).

Usage::

    huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct
    python test/integration/bench_prefill_cuda_graph.py \\
        --num-warmup 5 --num-iters 100

By default benchmarks the same 15 buckets the parity test covers. Override
with ``--bs-list`` and ``--num-tokens-list`` for a narrower sweep.

This is in-process timing, not an HTTP load test (cf. test/bagel/benchmark_bagel.py).
The model + runner are brought up once and shared across all buckets.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mminf.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mminf.engine.kv_cache_engine import KVCacheEngine  # noqa: E402
from mminf.engine.cuda_graph_runner import CudaGraphKey, CudaGraphRunner  # noqa: E402
from mminf.engine.kv_store import TransferEngineInfo  # noqa: E402
from mminf.model.submodule_base import ARNodeInputs, ModelInputsFromEngine  # noqa: E402

QWEN3_OMNI_REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


class _StubTransferEngine:
    """Single-process stand-in for the Mooncake TransferEngine."""

    def __init__(self):
        self.registered: list[tuple[int, int]] = []

    def register_memory(self, ptr: int, nbytes: int) -> int:
        self.registered.append((ptr, nbytes))
        return 0

    def unregister_memory(self, ptr: int) -> int:  # noqa: ARG002
        return 0

    def get_async_reader(self, device):  # noqa: ARG002
        return None

    def batch_transfer_sync_read(self, *args, **kwargs):
        raise RuntimeError("stub: no transfers expected in this benchmark")


def _bring_up_thinker(cache_dir: str | None = None):
    """Load Qwen3-Omni Thinker, build KVCacheEngine, capture CUDA graphs.

    Skips ``engine.warmup()``'s ``_compile_submodules`` step so the eager
    side and the captured graph use the same uncompiled kernels (matches the
    parity test's setup — eager numbers here reflect what the engine would
    run without torch.compile).
    """
    from mminf.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    # cuda_graph_runner calls torch.cuda.set_device(self.device) inside the
    # capture path; it rejects a bare torch.device("cuda"). Use the explicit
    # current-device index, matching how production workers pass it.
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    model = Qwen3OmniModel(model_path_hf=QWEN3_OMNI_REPO, cache_dir=cache_dir)
    thinker = model.get_submodule("Thinker", device=str(device))
    assert thinker is not None

    kv_cfgs = [c for c in model.get_kv_cache_config() if c.nodes and "Thinker" in c.nodes]
    assert len(kv_cfgs) == 1
    kv_cfg = kv_cfgs[0]
    kv_cfg.max_num_pages = 256

    engine = KVCacheEngine(autocast_dtype=torch.bfloat16)
    transfer_info = TransferEngineInfo(
        my_entity_id="bench",
        my_session_id="bench_session",
        transfer_engine=_StubTransferEngine(),
    )
    engine.load_model(
        submodules={"Thinker": thinker.to(device)},
        kv_cache_config=[kv_cfg],
        device=device,
        transfer_engine_info=transfer_info,
        kv_cache_type=torch.bfloat16,
    )

    submod_mgmt = engine.submodule_management["Thinker"]
    kv_mgmt = submod_mgmt.kv_management
    runner = CudaGraphRunner(
        submodule_name="Thinker",
        submodule=submod_mgmt.submodule,
        kv_cache_config=kv_mgmt.kv_cache_config,
        alloc_manager=kv_mgmt.alloc_manager,
        sampler=submod_mgmt.sampler,
        buffer_manager=kv_mgmt.buffer_manager,
        device=device,
        autocast_dtype=torch.bfloat16,
    )
    runner.warmup_and_capture()
    submod_mgmt.cuda_graph_runner = runner
    return engine, runner, submod_mgmt.submodule, device


def _make_inputs(
    bs: int,
    total_tokens: int,
    submodule,
    device: torch.device,
    seed: int = 0,
) -> tuple[list[str], list[ARNodeInputs]]:
    """Build bs ARNodeInputs whose seq_lens sum to total_tokens.

    Embeds via embed_tokens on random in-vocab token IDs (kept in [0, 10000)
    to avoid special tokens). Realistic-magnitude embeds keep timings
    representative — torch.randn N(0, 1) embeds saturate non-linearities
    and aren't what the model would see in production.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    request_ids = [f"req_{uuid.uuid4().hex[:8]}" for _ in range(bs)]
    base = total_tokens // bs
    seq_lens = [base] * bs
    seq_lens[-1] += total_tokens - sum(seq_lens)
    safe_vocab_max = 10000
    embed_layer = submodule.model.model.embed_tokens
    inputs: list[ARNodeInputs] = []
    for sl in seq_lens:
        token_ids = torch.randint(
            0, safe_vocab_max, (sl,),
            dtype=torch.long, device=device, generator=g,
        )
        with torch.no_grad():
            embeds = embed_layer(token_ids).to(torch.bfloat16)
        pos_ids = torch.arange(
            sl, dtype=torch.float, device=device,
        ).unsqueeze(0).expand(3, -1).contiguous()
        masks = torch.stack([
            torch.zeros(sl, dtype=torch.bool, device=device),
            torch.ones(sl, dtype=torch.bool, device=device),
        ])
        inputs.append(ARNodeInputs(
            input_seq_len=sl,
            input_embeds=embeds,
            custom_pos_ids=pos_ids,
            tensor_inputs={"masks_for_talker": masks},
        ))
    return request_ids, inputs


def _make_per_request_info(request_ids: list[str]) -> dict[str, CurrentForwardPassInfo]:
    return {
        rid: CurrentForwardPassInfo(
            request_id=rid,
            graph_walk="prefill_text",
            requires_cfg=False,
            fwd_index=0,
            random_seed=42,
            max_tokens=1,
            sampling_config={},
            step_metadata={"audio_output": True, "is_last_prefill": True},
        )
        for rid in request_ids
    }


def _time_eager_one(
    engine: KVCacheEngine,
    submodule,
    bs: int,
    total_tokens: int,
    device: torch.device,
) -> float:
    """One timed eager prefill — per-rid sequential, the production path.

    forward_batched can't be timed here (asserts on a CUDA-graph-only
    qo_indptr_buf). _execute_sequential calls submodule.forward in a per-rid
    loop for prefill_text; we mirror that.
    """
    rids, inputs = _make_inputs(bs, total_tokens, submodule, device, seed=0)
    per_info = _make_per_request_info(rids)
    for rid in rids:
        engine.add_request(rid, ["main"])
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                for rid, inp in zip(rids, inputs, strict=True):
                    cache_mgr = engine._create_cache_manager([rid], "Thinker")
                    engine_inputs = ModelInputsFromEngine(
                        request_ids=[rid],
                        per_request_info={rid: per_info[rid]},
                        cache_manager=cache_mgr,
                    )
                    preprocessed = submodule.preprocess(
                        graph_walk="prefill_text",
                        engine_inputs=engine_inputs,
                        inputs=[inp],
                    )
                    _ = submodule.forward(
                        graph_walk="prefill_text",
                        engine_inputs=engine_inputs,
                        **preprocessed,
                    )
        torch.cuda.synchronize()
        return time.perf_counter() - t0
    finally:
        for rid in rids:
            engine.remove_request(rid)


def _time_graph_one(
    engine: KVCacheEngine,
    runner: CudaGraphRunner,
    submodule,
    bs: int,
    total_tokens: int,
    device: torch.device,
) -> float:
    rids, inputs = _make_inputs(bs, total_tokens, submodule, device, seed=0)
    per_info = _make_per_request_info(rids)
    for rid in rids:
        engine.add_request(rid, ["main"])
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                runner.run(
                    graph_walk="prefill_text",
                    requires_cfg=False,
                    request_ids=rids,
                    inputs=inputs,
                    per_request_info=per_info,
                    submodule=submodule,
                )
        torch.cuda.synchronize()
        return time.perf_counter() - t0
    finally:
        for rid in rids:
            engine.remove_request(rid)


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = lo + 1
    if hi >= len(s):
        return s[lo]
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def _bench_bucket(
    engine: KVCacheEngine,
    runner: CudaGraphRunner,
    submodule,
    bs: int,
    total_tokens: int,
    device: torch.device,
    num_warmup: int,
    num_iters: int,
) -> dict:
    """Time one (bs, total_tokens) bucket. Returns timing stats in milliseconds.

    total_tokens is the sum across the batch (matches CudaGraphKey.num_tokens).
    """
    key = CudaGraphKey(
        graph_walk="prefill_text",
        requires_cfg=False,
        bs=bs,
        num_tokens=total_tokens,
    )
    if key not in runner.graphs:
        return {"bs": bs, "total_tokens": total_tokens, "error": "no captured graph"}

    for _ in range(num_warmup):
        _time_eager_one(engine, submodule, bs, total_tokens, device)
        _time_graph_one(engine, runner, submodule, bs, total_tokens, device)

    eager_times = [
        _time_eager_one(engine, submodule, bs, total_tokens, device) * 1000
        for _ in range(num_iters)
    ]
    graph_times = [
        _time_graph_one(engine, runner, submodule, bs, total_tokens, device) * 1000
        for _ in range(num_iters)
    ]
    eager_med = statistics.median(eager_times)
    graph_med = statistics.median(graph_times)
    return {
        "bs": bs,
        "total_tokens": total_tokens,
        "eager_p10_ms": _percentile(eager_times, 10),
        "eager_p50_ms": eager_med,
        "eager_p90_ms": _percentile(eager_times, 90),
        "graph_p10_ms": _percentile(graph_times, 10),
        "graph_p50_ms": graph_med,
        "graph_p90_ms": _percentile(graph_times, 90),
        "speedup_p50": eager_med / graph_med if graph_med > 0 else float("inf"),
    }


def _print_results(results: list[dict]) -> None:
    print("\n" + "=" * 88)
    print("Qwen3-Omni Thinker prefill_text — eager vs CUDA graph (per-call latency, ms)")
    print("Note: total_tokens is the sum across the batch (split evenly per request).")
    print("=" * 88)
    header = (
        f"{'bs':>3}  {'total':>6}  {'eager p50':>10}  {'graph p50':>10}  "
        f"{'speedup':>8}  {'eager p90':>10}  {'graph p90':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['bs']:>3}  {r['total_tokens']:>6}  ({r['error']})")
            continue
        print(
            f"{r['bs']:>3}  {r['total_tokens']:>6}  "
            f"{r['eager_p50_ms']:>10.3f}  {r['graph_p50_ms']:>10.3f}  "
            f"{r['speedup_p50']:>7.2f}x  "
            f"{r['eager_p90_ms']:>10.3f}  {r['graph_p90_ms']:>10.3f}"
        )
    print("=" * 88)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-warmup", type=int, default=5,
                   help="Warmup iterations per bucket before timing (default: 5)")
    p.add_argument("--num-iters", type=int, default=100,
                   help="Timed iterations per bucket per path (default: 100, per plan §6.3)")
    p.add_argument("--bs-list", type=int, nargs="+", default=[1, 2, 4],
                   help="Batch sizes to benchmark (must be subset of captured set)")
    p.add_argument("--total-tokens-list", type=int, nargs="+",
                   default=[128, 256, 512, 1024, 2048],
                   help="Total-token bucket sizes — sum across the batch, split "
                        "evenly per request (must be subset of the captured set)")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="HuggingFace cache dir for the Qwen3-Omni snapshot")
    return p.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: CUDA required.")
        sys.exit(1)

    args = parse_args()
    cache_dir = args.cache_dir or os.environ.get("QWEN3_OMNI_CACHE_DIR")

    print("Bringing up Qwen3-Omni Thinker (this may take ~30-60s + ~50s capture)...")
    engine, runner, submodule, device = _bring_up_thinker(cache_dir)
    print(f"Ready. {len(runner.graphs)} captured graphs.")

    results: list[dict] = []
    for bs in args.bs_list:
        for tt in args.total_tokens_list:
            print(f"  benchmarking bs={bs} total_tokens={tt} ...", flush=True)
            results.append(_bench_bucket(
                engine, runner, submodule, bs, tt, device,
                num_warmup=args.num_warmup, num_iters=args.num_iters,
            ))

    _print_results(results)
    engine.shutdown()


if __name__ == "__main__":
    main()

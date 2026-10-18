"""Tune fused MoE Triton launch configs for one set of MoE sizes on this GPU.

Writes E=..,N=..,K=..,topk=..,device_name=<gpu>.json (into
mstar/utils/fused_moe/configs/ by default), which the Triton backend loads on
that GPU model, plus a timings file comparing default and tuned configs at
each token count.

Each candidate is timed as a full fused_experts call (align, both GEMMs,
activation, top-k sum, one config for both GEMMs) replayed from a CUDA graph.
Candidates whose output doesn't match the default config get dropped.

The search space follows vLLM's benchmarks/kernels/benchmark_moe.py, pruned
per token count (BLOCK_SIZE_M capped by rows per expert, LDS <= 64 KiB,
large-GEMM rules). --search staged picks tiles, warps and grouping first with
the AMD options at Inductor's ROCm defaults, then sweeps those options for the
best few. --search full times everything (hours per token count).

    python benchmark/fused_moe/tune_fused_moe.py --num-experts 128 \\
        --intermediate-size 768 --hidden-size 2048 --top-k 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import triton

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mstar.platforms import current_platform  # noqa: E402
from mstar.utils.fused_moe.config import (  # noqa: E402
    CONFIG_DIR,
    MoEProblem,
    config_file_name,
    device_config_name,
)
from mstar.utils.fused_moe.kernels import get_default_config  # noqa: E402
from mstar.utils.fused_moe.runner import fused_experts  # noqa: E402

DEFAULT_TOKEN_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# Inductor's ROCm GEMM defaults (ROCmGemmConfig). The staged search starts here.
AMD_BASE_OPTIONS = {"waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 2}
CONFIG_KEY_ORDER = (
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages",
    "waves_per_eu", "matrix_instr_nonkdim", "kpack",
)


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def tile_space(amd: bool) -> dict[str, list[int]]:
    if amd:
        return {
            "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_K": [16, 32, 64, 128, 256],
            "GROUP_SIZE_M": [1, 4, 8, 16, 32],
            "num_warps": [1, 2, 4, 8],
            "num_stages": [2],
        }
    return {
        "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
        "BLOCK_SIZE_N": [32, 64, 128, 256],
        "BLOCK_SIZE_K": [64, 128, 256],
        "GROUP_SIZE_M": [1, 16, 32, 64],
        "num_warps": [4, 8],
        "num_stages": [2, 3, 4, 5],
    }


AMD_OPTION_SPACE = {
    "waves_per_eu": [0, 1, 2, 4],
    "matrix_instr_nonkdim": [16, 32],
    "kpack": [1, 2],
}


def product(space: dict[str, list[int]]) -> list[dict[str, int]]:
    keys = list(space)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(space[k] for k in keys))]


def _valid_for_gemm(cfg: dict[str, int], rows: int, n: int, k: int, amd: bool) -> bool:
    """vLLM's ROCm pruning (``prune_rocm_configs``) for one GEMM of ``rows x n x k``."""
    bm, bn, bk = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    if bm * bn < 64:
        return False
    if n * 2 < bn and bn != 16:
        return False
    if cfg["GROUP_SIZE_M"] * bm > rows and cfg["GROUP_SIZE_M"] != 1:
        return False
    if k % bk:
        return False
    if (bm * bk + bk * bn) * 2 > 65536:
        return False
    if rows >= 2048 and n >= 2048 and (bm < 64 or bn < 64 or bk < 64 or cfg["num_warps"] < 4):
        return False
    # kpack=2 with 16-wide K tiles compiles but gives wrong results on gfx942
    # (Triton 3.7). The output check would drop them anyway.
    if cfg.get("kpack") == 2 and bk < 32:
        return False
    nonk = cfg.get("matrix_instr_nonkdim")
    if amd and nonk is not None:
        mfma = 16 if rows < 32 or n < 32 else 32
        if nonk > mfma or nonk > bm or nonk > bn:
            return False
    return True


def prune(configs: list[dict[str, int]], problem: MoEProblem, num_tokens: int, amd: bool) -> list[dict[str, int]]:
    rows = num_tokens * problem.top_k
    # A tile taller than the rows one expert receives only pads.
    rows_per_expert = math.ceil(rows / problem.num_experts)
    max_bm = max(16, triton.next_power_of_2(rows_per_expert))
    gemm1 = (rows, 2 * problem.intermediate_size, problem.hidden_size)
    gemm2 = (rows, problem.hidden_size, problem.intermediate_size)
    return [
        c for c in configs
        if c["BLOCK_SIZE_M"] <= max_bm
        and (_valid_for_gemm(c, *gemm1, amd) or _valid_for_gemm(c, *gemm2, amd))
    ]


def ordered(cfg: dict[str, int]) -> dict[str, int]:
    return {k: cfg[k] for k in CONFIG_KEY_ORDER if k in cfg}


# ---------------------------------------------------------------------------
# Running and timing one config
# ---------------------------------------------------------------------------


def make_inputs(problem: MoEProblem, num_tokens: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, dt = "cuda", torch.bfloat16
    e, n, k = problem.num_experts, problem.intermediate_size, problem.hidden_size
    hidden = torch.randn(num_tokens, k, device=dev, dtype=dt, generator=g)
    w1 = torch.randn(e, 2 * n, k, device=dev, dtype=dt, generator=g) * 0.02
    w2 = torch.randn(e, k, n, device=dev, dtype=dt, generator=g) * 0.02
    logits = torch.randn(num_tokens, e, device=dev, dtype=torch.float32, generator=g)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), problem.top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return hidden, w1, w2, weights, ids.to(torch.int32)


def run(inputs, config):
    hidden, w1, w2, weights, ids = inputs
    return fused_experts(hidden, w1, w2, weights, ids, config=config)


def time_config(inputs, config, graph_calls: int = 10, replays: int = 20) -> float:
    """Microseconds per ``fused_experts`` call, from replaying a captured graph."""
    run(inputs, config)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(inputs, config)  # warm on the capture stream
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(graph_calls):
                run(inputs, config)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    us = start.elapsed_time(end) * 1e3 / (replays * graph_calls)
    del graph
    return us


# ---------------------------------------------------------------------------
# Compile workers fill Triton's disk cache in parallel, so the timing loop in
# the main process only has to load kernels.
# ---------------------------------------------------------------------------

_WORKER_INPUTS = None


def _worker_init(problem: tuple, num_tokens: int) -> None:
    global _WORKER_INPUTS
    _WORKER_INPUTS = make_inputs(MoEProblem(*problem), num_tokens)


def _worker_compile(config: dict[str, int]) -> str | None:
    try:
        run(_WORKER_INPUTS, config)
        torch.cuda.synchronize()
        return None
    except Exception as e:  # invalid configs fail to compile or launch
        return f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"


def precompile(problem: MoEProblem, num_tokens: int, configs: list[dict], workers: int) -> dict[int, str]:
    if workers <= 0 or not configs:
        return {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_worker_init,
        initargs=(tuple(problem), num_tokens),
    ) as pool:
        errors = list(pool.map(_worker_compile, configs, chunksize=1))
    return {i: e for i, e in enumerate(errors) if e is not None}


def evaluate(inputs, reference, configs, problem, num_tokens, workers, log):
    failed = precompile(problem, num_tokens, configs, workers)
    results = []
    for i, cfg in enumerate(configs):
        if i in failed:
            continue
        try:
            out = run(inputs, cfg)
            torch.cuda.synchronize()
            if not torch.allclose(out.float(), reference.float(), atol=2e-2, rtol=2e-2):
                log(f"    mismatch, dropped: {ordered(cfg)}")
                continue
            results.append((time_config(inputs, cfg), cfg))
        except Exception as e:
            failed[i] = f"{type(e).__name__}"
    results.sort(key=lambda r: r[0])
    return results, len(failed)


def tune_one(problem: MoEProblem, num_tokens: int, args, log) -> dict:
    amd = current_platform.is_rocm()
    inputs = make_inputs(problem, num_tokens)
    default_cfg = get_default_config(
        M=num_tokens, E=problem.num_experts, N=2 * problem.intermediate_size,
        K=problem.hidden_size, top_k=problem.top_k,
    )
    reference = run(inputs, default_cfg).clone()
    default_us = time_config(inputs, default_cfg)
    t0 = time.time()

    tiles = prune(product(tile_space(amd)), problem, num_tokens, amd)
    if amd and args.search == "full":
        candidates = prune(
            [{**t, **o} for t in tiles for o in product(AMD_OPTION_SPACE)], problem, num_tokens, amd,
        )
        results, n_failed = evaluate(inputs, reference, candidates, problem, num_tokens, args.compile_workers, log)
        n_tried = len(candidates)
    elif amd:
        stage1 = prune([{**t, **AMD_BASE_OPTIONS} for t in tiles], problem, num_tokens, amd)
        res1, fail1 = evaluate(inputs, reference, stage1, problem, num_tokens, args.compile_workers, log)
        top = [cfg for _, cfg in res1[: args.refine_top]]
        stage2 = []
        for cfg in top:
            for opts in product(AMD_OPTION_SPACE):
                cand = {**cfg, **opts}
                if cand not in stage2 and cand not in top:
                    stage2.append(cand)
        stage2 = prune(stage2, problem, num_tokens, amd)
        res2, fail2 = evaluate(inputs, reference, stage2, problem, num_tokens, args.compile_workers, log)
        results = sorted(res1 + res2, key=lambda r: r[0])
        n_failed, n_tried = fail1 + fail2, len(stage1) + len(stage2)
    else:
        results, n_failed = evaluate(inputs, reference, tiles, problem, num_tokens, args.compile_workers, log)
        n_tried = len(tiles)

    if not results:
        raise RuntimeError(f"no config compiled and matched for M={num_tokens}")
    # Re-time the top few, single timings of close configs are noisy.
    finalists = [cfg for _, cfg in results[: args.final_top]]
    final = sorted(
        ((min(time_config(inputs, c) for _ in range(3)), c) for c in finalists), key=lambda r: r[0],
    )
    default_us = min([default_us] + [time_config(inputs, default_cfg) for _ in range(2)])
    best_us, best = final[0]
    return {
        "M": num_tokens, "best": ordered(best), "best_us": round(best_us, 2),
        "default": default_cfg, "default_us": round(default_us, 2),
        "speedup": round(default_us / best_us, 3), "tried": n_tried, "failed": n_failed,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-experts", type=int, required=True)
    p.add_argument("--intermediate-size", type=int, required=True, help="per expert and rank (w2.shape[2])")
    p.add_argument("--hidden-size", type=int, required=True)
    p.add_argument("--top-k", type=int, required=True)
    p.add_argument("--token-counts", type=int, nargs="+", default=DEFAULT_TOKEN_COUNTS)
    p.add_argument("--search", choices=["staged", "full"], default="staged")
    p.add_argument("--refine-top", type=int, default=5, help="how many configs get their AMD options swept (staged)")
    p.add_argument("--final-top", type=int, default=5, help="configs re-timed before picking")
    p.add_argument("--compile-workers", type=int, default=min(6, max(1, (os.cpu_count() or 2) // 2)))
    p.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    p.add_argument("--results", type=Path, default=None, help="timings JSON (default: in the working directory)")
    args = p.parse_args()

    problem = MoEProblem(args.num_experts, args.intermediate_size, args.hidden_size, args.top_k)
    device_name = device_config_name()
    out = args.config_dir / config_file_name(problem, device_name)
    results_path = args.results or Path(out.with_suffix(".timings.json").name)
    table = json.loads(out.read_text()) if out.is_file() else {}
    timings = json.loads(results_path.read_text()) if results_path.is_file() else {}

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"tuning {problem} on {device_name} (triton {triton.__version__}, torch {torch.__version__}, "
        f"hip {torch.version.hip}); search={args.search}")
    for m in args.token_counts:
        r = tune_one(problem, m, args, log)
        log(f"M={m:5d} default {r['default_us']:8.1f} us  best {r['best_us']:8.1f} us  "
            f"x{r['speedup']:.2f}  ({r['tried']} tried, {r['failed']} failed, {r['seconds']} s)  {r['best']}")
        table[str(m)] = r["best"]
        timings[str(m)] = r
        table["triton_version"] = triton.__version__
        args.config_dir.mkdir(parents=True, exist_ok=True)
        ms = sorted(int(k) for k in table if k.isdigit())
        out.write_text(json.dumps(
            {"triton_version": table["triton_version"], **{str(k): table[str(k)] for k in ms}}, indent=4,
        ) + "\n")
        results_path.write_text(json.dumps(timings, indent=2) + "\n")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()

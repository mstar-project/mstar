"""DeepSeek-V4.1's compressed sparse attention (CSA) on a GPU: what each of the resource's torch
operations costs, and library kernels against them. Not a CI test: it needs a GPU.

    python test/deepseek_v41/gpu_csa.py --mode bench --out bench.json
    python test/deepseek_v41/gpu_csa.py --mode kernels --reference <snapshot dir> --out kernels.json
    python test/deepseek_v41/gpu_csa.py --mode step-prepare --snapshot <dir>     (no GPU)
    python test/deepseek_v41/gpu_csa.py --mode step --snapshot <dir> --out step.json

``bench``: a decode step's scoring, selection, cache write and attention through ``CSAManager``,
over caches filled with random FP4 rows, at the reduced model's shapes and the checkpoint's
(64 heads x 512, 32 index heads x 128, top 512, a window of 128), for 1 to 64 rows and contexts
up to 65,536 tokens: milliseconds a call, median and p10/p90, with the gather-and-unpack of the
FP4 rows timed apart from the arithmetic on them.
``kernels``: on the same inputs, library kernels against the torch operations, in the units of
``gpu_parity.py`` (elements equal, error in the result's bf16 spacing), and their time:
FlashInfer's top-k (ties to the smaller index) against the stable sort; the Triton score kernel
of the DSA resource on the same keys held in bf16; and for the attention, on the entries gathered
to bf16, the checkpoint reference's own kernel (``inference/kernel.py``, tilelang) and the torch
stand-in the CPU parity tests hold the resource to bit for bit.
``step``: a whole decode step through the ``Engine`` of a model with the checkpoint's attention
shapes and the reduced model's depth (seven layers, small experts, random weights written by
``step-prepare``): milliseconds a step run eagerly and replayed from its captured graph, for 1 to
8 requests past 1,100 tokens of context, with the replay's logits checked against the eager ones.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, ".")

import torch

from mstar.engine.resources.csa.config import INDEX_PLANES, CSAConfig, Fp4Rows
from mstar.engine.resources.csa.manager import CSACache, CSAManager, CSARows, _load, _store, top_scores
from mstar.utils.mx_round import round_fp4, round_fp8

# (name, heads, head_dim, index heads, index head dim, topk, window, candidate blocks, block size)
SHAPES = {
    "reduced": dict(heads=4, dim=32, index_heads=16, index_dim=32, topk=8, window=8, cand_blocks=3, cand_size=4),
    "full": dict(heads=64, dim=512, index_heads=32, index_dim=128, topk=512, window=128, cand_blocks=2048,
                 cand_size=8),
}
PAGE = 16
# the checkpoint's attention at the reduced model's depth and expert size
STEP_SHAPE = dict(
    hidden_size=5120, num_attention_heads=64, head_dim=512, qk_rope_head_dim=64, q_lora_rank=1280, o_lora_rank=1024,
    o_groups=8, sliding_window=128, index_n_heads=32, index_head_dim=128, index_topk=512, candidate_topk_blocks=2048,
    candidate_block_size=8, moe_intermediate_size=64, max_position_embeddings=4096,
    rope_original_max_position_embeddings=1024)
STEP_PROMPT, STEP_STEPS = 1100, 24


class _Planes:
    """A paged cache's storage as the CSA resource reads it: latent and key planes by page."""

    def __init__(self, pages: int, latent_width: int, key_width: int, device):
        self.latents = torch.zeros(1, pages, PAGE, latent_width, dtype=torch.bfloat16, device=device)
        self.keys = torch.zeros(1, pages, PAGE, key_width, dtype=torch.bfloat16, device=device)

    def layer_view(self, plane: int) -> torch.Tensor:
        return self.latents[plane]

    def aux_view(self, name: str, plane: int) -> torch.Tensor:
        assert name == INDEX_PLANES
        return self.keys[plane]

    def spare_row(self):
        return PAGE - 1, None


def _timed(fn, iters: int, device) -> dict:
    cuda = device.type == "cuda"
    for _ in range(2):
        fn()
    laps = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize(device)
        laps.append((time.perf_counter() - t0) * 1e3)
    laps.sort()
    return {"ms": round(statistics.median(laps), 4), "p10": round(laps[len(laps) // 10], 4),
            "p90": round(laps[-1 - len(laps) // 10], 4), "iters": iters}


def _spacing(x: torch.Tensor) -> torch.Tensor:
    return torch.ldexp(torch.ones_like(x.float()), torch.frexp(x.float().abs().clamp_min(1e-30))[1] - 8)


def _ulps(got: torch.Tensor, want: torch.Tensor) -> dict:
    err = (got.float() - want.float()).abs() / _spacing(want)
    return {"equal": bool(torch.equal(got, want)), "differing": int((got != want).sum()),
            "elements": got.numel(), "max_ulp": err.max().item(), "mean_ulp": err.mean().item()}


def build(shape: dict, rows: int, context: int, ratio: int, device, captured_width: int | None = None):
    """A decode step of ``rows`` requests, each at ``context`` tokens, over one cache of
    ``ratio``: the resource, the step's rows and its inputs."""
    g = torch.Generator(device=device).manual_seed(0)
    latent_rows, key_rows = Fp4Rows(block=16, e4m3_scale=True), Fp4Rows(block=32)
    dim, index_dim = shape["dim"], shape["index_dim"]
    config = CSAConfig(caches={ratio: "cache"}, topk=shape["topk"], sm_scale=dim ** -0.5, index_head_dim=index_dim,
                       latent_dim=dim, latent_rows=latent_rows, key_rows=key_rows)
    groups = context // ratio
    pages_each = -(-groups // PAGE)
    num_pages = 1 + rows * pages_each
    planes = _Planes(num_pages, -(-latent_rows.nbytes(dim) // 2), -(-key_rows.nbytes(index_dim) // 2), device)
    # a block of random rows, repeated over the cache
    block = 4096
    latent = torch.randn(block, dim, generator=g, device=device)
    keys = torch.randn(block, index_dim, generator=g, device=device)
    for values, fmt, view in ((latent, latent_rows, planes.latents[0]), (keys, key_rows, planes.keys[0])):
        flat = view.view(-1, view.shape[-1])
        packed = _store(values, fmt, view).to(view.dtype)
        for r0 in range(0, flat.shape[0], block):
            flat[r0:r0 + block] = packed[:flat.shape[0] - r0]
    manager = CSAManager(config, {ratio: PAGE}, device, {ratio: captured_width or groups})
    manager._kv = {ratio: planes}

    def dev(values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.long, device=device)

    pos = dev([context - 1] * rows)
    req = torch.arange(rows, device=device)
    pages = torch.arange(1, num_pages, device=device)
    step = CSARows(
        pos=pos, row_req=req, first=req, end=pos + 1, req_first=req, req_rows=torch.ones_like(req),
        caches={ratio: CSACache(pages, req * pages_each, PAGE, num_pages)},
        spans=[(i, 1, context - 1) for i in range(rows)], one_row_each=True,
        widths={ratio: captured_width} if captured_width else None)
    manager._current = {"main": step}
    heads, window = shape["heads"], shape["window"]
    inputs = types.SimpleNamespace(
        q=torch.randn(rows, heads, dim, generator=g, device=device).to(torch.bfloat16),
        sink=torch.randn(heads, generator=g, device=device),
        index_q=round_fp4(torch.randn(rows, shape["index_heads"], index_dim, generator=g, device=device)
                          .to(torch.bfloat16), 32),
        index_w=(torch.randn(rows, shape["index_heads"], generator=g, device=device) * 0.05).to(torch.bfloat16),
        window=round_fp8(torch.randn(rows * window + rows, dim, generator=g, device=device).to(torch.bfloat16), 32),
        window_idx=(req[:, None] * window + torch.arange(window, device=device)[None, :]),
        window_cols=torch.full((rows,), window, device=device),
        latent=torch.randn(rows, dim, generator=g, device=device).to(torch.bfloat16),
        keys=torch.randn(rows, index_dim, generator=g, device=device).to(torch.bfloat16),
    )
    return manager, planes, step, inputs


def _cases(args):
    for name in args.shapes.split(","):
        contexts = (64, 256) if name == "reduced" else tuple(int(c) for c in args.contexts.split(","))
        for ratio in (1, 2):
            for context in contexts:
                for rows in (int(r) for r in args.rows.split(",")):
                    yield name, SHAPES[name], ratio, context, rows


def _bench_case(shape: dict, ratio: int, context: int, rows: int, args, device) -> dict:
    manager, planes, step, x = build(shape, rows, context, ratio, device)
    groups = context // ratio
    iters = args.iters if rows * groups < 1 << 20 else max(5, args.iters // 4)
    row: dict[str, dict] = {}
    scores = manager.scores(ratio, 0, x.index_q, x.index_w)
    slots = manager.select(ratio, scores)
    keep = manager.candidates(ratio, scores, shape["cand_blocks"], shape["cand_size"])
    row["scores"] = _timed(lambda: manager.scores(ratio, 0, x.index_q, x.index_w), iters, device)
    manager.config.score_kernel = False
    row["scores_torch"] = _timed(lambda: manager.scores(ratio, 0, x.index_q, x.index_w), iters, device)
    manager.config.score_kernel = True
    row["select"] = _timed(lambda: manager.select(ratio, scores), iters, device)
    row["candidates"] = _timed(
        lambda: manager.candidates(ratio, scores, shape["cand_blocks"], shape["cand_size"]), iters, device)
    row["select_in_candidates"] = _timed(lambda: manager.select(ratio, scores, keep), iters, device)
    row["write"] = _timed(lambda: manager.write(ratio, 0, latent=x.latent, keys=x.keys), iters, device)
    row["attend"] = _timed(
        lambda: manager.attend(x.q, x.sink, x.window, x.window_idx, x.window_cols, ratio, 0, slots), iters, device)
    row["attend_window_only"] = _timed(
        lambda: manager.attend(x.q, x.sink, x.window, x.window_idx, x.window_cols), iters, device)
    # the FP4 rows alone: gathered and unpacked, as scores and attend read them
    key_flat = planes.keys[0].view(-1, planes.keys.shape[-1])
    latent_flat = planes.latents[0].view(-1, planes.latents.shape[-1])
    group = torch.arange(groups, device=device)
    some = min(rows, max(1, manager.config.score_bytes // (4 * groups * shape["index_dim"])))
    key_slots = step.slots(ratio, step.row_req[:some, None], group[None, :])
    row["load_keys"] = dict(_timed(
        lambda: _load(key_flat[key_slots], manager.config.key_rows, shape["index_dim"]), iters, device),
        rows=some, launches_a_step=-(-rows // some))
    live = slots.clamp_min(0)
    row["load_latents"] = _timed(
        lambda: _load(latent_flat[live], manager.config.latent_rows, shape["dim"]), iters, device)
    row["meta"] = {"groups": groups, "picks": slots.shape[1]}
    return row


def mode_bench(args, device) -> dict:
    out: dict[str, dict] = {}
    for name, shape, ratio, context, rows in _cases(args):
        label = f"{name}/ratio{ratio}/ctx{context}/rows{rows}"
        row = out[label] = _bench_case(shape, ratio, context, rows, args, device)
        print(f"{label}: " + json.dumps({k: v.get("ms", v) for k, v in row.items()}), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _reference_kernels(reference: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("dsv41_reference_kernel", Path(reference) / "inference" / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _kernels_case(shape: dict, ratio: int, context: int, rows: int, args, device, reference) -> dict:
    manager, planes, step, x = build(shape, rows, context, ratio, device)
    groups, iters = context // ratio, args.iters
    row: dict[str, dict] = {}
    # the torch path is what each kernel is held to
    manager.config.score_kernel = False
    scores = manager.scores(ratio, 0, x.index_q, x.index_w)
    if device.type == "cuda" and "scores" in set(args.only.split(",")):
        # -- scores: the resource's own kernel, reading the FP4 keys in place
        try:
            manager.config.score_kernel = True
            got = manager.scores(ratio, 0, x.index_q, x.index_w)
            seen = torch.isfinite(scores)
            row["scores/fp4_kernel"] = dict(
                _ulps(got[seen], scores[seen]), equal=bool(torch.equal(got, scores)),
                same_topk=bool(torch.equal(top_scores(got, min(shape["topk"], groups))[1],
                                           top_scores(scores, min(shape["topk"], groups))[1])))
            row["scores/fp4_kernel/time"] = _timed(
                lambda: manager.scores(ratio, 0, x.index_q, x.index_w), iters, device)
        except Exception as error:  # noqa: BLE001 - a kernel that does not launch is a result
            row["scores/fp4_kernel"] = {"error": f"{type(error).__name__}: {str(error)[-300:]}"}
        manager.config.score_kernel = False
    # scores with exact ties at the cut: many groups share a few values
    tied = (scores * 4).round() / 4
    k = min(shape["topk"], groups)

    only = set(args.only.split(","))
    # -- top-k: FlashInfer (ties to the smaller index) against the stable sort
    try:
        if "topk" not in only:
            raise LookupError("skipped")
        import flashinfer

        def fi_topk(values):
            return flashinfer.top_k(values.contiguous(), k, deterministic=True,
                                    tie_break=flashinfer.TopKTieBreak.SMALL, dsa_graph_safe=True)

        for kind, values in (("scores", scores), ("tied", tied)):
            want = top_scores(values, k)[1].sort(-1).values
            got = fi_topk(values)[1].long().sort(-1).values
            row[f"topk/{kind}"] = {
                "same_picks": bool(torch.equal(got, want)), "rows_differing": int((got != want).any(-1).sum()),
                "rows": rows, "k": k, "groups": groups,
                "distinct_values_a_row": int(values[0].unique().numel())}
        row["topk/flashinfer"] = _timed(lambda: fi_topk(scores), iters, device)
        row["topk/stable_sort"] = _timed(lambda: top_scores(scores, k), iters, device)
    except Exception as error:  # noqa: BLE001 - a kernel that does not launch is a result
        row["topk/flashinfer"] = {"error": f"{type(error).__name__}: {str(error)[-200:]}"}

    # -- scores: the DSA resource's Triton kernel on the same keys held in bf16
    if shape["index_heads"] >= 16 and shape["index_dim"] >= 16 and device.type == "cuda" and "scores" in only:
        try:
            from mstar.engine.resources.dsa.kernels import decode_scores

            key_flat = planes.keys[0].view(-1, planes.keys.shape[-1])
            plane = torch.empty(planes.keys.shape[1], PAGE, shape["index_dim"], dtype=torch.bfloat16,
                                device=device)
            flat = plane.view(-1, shape["index_dim"])
            for r0 in range(0, flat.shape[0], 1 << 16):
                flat[r0:r0 + (1 << 16)] = _load(
                    key_flat[r0:r0 + (1 << 16)], manager.config.key_rows, shape["index_dim"]).to(plane.dtype)
            cache = step.caches[ratio]
            seen = step.groups(ratio)
            kernel_args = (x.index_q.contiguous(), x.index_w.float().contiguous(), plane, cache.pages.int(),
                           cache.page_start.int(), step.row_req.int(), (seen - 1).int(), groups, 1)
            got = decode_scores(*kernel_args)
            row["scores/dsa_kernel_bf16_keys"] = dict(
                _ulps(got, scores), max_abs=(got - scores).abs().max().item(),
                same_topk=bool(torch.equal(top_scores(got, k)[1].sort(-1).values,
                                           top_scores(scores, k)[1].sort(-1).values)),
                key_bytes=2 * shape["index_dim"], fp4_key_bytes=manager.config.key_rows.nbytes(shape["index_dim"]))
            row["scores/dsa_kernel_bf16_keys/time"] = _timed(lambda: decode_scores(*kernel_args), iters, device)
            row["scores/torch_fp4_keys/time"] = _timed(
                lambda: manager.scores(ratio, 0, x.index_q, x.index_w), max(5, iters // 4), device)
            del plane
        except Exception as error:  # noqa: BLE001
            row["scores/dsa_kernel_bf16_keys"] = {"error": f"{type(error).__name__}: {str(error)[-200:]}"}

    # -- attention: the reference's own kernel on the row's entries gathered to bf16
    slots = manager.select(ratio, scores)
    want = manager.attend(x.q, x.sink, x.window, x.window_idx, x.window_cols, ratio, 0, slots)
    row["attend/torch/time"] = _timed(
        lambda: manager.attend(x.q, x.sink, x.window, x.window_idx, x.window_cols, ratio, 0, slots),
        max(5, iters // 4), device)
    if device.type == "cuda" and "attend" in only:
        try:
            manager.config.attend_kernel = True

            def on_kernel():
                return manager.attend(x.q, x.sink, x.window, x.window_idx, x.window_cols, ratio, 0, slots)

            row["attend/triton_kernel"] = _ulps(on_kernel(), want)
            row["attend/triton_kernel/time"] = _timed(on_kernel, iters, device)
        except Exception as error:  # noqa: BLE001 - a kernel that does not launch is a result
            row["attend/triton_kernel"] = {"error": f"{type(error).__name__}: {str(error)[-300:]}"}
        manager.config.attend_kernel = False
    if reference is not None and "attend" in only:
        try:
            latent_flat = planes.latents[0].view(-1, planes.latents.shape[-1])

            def gather():
                picked = _load(latent_flat[slots.clamp_min(0)], manager.config.latent_rows, shape["dim"])
                entries = torch.cat([x.window[x.window_idx], picked.to(torch.bfloat16)], dim=1)
                idx = torch.arange(entries.shape[1], device=device, dtype=torch.int32)[None, :].repeat(rows, 1)
                idx[:, shape["window"]:] = torch.where(slots >= 0, idx[:, shape["window"]:], -1)
                return entries, idx

            entries, idx = gather()
            from mstar.model.deepseek_v41 import _testing as T

            standin = T.reference_kernels().sparse_attn(
                x.q[:, None], entries, x.sink, idx[:, None], shape["dim"] ** -0.5)[:, 0]
            row["attend/reference_standin"] = _ulps(want, standin)

            def kernel():
                return reference.sparse_attn(x.q[:, None], entries, x.sink, idx[:, None], shape["dim"] ** -0.5)[:, 0]

            row["attend/reference_kernel"] = _ulps(kernel(), want)
            row["attend/reference_kernel_vs_standin"] = _ulps(kernel(), standin)
            row["attend/reference_kernel/time"] = _timed(kernel, iters, device)
            row["attend/gather_to_bf16/time"] = _timed(gather, iters, device)
        except Exception as error:  # noqa: BLE001 - a shape the kernel does not compile for is a result
            row["attend/reference_kernel"] = {"error": f"{type(error).__name__}: {str(error)[-200:]}"}
    return row


def mode_kernels(args, device) -> dict:
    out: dict[str, dict] = {}
    reference = None
    if args.reference:
        try:
            reference = _reference_kernels(args.reference)
        except Exception as error:  # noqa: BLE001 - the rows that need no tilelang stand
            out["reference"] = {"error": f"{type(error).__name__}: {str(error)[-200:]}"}
    for name, shape, ratio, context, rows in _cases(args):
        label = f"{name}/ratio{ratio}/ctx{context}/rows{rows}"
        row = out[label] = _kernels_case(shape, ratio, context, rows, args, device, reference)
        print(f"{label}: " + json.dumps(row), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def mode_step_prepare(args, device) -> dict:
    from mstar.model.deepseek_v41 import _testing as T
    from mstar.model.deepseek_v41.config import DeepseekV41Config

    t0 = time.perf_counter()
    config, _ = T.write_snapshot(args.snapshot, DeepseekV41Config.reduced(**STEP_SHAPE), seed=0)
    size = sum(f.stat().st_size for f in Path(args.snapshot).glob("*.safetensors"))
    return {"snapshot": {"layers": config.num_hidden_layers, "gib": round(size / 2**30, 2),
                         "seconds": round(time.perf_counter() - t0, 1)}}


def mode_step(args, device) -> dict:
    from mstar.engine.resources.kv import manager as kv_manager
    from mstar.model.deepseek_v41 import _testing as T
    from mstar.model.deepseek_v41.config import DeepseekV41Config

    kv_manager.KVTransferManager = T.StubTransfer
    config = DeepseekV41Config.reduced(**STEP_SHAPE)
    out: dict[str, dict] = {}
    for dtype in ("bfloat16",):
        # the model takes its shapes from the snapshot's config.json
        engine, sampler, _ = T.build_engine(
            args.snapshot, getattr(torch, dtype), device=str(device), max_seq_len=2048, max_requests=8,
            resources={"csa": {"attend_kernel": args.attend_kernel, "score_kernel": not args.no_score_kernel}})
        buckets = T.captured_buckets(engine)
        for requests in (1, 4, 8):
            ids = {f"r{i}": T.random_ids(config, STEP_PROMPT + 2 * STEP_STEPS, seed=i).to(device)
                   for i in range(requests)}
            for rid, tokens in ids.items():
                T.add_request(engine, rid)
                T.run_step(engine, sampler, "prefill", {rid: tokens[:STEP_PROMPT]})
            laps: dict[str, list[float]] = {"eager": [], "replay": []}
            logits: dict[str, list[torch.Tensor]] = {"eager": [], "replay": []}
            replayed = sampler.replayed
            # the same request decodes on: eager steps, then replayed ones (other tokens, so only the
            # times compare); then both from one more engine state for the logits
            for kind, start in (("eager", 0), ("replay", STEP_STEPS)):
                for i in range(STEP_STEPS):
                    step = {rid: tokens[STEP_PROMPT + start + i].view(1) for rid, tokens in ids.items()}
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    if kind == "eager":
                        with T.eager_steps(engine):
                            got = T.run_step(engine, sampler, "decode", step)
                    else:
                        got = T.run_step(engine, sampler, "decode", step)
                    torch.cuda.synchronize(device)
                    laps[kind].append((time.perf_counter() - t0) * 1e3)
                    logits[kind].append(torch.stack([got[rid] for rid in ids]))
            for rid in ids:
                engine.remove_request(rid)
            row = {kind: {"ms": round(statistics.median(times[2:]), 3), "p10": round(sorted(times[2:])[2], 3),
                          "p90": round(sorted(times[2:])[-3], 3)} for kind, times in laps.items()}
            row.update(replayed_steps=sampler.replayed - replayed, steps=STEP_STEPS, layers=config.num_hidden_layers,
                       finite=bool(all(torch.isfinite(t).all() for kind in logits for t in logits[kind])),
                       buckets=len(buckets))
            out[f"{dtype}/requests{requests}"] = row
            print(f"{dtype}/requests{requests}: " + json.dumps(row), flush=True)
        # replayed against eager logits at this shape: one request, the same tokens both ways
        ids = T.random_ids(config, STEP_PROMPT + 8, seed=99).to(device)
        with T.eager_steps(engine):
            eager = T.run_request(engine, sampler, ids, STEP_PROMPT, rid="e")
        replay = T.run_request(engine, sampler, ids, STEP_PROMPT, rid="g")
        out[f"{dtype}/replay-vs-eager"] = {
            "equal": bool(torch.equal(replay, eager)), "gap": T.logit_gap(replay.float().cpu(), eager.float().cpu()),
            "steps": int(eager.shape[0])}
        print(f"{dtype}/replay-vs-eager: " + json.dumps(out[f"{dtype}/replay-vs-eager"]), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("bench", "kernels", "step-prepare", "step"), required=True)
    ap.add_argument("--snapshot", help="step: the directory step-prepare wrote")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shapes", default="reduced,full")
    ap.add_argument("--rows", default="1,8,32,64")
    ap.add_argument("--contexts", default="4096,65536", help="tokens a request holds, at the full shape")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--reference", help="a snapshot directory holding inference/ (kernels: the attention kernel)")
    ap.add_argument("--only", default="topk,scores,attend", help="kernels: which comparisons to run")
    ap.add_argument("--attend-kernel", action="store_true", help="step: the softmax on its Triton kernel")
    ap.add_argument("--no-score-kernel", action="store_true", help="step: the index scores in torch")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.mode == "step-prepare":
        print(json.dumps(mode_step_prepare(args, None)))
        return 0
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    with torch.no_grad():
        rows = {"bench": mode_bench, "kernels": mode_kernels, "step": mode_step}[args.mode](args, device)
    cuda = device.type == "cuda"
    meta = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device,
            "gpu": torch.cuda.get_device_name(device) if cuda else None,
            "load_average": os.getloadavg()[0], "page": PAGE}
    Path(args.out).write_text(json.dumps({"mode": args.mode, "meta": meta, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

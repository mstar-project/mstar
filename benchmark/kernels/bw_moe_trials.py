"""flashinfer's Blackwell fp8 MoE (trtllm-gen, DeepSeek fp8 block scales) on M*'s block-fp8 experts: which layout
and routing arguments reproduce them, and what the kernel buys GLM's MoE block. sm_100 / sm_103 only; synthetic
weights at GLM-5.3's per-rank shapes, no checkpoint.

1. Variants: every gate/up order, weight layout, activation-scale orientation and routing input against a
   dequantized fp32 reference and the Triton prefill kernels; MATCH where the relative L2 distance is within
   fused_moe.trtllm_fp8.PROBE_TOLERANCE of Triton and REFERENCE_TOLERANCE of the reference. ``prod`` is the
   layout the serving path ships (trtllm_fp8.pack / experts).
2. Routing: the kernel's own DeepSeekV3 router (from logits: sigmoid, correction bias, n_group / topk_group,
   routed scale) against GLM's noaux_tc router, ids per token and weights.
3. Timing (unless --no-time): GLM's MoE block (router, routed and shared experts) per token count, CUDA-graph
   replays: the Triton kernels vs the trtllm kernel at its default tactics and after flashinfer's autotuner.

    python -m benchmark.kernels.bw_moe_trials                       # GLM-5.3 TP8: E=256 H=6144 I=256 top-8
    python -m benchmark.kernels.bw_moe_trials --shape 256,6144,512,8 --tokens 1,8,64,512 --out trials.jsonl
"""

from __future__ import annotations

import argparse
import json
import traceback
from unittest import mock

import torch
import torch.nn.functional as F

from mstar.utils.fused_moe import trtllm_fp8
from mstar.utils.fused_moe.trtllm_fp8 import ExpertWeights

FP8 = torch.float8_e4m3fn
# from the fp32 reference, which quantizes nothing: e4m3 activations and intermediates alone cost a few percent
REFERENCE_TOLERANCE = 0.1


def block_quant(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``w (E, N, K)`` as e4m3 bytes (uint8) and fp32 scales per 128 x 128 block, as checkpoints store it."""
    e, n, k = w.shape
    tiles = w.float().reshape(e, n // 128, 128, k // 128, 128)
    scale = (tiles.abs().amax(dim=(2, 4)) / torch.finfo(FP8).max).clamp_min(1e-12)
    q = (tiles / scale[:, :, None, :, None]).to(FP8).reshape(e, n, k)
    return q.view(torch.uint8), scale


def random_experts(e: int, hidden: int, inter: int, device, seed: int = 0, chunk: int = 16) -> ExpertWeights:
    """Block-fp8 experts in M*'s layout: gate/up ``(E, 2I, H)``, down ``(E, H, I)``, weights ~ N(0, 0.02)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    parts: dict[str, list[torch.Tensor]] = {"w13": [], "s13": [], "w2": [], "s2": []}
    for start in range(0, e, chunk):
        n = min(chunk, e - start)
        for name, shape in (("13", (n, 2 * inter, hidden)), ("2", (n, hidden, inter))):
            q, s = block_quant(torch.randn(shape, device=device, generator=gen) * 0.02)
            parts["w" + name].append(q)
            parts["s" + name].append(s)
    return ExpertWeights(*(torch.cat(parts[k]) for k in ("w13", "s13", "w2", "s2")))


def dequantized(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """One expert's ``(N, K)`` e4m3 bytes times its 128 x 128 block scales, fp32."""
    return w.view(FP8).float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)


def reference(x: torch.Tensor, raw: ExpertWeights, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    """The routed experts in fp32 from the dequantized weights: SwiGLU over ``[gate; up]``, weighted sum per token."""
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in topk_ids.unique().tolist():
        tok, slot = torch.where(topk_ids == e)
        gate, up = (xf[tok] @ dequantized(raw.w13[e], raw.s13[e]).T).chunk(2, dim=-1)
        y = (F.silu(gate) * up) @ dequantized(raw.w2[e], raw.s2[e]).T
        out.index_add_(0, tok, y * topk_weights[tok, slot, None].float())
    return out


def glm_router(logits: torch.Tensor, bias: torch.Tensor, top_k: int, scale: float = 2.5):
    """GLM's noaux_tc routing (NoAuxTcRouter's torch path): sigmoid, top-k of score + bias, the unbiased scores
    renormalized, times ``scale``."""
    scores = logits.float().sigmoid()
    ids = torch.topk(scores + bias.float(), top_k, dim=-1, sorted=False).indices
    w = scores.gather(1, ids)
    return w / (w.sum(-1, keepdim=True) + 1e-20) * scale, ids


def _kernel_args(raw: ExpertWeights, *, swap: bool, tile: int | None, block_major: bool):
    """gemm1/gemm2 weights and scales: halves swapped or not, rows shuffled for epilogue tile ``tile`` (None:
    unshuffled), blocked along K or K-major."""
    from flashinfer.utils import get_shuffle_matrix_a_row_indices

    w13, s13, w2 = raw.w13, raw.s13, raw.w2
    if swap:
        w13, s13 = trtllm_fp8.swap_halves(w13), trtllm_fp8.swap_halves(s13).contiguous()
    if tile is not None:
        def rows(m):
            return get_shuffle_matrix_a_row_indices(torch.empty(m, 128, dtype=torch.uint8), tile)
        if block_major:
            w13 = trtllm_fp8.to_block_major_k(w13, rows(w13.shape[1]))
            w2 = trtllm_fp8.to_block_major_k(w2, rows(w2.shape[1]))
        else:
            w13, w2 = w13[:, rows(w13.shape[1]).to(w13.device)], w2[:, rows(w2.shape[1]).to(w2.device)]
    return ExpertWeights(w13.contiguous(), s13, w2.contiguous(), raw.s2)


def _routed(x, packed, w, ids, *, shuffled, block_major, scale="t", weights_dtype=torch.float32, method=1,
            packed_ids=False):
    from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe

    from mstar.utils.quant_fp8 import per_token_group_quant_fp8

    xq, xs = per_token_group_quant_fp8(x, 128)
    xs = {"t": xs.t().contiguous(), "t-view": xs.t(), "as-is": xs}[scale]
    ids32 = ids.to(torch.int32).contiguous()
    if packed_ids:
        routing = (ids32 << 16) | w.to(torch.bfloat16).view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
    else:
        routing = (ids32, w.to(weights_dtype).contiguous())
    e = packed.w13.shape[0]
    return trtllm_fp8_block_scale_routed_moe(
        topk_ids=routing, routing_bias=None, hidden_states=xq, hidden_states_scale=xs,
        gemm1_weights=packed.w13.view(FP8), gemm1_weights_scale=packed.s13,
        gemm2_weights=packed.w2.view(FP8), gemm2_weights_scale=packed.s2,
        num_experts=e, top_k=ids.shape[1], n_group=None, topk_group=None,
        intermediate_size=packed.s2.shape[2] * 128, local_expert_offset=0, local_num_experts=e,
        routed_scaling_factor=None, routing_method_type=method, use_shuffled_weight=shuffled,
        weight_layout=trtllm_fp8.BLOCK_MAJOR_K if block_major else trtllm_fp8.MAJOR_K)


def _from_logits(x, packed, logits, bias, top_k, *, n_group, topk_group, bias_dtype=torch.float32):
    """The kernel's own DeepSeekV3 routing; returns (out, the expert ids it picked)."""
    from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

    from mstar.utils.quant_fp8 import per_token_group_quant_fp8

    xq, xs = per_token_group_quant_fp8(x, 128)
    e = packed.w13.shape[0]
    replay = torch.full((x.shape[0], top_k), -1, dtype=torch.int16, device=x.device)
    out = trtllm_fp8_block_scale_moe(
        routing_logits=logits.float(), routing_bias=bias.to(bias_dtype), hidden_states=xq,
        hidden_states_scale=xs.t().contiguous(), gemm1_weights=packed.w13.view(FP8), gemm1_weights_scale=packed.s13,
        gemm2_weights=packed.w2.view(FP8), gemm2_weights_scale=packed.s2, num_experts=e, top_k=top_k,
        n_group=n_group, topk_group=topk_group, intermediate_size=packed.s2.shape[2] * 128, local_expert_offset=0,
        local_num_experts=e, routed_scaling_factor=2.5, routing_method_type=2, use_shuffled_weight=True,
        weight_layout=trtllm_fp8.BLOCK_MAJOR_K, routing_replay_out=replay)
    return out, replay.long()


def _same_sets(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of tokens whose expert id sets agree."""
    return (a.sort(-1).values == b.sort(-1).values).all(-1).float().mean().item()


def variants(raw: ExpertWeights, top_k: int):
    """(name, what it tries, run(x, w, ids, logits, bias) -> (out, ids it routed to or None))."""
    prod = trtllm_fp8.pack(raw)
    shuffled = dict(shuffled=True, block_major=True)
    lazy: dict[str, ExpertWeights] = {"prod": prod}

    def weights(name, **kw):
        if name not in lazy:
            lazy[name] = _kernel_args(raw, **kw)
        return lazy[name]

    def routed(name, prep, **kw):
        return lambda x, w, ids, logits, bias: (_routed(x, weights(name, **prep), w, ids, **kw), None)

    up_gate_64 = dict(swap=True, tile=64, block_major=True)
    out = [
        ("prod", "the serving path: trtllm_fp8.pack + trtllm_fp8.experts ([up; gate], shuffled 64, BlockMajorK)",
         lambda x, w, ids, logits, bias: (trtllm_fp8.experts(x, prod, w, ids), None)),
        ("gate-up", "[gate; up] as M* stores it, shuffled 64, BlockMajorK",
         routed("gate-up", dict(swap=False, tile=64, block_major=True), **shuffled)),
        ("tile128", "[up; gate], shuffled for epilogue tile 128, BlockMajorK",
         routed("tile128", dict(swap=True, tile=128, block_major=True), **shuffled)),
        ("majork", "[up; gate], unshuffled K-major (flashinfer's unified runner for DeepSeek fp8)",
         routed("majork", dict(swap=True, tile=None, block_major=False), shuffled=False, block_major=False)),
        ("majork-gate-up", "[gate; up], unshuffled K-major",
         routed("majork-gate-up", dict(swap=False, tile=None, block_major=False), shuffled=False, block_major=False)),
        ("shuffled-majork", "[up; gate], shuffled 64 but K-major",
         routed("shuffled-majork", dict(swap=True, tile=64, block_major=False), shuffled=True, block_major=False)),
        ("scale-t-view", "prod, activation scales as a transposed view (not contiguous)",
         routed("prod", up_gate_64, **shuffled, scale="t-view")),
        ("scale-as-is", "prod, activation scales (tokens, H / 128) untransposed",
         routed("prod", up_gate_64, **shuffled, scale="as-is")),
        ("bf16-weights", "prod, routing weights in bf16", routed("prod", up_gate_64, **shuffled,
                                                                 weights_dtype=torch.bfloat16)),
        ("packed-ids", "prod, routing packed as (id << 16) | bf16 weight", routed("prod", up_gate_64, **shuffled,
                                                                                  packed_ids=True)),
    ]
    for method, label in ((0, "Default"), (2, "DeepSeekV3"), (5, "TopK")):
        out.append((f"method-{method}", f"prod with routing_method_type {method} ({label}) on precomputed routing",
                    routed("prod", up_gate_64, **shuffled, method=method)))
    for n_group, topk_group in ((1, 1), (0, 0), (None, None)):
        for bias_dtype in (torch.float32, torch.bfloat16):
            name = f"logits-g{n_group}-{str(bias_dtype)[6:]}"
            out.append((name, f"the kernel's DeepSeekV3 routing from fp32 logits, n_group={n_group} topk_group="
                              f"{topk_group}, bias {bias_dtype}, routed scale 2.5",
                        lambda x, w, ids, logits, bias, g=n_group, tg=topk_group, bd=bias_dtype: _from_logits(
                            x, prod, logits, bias, top_k, n_group=g, topk_group=tg, bias_dtype=bd)))
    return out


def run_variants(e, hidden, inter, top_k, tokens, dev, log):
    from mstar.utils.fused_moe import prefill

    raw = random_experts(e, hidden, inter, dev)
    gen = torch.Generator(device=dev).manual_seed(1)
    gate_w = torch.randn(e, hidden, device=dev, generator=gen) * 0.05
    bias = torch.randn(e, device=dev, generator=gen) * 0.1
    cases = []
    for t in tokens:
        x = torch.randn(t, hidden, device=dev, generator=gen).to(torch.bfloat16)
        logits = x.float() @ gate_w.T
        w, ids = glm_router(logits, bias, top_k)
        want = reference(x, raw, w, ids)
        tri = prefill.experts(x, raw.w13, raw.w2, raw.s13, raw.s2, w, ids, block_size=(128, 128))
        cases.append((x, logits, w, ids, want, tri))
        log({"check": "triton-vs-reference", "tokens": t, "rel_l2": round(trtllm_fp8.rel_l2(tri, want), 5)})
    tol, matches = trtllm_fp8.PROBE_TOLERANCE, []
    for name, what, run in variants(raw, top_k):
        row = {"variant": name, "what": what}
        try:
            errs, agree = [], []
            for x, logits, w, ids, want, tri in cases:
                out, picked = run(x, w, ids, logits, bias)
                errs.append((trtllm_fp8.rel_l2(out, want), trtllm_fp8.rel_l2(out, tri)))
                if picked is not None:
                    agree.append(_same_sets(picked, ids))
            row["rel_l2_vs_reference"] = [round(a, 4) for a, _ in errs]
            row["rel_l2_vs_triton"] = [round(b, 4) for _, b in errs]
            if agree:
                row["routing_agreement"] = [round(a, 4) for a in agree]
            row["result"] = "MATCH" if all(a <= REFERENCE_TOLERANCE and b <= tol for a, b in errs) else "mismatch"
        except Exception as exc:  # noqa: BLE001 - a variant the kernel refuses is a result too
            row["result"] = f"error: {type(exc).__name__}: {str(exc)[:200]}"
        if row["result"] == "MATCH":
            matches.append(name)
        log(row)
        torch.cuda.synchronize()
    log({"summary": "variants", "tokens": list(tokens), "match": matches})
    return raw, gate_w, bias


def run_routing(gate_w, bias, top_k, tokens, dev, log):
    """flashinfer's standalone trtllm-gen routing (the kernels the fused MoE runs) against GLM's router."""
    from flashinfer.fused_moe.trtllm_gen_routing import trtllm_gen_routing
    from flashinfer.tllm_enums import RoutingMethodType

    gen = torch.Generator(device=dev).manual_seed(2)
    for t in tokens:
        logits = torch.randn(t, gate_w.shape[1], device=dev, generator=gen) @ gate_w.T
        w, ids = glm_router(logits, bias, top_k)
        for n_group, topk_group in ((1, 1), (0, 0)):
            row = {"check": "routing", "tokens": t, "n_group": n_group, "topk_group": topk_group}
            try:
                r = trtllm_gen_routing(logits, bias, RoutingMethodType.DeepSeekV3, top_k, n_group=n_group,
                                       topk_group=topk_group, routed_scaling_factor=2.5)
                picked = r.topk_ids.long()
                row["id_sets_agree"] = round(_same_sets(picked, ids), 4)
                ours = torch.zeros_like(logits).scatter_(1, ids, w)
                theirs = torch.zeros_like(logits).scatter_(1, picked.clamp_min(0), r.topk_weights.float())
                row["weights_rel_l2"] = round(trtllm_fp8.rel_l2(theirs, ours), 5)
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            log(row)


def glm_blocks(e, hidden, inter, top_k, raw, gate_w, bias, dev):
    """GLM's MoE block at these per-rank shapes (bf16, one shared expert of the routed size) twice over the
    same weights: Triton kernels, and the trtllm kernel where its probe passes."""
    from mstar.model.glm52.components.moe import Glm52SparseMoeBlock
    from mstar.model.glm52.config import Glm52ModelConfig
    from mstar.model.glm52.weight_loader import restore_fp32_params
    from mstar.utils.fp8_block import Fp8BlockQuantConfig

    cfg = Glm52ModelConfig(hidden_size=hidden, moe_intermediate_size=inter, n_routed_experts=e,
                           num_experts_per_tok=top_k, n_shared_experts=1,
                           quantization_config=Fp8BlockQuantConfig(weight_block_size=(128, 128)))
    blocks = []
    for take in (False, True):
        with torch.device("meta"):
            block = Glm52SparseMoeBlock(cfg)
        block = block.to(torch.bfloat16)
        restore_fp32_params(block)
        block.to_empty(device=dev)
        torch.manual_seed(3)
        for prm in block.shared_expert.parameters():
            prm.data.normal_(0, 0.02)
        block.gate.weight.data.copy_(gate_w)
        block.gate.e_score_correction_bias.data.copy_(bias)
        exp = block.experts
        for name, t in zip(("gate_up_proj_fp8", "gate_up_proj_scale_inv", "down_proj_fp8", "down_proj_scale_inv"),
                           raw, strict=True):
            getattr(exp, name).data.copy_(t)
        if take:
            block.process_weights_after_loading(dev)
        else:
            with mock.patch.object(trtllm_fp8, "take", lambda *a, **k: None):
                block.process_weights_after_loading(dev)
        blocks.append(block.eval())
    return blocks


def run_timing(e, hidden, inter, top_k, raw, gate_w, bias, tokens, dev, log):
    from flashinfer.autotuner import autotune

    from benchmark.kernels.timing import time_graph

    tri, trt = glm_blocks(e, hidden, inter, top_k, raw, gate_w, bias, dev)
    log({"check": "glm-block", "trtllm_taken": trt._trtllm, "triton_block_trtllm": tri._trtllm})
    if not trt._trtllm:
        return
    gen = torch.Generator(device=dev).manual_seed(4)
    xs = {t: torch.randn(t, hidden, device=dev, generator=gen).to(torch.bfloat16) for t in tokens}
    rows = {}
    with torch.inference_mode():
        for t, x in xs.items():
            want = tri(x)
            rows[t] = {"tokens": t, "triton_us": round(time_graph(lambda: tri(x)), 2),  # noqa: B023
                       "trtllm_us": round(time_graph(lambda: trt(x)), 2),  # noqa: B023
                       "rel_l2_trtllm_vs_triton": round(trtllm_fp8.rel_l2(trt(x), want), 4)}
        try:
            with autotune(True):
                for x in xs.values():
                    trt(x)
            for t, x in xs.items():
                rows[t]["trtllm_tuned_us"] = round(time_graph(lambda: trt(x)), 2)  # noqa: B023
        except Exception:  # noqa: BLE001 - the untuned numbers still stand
            log({"check": "autotune", "error": traceback.format_exc(limit=3)[-400:]})
    for row in rows.values():
        best = min(row.get("trtllm_tuned_us", row["trtllm_us"]), row["trtllm_us"])
        row["speedup"] = round(row["triton_us"] / best, 3)
        log({"check": "glm-block-timing", "shape": [e, hidden, inter, top_k], **row})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", default="256,6144,256,8", help="experts,hidden,per-rank intermediate,top-k")
    p.add_argument("--check-tokens", default="3,64,300")
    p.add_argument("--tokens", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192")
    p.add_argument("--no-time", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    e, hidden, inter, top_k = (int(v) for v in args.shape.split(","))
    dev = torch.device("cuda", torch.cuda.current_device())
    gpu = torch.cuda.get_device_name(dev)
    out = open(args.out, "a") if args.out else None  # noqa: SIM115

    def log(row):
        row = {"gpu": gpu, **row}
        print(json.dumps(row), flush=True)
        if out:
            out.write(json.dumps(row) + "\n")
            out.flush()

    import flashinfer

    log({"start": args.shape, "capability": torch.cuda.get_device_capability(dev), "torch": torch.__version__,
         "flashinfer": flashinfer.__version__})
    check = [int(t) for t in args.check_tokens.split(",")]
    raw, gate_w, bias = run_variants(e, hidden, inter, top_k, check, dev, log)
    run_routing(gate_w, bias, top_k, check, dev, log)
    if not args.no_time:
        run_timing(e, hidden, inter, top_k, raw, gate_w, bias, [int(t) for t in args.tokens.split(",")], dev, log)


if __name__ == "__main__":
    main()

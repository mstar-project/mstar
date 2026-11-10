"""Block-fp8 routed experts on Blackwell (sm_100 / sm_103): flashinfer's trtllm-gen fused MoE
(``trtllm_fp8_block_scale_routed_moe``, DeepSeek fp8: e4m3 weights with fp32 scales per 128 x 128
block, activations quantized per 128 columns), routed by the caller's own router.

The kernel's layout is vLLM 0.33's for it: gemm1 rows ``[up; gate]`` where M* stores ``[gate; up]``
(both halves of the weight and of its scales swapped); each weight's rows shuffled for the kernel's
64-row epilogue tile and blocked along K, ``(E, K / 128, rows, 128)``; the scales otherwise as
stored (the shuffle keeps rows inside their 128-row block). Activation scales go in transposed,
``(H / 128, tokens)``. The repacked weights are the only copy, so a block that takes this path
runs it at every token count.

:func:`take` repacks a block's experts only where its probe passed: once per device and shape,
the kernel against the Triton prefill kernels on the block's own weights. A rank whose probe
fails keeps Triton; both paths return the same partial with no collective inside, so ranks that
disagree still reduce in lockstep.
"""
from __future__ import annotations

import logging
import os
import time
from typing import NamedTuple

import torch

from mstar.platforms import current_platform

logger = logging.getLogger(__name__)

BLOCK = 128
# the epilogue tile vLLM shuffles DeepSeek-fp8 MoE weights for (``shuffle_matrix_a``'s 16-row groups)
EPILOGUE_TILE_M = 64
# flashinfer.tllm_enums.WeightLayout
MAJOR_K, BLOCK_MAJOR_K = 0, 2
# Relative L2 distance from the Triton kernels the probe accepts. Both quantize activations to
# e4m3 per 128 columns, so they differ by rounding (~1e-2); a wrong layout is off by O(1).
PROBE_TOLERANCE = 0.05
PROBE_TOKENS = (8, 200)

# The token counts FlashInfer profiles the kernel's tactics at, at load, keeping the fastest per count for the
# process (vLLM's boot autotunes it too): the decode batches, so the first boot pays ~a minute, not ~5 for every
# bucket; other counts keep the default tactic. Each is one of FlashInfer's own buckets, which calls look up.
TUNE_TOKENS = (1, 2, 4, 8, 16, 32, 64)

# (device, w13 shape, w2 shape, top_k) -> None where the probe passed, else why not
_VERDICTS: dict[tuple, str | None] = {}
_LOGGED: set[tuple] = set()
# (device, w13 shape, top_k, tokens) -> the autotune's summary
_TUNED: dict[tuple, str] = {}
# (device, w13 shape, top_k, scale) -> None where the in-kernel routing's probe passed, else why not
_ROUTING: dict[tuple, str | None] = {}
# where the kernel routes itself, its outputs within this of the same kernel on the reference router's ids and
# weights: they differ by the kernel's bf16 routing weights (~0.4%), a wrong routing by O(1)
ROUTING_TOLERANCE = 0.02


class ExpertWeights(NamedTuple):
    """A block's routed experts: e4m3 bytes (uint8 containers) with fp32 scales per 128 x 128 block,
    gate/up ``(E, 2I, H)`` / ``(E, 2I / 128, H / 128)`` and down ``(E, H, I)`` / ``(E, H / 128, I / 128)``
    as M* stores them, or in the kernel's layout from :func:`pack`."""

    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor


def swap_halves(x: torch.Tensor) -> torch.Tensor:
    """``x (E, 2n, ...)`` with its two halves along dim 1 swapped; its own inverse."""
    e, rows = x.shape[:2]
    return x.reshape(e, 2, rows // 2, *x.shape[2:]).flip(1).reshape(x.shape)


def to_block_major_k(w: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``w (E, M, K)`` as ``(E, K / 128, M, 128)``, row ``r`` of each K block taken from row ``rows[r]``."""
    e, m, k = w.shape
    blocked = w.view(e, m, k // BLOCK, BLOCK).permute(0, 2, 1, 3)
    return blocked[:, :, rows.to(w.device), :].contiguous()


def from_block_major_k(b: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`to_block_major_k`."""
    e, kb, m, block = b.shape
    inverse = torch.empty_like(rows)
    inverse[rows] = torch.arange(m, device=rows.device)
    return b[:, :, inverse.to(b.device), :].permute(0, 2, 1, 3).reshape(e, m, kb * block)


def shuffle_rows(m: int) -> torch.Tensor:
    """flashinfer's row shuffle for ``m`` rows at :data:`EPILOGUE_TILE_M` (``row_indices[new] = old``)."""
    from flashinfer.utils import get_shuffle_matrix_a_row_indices

    return get_shuffle_matrix_a_row_indices(torch.empty(m, BLOCK, dtype=torch.uint8), EPILOGUE_TILE_M)


def pack(raw: ExpertWeights, rows_of=None) -> ExpertWeights:
    """``raw`` in the kernel's layout, rows shuffled by ``rows_of(m)`` (default :func:`shuffle_rows`).
    The gate/up swap is a row permutation too, so it rides the shuffle's gather: one copy per weight."""
    rows_of = rows_of or shuffle_rows
    two_i = raw.w13.shape[1]
    swap = (torch.arange(two_i) + two_i // 2) % two_i
    w13 = to_block_major_k(raw.w13, swap[rows_of(two_i)])
    w2 = to_block_major_k(raw.w2, rows_of(raw.w2.shape[1]))
    return ExpertWeights(w13, swap_halves(raw.s13).contiguous(), w2, raw.s2)


def unavailable(device: torch.device, raw: ExpertWeights, block_size) -> str | None:
    """Why the kernel can't serve these experts here, or None."""
    if current_platform.cuda_capability(device) not in ((10, 0), (10, 3)):
        return "it runs on sm_100 / sm_103"
    if tuple(block_size) != (BLOCK, BLOCK):
        return f"it takes {BLOCK} x {BLOCK} scale blocks, not {tuple(block_size)}"
    _, two_i, hidden = raw.w13.shape
    if two_i % (2 * BLOCK) or hidden % BLOCK:
        return f"hidden {hidden} and intermediate {two_i // 2} must be multiples of {BLOCK}"
    try:
        from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - no flashinfer, or one without the kernel
        return f"flashinfer's trtllm fp8 MoE does not import ({exc!r})"
    return None


def experts(x: torch.Tensor, packed: ExpertWeights, topk_weights: torch.Tensor,
            topk_ids: torch.Tensor) -> torch.Tensor:
    """The routed experts of ``x (T, H)``, weighted and summed per token, from :func:`pack`'s layout."""
    return torch.ops.mstar.trtllm_fp8_moe(x, packed.w13, packed.s13, packed.w2, packed.s2,
                                         topk_weights, topk_ids)


# One opaque op to dynamo: flashinfer's wrapper runs its autotuner's host logic per call.
@torch.library.custom_op("mstar::trtllm_fp8_moe", mutates_args=())
def _experts_op(
    x: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    return launch(x, ExpertWeights(w13, s13, w2, s2), topk_weights, topk_ids)


@_experts_op.register_fake
def _(x, w13, s13, w2, s2, topk_weights, topk_ids):
    return torch.empty_like(x)


def deepseek_v3_route(logits: torch.Tensor, bias: torch.Tensor, top_k: int,
                      scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weights, ids)`` as the kernel's DeepSeekV3 routing with one expert group computes them, in fp32:
    experts by sigmoid(logit) + bias, weights the selected sigmoid scores renormalized (+ 1e-20) times
    ``scale``. That is GLM's noaux_tc router (``NoAuxTcRouter`` with ``norm_topk_prob``)."""
    scores = logits.float().sigmoid()
    ids = torch.topk(scores + bias.float().unsqueeze(0), k=top_k, dim=-1).indices
    w = scores.gather(1, ids)
    return w * scale / (w.sum(dim=-1, keepdim=True) + 1e-20), ids


def experts_from_logits(x: torch.Tensor, packed: ExpertWeights, logits: torch.Tensor, bias: torch.Tensor,
                        top_k: int, scale: float) -> torch.Tensor:
    """:func:`experts` with the routing done inside the kernel (DeepSeekV3, one group) from the router's fp32
    ``(T, E)`` logits and ``bias``: no top-k, id cast or weight copies of its own."""
    return torch.ops.mstar.trtllm_fp8_moe_logits(x, packed.w13, packed.s13, packed.w2, packed.s2,
                                                logits, bias, top_k, scale)


@torch.library.custom_op("mstar::trtllm_fp8_moe_logits", mutates_args=())
def _experts_logits_op(
    x: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    scale: float,
) -> torch.Tensor:
    return launch_from_logits(x, ExpertWeights(w13, s13, w2, s2), logits, bias, top_k, scale)


@_experts_logits_op.register_fake
def _(x, w13, s13, w2, s2, logits, bias, top_k, scale):
    return torch.empty_like(x)


def launch_from_logits(x, packed: ExpertWeights, logits, bias, top_k: int, scale: float) -> torch.Tensor:
    """One call of the monolithic kernel, routing from the logits (vLLM routes GLM-5.x on Blackwell this way)."""
    from flashinfer.fused_moe import trtllm_fp8_block_scale_moe
    from flashinfer.tllm_enums import RoutingMethodType

    from mstar.utils.quant_fp8 import per_token_group_quant_fp8

    e, hidden = packed.w13.shape[0], x.shape[1]
    xq, xs = per_token_group_quant_fp8(x.contiguous(), BLOCK)
    out = trtllm_fp8_block_scale_moe(
        logits.float().contiguous(), bias, xq, xs.t().contiguous(),
        packed.w13.view(torch.float8_e4m3fn), packed.s13, packed.w2.view(torch.float8_e4m3fn), packed.s2,
        e, top_k, 1, 1, packed.s2.shape[2] * BLOCK, 0, e, scale,
        routing_method_type=int(RoutingMethodType.DeepSeekV3), use_shuffled_weight=True,
        weight_layout=BLOCK_MAJOR_K,
    )
    out = out[0] if isinstance(out, (list, tuple)) else out
    assert out.shape == (x.shape[0], hidden)
    return out if out.dtype == x.dtype else out.to(x.dtype)


def take_routing(packed: ExpertWeights, hidden: int, bias: torch.Tensor, top_k: int, scale: float) -> bool:
    """Whether the kernel routes these experts itself: once per device and shape, on random logits and the
    block's own bias at 1, 8 and 200 tokens, against the same kernel fed :func:`deepseek_v3_route`'s ids and
    weights, within :data:`ROUTING_TOLERANCE`; then its tactics are tuned in that call form. A failure or an
    exception logs and keeps the router outside the kernel. Ranks decide alone; neither path has a collective."""
    device = packed.w13.device
    key = (device, tuple(packed.w13.shape), top_k, float(scale))
    if key not in _ROUTING:
        try:
            gen = torch.Generator(device=device).manual_seed(1)
            e, worst = packed.w13.shape[0], 0.0
            for tokens in (1, 8, 200):
                x = torch.randn(tokens, hidden, device=device, generator=gen).to(torch.bfloat16)
                logits = 2 * torch.randn(tokens, e, device=device, generator=gen)
                w, ids = deepseek_v3_route(logits, bias, top_k, scale)
                worst = max(worst, rel_l2(experts_from_logits(x, packed, logits, bias, top_k, scale),
                                          experts(x, packed, w, ids)))
            _ROUTING[key] = (None if worst <= ROUTING_TOLERANCE
                             else f"relative error {worst:.3g}, above {ROUTING_TOLERANCE}")
        except Exception as exc:  # noqa: BLE001 - an unproven call form keeps the outside router
            _ROUTING[key] = f"{type(exc).__name__}: {str(exc)[:200]}"
        if _ROUTING[key] is not None:
            logger.warning("Blackwell fp8 MoE: routing stays outside the kernel for experts %s on %s: %s",
                           key[1], device, _ROUTING[key])
        else:
            tuned = autotune(packed, hidden, top_k, device, routing=(bias, float(scale)))
            logger.info("Blackwell fp8 MoE: the kernel routes experts %s on %s itself (DeepSeekV3, one group; "
                        "probe within %s; %s)", key[1], device, ROUTING_TOLERANCE, tuned)
    return _ROUTING[key] is None


def launch(x, packed: ExpertWeights, topk_weights, topk_ids, *, shuffled: bool = True,
           routing_method: int = 1, scale_t: bool = True) -> torch.Tensor:
    """One call of the kernel. The keywords are the production layout; the B300 trials
    (benchmark/kernels/bw_moe_trials.py) try the others. The routing method is unused on
    precomputed routing (vLLM passes Renormalize, 1)."""
    from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe

    from mstar.utils.quant_fp8 import per_token_group_quant_fp8

    e, hidden = packed.w13.shape[0], x.shape[1]
    inter = packed.s2.shape[2] * BLOCK
    xq, xs = per_token_group_quant_fp8(x.contiguous(), BLOCK)
    out = trtllm_fp8_block_scale_routed_moe(
        topk_ids=(topk_ids.to(torch.int32).contiguous(), topk_weights.contiguous()),
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xs.t().contiguous() if scale_t else xs,
        gemm1_weights=packed.w13.view(torch.float8_e4m3fn),
        gemm1_weights_scale=packed.s13,
        gemm2_weights=packed.w2.view(torch.float8_e4m3fn),
        gemm2_weights_scale=packed.s2,
        num_experts=e,
        top_k=topk_ids.shape[1],
        n_group=None,
        topk_group=None,
        intermediate_size=inter,
        local_expert_offset=0,
        local_num_experts=e,
        routed_scaling_factor=None,
        routing_method_type=routing_method,
        use_shuffled_weight=shuffled,
        weight_layout=BLOCK_MAJOR_K if shuffled else MAJOR_K,
    )
    assert out.shape == (x.shape[0], hidden)
    return out if out.dtype == x.dtype else out.to(x.dtype)


def rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    """``||got - want|| / ||want||`` in fp32; inf if ``got`` has a non-finite element."""
    got, want = got.float(), want.float()
    if not torch.isfinite(got).all():
        return float("inf")
    return ((got - want).norm() / want.norm().clamp_min(1e-30)).item()


def probe_routing(tokens: int, num_experts: int, top_k: int, device, generator):
    """Distinct random experts per token, positive fp32 weights summing to 2.5 (GLM's routed scale)."""
    ids = torch.rand(tokens, num_experts, device=device, generator=generator).topk(top_k, dim=-1).indices
    w = torch.rand(tokens, top_k, device=device, generator=generator) + 0.1
    return w / w.sum(dim=-1, keepdim=True) * 2.5, ids


def _probe(raw: ExpertWeights, packed: ExpertWeights, top_k: int) -> str | None:
    """None where the kernel, on ``packed``, matches the Triton prefill kernels on ``raw`` at every
    :data:`PROBE_TOKENS` count within :data:`PROBE_TOLERANCE`; else what it got."""
    from mstar.utils.fused_moe import prefill

    dev = raw.w13.device
    e, _, hidden = raw.w13.shape
    gen = torch.Generator(device=dev).manual_seed(0)
    for tokens in PROBE_TOKENS:
        x = torch.randn(tokens, hidden, device=dev, generator=gen).to(torch.bfloat16)
        w, ids = probe_routing(tokens, e, top_k, dev, gen)
        want = prefill.experts(x, raw.w13, raw.w2, raw.s13, raw.s2, w, ids, block_size=(BLOCK, BLOCK))
        err = rel_l2(experts(x, packed, w, ids), want)
        if not err <= PROBE_TOLERANCE:
            return f"relative error {err:.3g} at {tokens} tokens, above {PROBE_TOLERANCE}"
    return None


def _tune_cache_path(name: str, device: torch.device) -> str | None:
    """One tactic file per GPU (its UUID) and kernel shape, under ``$XDG_CACHE_HOME/mstar/flashinfer_moe_tune``,
    so a later boot on the same GPU loads the tactics instead of profiling them."""
    if device.type != "cuda":
        return None
    import flashinfer

    uuid = str(torch.cuda.get_device_properties(device).uuid)
    major, minor = torch.cuda.get_device_capability(device)
    root = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "mstar",
                        "flashinfer_moe_tune")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"fi{flashinfer.__version__}_sm{major}{minor}_{name}_{uuid}.json")


def autotune(packed: ExpertWeights, hidden: int, top_k: int, device: torch.device,
             tokens: tuple[int, ...] = TUNE_TOKENS, routing: tuple[torch.Tensor, float] | None = None) -> str:
    """Profile the kernel's tactics at ``tokens`` inside ``flashinfer.autotune``: one call at the largest count
    tunes every count, and the calls after it (CUDA graph captures included) look the tactics up. Once per process
    and shape, the experts of a model's layers sharing it; the tactics persist per GPU (:func:`_tune_cache_path`).
    Returns a summary for the log; a failure is logged and keeps the default tactics, never failing the load.
    With ``routing`` (bias, scale) the call that is tuned routes in the kernel (:func:`experts_from_logits`)."""
    e = packed.w13.shape[0]
    key = (device, tuple(packed.w13.shape), top_k, tuple(tokens), routing is not None)
    if key in _TUNED:
        return _TUNED[key]
    try:
        from flashinfer.autotuner import autotune as fi_autotune

        path = _tune_cache_path(f"fp8block_E{e}_I{packed.s2.shape[2] * BLOCK}_H{hidden}_K{top_k}"
                                + ("_dsv3" if routing is not None else ""), device)
        gen = torch.Generator(device=device).manual_seed(0)
        m = max(tokens)
        x = torch.randn(m, hidden, device=device, generator=gen).to(torch.bfloat16)
        w, ids = probe_routing(m, e, top_k, device, gen)
        t = time.time()
        with fi_autotune(True, cache=path, tuning_buckets=tuple(tokens)):
            if routing is None:
                experts(x, packed, w, ids)
            else:
                experts_from_logits(x, packed, 2 * torch.randn(m, e, device=device, generator=gen), routing[0],
                                    top_k, routing[1])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        summary = (f"tactics autotuned at {len(tokens)} token counts {tokens[0]}..{tokens[-1]} in "
                   f"{time.time() - t:.0f} s")
    except Exception as exc:  # noqa: BLE001 - the default tactics still serve
        summary = f"autotune failed, default tactics: {type(exc).__name__}: {str(exc)[:200]}"
        logger.warning("Blackwell fp8 MoE %s", summary)
    _TUNED[key] = summary
    return summary


def take(raw: ExpertWeights, top_k: int, block_size) -> ExpertWeights | None:
    """``raw`` in the kernel's layout where the kernel serves it on this device, else None. The
    probe runs (and logs) once per device and expert shape; the caller swaps the result in for ``raw``."""
    device = raw.w13.device
    key = (device, tuple(raw.w13.shape), tuple(raw.w2.shape), top_k)
    if key not in _VERDICTS:
        reason = unavailable(device, raw, block_size)
        if reason is not None:
            _VERDICTS[key] = reason
            if current_platform.cuda_capability(device) in ((10, 0), (10, 3)):
                logger.warning("Blackwell fp8 MoE: the routed experts %s keep the Triton kernels: %s",
                               key[1], reason)
            return None
    elif _VERDICTS[key] is not None:
        return None
    try:
        packed = pack(raw)
        if key not in _VERDICTS:
            _VERDICTS[key] = _probe(raw, packed, top_k)
    except Exception as exc:  # noqa: BLE001 - any failure of an unproven kernel keeps Triton
        if key in _VERDICTS:  # a later block of a passed shape: it alone keeps Triton
            logger.warning("Blackwell fp8 MoE: repacking experts %s failed (%r); this block keeps the Triton "
                           "kernels", key[1], exc)
            return None
        _VERDICTS[key] = f"{type(exc).__name__}: {exc}"
    verdict = _VERDICTS[key]
    if verdict is not None:
        logger.warning("Blackwell fp8 MoE: flashinfer's trtllm kernel failed its probe on %s for experts %s "
                       "(%s); the routed experts keep the Triton kernels", device, key[1], verdict)
        return None
    if key not in _LOGGED:
        _LOGGED.add(key)
        tuned = autotune(packed, raw.w13.shape[2], top_k, device)
        logger.info("Blackwell fp8 MoE: flashinfer's trtllm kernel passed its probe on %s for experts %s; "
                    "it runs the routed experts at every token count (%s)", device, key[1], tuned)
    return packed

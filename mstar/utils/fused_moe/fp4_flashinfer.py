"""Routed experts in MXFP4 or NVFP4 on B200/B300 (sm_100 / sm_103): FlashInfer's TRT-LLM FP4 MoE.

The checkpoint's packed experts (``moe.QuantExperts``: e2m1 bytes ``(E, rows, K/2)`` and row-major block scales)
are put in the kernel's layout once, in place, after the load: gate/up rows reordered to ``[up; gate]``, each
expert's bytes and scales permuted for the MMA (``_maybe_get_cached_w3_w1_permute_indices``,
``get_w2_permute_indices_with_cache``) and the scales interleaved (``block_scale_interleave``) — the steps of
FlashInfer 0.7.2's ``prepare_trtllm_fp4_weights`` minus its quantizer, which the checkpoint has already been
through. ``trtllm_fp4_block_scale_routed_moe`` then takes M*'s own routing (ids and weights).

- MXFP4: bf16 activations (``W4A16``) where trtllm-gen has the activation (SwiGLU); Kimi's SiTU it ships only
  with ``activation="mxfp8"``, the activations quantized per 32 to e4m3 (``W4A8``: vLLM's
  ``FLASHINFER_TRTLLM_MXFP4_MXFP8``, what its kimi-k31 image serves K3.1's experts with on sm_103, SiTU fused into
  FC1). ``choose`` picks the activations and, for W4A8, probes the kernel on a copy of a few experts first.
- NVFP4 (modelopt, nvidia/*-NVFP4): activations quantized to NVFP4 with the layer's calibrated input scale (the
  largest of its experts'), dequant scalars per expert as vLLM's modelopt path sets them.

Routing: M*'s router's ids and weights go in as computed (the unpacked precomputed form). With ids given,
FlashInfer's routing runs its post-top-k pipeline only (``trtllm_fused_moe_routing_custom.cuh``: "No
routing-method-specific logic needed"), so the routing method neither re-normalizes nor re-activates them and
Kimi's weights (sigmoid scores of the top-k, renormalized, times ``routed_scaling_factor``, as the HF reference
computes) reach the finalize unchanged; ``TopK`` (no pre- or post-processing) is passed, the method that means it.

``test_quantization_int4.py``'s CUDA cases compare it with the CPU reference on the B300.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

_EPILOGUE_TILE_M = 128  # the TRT-LLM kernels' tile along a GEMM's rows
_SUPPORTED = ((10, 0), (10, 3))
# FlashInfer's tuning range for the FP4 MoE: its tactics are kept per "hybrid" token bucket up to this many tokens
# (1..256 in powers of 2, then steps of 256, 512, ...), a call's tokens rounded up to one; every call and the boot
# tuning pass the same value, so a tuned bucket is the one a call looks up (8192: FlashInfer's default and
# Kimi K3.1's prefill chunk)
TUNE_MAX_TOKENS = 8192
# the buckets the boot tuning profiles by default: every one a call can look up (FlashInfer's hybrid buckets under
# TUNE_MAX_TOKENS): decode 1..256, and prefill in steps of 256 to 2048, of 512 to 4096, then 8192. M* captures prefill
# at 512..8192 tokens (prefill_token_buckets; 6144 looks up 8192) and runs other counts eager, so every prefill
# bucket is served (B300 slot 8: a 3072-token step on the default tactic); a bucket left out keeps that default
TUNE_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 8192)


def unsupported_activation(activation, fmt: str = "mxfp4", activation_dtype: str = "bf16") -> str | None:
    """Why the TRT-LLM FP4 MoE can't run the experts' gated activation, None when it can: SwiGLU, or a SiTU
    with both betas (its ``activation_type`` Situ takes them per expert as ``gemm1_alpha`` / ``gemm1_beta``),
    except SiTU on MXFP4 weights with bf16 activations: FlashInfer 0.7.2.post1's trtllm-gen ships no
    ``MxE2m1 x Bfloat16`` SiTuGlu batched GEMM (B300 04c: "No kernel found" at decode, tiles 8 and 16), so
    Kimi's W4A16 experts take the Triton kernel. Its Kimi W4A8 SiTU MoE (``backend="cake"``, sm_103a,
    MXFP4 x MXFP8) is not wired here; trtllm-gen's MXFP4 x MXFP8 SiTU (``activation_dtype="mxfp8"``) is."""
    from mstar.utils.situ import SiTU

    if activation == "silu":
        return None
    if isinstance(activation, SiTU) and activation.linear_beta is not None:
        if fmt == "mxfp4" and activation_dtype == "bf16":
            return "FlashInfer's trtllm-gen has no MXFP4 x bf16 SiTU kernel"
        return None
    return f"the TRT-LLM FP4 MoE has no {activation!r} activation"


def _gated_act(activation, num_experts: int, device: torch.device):
    """``(activation_type, gemm1_alpha, gemm1_beta)``. FlashInfer's SiTU is ``beta*tanh(x0/beta) *
    alpha*tanh(x1/alpha)*sigmoid(x1)`` over ``[up; gate]`` rows (x0 up, x1 gate): ``alpha`` is the gate's
    beta, ``beta`` the up half's linear beta."""
    from flashinfer.tllm_enums import ActivationType

    from mstar.utils.situ import SiTU

    if not isinstance(activation, SiTU):
        return ActivationType.Swiglu.value, None, None
    full = lambda v: torch.full((num_experts,), float(v), dtype=torch.float32, device=device)  # noqa: E731
    return ActivationType.Situ.value, full(activation.beta), full(activation.linear_beta)


def choose(block, device: torch.device) -> tuple[str | None, str]:
    """``(activation dtype, detail)`` when the TRT-LLM FP4 MoE serves ``block``'s experts on ``device``,
    ``(None, why not)`` when it doesn't. MXFP4 experts take bf16 activations where trtllm-gen has the activation,
    else MXFP8 (Kimi's SiTU: no ``MxE2m1 x Bfloat16`` SiTuGlu kernel ships, the ``MxE4m3`` one does); the MXFP8
    choice is probed first (``probe``), since only a launch shows a missing cubin or a layout mismatch."""
    reason = unavailable_reason(device)
    if reason is not None:
        return None, reason
    fmt, activation = block.experts_packed.name, block._experts_activation
    dtype = "bf16"
    if fmt == "mxfp4" and unsupported_activation(activation, fmt, "bf16") is not None:
        dtype = "mxfp8"
    reason = unsupported_activation(activation, fmt, dtype)
    if reason is not None:
        return None, reason
    if dtype == "mxfp8":
        # one probe per process and shape: a model's ~90 MoE layers share the verdict (the first layer's experts)
        key = (fmt, tuple(block._experts_dims()), repr(activation), dtype, str(device))
        if key not in _PROBED:
            _PROBED[key] = probe(block, device, dtype)
        err, detail = _PROBED[key]
        if err is not None:
            return None, err
        return dtype, detail
    return dtype, ""


# (format, expert dims, activation, activation dtype, device) -> the probe's verdict
_PROBED: dict[tuple, tuple[str | None, str]] = {}


class KernelRouting(NamedTuple):
    """A noaux_tc router the TRT-LLM MoE runs itself from the logits (``TrtllmFp4Experts.from_logits``)."""
    bias: torch.Tensor   # (E,) fp32 e_score_correction_bias
    top_k: int
    scale: float         # routed_scaling_factor


def kernel_routing(block) -> tuple[KernelRouting | None, str]:
    """``(routing, "")`` when ``block``'s router can run inside the TRT-LLM MoE, ``(None, why not)`` otherwise.

    The kernel's DeepSeekV3 routing with ``n_group = topk_group = 1`` is FlashInfer's routingCustom with the
    SigmoidBias preprocess and ScaledSumNormalize postprocess (``trtllm_fused_moe_routing_runner.cu``): experts by
    sigmoid(logit) + bias, top-k, weights = sigmoid(logit) x routed_scaling_factor / (sum over the top-k + 1e-20)
    (``deepseek_v3_route``), noaux_tc renormalized exactly as the HF reference routes K3.1 (vLLM routes K3.1 this
    way: sigmoid + bias + renormalize with one expert group selects DeepSeekV3). The weights come out bf16. Only a
    tensor parallel block: every expert's logit is local and no sentinel id is passed."""
    gate = getattr(block, "gate", None)
    bias = getattr(gate, "e_score_correction_bias", None)
    if bias is None or not hasattr(gate, "routed_scaling_factor"):
        return None, "the router is not noaux_tc (sigmoid + e_score_correction_bias)"
    if not getattr(gate, "norm_topk_prob", False):
        return None, "the router does not renormalize its top-k (DeepSeekV3 routing always does)"
    total = getattr(block, "num_experts", None)
    if getattr(block, "num_local_experts", total) != total:
        return None, "expert parallel: another rank's logits are not this kernel's experts"
    e, top_k = block._experts_dims()[0], int(gate.top_k)
    if e % 4 or not 0 < top_k < e or top_k > (8 if e <= 384 else 22):
        return None, f"DeepSeekV3 routing takes experts in 4s and top_k <= {8 if e <= 384 else 22} ({e}, {top_k})"
    return KernelRouting(bias.detach().float().contiguous(), top_k, float(gate.routed_scaling_factor)), ""


def deepseek_v3_route(logits: torch.Tensor, bias: torch.Tensor, top_k: int,
                      scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weights, ids)`` as the kernel's DeepSeekV3 routing (one group) computes them, in fp32 (the kernel's
    weights are bf16): the probe's reference and the CPU tests' contract."""
    scores = logits.float().sigmoid()
    ids = torch.topk(scores + bias.float().unsqueeze(0), k=top_k, dim=-1).indices
    w = scores.gather(1, ids)
    return w * scale / (w.sum(dim=-1, keepdim=True) + 1e-20), ids


class _ProbeBlock:
    """The attributes ``TrtllmFp4Experts.convert`` reads off a MoE block, over copies of its first experts."""

    def __init__(self, block, n: int):
        self.experts = torch.nn.Module()
        for name in ("gate_up_proj_packed", "down_proj_packed", "gate_up_proj_scale", "down_proj_scale"):
            src = getattr(block.experts, name)
            setattr(self.experts, name, torch.nn.Parameter(src[:n].clone(), requires_grad=False))
        self.experts_packed = block.experts_packed
        self.activation = block._experts_activation
        _, two_inter, hidden = block._experts_dims()
        self._dims = (n, two_inter, hidden)

    def _experts_dims(self):
        return self._dims


@torch.no_grad()
def probe(block, device: torch.device, activation: str, experts: int = 16, tokens: tuple[int, ...] = (1, 32, 256, 1024),
          top_k: int = 4, tol: float = 0.1, routing: KernelRouting | None = None) -> tuple[str | None, str]:
    """``(None, detail)`` when the kernel, with ``activation``, agrees with the Triton W4A16 kernel on copies of
    ``block``'s first ``experts`` MXFP4 experts (converted as ``convert`` converts the block), rel L2 below ``tol``
    at every count in ``tokens`` (trtllm-gen picks its tile from the tokens per expert: these span decode's
    smallest to prefill's largest); MXFP8 rounds both GEMMs' inputs to e4m3 per 32, a few percent, while a wrong
    layout or activation is order 1. ``(why not, "")`` otherwise, an exception included: no cubin for these
    dtypes or this tile ("No kernel found", B300 04c), a FlashInfer without the op or the activation.

    With ``routing``, the kernel routes each case itself from random logits (``from_logits``, the block's bias
    over the probed experts) and the reference takes ``deepseek_v3_route``'s ids and weights: the in-kernel
    routing checked against the HF formula on the GPU that serves it."""
    from mstar.model.components.int4_formats import KIND_E2M1
    from mstar.utils.fused_moe.w4a16 import fused_experts_w4a16
    from mstar.utils.situ import activation_spec

    try:
        e = min(experts, block._experts_dims()[0])
        e -= e % 4  # the routing kernel takes expert counts in 4s, past top_k
        top_k = min(top_k, e // 2)
        if top_k < 1:
            return f"too few experts ({block._experts_dims()[0]}) to probe", ""
        shim = _ProbeBlock(block, e)
        hidden = shim._dims[2]
        g = torch.Generator(device=device).manual_seed(0)
        sub = None if routing is None else KernelRouting(routing.bias[:e].to(device), top_k, routing.scale)
        cases = []
        for m in tokens:
            x = torch.randn(m, hidden, generator=g, device=device).to(torch.bfloat16)
            if sub is None:
                logits = None
                ids = torch.rand(m, e, generator=g, device=device).topk(top_k, dim=-1).indices.to(torch.int32)
                weights = torch.rand(m, top_k, generator=g, device=device).softmax(-1)
            else:
                logits = 2 * torch.randn(m, e, generator=g, device=device)
                weights, ids = deepseek_v3_route(logits, sub.bias, top_k, sub.scale)
                ids = ids.to(torch.int32)
            cases.append((x, ids, weights, logits))
        ex = shim.experts
        wants = [fused_experts_w4a16(
            x, ex.gate_up_proj_packed.view(torch.uint8), ex.down_proj_packed.view(torch.uint8),
            ex.gate_up_proj_scale.view(torch.uint8), ex.down_proj_scale.view(torch.uint8), None, None, weights, ids,
            KIND_E2M1, 32, activation_spec(shim.activation), False) for x, ids, weights, _ in cases]
        kernel = TrtllmFp4Experts.convert(shim, device, activation=activation)
        kernel.routing = sub
        errs = []
        for (x, ids, weights, logits), want in zip(cases, wants, strict=True):
            got = kernel(x, weights, ids, False) if logits is None else kernel.from_logits(x, logits)
            torch.cuda.synchronize(device)
            err = ((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12)).item()
            errs.append(err if bool(torch.isfinite(got).all()) else float("inf"))
    except Exception as exc:  # noqa: BLE001 - any failure keeps the experts off this kernel
        return f"the probe failed: {type(exc).__name__}: {str(exc)[:300]}", ""
    worst = max(errs)
    if not worst < tol:
        return (f"the probe disagrees with the Triton W4A16 kernel (rel L2 {worst:.3g} at "
                f"{tokens[errs.index(worst)]} tokens over {e} experts)"), ""
    what = "routing in the kernel, " if routing is not None else ""
    return None, f"{what}probe rel L2 <= {worst:.3g} vs Triton W4A16 at {list(tokens)} tokens"


def probe_routing(block, device: torch.device, activation: str, routing: KernelRouting) -> tuple[str | None, str]:
    """``probe`` with the kernel routing from the logits, once per process and shape."""
    key = ("routing", block.experts_packed.name, tuple(block._experts_dims()), routing.top_k, routing.scale,
           activation, str(device))
    if key not in _PROBED:
        _PROBED[key] = probe(block, device, activation, routing=routing)
    return _PROBED[key]


def routing_experts(num_local: int, skip_invalid: bool) -> int:
    """The expert count the routing kernel is given: this rank's, or with ``skip_invalid`` one past the sentinel id
    ``num_local`` (padding rows, another rank's experts), rounded up to the 4 the routing kernel requires
    (``num_experts % 4 == 0``; 113 at EP8 would refuse); ids past ``local_num_experts`` are never local, so their
    slots are skipped either way."""
    return -(-(num_local + 1) // 4) * 4 if skip_invalid else num_local


def unavailable_reason(device: torch.device) -> str | None:
    from mstar.platforms import current_platform

    capability = current_platform.cuda_capability(device) if device.type == "cuda" else None
    if capability is None or tuple(capability) not in _SUPPORTED:
        return "FlashInfer's TRT-LLM FP4 MoE runs on sm_100 / sm_103 (B200, B300)"
    try:
        from flashinfer.fused_moe.core import trtllm_fp4_block_scale_routed_moe  # noqa: F401
    except ImportError as e:
        return f"FlashInfer's TRT-LLM FP4 MoE is not importable ({e})"
    return None


def _shuffle(w: torch.Tensor, s: torch.Tensor, gated: bool, sf_block: int, cache: dict) -> None:
    """One expert's ``w (rows, K/2)`` and ``s (rows, K/sf_block)`` bytes, permuted in place for the MMA."""
    from flashinfer.fused_moe.core import _maybe_get_cached_w3_w1_permute_indices, get_w2_permute_indices_with_cache
    from flashinfer.quantization.fp4_quantization import block_scale_interleave

    permute = _maybe_get_cached_w3_w1_permute_indices if gated else get_w2_permute_indices_with_cache
    kw = {"is_gated_act_gemm": True} if gated else {}
    p = permute(cache, w, _EPILOGUE_TILE_M, **kw)
    w.copy_(w[p.to(w.device)].contiguous())
    p_sf = permute(cache, s, _EPILOGUE_TILE_M, num_elts_per_sf=16, **kw)
    s.view(-1).copy_(block_scale_interleave(s[p_sf.to(s.device)].contiguous()).view(-1))


class TrtllmFp4Experts:
    """The converted experts of one MoE block and the call that runs them."""

    def __init__(self, block, fmt: str, activation: str = "bf16"):
        self.block, self.fmt, self.activation = block, fmt, activation
        # set when the kernel routes from the logits itself (``kernel_routing``); the caller then calls
        # ``from_logits`` instead of passing ids and weights
        self.routing: KernelRouting | None = None

    @classmethod
    @torch.no_grad()
    def convert(cls, block, device: torch.device, activation: str = "bf16") -> TrtllmFp4Experts:
        fmt = block.experts_packed.name
        e, two_inter, hidden = block._experts_dims()
        inter = two_inter // 2
        if hidden % 128 or inter % 128:
            raise ValueError(f"the TRT-LLM FP4 MoE needs hidden and expert widths in 128s; got {hidden}, {inter}")
        experts = block.experts
        w1, w2 = ((experts.gate_up_proj_packed, experts.down_proj_packed) if fmt == "mxfp4"
                  else (experts.gate_up_proj, experts.down_proj))
        s1, s2 = experts.gate_up_proj_scale, experts.down_proj_scale
        sf_block = 32 if fmt == "mxfp4" else 16
        cache: dict = {}
        for i in range(e):
            # [gate; up] -> [up; gate], the kernel's order
            w1[i].copy_(torch.cat([w1[i, inter:], w1[i, :inter]]))
            s1[i].copy_(torch.cat([s1[i, inter:], s1[i, :inter]]))
            _shuffle(w1[i], s1[i], True, sf_block, cache)
            _shuffle(w2[i], s2[i], False, sf_block, cache)
        out = cls(block, fmt, activation)
        out.gated = _gated_act(getattr(block, "activation", "silu"), e, device)
        ones = torch.ones(e, dtype=torch.float32, device=device)
        if fmt == "mxfp4":
            out.scalars = (ones, ones, ones)
            out.input_global = None
        else:
            f32 = lambda t: t.view(torch.float32)  # noqa: E731
            in1 = f32(experts.gate_up_proj_input_scale).max()          # one activation scale for every expert
            ws1 = f32(experts.gate_up_proj_scale_2)[:, 0].contiguous()   # gate == up (checked at load)
            in2 = f32(experts.down_proj_input_scale)[:, 0].contiguous()
            ws2 = f32(experts.down_proj_scale_2)[:, 0].contiguous()
            alpha1 = in1 * ws1
            out.scalars = (alpha1 / in2, alpha1, in2 * ws2)
            out.input_global = (1.0 / in1).reshape(1)
        return out

    def takes_mxfp8(self) -> bool:
        """Whether ``from_logits`` takes ``flat`` as MXFP8 ``(values, ue8m0 scales)`` (``mxfp8_cast``), as
        ``_hidden`` would quantize it."""
        return self.fmt == "mxfp4" and self.activation == "mxfp8"

    def _hidden(self, flat):
        if isinstance(flat, tuple):  # already quantized (``takes_mxfp8``): values, scales in rows of K/32
            q, sf = flat
            return q, sf.view(torch.float8_e4m3fn).reshape(q.shape[0], -1)
        if self.fmt == "nvfp4":
            from flashinfer import fp4_quantize

            q, sf = fp4_quantize(flat, global_scale=self.input_global, sf_vec_size=16, sf_use_ue8m0=False,
                                 is_sf_swizzled_layout=False)
            return q, sf.view(torch.float8_e4m3fn).reshape(flat.shape[0], -1)
        if self.activation == "mxfp8":
            from flashinfer.quantization.fp8_quantization import mxfp8_quantize

            q, sf = mxfp8_quantize(flat, is_sf_swizzled_layout=False)
            return q, sf.view(torch.float8_e4m3fn).reshape(flat.shape[0], -1)
        return flat, None

    def __call__(self, flat: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor,
                 skip_invalid: bool) -> torch.Tensor:
        from flashinfer.fused_moe.core import trtllm_fp4_block_scale_routed_moe

        experts = self.block.experts
        e, two_inter, _ = self.block._experts_dims()
        w1, w2 = ((experts.gate_up_proj_packed, experts.down_proj_packed) if self.fmt == "mxfp4"
                  else (experts.gate_up_proj, experts.down_proj))
        from flashinfer.tllm_enums import RoutingMethodType

        hidden, hidden_scale = self._hidden(flat)
        out1, out1_gate, out2 = self.scalars
        act_type, alpha, beta = self.gated
        # precomputed ids and weights: the routing method is not applied to them (module docstring)
        return trtllm_fp4_block_scale_routed_moe(
            (selected_experts.to(torch.int32).contiguous(), routing_weights.to(torch.float32).contiguous()), None,
            hidden, hidden_scale,
            w1, experts.gate_up_proj_scale.view(torch.float8_e4m3fn), None, alpha, beta, None,
            w2, experts.down_proj_scale.view(torch.float8_e4m3fn), None, out1, out1_gate, out2,
            routing_experts(e, skip_invalid), selected_experts.shape[1], None, None, two_inter // 2, 0, e, None,
            routing_method_type=int(RoutingMethodType.TopK), do_finalize=True, activation_type=act_type,
            tune_max_num_tokens=TUNE_MAX_TOKENS, **self._scale_layout(),
        )[0]

    def from_logits(self, flat, logits: torch.Tensor, finalize: bool = True):
        """The routed experts' weighted sum with the routing done in the kernel (``self.routing``: DeepSeekV3,
        one expert group) from the router's ``(tokens, E)`` logits: no top-k, gather or renormalize launches of
        its own. A tensor parallel block's call (``kernel_routing``). ``flat`` may come quantized where
        ``takes_mxfp8``. ``finalize=False`` leaves the top-k sum
        to the caller (``CommGroup.allreduce_moe_finalize_rmsnorm`` folds it into the all-reduce) and returns
        ``(permuted rows, routing weights [tokens, top_k], expanded_idx_to_permuted_idx [tokens, top_k])``."""
        from flashinfer.fused_moe.core import trtllm_fp4_block_scale_moe
        from flashinfer.tllm_enums import RoutingMethodType

        r = self.routing
        experts = self.block.experts
        e, two_inter, _ = self.block._experts_dims()
        w1, w2 = ((experts.gate_up_proj_packed, experts.down_proj_packed) if self.fmt == "mxfp4"
                  else (experts.gate_up_proj, experts.down_proj))
        hidden, hidden_scale = self._hidden(flat)
        out1, out1_gate, out2 = self.scalars
        act_type, alpha, beta = self.gated
        out = trtllm_fp4_block_scale_moe(
            logits.float().contiguous(), r.bias, hidden, hidden_scale,
            w1, experts.gate_up_proj_scale.view(torch.float8_e4m3fn), None, alpha, beta, None,
            w2, experts.down_proj_scale.view(torch.float8_e4m3fn), None, out1, out1_gate, out2,
            e, r.top_k, 1, 1, two_inter // 2, 0, e, r.scale,
            routing_method_type=int(RoutingMethodType.DeepSeekV3), do_finalize=finalize, activation_type=act_type,
            tune_max_num_tokens=TUNE_MAX_TOKENS, **self._scale_layout(),
        )
        if finalize:
            return out[0]
        # the launcher's expanded index is flat, [tokens * top_k] (its routing buffer): the finalize fold reads
        # top_k off its last dim, so a flat one would index past it (tokens from the logits: ``flat`` may be MXFP8's
        # (values, scales))
        permuted, weights, idx = out[:3]
        return permuted, weights, idx.view(logits.shape[0], r.top_k)

    def _scale_layout(self) -> dict:
        """The activation scales' layout, named where FlashInfer takes it (omitting it is deprecated there):
        ``_hidden``'s are linear (one row of K/32 or K/16 per token)."""
        if self.fmt == "mxfp4" and self.activation == "bf16":
            return {}
        try:
            from flashinfer.tllm_enums import SfLayout
        except ImportError:
            return {}
        return {"hidden_states_scale_layout": SfLayout.layout_linear}


# (format, expert dims, top_k, activation dtype, skip_invalid, routed, finalize, buckets, device) -> the summary
_TUNED: dict[tuple, str] = {}


def tune_buckets(tokens=None) -> tuple[int, ...]:
    """``tokens`` (default ``TUNE_TOKENS``) as FlashInfer's tuning buckets: the ones that are hybrid buckets under
    ``TUNE_MAX_TOKENS`` (any other count would be tuned under a key no call looks up)."""
    from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

    hybrid = set(get_hybrid_num_tokens_buckets(TUNE_MAX_TOKENS))
    wanted = sorted({int(t) for t in (tokens or TUNE_TOKENS)})
    dropped = [t for t in wanted if t not in hybrid]
    if dropped:
        raise ValueError(f"routed-expert tune tokens {dropped} are not FlashInfer's hybrid buckets under "
                         f"{TUNE_MAX_TOKENS} tokens: {sorted(hybrid)}")
    return tuple(wanted)


def _tune_cache_path(key: str, device: torch.device) -> str:
    """One tactic file per GPU (by its UUID: a rank's own, kept across boots) and kernel shape, under
    ``MSTAR_FLASHINFER_TUNE_DIR`` (default ``$XDG_CACHE_HOME/mstar/flashinfer_moe_tune``)."""
    import os
    import re

    import flashinfer

    root = os.environ.get("MSTAR_FLASHINFER_TUNE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "mstar", "flashinfer_moe_tune")
    props = torch.cuda.get_device_properties(device)
    gpu = str(getattr(props, "uuid", "")) or f"dev{device.index}"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"fi{flashinfer.__version__}_sm{props.major}{props.minor}_{key}_{gpu}")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, name + ".json")


@torch.no_grad()
def autotune(block, kernel: TrtllmFp4Experts, device: torch.device, tokens=None, skip_invalid: bool = False,
             finalize: bool = True) -> str:
    """vLLM's boot autotune of the FP4 MoE (its ``kernel_warmup``): one call at the largest bucket inside
    ``flashinfer.autotune`` profiles every tactic at every bucket in ``tokens`` and keeps the fastest per bucket for
    the process (calls after it, CUDA graph captures included, look them up; an untuned bucket keeps the default
    tactic). The tactics persist in a per-GPU file, so a later boot on the same GPU, FlashInfer and shapes loads
    them instead of profiling. Once per process and shape: a model's ~90 MoE layers share it. Returns a summary
    for the backend log line; a failure is logged and leaves the default tactics (never fails the load).
    ``finalize=False`` tunes ``from_logits(finalize=False)``, a call of its own to FlashInfer's tuner (its output
    is a ``(tokens, 0)`` placeholder, so the finalized call's tactics don't cover it), into the same file: entering
    FlashInfer's ``autotune(cache=...)`` drops the tactics loaded from any other file, the finalized call's too."""
    import time

    e, two_inter, hidden = block._experts_dims()
    top_k = int(getattr(block, "num_experts_per_tok", 8) or 8)
    try:
        buckets = tune_buckets(tokens)
    except ValueError as exc:
        return f"autotune skipped: {exc}"
    except Exception as exc:  # noqa: BLE001 - no FlashInfer to tune: the default tactics
        return f"autotune skipped: {type(exc).__name__}: {str(exc)[:200]}"
    routed = kernel.routing is not None
    if not finalize and not routed:
        return "autotune skipped: only the kernel-routed call can leave its finalize"
    key = (kernel.fmt, (e, two_inter, hidden), top_k, kernel.activation, skip_invalid, routed, finalize, buckets,
           str(device))
    if key in _TUNED:
        return _TUNED[key]
    try:
        from flashinfer.autotuner import autotune as fi_autotune

        name = (f"{kernel.fmt}x{kernel.activation}_E{e}_I{two_inter // 2}_H{hidden}_K{top_k}"
                + ("_ep" if skip_invalid else "") + ("_dsv3" if routed else ""))
        path = _tune_cache_path(name, device)
        g = torch.Generator(device=device).manual_seed(0)
        m = max(buckets)
        x = torch.randn(m, hidden, generator=g, device=device).to(torch.bfloat16)
        ids = torch.rand(m, e, generator=g, device=device).topk(min(top_k, e), dim=-1).indices.to(torch.int32)
        weights = torch.rand(m, ids.shape[1], generator=g, device=device).softmax(-1)
        t = time.time()
        with fi_autotune(True, cache=path, tuning_buckets=buckets):
            # the call the block serves: its tactics are keyed by it (routing in the kernel or ids passed in)
            if routed:
                kernel.from_logits(x, torch.randn(m, e, generator=g, device=device), finalize=finalize)
            else:
                kernel(x, weights, ids, skip_invalid)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        summary = f"autotuned {len(buckets)} buckets {buckets[0]}..{buckets[-1]} in {time.time() - t:.0f} s ({path})"
    except Exception as exc:  # noqa: BLE001 - the default tactics still serve
        import logging

        summary = f"autotune failed, default tactics: {type(exc).__name__}: {str(exc)[:200]}"
        logging.getLogger(__name__).warning("FlashInfer FP4 MoE %s", summary)
    _TUNED[key] = summary
    return summary

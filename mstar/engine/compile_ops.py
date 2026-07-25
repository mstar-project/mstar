"""torch.library custom ops that make otherwise-untraceable engine calls into
opaque-but-traceable graph nodes, so ``torch.compile(fullgraph=False)`` stops
graph-breaking on them.

Background. The compiled Thinker (``forward_batched`` under
``torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=False)`` then a
full manual CUDA-graph capture) calls several ``@torch.compiler.disable``
methods on :class:`BatchedCacheManager` per layer -- most importantly
``run_attention`` (the FlashInfer paged-attention wrapper). Each disabled call
forces a graph break, which is a fusion boundary: Inductor cannot fuse the
projection / norm / residual chains on either side of it, and the break also
fragments neighbouring traces (the same mechanism the pure-torch RMSNorm fix
exploited to remove ~800 breaks for +4.5%).

A registered custom op is the opposite of ``@torch.compiler.disable``: dynamo
keeps tracing the surrounding code into ONE graph and inserts the op as an
opaque node it never looks inside. Same runtime kernels, but the fusion
boundary at the call disappears. This mirrors vLLM's ``unified_attention``
custom op, which fetches its KV cache + attention metadata from a global
forward-context set OUTSIDE the compiled region and takes only plain tensors +
a layer identifier across the op boundary.

The op cannot receive the stateful ``BatchedCacheManager`` as an argument
(dynamo would trace into it, defeating the point, and custom-op schemas only
admit tensors / scalars). So the manager is published to a module global by the
non-compiled driver just before it invokes the compiled forward -- exactly
vLLM's forward-context pattern.

Gated by ``MSTAR_CUSTOM_OPS=1``; default-off, so the baseline boot path is
untouched.
"""
from __future__ import annotations

import os

import torch

# --------------------------------------------------------------------------
# Active-manager registry (the "forward context").
#
# Set by the non-compiled driver right before it calls a compiled forward
# (see cuda_graph_runner capture loop and BatchedCacheManager.plan_attention).
# The op body reads it at eager/capture time only: during CUDA-graph replay the
# op body does not re-run (its kernels were recorded), so the global matters
# solely on the warmup / capture / eager-serve paths, each of which sets it
# just-in-time.
# --------------------------------------------------------------------------
_ACTIVE_MANAGER = None


def set_active_manager(mgr) -> None:
    global _ACTIVE_MANAGER
    _ACTIVE_MANAGER = mgr


def get_active_manager():
    mgr = _ACTIVE_MANAGER
    assert mgr is not None, (
        "compile_ops: no active BatchedCacheManager. A custom op ran without a "
        "manager published by the driver -- set_active_manager was not called "
        "on this forward path."
    )
    return mgr


def custom_ops_enabled() -> bool:
    return os.environ.get("MSTAR_CUSTOM_OPS", "0") == "1"


# --------------------------------------------------------------------------
# mstar::run_attention -- FlashInfer paged attention + KV write.
#
# Wraps BatchedCacheManager.run_attention. The KV-cache write is a side effect
# on a tensor fetched from the active manager (not an op argument), so it is
# invisible to dynamo -- which is fine: the op's OUTPUT feeds o_proj (no DCE),
# and adjacent layers are strictly data-ordered through the residual stream, so
# there is no reorder hazard. Under the manual CUDA-graph capture, execution
# order is program order regardless. q/k/v are not mutated in place (set_kv_cache
# copies out), so mutates_args is empty.
# --------------------------------------------------------------------------
@torch.library.custom_op("mstar::run_attention", mutates_args=())
def run_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_idx: int
) -> torch.Tensor:
    return get_active_manager().run_attention(q=q, k=k, v=v, layer_idx=layer_idx)


@run_attention.register_fake
def _run_attention_fake(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_idx: int
) -> torch.Tensor:
    # Output shape == q shape: [total_tokens, num_q_heads, head_dim].
    return torch.empty_like(q)


# --------------------------------------------------------------------------
# mstar::fused_experts_fp8 -- block-fp8 w8a8 grouped-GEMM MoE.
#
# Wraps mstar.utils.fused_moe.fp8.fused_experts_fp8, the steady-state kernel.
# Unlike run_attention this op needs no forward-context: the fp8 expert weights
# (w1/s1/w2/s2) are passed in as plain tensors. The one-time lazy quantization
# that produces them (which mutates module state -- frees the bf16 originals --
# and must not be traced) is hoisted OUT of the compiled forward by
# moe.prequantize_fp8_experts, run before compile. By trace time the weights are
# cached, so the caller reads them as ordinary tensors and this op is the only
# thing left at the call site -- no @torch.compiler.disable, no break.
# --------------------------------------------------------------------------
@torch.library.custom_op("mstar::fused_experts_fp8", mutates_args=())
def fused_experts_fp8(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor, w1_scale: torch.Tensor,
    w2_fp8: torch.Tensor, w2_scale: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    reduce_results: bool,
) -> torch.Tensor:
    from mstar.utils.fused_moe.fp8 import fused_experts_fp8 as _impl
    return _impl(
        hidden_states, w1_fp8, w1_scale, w2_fp8, w2_scale,
        topk_weights, topk_ids, reduce_results=reduce_results,
    )


@fused_experts_fp8.register_fake
def _fused_experts_fp8_fake(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor, w1_scale: torch.Tensor,
    w2_fp8: torch.Tensor, w2_scale: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    reduce_results: bool,
) -> torch.Tensor:
    # reduce_results: sum over top_k -> hidden shape. Otherwise the unreduced
    # per-expert partials: [num_tokens, top_k, hidden].
    if reduce_results:
        return torch.empty_like(hidden_states)
    return hidden_states.new_empty(
        (hidden_states.shape[0], topk_ids.shape[1], hidden_states.shape[1])
    )


# --------------------------------------------------------------------------
# mstar::apply_rope -- FlashInfer 1D llama3.1 RoPE (Talker / code-predictor).
#
# Wraps BatchedCacheManager.apply_rope (the @torch.compiler.disable base
# _apply_rope path used by non-MRoPE attention). Same forward-context idea as
# run_attention: pos_ids live on the active manager, so the op takes only q/k +
# the scalar rope params. The manager's apply_rope mutates q/k in place, so the
# op clones its inputs first and rotates the clones -- the op itself reports no
# input mutation (mutates_args=()) and returns fresh tensors, which keeps the
# custom-op aliasing contract clean. Rope params are always concrete here
# (ParallelAttention defaults 1.0/1.0/8192), so the llama3.1 branch is taken.
# --------------------------------------------------------------------------
@torch.library.custom_op("mstar::apply_rope", mutates_args=())
def apply_rope(
    q: torch.Tensor, k: torch.Tensor,
    rope_theta: float, rope_scale: float,
    low_freq_factor: float, high_freq_factor: float, old_context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_active_manager().apply_rope(
        q.clone(), k.clone(),
        rope_theta=rope_theta, rope_scale=rope_scale,
        low_freq_factor=low_freq_factor, high_freq_factor=high_freq_factor,
        old_context_len=old_context_len,
    )


@apply_rope.register_fake
def _apply_rope_fake(
    q: torch.Tensor, k: torch.Tensor,
    rope_theta: float, rope_scale: float,
    low_freq_factor: float, high_freq_factor: float, old_context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # apply_rope returns q/k rotated at the input dtype/shape.
    return torch.empty_like(q), torch.empty_like(k)

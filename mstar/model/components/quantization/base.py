"""Model-agnostic quantization seams for mstar.

mstar has no vLLM-style quant-method abstraction. This module introduces the
minimal seam needed to bolt a kernel backend (currently Marlin W4A16 for the
routed experts) onto a model without the model code knowing which kernel runs:

* :class:`FusedMoEQuantizeMethod` — the interface an MoE block delegates its
  quantized routed-expert GEMM to. A block holds one instance, calls
  :meth:`~FusedMoEQuantizeMethod.prepare` once post-load to transform its loaded
  packed params into the backend's kernel layout, then calls
  :meth:`~FusedMoEQuantizeMethod.apply` each forward. Kimi's
  :class:`~mstar.model.components.quantization.marlin_moe.MarlinMoEMethod` is the
  first implementation; another MoE model can reuse it verbatim.

* :func:`process_weights_after_loading` — a generic post-load pass. mstar builds
  a module on ``meta``, ``to_empty``\\s it, and loads weights, but has no hook to
  finalize a kernel layout on the real device afterwards (Marlin needs a one-time
  repack + workspace alloc). This walker calls a ``process_weights_after_loading``
  method on every submodule that exposes one; it is a no-op for a plain bf16 model.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class FusedMoEQuantizeMethod(Protocol):
    """Kernel backend for a routed-expert (fused-MoE) W4A16 GEMM.

    Implementations own their kernel-format weights after :meth:`prepare` and are
    otherwise stateless. The MoE block that holds the method is responsible for
    freeing the source packed params once :meth:`prepare` has consumed them.
    """

    def prepare(
        self,
        w13_packed: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_packed: torch.Tensor,
        w2_scale: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Transform the loaded compressed-tensors packed expert weights into the
        backend's runtime layout (e.g. Marlin repack), storing them internally.

        ``w13_packed``/``w2_packed`` are int32 ``(E, N, K // pack_factor)`` and
        ``w13_scale``/``w2_scale`` are ``(E, N, K // group_size)`` — the layout
        Kimi's Hook B packed params already carry.
        """
        ...

    def apply(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation: str = "silu",
        reduce_results: bool = True,
    ) -> torch.Tensor:
        """Run the routed-expert GEMM on the prepared weights.

        Mirrors :func:`mstar.utils.fused_moe.fused_experts`: returns
        ``(tokens, hidden)`` when ``reduce_results`` else the per-slot
        ``(tokens, top_k, hidden)`` tensor the TP path all-reduces before folding.
        """
        ...


def process_weights_after_loading(root: nn.Module, device: torch.device) -> None:
    """Finalize kernel layouts across a freshly-loaded module tree.

    Call once after ``load_weights`` and before ``eval()``/CUDA-graph capture. Any
    submodule exposing a ``process_weights_after_loading(device)`` method gets it
    invoked (e.g. a Marlin MoE block repacks its packed experts + allocates a
    workspace). Modules without the method are skipped, so this is a no-op for a
    plain bf16 model.
    """
    for module in root.modules():
        hook = getattr(module, "process_weights_after_loading", None)
        if callable(hook):
            hook(device)

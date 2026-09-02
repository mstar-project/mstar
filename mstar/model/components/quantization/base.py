"""Model-agnostic quantization hooks for post-load kernel layout fixes."""
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
    workspace).
    """
    for module in root.modules():
        hook = getattr(module, "process_weights_after_loading", None)
        if callable(hook):
            hook(device)

"""Kernels for an MoE layer's routed experts.

A FusedMoEBackend takes the tokens, the stacked expert weights (w1 is
(E, 2 * inter, hidden), gate then up, w2 is (E, hidden, inter)) and the
router's top-k pick, and returns the routing-weighted sum over those experts.
A backend with its own weight layout converts the weights once after loading
(prepare_weights).

MoEBackend is what a model or deployment picks. AUTO is Triton everywhere,
which has measured best so far.
"""

from __future__ import annotations

import enum
import functools
from typing import Any

import torch
from torch import nn

from mstar.utils.fused_moe.config import (
    MoEProblem,
    device_config_name,
    load_tuned_configs,
    nearest_config,
)


class MoEBackend(str, enum.Enum):
    AUTO = "auto"
    TRITON = "triton"  # mstar's Triton grouped GEMM, any GPU Triton targets


class FusedMoEBackend:
    """Computes the routed experts of one MoE layer."""

    name: str = "?"

    def __init__(self, problem: MoEProblem):
        self.problem = problem

    def prepare_weights(self, w1: nn.Parameter, w2: nn.Parameter) -> None:
        """Convert freshly loaded expert weights, in place, to this backend's layout."""

    def __call__(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """``(tokens, hidden)`` sum of the ``top_k`` experts' outputs, each
        scaled by its routing weight."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.problem})"


@functools.cache
def _tuned_configs(problem: MoEProblem, device_name: str | None) -> dict[int, dict[str, Any]] | None:
    return load_tuned_configs(problem, device_name)


class TritonFusedMoE(FusedMoEBackend):
    """fused_experts, with a tuned launch config when configs/ has one for
    this GPU and these sizes."""

    name = "triton"

    def __init__(self, problem: MoEProblem):
        super().__init__(problem)
        from mstar.platforms import current_platform

        self._tuned = _tuned_configs(problem, device_config_name())
        # AMD recommends 2 stages for Triton GEMMs on CDNA3 (more can run out
        # of LDS). It's already Triton's HIP default, pinned here anyway.
        self._untuned_extras = {"num_stages": 2} if current_platform.is_rocm() else {}

    def config(self, num_tokens: int) -> dict[str, Any] | None:
        """Launch config for num_tokens tokens. None means the kernel's default."""
        if self._tuned is not None:
            return nearest_config(self._tuned, num_tokens)
        if not self._untuned_extras:
            return None
        from mstar.utils.fused_moe.kernels import get_default_config

        p = self.problem
        return {
            **get_default_config(
                M=num_tokens, E=p.num_experts, N=2 * p.intermediate_size, K=p.hidden_size, top_k=p.top_k,
            ),
            **self._untuned_extras,
        }

    def __call__(self, hidden_states, w1, w2, topk_weights, topk_ids):
        from mstar.utils.fused_moe.runner import fused_experts

        return fused_experts(
            hidden_states, w1, w2, topk_weights, topk_ids,
            config=self.config(hidden_states.shape[0]),
        )


def make_moe_backend(kind: MoEBackend | str, problem: MoEProblem) -> FusedMoEBackend:
    """Build the backend named by kind. Raises if it can't run here."""
    kind = MoEBackend(kind)
    if kind in (MoEBackend.AUTO, MoEBackend.TRITON):
        return TritonFusedMoE(problem)
    raise ValueError(f"unknown MoE backend {kind!r}")

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
from typing import Any, NamedTuple

import torch
from torch import nn

from mstar.utils.fused_moe import kernels


class MoEProblem(NamedTuple):
    """Sizes of one MoE layer's expert GEMMs, minus the token count."""

    num_experts: int
    intermediate_size: int  # per expert and rank, i.e. w2.shape[2]
    hidden_size: int
    top_k: int


class MoEBackend(str, enum.Enum):
    AUTO = "auto"
    TRITON = "triton"  # mstar's Triton grouped GEMM, any GPU Triton targets
    AITER = "aiter"  # AITER's CK / asm fused MoE, AMD Instinct GPUs


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
        skip_invalid: bool = False,
    ) -> torch.Tensor:
        """``(tokens, hidden)`` sum of the ``top_k`` experts' outputs, each
        scaled by its routing weight. With ``skip_invalid``, ``topk_ids`` may
        hold the id ``E`` (one past the last expert), and those slots add
        nothing."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.problem})"


class TritonFusedMoE(FusedMoEBackend):
    """fused_experts, with get_config's tuned launch configs when configs/ has
    a table for this GPU and these sizes."""

    name = "triton"

    def __init__(self, problem: MoEProblem):
        super().__init__(problem)
        from mstar.platforms import current_platform

        # Looked up once here rather than through get_config per call: calls
        # run inside compiled forwards, where the file lookup breaks the graph.
        self._tuned = kernels.tuned_configs(problem.num_experts, 2 * problem.intermediate_size, problem.hidden_size)
        # AMD recommends 2 stages for Triton GEMMs on CDNA3 (more can run out
        # of LDS). It's already Triton's HIP default, pinned here anyway.
        self._untuned_extras = {"num_stages": 2} if current_platform.is_rocm() else {}

    def config(self, num_tokens: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Gate/up and down launch configs for num_tokens tokens. None leaves
        the choice to fused_experts, which then runs the kernel's default."""
        if self._tuned is not None:
            return kernels.nearest_config(self._tuned, num_tokens)
        if not self._untuned_extras:
            return None
        p = self.problem
        config = {
            **kernels.get_default_config(
                M=num_tokens, E=p.num_experts, N=2 * p.intermediate_size, K=p.hidden_size, top_k=p.top_k,
            ),
            **self._untuned_extras,
        }
        return config, config

    def __call__(self, hidden_states, w1, w2, topk_weights, topk_ids, skip_invalid=False):
        from mstar.utils.fused_moe.runner import fused_experts

        return fused_experts(
            hidden_states, w1, w2, topk_weights, topk_ids, skip_invalid=skip_invalid,
            config=self.config(hidden_states.shape[0]),
        )


def make_moe_backend(kind: MoEBackend | str, problem: MoEProblem) -> FusedMoEBackend:
    """Build the backend named by kind. Raises if it can't run here."""
    kind = MoEBackend(kind)
    if kind in (MoEBackend.AUTO, MoEBackend.TRITON):
        return TritonFusedMoE(problem)
    if kind is MoEBackend.AITER:
        from mstar.utils.fused_moe.backend_aiter import AiterFusedMoE

        return AiterFusedMoE(problem)
    raise ValueError(f"unknown MoE backend {kind!r}")

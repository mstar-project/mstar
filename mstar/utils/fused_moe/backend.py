"""Kernels for an MoE layer's routed experts.

A FusedMoEBackend takes the tokens, the stacked expert weights (w1 is
(E, 2 * inter, hidden), gate then up, w2 is (E, hidden, inter)) and the
router's top-k pick, and returns the routing-weighted sum over those experts.
A backend with its own weight layout converts the weights once after loading
(prepare_weights).

MoEBackend is what a model or deployment picks. AUTO is Triton, except for
the expert sizes and GPUs in AITER_PROBLEMS.
"""

from __future__ import annotations

import enum
import functools
import importlib.util
import logging
from typing import NamedTuple

import torch
from torch import nn

from mstar.kernels.table import device_key

logger = logging.getLogger(__name__)


class MoEProblem(NamedTuple):
    """Sizes of one MoE layer's expert GEMMs, minus the token count."""

    num_experts: int
    intermediate_size: int  # per expert and rank, i.e. w2.shape[2]
    hidden_size: int
    top_k: int


class MoEBackend(str, enum.Enum):
    AUTO = "auto"  # see auto_backend
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
        alignment=None,
    ) -> torch.Tensor:
        """``(tokens, hidden)`` sum of the ``top_k`` experts' outputs, each
        scaled by its routing weight. With ``skip_invalid``, ``topk_ids`` may
        hold the id ``E`` (one past the last expert), and those slots add
        nothing. ``alignment`` is a block alignment the router already computed
        (``fused_experts``'s)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.problem})"


class TritonFusedMoE(FusedMoEBackend):
    """fused_experts, which takes its tiles from the GPU's ``moe_gemm`` table
    (``mstar.kernels``) when it has these sizes, else the kernel's defaults."""

    name = "triton"

    def __call__(self, hidden_states, w1, w2, topk_weights, topk_ids, skip_invalid=False, alignment=None):
        from mstar.utils.fused_moe.runner import fused_experts

        return fused_experts(
            hidden_states, w1, w2, topk_weights, topk_ids, skip_invalid=skip_invalid, alignment=alignment,
        )


# Expert sizes AITER's fused MoE runs faster than the Triton kernels with their
# tuned table, by GPU (the table file stem, ``DeviceKey.stems()[0]``). AITER needs its own weight
# layout and a layer keeps one, so the pick holds for every token count. The
# Thinker on MI325X: 4-16 % faster from 2 to 4096 tokens, 3 % slower at 1.
AITER_PROBLEMS: dict[str, frozenset[MoEProblem]] = {
    "AMD_Instinct_MI325X": frozenset({
        MoEProblem(num_experts=128, intermediate_size=768, hidden_size=2048, top_k=8),  # Qwen3-Omni Thinker
    }),
}


def auto_backend(problem: MoEProblem, skips_rows: bool = False) -> MoEBackend:
    """The backend AUTO runs for ``problem`` here. ``skips_rows`` is for layers
    that pass ``skip_invalid``, which only Triton supports."""
    from mstar.platforms import current_platform

    if skips_rows or not current_platform.is_rocm():
        return MoEBackend.TRITON
    key = device_key()
    if key is None or problem not in AITER_PROBLEMS.get(key.stems()[0], ()):
        return MoEBackend.TRITON
    if importlib.util.find_spec("aiter") is None:
        return MoEBackend.TRITON
    _log_auto_aiter(problem)
    return MoEBackend.AITER


@functools.cache
def _log_auto_aiter(problem: MoEProblem) -> None:
    # Named as the Triton table files are, once per expert size rather than per layer.
    logger.info("fused MoE: auto runs AITER for E=%d,N=%d,K=%d", problem.num_experts,
                2 * problem.intermediate_size, problem.hidden_size)


def make_moe_backend(kind: MoEBackend | str, problem: MoEProblem, skips_rows: bool = False) -> FusedMoEBackend:
    """Build the backend named by kind. Raises if it can't run here."""
    kind = MoEBackend(kind)
    if kind is MoEBackend.AUTO:
        kind = auto_backend(problem, skips_rows)
    if kind is MoEBackend.TRITON:
        return TritonFusedMoE(problem)
    if kind is MoEBackend.AITER:
        from mstar.utils.fused_moe.backend_aiter import AiterFusedMoE

        return AiterFusedMoE(problem)
    raise ValueError(f"unknown MoE backend {kind!r}")

"""Marlin W4A16 backend for the routed-expert (fused-MoE) GEMM.

Implements :class:`~mstar.model.components.quantization.base.FusedMoEQuantizeMethod`
for symmetric INT4 (compressed-tensors ``pack-quantized``, group-wise). It consumes
the packed params an MoE block already loaded (the "Hook B" layout Kimi uses —
``(E, N, K // pack_factor)`` int32 weights + ``(E, N, K // group_size)`` bf16 group
scales), repacks them once into Marlin's tiled layout, and thereafter runs the
vendored Marlin kernels (:mod:`mstar.utils.marlin`).

Model-agnostic: any MoE block with the same packed-param convention can hold one.
"""
from __future__ import annotations

import torch

from mstar.utils.marlin import ops as marlin_ops


class MarlinMoEMethod:
    """Marlin routed-expert GEMM backend (symmetric INT4, group-wise).

    Stateful: :meth:`prepare` repacks the loaded packed experts into Marlin
    layout and stores them (plus the workspace) on the instance; the source
    packed params can then be freed by the owning block. :meth:`apply` runs the
    two Marlin GEMMs.
    """

    def __init__(self, *, num_bits: int = 4, group_size: int = 32) -> None:
        if num_bits != 4:
            raise ValueError(f"MarlinMoEMethod supports INT4 only, got num_bits={num_bits}")
        self.num_bits = num_bits
        self.group_size = group_size
        self.pack_factor = 32 // num_bits
        # Populated by prepare():
        self.w13_qweight: torch.Tensor | None = None
        self.w2_qweight: torch.Tensor | None = None
        self.w13_scale: torch.Tensor | None = None
        self.w2_scale: torch.Tensor | None = None
        self.workspace: torch.Tensor | None = None

    def prepare(
        self,
        w13_packed: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_packed: torch.Tensor,
        w2_scale: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Repack Hook-B packed experts into Marlin layout.

        mstar packs experts along the input axis as ``(E, N_out, K_in // pack)``
        (compressed-tensors), whereas Marlin's ``gptq_marlin_moe_repack`` wants
        GPTQ-style ``(E, K_in // pack, N_out)`` — hence the transpose before each
        repack. Scales are permuted the same way. ``w13`` is the fused gate+up
        (``N_out = 2 * shard_inter``, ``K_in = hidden``); ``w2`` is the down
        projection (``N_out = hidden``, ``K_in = shard_inter``).
        """
        pf, gs = self.pack_factor, self.group_size
        E, two_inter, hidden_over_pack = w13_packed.shape
        hidden = hidden_over_pack * pf
        _, w2_hidden, inter_over_pack = w2_packed.shape
        inter = inter_over_pack * pf
        assert w2_hidden == hidden, f"w2 dim1 {w2_hidden} != hidden {hidden}"

        # gate_up: (E, 2*inter, hidden/pack) -> (E, hidden/pack, 2*inter) -> marlin.
        w13_t = w13_packed.transpose(1, 2).contiguous()
        self.w13_qweight = marlin_ops.gptq_marlin_moe_repack(
            w13_t, size_k=hidden, size_n=two_inter, num_bits=self.num_bits
        )
        w13_s = w13_scale.transpose(1, 2).contiguous()  # (E, hidden/gs, 2*inter)
        self.w13_scale = marlin_ops.marlin_moe_permute_scales(
            w13_s, size_k=hidden, size_n=two_inter, group_size=gs
        )

        # down: (E, hidden, inter/pack) -> (E, inter/pack, hidden) -> marlin.
        w2_t = w2_packed.transpose(1, 2).contiguous()
        self.w2_qweight = marlin_ops.gptq_marlin_moe_repack(
            w2_t, size_k=inter, size_n=hidden, num_bits=self.num_bits
        )
        w2_s = w2_scale.transpose(1, 2).contiguous()  # (E, inter/gs, hidden)
        self.w2_scale = marlin_ops.marlin_moe_permute_scales(
            w2_s, size_k=inter, size_n=hidden, group_size=gs
        )

        self.workspace = marlin_ops.marlin_make_workspace(torch.device(device))

    def apply(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation: str = "silu",
        reduce_results: bool = True,
    ) -> torch.Tensor:
        assert self.w13_qweight is not None, "MarlinMoEMethod.apply before prepare()"
        return marlin_ops.fused_marlin_moe(
            x,
            self.w13_qweight,
            self.w2_qweight,
            self.w13_scale,
            self.w2_scale,
            topk_weights,
            topk_ids,
            self.workspace,
            activation=activation,
            reduce_results=reduce_results,
        )

    @staticmethod
    def shapes_are_legal(hidden: int, shard_inter: int, group_size: int) -> bool:
        """Whether the per-rank expert GEMM shapes satisfy Marlin's tile rules.

        Marlin needs ``n % 64 == 0`` and ``k % 128 == 0`` on each GEMM, plus
        ``k % group_size == 0``. gate_up: k=hidden, n=2*shard_inter; down:
        k=shard_inter, n=hidden.
        """
        if group_size not in (-1, 32, 64, 128):
            return False
        checks = [
            hidden % 128 == 0,            # gate_up K
            (2 * shard_inter) % 64 == 0,  # gate_up N
            shard_inter % 128 == 0,       # down K
            hidden % 64 == 0,             # down N
        ]
        if group_size != -1:
            checks += [hidden % group_size == 0, shard_inter % group_size == 0]
        return all(checks)

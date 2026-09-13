"""The gated delta-net mixer, sharded across tensor-parallel ranks.

Sharding is by head: k-heads for q and k, v-heads for v, z, the gates, the
decay parameters and the recurrent state. Head *dims* — ``head_k_dim``,
``head_v_dim``, and so the gated norm's weight — do not shard.

The trap is ``[q|k|v]``. It is a single checkpoint tensor over ``conv_dim``, so
slicing it by rank means slicing each of the three blocks: the obvious
``chunk(conv_dim, tp)`` hands rank 0 the whole of q plus half of k, at every
shape check's blessing. ``_shard_blocks`` is that slice, and it is why the conv
weight and ``in_proj_qkv`` carry loaders of their own.

The engine shards to match without being told: ``DeltaNetGeometry.to_blocks``
marks both pool blocks ``shard_dims=(0,)`` and the pool divides them at build,
so a model declares its geometry whole and the rank arithmetic happens once.

Only ``__init__`` differs from the base — the forward is written against
``self.num_k_heads`` / ``self.key_dim`` and friends, which are this rank's
share, so it is inherited rather than repeated.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.distributed.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.linear_attn import GatedDeltaNet, GDNProjLayout


def _shard_blocks(
    tensor: torch.Tensor, block_sizes: list[int], rank: int, world: int,
) -> torch.Tensor:
    """One rank's slice of a tensor whose dim 0 is ``[block0|block1|...]``.

    Each block is divided separately and the pieces re-joined, so the result
    is this rank's ``[q|k|v]`` rather than a contiguous cut across the three.
    """
    parts, offset = [], 0
    for size in block_sizes:
        per = divide(size, world)
        parts.append(tensor.narrow(0, offset + rank * per, per))
        offset += size
    return torch.cat(parts, dim=0)


class _FusedBlockColumnParallelLinear(MergedColumnParallelLinear):
    """A merged column-parallel linear whose checkpoint tensor arrives whole.

    The parent wants one call per block with a ``loaded_shard_id``; the
    delta-net checkpoint stores a single ``[q|k|v]`` weight. Splitting here
    keeps the weight loader's names one-to-one with the checkpoint's.
    """

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ):
        if loaded_shard_id is not None:
            return super().weight_loader(param, loaded_weight, loaded_shard_id)
        offset = 0
        for shard_id, size in enumerate(self.output_sizes):
            super().weight_loader(
                param, loaded_weight.narrow(0, offset, size), shard_id,
            )
            offset += size
        return None


class ParallelGatedDeltaNet(GatedDeltaNet):
    """``GatedDeltaNet`` over one rank's heads.

    Takes the checkpoint's head counts, not the rank's — the division happens
    here, so callers declare the model once and the same arguments serve any
    degree, ``CommGroup.trivial()`` included.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        layout: GDNProjLayout = GDNProjLayout.SPLIT,
        proj_bias: bool = False,
        conv_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        linear_attn_key: str = "linear_attn",
        state_key: str = "gdn_state",
        comm_group: CommGroup | None = None,
    ):
        if comm_group is None:
            comm_group = CommGroup.trivial()
        tp = comm_group.world_size
        if layout is GDNProjLayout.FUSED and tp > 1:
            # The fused projection is head-interleaved, so a rank's slice is a
            # gather of per-k-head groups rather than a narrow. Nothing ships
            # this layout yet, so it is unwritten rather than wrong.
            raise NotImplementedError(
                "the fused delta-net layout has no tensor-parallel weight "
                "loader yet; it is head-interleaved, so sharding it is a "
                "gather per k-head, not a narrow"
            )

        # The base builds this rank's share, so every view in its forward is
        # already local; only the projections need replacing below, and they
        # are replaced at the same local width.
        super().__init__(
            hidden_size=hidden_size,
            num_k_heads=divide(num_k_heads, tp),
            num_v_heads=divide(num_v_heads, tp),
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            layout=layout,
            proj_bias=proj_bias,
            conv_bias=conv_bias,
            rms_norm_eps=rms_norm_eps,
            linear_attn_key=linear_attn_key,
            state_key=state_key,
        )
        self.comm_group = comm_group
        # Totals describe the checkpoint, which is what the loaders slice.
        self.total_num_k_heads = num_k_heads
        self.total_num_v_heads = num_v_heads
        self.total_key_dim = num_k_heads * head_k_dim
        self.total_value_dim = num_v_heads * head_v_dim
        self._qkv_blocks = [
            self.total_key_dim, self.total_key_dim, self.total_value_dim,
        ]
        # the fused layout keeps the base's plain projections; it refuses tp>1
        # above, so there is nothing to shard
        if layout is GDNProjLayout.SPLIT:
            self.in_proj_qkv = _FusedBlockColumnParallelLinear(
                comm_group=comm_group, input_size=hidden_size,
                output_sizes=self._qkv_blocks, bias=proj_bias,
            )
            self.in_proj_z = ColumnParallelLinear(
                comm_group=comm_group, input_size=hidden_size,
                output_size=self.total_value_dim, bias=proj_bias,
            )
            self.in_proj_a = ColumnParallelLinear(
                comm_group=comm_group, input_size=hidden_size,
                output_size=num_v_heads, bias=proj_bias,
            )
            self.in_proj_b = ColumnParallelLinear(
                comm_group=comm_group, input_size=hidden_size,
                output_size=num_v_heads, bias=proj_bias,
            )
        self.out_proj = RowParallelLinear(
            comm_group=comm_group, input_size=self.total_value_dim,
            output_size=hidden_size, bias=proj_bias,
            input_is_parallel=True, reduce_results=True,
        )
        self._attach_weight_loaders()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _conv_weight_loader(
        self, param: nn.Parameter, loaded: torch.Tensor, shard_id=None,
    ) -> None:
        """The depthwise conv runs over ``[q|k|v]``, so it slices per block."""
        assert shard_id is None
        param.data.copy_(
            _shard_blocks(
                loaded, self._qkv_blocks,
                self.comm_group.rank, self.comm_group.world_size,
            ).view_as(param.data)
        )

    def _head_loader(
        self, param: nn.Parameter, loaded: torch.Tensor, shard_id=None,
    ) -> None:
        """One value per v-head — a plain narrow, unlike the conv."""
        assert shard_id is None
        per = param.data.shape[0]
        param.data.copy_(
            loaded.narrow(0, self.comm_group.rank * per, per).to(param.dtype)
        )

    def _attach_weight_loaders(self) -> None:
        """Bind the loaders that are not a parallel linear's own.

        Re-run from the base's ``_apply``: ``.to(device)`` re-allocates
        Parameters and drops attribute attachments, and it happens before
        weights load. Same reason ``ColumnParallelLinear`` re-attaches.
        """
        conv = getattr(self, "conv1d", None)
        if conv is not None:
            conv.weight.weight_loader = self._conv_weight_loader
        for name in ("A_log", "dt_bias"):
            param = getattr(self, name, None)
            if param is not None:
                param.weight_loader = self._head_loader

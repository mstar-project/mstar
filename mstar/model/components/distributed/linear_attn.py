"""The gated delta-net mixer, sharded across tensor-parallel ranks.

Sharding is by head (k-heads for q/k, v-heads for v, z, gates, decay and
state); head dims, and so the gated norm's weight, do not shard.

``[q|k|v]`` is one checkpoint tensor, so each of its three blocks must be
sliced separately: a plain ``chunk(conv_dim, tp)`` would give rank 0 all of q
plus half of k and still pass shape checks. ``_shard_blocks`` does this, hence
the own loaders on the conv weight and ``in_proj_qkv``.

The state pool shards to match via ``DeltaNetGeometry.to_blocks``
(``shard_dims=(0,)``). The forward is inherited: it reads the rank-local
``self.num_k_heads`` / ``self.key_dim`` etc.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.distributed.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.linear_attn import (
    SPLIT_SHARD_BLOCKS,
    GatedDeltaNet,
    GDNProjLayout,
    gate_pad,
)


def _shard_blocks(
    tensor: torch.Tensor, block_sizes: list[int], rank: int, world: int,
) -> torch.Tensor:
    """One rank's slice of a tensor whose dim 0 is ``[block0|block1|...]``,
    divided block by block and re-joined."""
    parts, offset = [], 0
    for size in block_sizes:
        per = divide(size, world)
        parts.append(tensor.narrow(0, offset + rank * per, per))
        offset += size
    return torch.cat(parts, dim=0)


class _FusedBlockColumnParallelLinear(MergedColumnParallelLinear):
    """A merged column-parallel linear whose checkpoint tensors span blocks.

    ``shard_map`` names the blocks each checkpoint tensor covers (``[q|k|v]``
    is one tensor), so loader names match the checkpoint's.
    """

    def __init__(self, *, shard_map: dict[str, tuple[int, ...]], **kwargs):
        # before super(), which attaches the bound loader below
        self._shard_map = shard_map
        super().__init__(**kwargs)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | int | None = None,
    ):
        # an int is already a block index — only names go through the map
        blocks = (
            (loaded_shard_id,) if isinstance(loaded_shard_id, int)
            else self._shard_map[loaded_shard_id]
        )
        offset = 0
        for block in blocks:
            size = self.output_sizes[block]
            super().weight_loader(
                param, loaded_weight.narrow(0, offset, size), block,
            )
            offset += size


class ParallelGatedDeltaNet(GatedDeltaNet):
    """``GatedDeltaNet`` over one rank's heads.

    Takes the checkpoint's (total) head counts and divides them here.
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
            raise NotImplementedError(
                "the fused delta-net layout has no tensor-parallel weight "
                "loader yet; it is head-interleaved, so sharding it is a "
                "gather per k-head, not a narrow"
            )

        # The base builds this rank's share; only the projections are
        # replaced below, at the same local width.
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
        # Totals describe the checkpoint, which the loaders slice.
        self.total_num_k_heads = num_k_heads
        self.total_num_v_heads = num_v_heads
        self.total_key_dim = num_k_heads * head_k_dim
        self.total_value_dim = num_v_heads * head_v_dim
        self._qkv_blocks = [
            self.total_key_dim, self.total_key_dim, self.total_value_dim,
        ]
        # the fused layout keeps the base's projections (tp>1 refused above)
        if layout is GDNProjLayout.SPLIT:
            # Widths are the checkpoint's; the pad is sized from the *local*
            # head count and scaled up, since alignment is per rank.
            self.in_proj_fused = _FusedBlockColumnParallelLinear(
                comm_group=comm_group, input_size=hidden_size,
                output_sizes=[
                    self.total_key_dim, self.total_key_dim,
                    self.total_value_dim, self.total_value_dim,
                    num_v_heads, gate_pad(self.num_v_heads) * tp, num_v_heads,
                ],
                bias=proj_bias, shard_map=SPLIT_SHARD_BLOCKS,
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

        Re-run from the base's ``_apply``, since ``.to(device)`` drops attached
        attributes before weights load.
        """
        conv = getattr(self, "conv1d", None)
        if conv is not None:
            conv.weight.weight_loader = self._conv_weight_loader
        for name in ("A_log", "dt_bias"):
            param = getattr(self, name, None)
            if param is not None:
                param.weight_loader = self._head_loader
        # the rank's pad block, sized from its local head count
        self._zero_gate_pad()

"""Kimi delta attention sharded by head across tensor-parallel ranks.

q/k/v, the gates' up-projections, ``dt_bias``, ``A_log`` and ``b_proj`` shard by row, ``o_proj``
by column and the conv per q/k/v block; the head-dim weights (``f_a_proj``, ``g_a_proj``, the
norm) are replicated. ``o_proj``'s partials are summed once per layer. A loader takes a
checkpoint tensor whole or already sliced to this rank; the conv's take a shard id, ``q`` /
``k`` / ``v``, since checkpoints store three.
"""
from __future__ import annotations

from dataclasses import replace
from functools import partial

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.kda import KDAConfig, KimiDeltaAttention


def _slice_rows_loader(
    tp_rank: int, tp_size: int, full_rows: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | None = None,
) -> None:
    """Head-block row shard of a checkpoint tensor whose dim 0 is ``full_rows``
    (q/k/v/f_b/g_b/b projections, dt_bias, A_log)."""
    del loaded_shard_id
    rows = divide(full_rows, tp_size)
    if loaded_weight.shape[0] == full_rows:
        loaded_weight = loaded_weight[tp_rank * rows:(tp_rank + 1) * rows]
    elif loaded_weight.shape[0] != rows:
        raise ValueError(
            f"KDA tensor has {loaded_weight.shape[0]} rows; expected the full "
            f"{full_rows} or the per-rank {rows}"
        )
    param.data.copy_(loaded_weight)


def _slice_cols_loader(
    tp_rank: int, tp_size: int, full_cols: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | None = None,
) -> None:
    """Contraction-dim (dim 1) twin of :func:`_slice_rows_loader` for the
    row-parallel ``o_proj``."""
    del loaded_shard_id
    cols = divide(full_cols, tp_size)
    if loaded_weight.shape[1] == full_cols:
        loaded_weight = loaded_weight[:, tp_rank * cols:(tp_rank + 1) * cols]
    elif loaded_weight.shape[1] != cols:
        raise ValueError(
            f"KDA o_proj has {loaded_weight.shape[1]} cols; expected the full "
            f"{full_cols} or the per-rank {cols}"
        )
    param.data.copy_(loaded_weight)


class ParallelKimiDeltaAttention(KimiDeltaAttention):
    """``KimiDeltaAttention`` over one rank's heads, with the checkpoint loaders. Takes the
    checkpoint's total head count."""

    _CONV_SHARD_ROW = {"q": 0, "k": 1, "v": 2}

    def __init__(
        self,
        config: KDAConfig,
        comm_group: CommGroup | None = None,
        dtype: torch.dtype | None = None,
        *,
        kda_key: str = "kda",
        state_key: str = "kda_state",
    ) -> None:
        if dtype is None:
            dtype = torch.get_default_dtype()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        self.tp_size = comm_group.world_size
        self.tp_rank = comm_group.rank
        self.total_num_heads = config.num_heads
        # the base sees this rank's heads only, and its widths follow
        local = replace(config, num_heads=divide(config.num_heads, self.tp_size))
        super().__init__(local, dtype=dtype, kda_key=kda_key, state_key=state_key)
        self.total_qkv_dim = self.total_num_heads * self.head_dim
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        """Bound to the parameters; ``_apply`` re-runs this, since ``.to(...)`` re-allocates
        them before weights load."""
        rank, size = self.tp_rank, self.tp_size
        rows = partial(_slice_rows_loader, rank, size, self.total_qkv_dim)
        self.conv1d.weight.weight_loader = self._conv_weight_loader
        self.q_proj.weight.weight_loader = rows
        self.k_proj.weight.weight_loader = rows
        self.v_proj.weight.weight_loader = rows
        self.forget_gate.f_b_proj.weight.weight_loader = rows
        self.forget_gate.dt_bias.weight_loader = rows
        if self.full_rank_output_gate:
            self.g_proj.weight.weight_loader = rows
        else:
            self.g_b_proj.weight.weight_loader = rows
        heads = partial(_slice_rows_loader, rank, size, self.total_num_heads)
        self.forget_gate.A_log.weight_loader = heads
        self.b_proj.weight.weight_loader = heads
        self.o_proj.weight.weight_loader = partial(
            _slice_cols_loader, rank, size, self.total_qkv_dim,
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders()
        return result

    def _conv_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        branch = self._CONV_SHARD_ROW.get(loaded_shard_id)
        if branch is None:
            raise ValueError(
                f"KDA conv shard id must be q/k/v, got {loaded_shard_id!r}")
        full = (self.total_qkv_dim, 1, self.conv_kernel_size)
        local = (self.qkv_dim, 1, self.conv_kernel_size)
        if tuple(loaded_weight.shape) == full:
            loaded_weight = loaded_weight[
                self.tp_rank * self.qkv_dim:(self.tp_rank + 1) * self.qkv_dim
            ]
        elif tuple(loaded_weight.shape) != local:
            raise ValueError(
                f"{loaded_shard_id}_conv1d.weight has shape "
                f"{tuple(loaded_weight.shape)}; expected {full} or the per-rank {local}"
            )
        rows = slice(branch * self.qkv_dim, (branch + 1) * self.qkv_dim)
        param.data[rows].copy_(loaded_weight)

    # -- TP: sum the o_proj partials once per layer -----------------------

    def _project_out(self, out: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return self.o_proj(out)
        plan = self.mix.current_plan()
        if plan.is_decode or plan.is_verify:
            # o_proj writes its partial straight into the all-reduce buffer
            buf = self.comm_group.all_reduce_buffer(
                hidden_states.shape, self.o_proj.weight.dtype, hidden_states.device)
            return self.comm_group.all_reduce(
                torch.mm(out, self.o_proj.weight.t(), out=buf))
        return self.comm_group.all_reduce(self.o_proj(out))

    def prefill(self, hidden_states, recurrent_state=None, conv_state=None, attention_mask=None):
        output, recurrent_state, conv_state = super().prefill(
            hidden_states, recurrent_state=recurrent_state,
            conv_state=conv_state, attention_mask=attention_mask,
        )
        if self.tp_size > 1:
            output = self.comm_group.all_reduce(output)
        return output, recurrent_state, conv_state

    def decode_step(self, hidden_states, recurrent_state, conv_state):
        output = super().decode_step(hidden_states, recurrent_state, conv_state)
        if self.tp_size > 1:
            output = self.comm_group.all_reduce(output)
        return output

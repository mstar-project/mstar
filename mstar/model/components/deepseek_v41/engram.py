"""DeepSeek V4.1's sharded n-gram lookup and residual-stream gate."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.fp8_linear import Fp8Linear


class DeepSeekV41EngramEmbedding(nn.Module):
    """Row-sharded E4M3 table with one E8M0 scale per 32 columns."""

    def __init__(
        self,
        num_embeddings: int,
        head_dim: int,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings < 1 or head_dim < 1 or head_dim % 32:
            raise ValueError("Engram needs rows and a head dimension divisible by 32")
        self.comm_group = comm_group or CommGroup.trivial()
        self.num_embeddings = num_embeddings
        self.head_dim = head_dim
        self.rows_per_rank = (num_embeddings + self.comm_group.world_size - 1) // self.comm_group.world_size
        self.start = self.comm_group.rank * self.rows_per_rank
        self.end = self.start + self.rows_per_rank
        # Keep the storage as bytes so a parent .to(bfloat16) cannot round it.
        self.weight = nn.Parameter(torch.empty(self.rows_per_rank, head_dim, dtype=torch.uint8), requires_grad=False)
        self.scale = nn.Parameter(
            torch.empty(self.rows_per_rank, head_dim // 32, dtype=torch.uint8),
            requires_grad=False,
        )
        self._attach_loaders()

    def _attach_loaders(self) -> None:
        self.weight.weight_loader = self._load_table_shard
        self.scale.weight_loader = self._load_table_shard

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_loaders()
        return result

    def _load_table_shard(self, param: nn.Parameter, loaded: torch.Tensor, shard_id=None) -> None:
        expected_dtype = torch.float8_e4m3fn if param is self.weight else torch.float8_e8m0fnu
        if loaded.dtype != expected_dtype:
            raise ValueError(f"Engram table needs {expected_dtype} checkpoint values")
        if shard_id is not None or loaded.shape[1:] != param.shape[1:]:
            raise ValueError("Engram table weight shape does not match the declared layout")
        count = max(0, min(self.num_embeddings - self.start, self.rows_per_rank))
        if loaded.shape[0] == self.num_embeddings:
            local = loaded.narrow(0, self.start, count) if count else loaded[:0]
        elif loaded.shape[0] == count:
            # The safetensors iterator can read just this rank's rows. The
            # released table is too large to materialize whole on each rank.
            local = loaded
        else:
            raise ValueError("Engram table weight shape does not match the declared layout")
        if count:
            param.data[:count].copy_(local.view(torch.uint8))
        if count < self.rows_per_rank:
            param.data[count:].zero_()

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        outside = (indices < self.start) | (indices >= self.end)
        local = (indices - self.start).masked_fill(outside, 0)
        values = F.embedding(local, self.weight.view(torch.float8_e4m3fn)).float().unflatten(-1, (-1, 32))
        scales = F.embedding(local, self.scale.view(torch.float8_e8m0fnu)).float().unsqueeze(-1)
        values = (values * scales).flatten(-2).to(torch.bfloat16)
        values = values.masked_fill(outside.unsqueeze(-1), 0)
        if self.comm_group.world_size > 1:
            values = self.comm_group.all_reduce(values)
        return values


class DeepSeekV41Engram(nn.Module):
    """Gate one n-gram value into each hyper-connection stream."""

    def __init__(
        self,
        num_embeddings: int,
        head_dim: int,
        n_hash_cols: int,
        hidden_size: int,
        hc_mult: int,
        *,
        comm_group: CommGroup | None = None,
        norm_eps: float = 1e-20,
        wkv_reference: bool = False,
    ) -> None:
        super().__init__()
        if n_hash_cols < 1 or hidden_size < 1 or hc_mult < 1:
            raise ValueError("Engram dimensions must be positive")
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.n_hash_cols = n_hash_cols
        self.norm_eps = norm_eps
        self.embed = DeepSeekV41EngramEmbedding(num_embeddings, head_dim, comm_group)
        self.wkv = Fp8Linear(
            n_hash_cols * head_dim,
            hidden_size * (hc_mult + 1),
            (32, 32),
            reference=wkv_reference,
        )
        self.q_weight = nn.Parameter(torch.ones(hc_mult, hidden_size, dtype=torch.float32))
        self.k_weight = nn.Parameter(torch.ones(hc_mult, hidden_size, dtype=torch.float32))

    def _apply(self, fn, recurse=True):
        kept = [
            (module, name, getattr(module, name).detach())
            for module, name in ((self, "q_weight"), (self, "k_weight"), (self.wkv, "weight_scale_inv"))
        ]
        result = super()._apply(fn, recurse=recurse)
        for module, name, original in kept:
            param = getattr(module, name)
            param.data = param.data.float() if original.is_meta else original.to(param.device, dtype=torch.float32)
        return result

    def load_wkv_pair(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        """Load the reference's E4M3 weight and E8M0 32x32 block scales."""
        if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu:
            raise ValueError("Engram wkv needs E4M3 weights and E8M0 scales")
        if weight.shape != self.wkv.weight.shape or scale.shape != self.wkv.weight_scale_inv.shape:
            raise ValueError("Engram wkv shape does not match the declared layout")
        self.wkv.weight.weight_loader(self.wkv.weight, weight)
        self.wkv.weight_scale_inv.weight_loader(self.wkv.weight_scale_inv, scale.float())

    def forward(
        self,
        streams: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.hidden_size, self.hidden_size], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        stream = streams.float()
        weight = self.q_weight.float() * self.k_weight.float()
        rstd = torch.rsqrt(stream.square().mean(-1) + self.norm_eps)
        rstd = rstd * torch.rsqrt(key.square().mean(-1) + self.norm_eps)
        dot = (stream * weight * key).sum(-1) * rstd * self.hidden_size**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (stream + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(streams.dtype)

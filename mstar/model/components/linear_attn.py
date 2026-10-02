"""Gated delta-net mixer — the linear-attention counterpart to ``Attention``.

Holds the weights; the recurrent state and kernels belong to the engine,
reached through ``LinearAttnCallable``.

Two checkpoint layouts. ``SPLIT`` (Qwen3.5) has separate ``in_proj_qkv`` /
``in_proj_z`` / ``in_proj_a`` / ``in_proj_b``, held here as one
``in_proj_fused`` GEMM (see ``SPLIT_SHARD_BLOCKS``). ``FUSED`` (Qwen3-Next)
packs ``in_proj_qkvz`` and ``in_proj_ba`` **head-interleaved** by k-head, which
``_project_fused`` undoes and which needs its own loader under TP. Everything
after the flat ``[q|k|v]`` is shared.

Head counts are whatever the caller passes; ``ParallelGatedDeltaNet`` passes
one rank's share, so this module has no distributed import.
"""
from __future__ import annotations

from enum import Enum

import torch
from torch import nn

from mstar.engine.resources.convenience import LinearAttnCallable
from mstar.model.components.norm import RMSNormGated


class GDNProjLayout(Enum):
    SPLIT = "split"
    FUSED = "fused"


# bf16 elements in the 32-byte alignment FlashInfer's bf16 GDN decode kernel
# checks on every call (the default kernel when K = V = 128).
_ALIGN = 16


def gate_pad(num_v_heads: int) -> int:
    """Padding between the ``a`` and ``b`` blocks, in elements.

    The decode kernel takes ``b`` as a raw slice of the fused projection and
    checks it for 32-byte alignment; ``b`` starts ``num_v_heads`` after ``a``,
    so this pad realigns it. Never loaded or read. Sized per rank under TP
    (16 v-heads at TP2 leaves 8, only half a 32-byte line).
    """
    return -num_v_heads % _ALIGN


# Blocks of the fused SPLIT projection, [q|k|v|z|a|pad|b], per checkpoint
# tensor. Shared with the TP subclass, which shards block by block.
SPLIT_SHARD_BLOCKS: dict[str, tuple[int, ...]] = {
    "qkv": (0, 1, 2),
    "z": (3,),
    "a": (4,),
    "b": (6,),
}


class GatedDeltaNet(nn.Module):
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
    ):
        super().__init__()
        # resource labels this layer calls; see components/attention.py
        self._linear_attn_key = linear_attn_key
        self._state_key = state_key
        self.attn = None
        self.pool = None

        self.hidden_size = hidden_size
        self.layout = layout
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim

        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        # v and z carry this many heads per k-head in the fused layout
        self.v_per_k = num_v_heads // num_k_heads

        # [q|k|v|z|a|pad|b], indexed by SPLIT_SHARD_BLOCKS
        self.in_proj_blocks = [
            self.key_dim, self.key_dim, self.value_dim, self.value_dim,
            num_v_heads, gate_pad(num_v_heads), num_v_heads,
        ]
        if layout is GDNProjLayout.SPLIT:
            # One GEMM rather than four: the 2560x32 gate projections run at
            # ~19 GB/s alone at decode; absorbed, they cost 0.3% more bytes.
            self.in_proj_fused = nn.Linear(
                hidden_size, sum(self.in_proj_blocks), bias=proj_bias,
            )
        else:
            self.in_proj_qkvz = nn.Linear(
                hidden_size, self.conv_dim + self.value_dim, bias=proj_bias
            )
            self.in_proj_ba = nn.Linear(hidden_size, 2 * num_v_heads, bias=proj_bias)

        # depthwise; the checkpoint stores [conv_dim, 1, width]
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, kernel_size=conv_kernel_size,
            groups=self.conv_dim, bias=conv_bias,
        )
        # fp32 in the checkpoint, and the kernels require it
        self.A_log = nn.Parameter(
            torch.zeros(self.num_v_heads, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(
            torch.zeros(self.num_v_heads, dtype=torch.float32)
        )

        # weight is [head_v_dim], a head dim, so it does not shard
        self.norm = RMSNormGated(head_v_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=proj_bias)
        self._attach_weight_loaders()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _attach_weight_loaders(self) -> None:
        """Bind the fused projection's loader (needed even unsharded).

        ``_apply`` re-runs this because ``.to(...)`` re-allocates Parameters and
        drops attached attributes, and that happens before weights load.
        """
        proj = getattr(self, "in_proj_fused", None)
        if proj is not None:
            proj.weight.weight_loader = self._fused_in_proj_loader
        self._zero_gate_pad()

    def _zero_gate_pad(self) -> None:
        """Zero the never-loaded pad block: its output is discarded, but NaNs
        in a weight trip finiteness checks and quantization passes."""
        proj = getattr(self, "in_proj_fused", None)
        pad = self.in_proj_blocks[5]
        if proj is None or pad == 0:
            return
        offset = sum(self.in_proj_blocks[:5])
        with torch.no_grad():
            proj.weight[offset:offset + pad].zero_()

    def _apply(self, fn, recurse: bool = True):
        """Keep the fp32 parameters fp32 through any ``.to(dtype)``.

        FlashInfer's GDN kernels assert on these three; the engine's bf16
        module cast at load would otherwise downcast them and fail at the first
        decode.
        """
        out = super()._apply(fn, recurse)
        for param in (
            getattr(self, "A_log", None),
            getattr(self, "dt_bias", None),
            getattr(getattr(self, "norm", None), "weight", None),
        ):
            if param is not None and param.dtype != torch.float32:
                param.data = param.data.float()
        self._attach_weight_loaders()
        return out

    def bind_resources(self, resources: dict) -> None:
        """Resolve the resources this layer calls. See
        ``NodeSubmodule.bind_node_resources``."""
        self.attn = resources.get(self._linear_attn_key)
        self.pool = resources.get(self._state_key)
        # see Attention.bind_resources
        self.mix = LinearAttnCallable(pool=self.pool, attn=self.attn)

    def _project_split(self, x: torch.Tensor):
        """Qwen3.5: one GEMM over [q|k|v|z|a|pad|b], then split.

        The splits are strided column views; the conv kernels take explicit
        strides and ``z``'s reshape splits a unit-stride dim, so nothing copies.
        """
        num_tokens = x.shape[0]
        qkv, z, a, _pad, b = torch.split(
            self.in_proj_fused(x),
            [
                self.conv_dim, self.value_dim, self.num_v_heads,
                gate_pad(self.num_v_heads), self.num_v_heads,
            ],
            dim=-1,
        )
        z = z.reshape(num_tokens, self.num_v_heads, self.head_v_dim)
        return qkv, z, a, b

    def _fused_in_proj_loader(
        self, param: nn.Parameter, loaded: torch.Tensor, shard_id: str,
    ) -> None:
        """Place one checkpoint projection into the fused weight, unsharded."""
        blocks = SPLIT_SHARD_BLOCKS[shard_id]
        offset = sum(self.in_proj_blocks[: blocks[0]])
        param.data.narrow(0, offset, loaded.shape[0]).copy_(loaded)

    def _project_fused(self, x: torch.Tensor):
        """Qwen3-Next: one projection each, head-interleaved.

        Rows group by k-head as [q, k, v_per_k v heads, v_per_k z heads].
        """
        num_tokens = x.shape[0]
        group_qkvz = (
            2 * self.head_k_dim + 2 * self.v_per_k * self.head_v_dim
        )
        qkvz = self.in_proj_qkvz(x).view(num_tokens, self.num_k_heads, group_qkvz)
        ba = self.in_proj_ba(x).view(num_tokens, self.num_k_heads, 2 * self.v_per_k)

        q, k, v, z = torch.split(
            qkvz,
            [
                self.head_k_dim,
                self.head_k_dim,
                self.v_per_k * self.head_v_dim,
                self.v_per_k * self.head_v_dim,
            ],
            dim=-1,
        )
        b, a = torch.split(ba, [self.v_per_k, self.v_per_k], dim=-1)

        # the conv runs over a flat [q|k|v], so undo the interleave here
        qkv = torch.cat(
            [
                q.reshape(num_tokens, -1),
                k.reshape(num_tokens, -1),
                v.reshape(num_tokens, -1),
            ],
            dim=-1,
        )
        z = z.reshape(num_tokens, self.num_v_heads, self.head_v_dim)
        return qkv, z, a.reshape(num_tokens, -1), b.reshape(num_tokens, -1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The label and layer index are cursors set by the caller running the
        stack (``mix.bind_step`` once, then ``mix.set_layer_idx`` per layer),
        as in ``Attention.forward``."""
        num_tokens = hidden_states.shape[0]
        project = (
            self._project_split
            if self.layout is GDNProjLayout.SPLIT
            else self._project_fused
        )
        qkv, z, a, b = project(hidden_states)

        # [conv_dim, 1, width] -> [conv_dim, width] for the kernel
        qkv = self.mix.conv(
            qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
        )
        q, k, v = torch.split(
            qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        q = q.view(num_tokens, self.num_k_heads, self.head_k_dim)
        k = k.view(num_tokens, self.num_k_heads, self.head_k_dim)
        v = v.view(num_tokens, self.num_v_heads, self.head_v_dim)

        core = self.mix(q, k, v, a, b, self.A_log, self.dt_bias)
        core = self.norm(core, z)
        return self.out_proj(core.reshape(num_tokens, self.value_dim))

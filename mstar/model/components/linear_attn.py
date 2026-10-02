"""Gated delta-net mixer — the linear-attention counterpart to ``Attention``.

Holds the projections, the depthwise conv weight, the decay parameters and the
gated output norm; the recurrent state and the kernels behind it belong to the
engine, reached through ``LinearAttnCallable``.

Two checkpoint layouts, because the family disagrees:

``SPLIT`` (Qwen3.5) keeps ``in_proj_qkv`` / ``in_proj_z`` / ``in_proj_a`` /
``in_proj_b`` apart, and ``in_proj_qkv`` is already the flat ``[q|k|v]`` the
conv wants. We hold them as one ``in_proj_fused`` GEMM anyway and let the
loaders place each at its offset — see ``SPLIT_SHARD_BLOCKS``.

``FUSED`` (Qwen3-Next) packs one ``in_proj_qkvz`` and one ``in_proj_ba``, and
they are **head-interleaved** rather than block-concatenated: the rows group by
k-head, each group holding that head's q, k, and its share of v and z. Undoing
that is what ``_project_fused`` does, and it is also why the fused layout needs
a weight loader of its own under tensor parallelism.

Everything downstream of the flat ``[q|k|v]`` is shared between the two.

Head counts here are whatever the caller passes. ``ParallelGatedDeltaNet`` in
``components/distributed`` passes one rank's share and swaps the projections,
so this module stays free of any distributed import — the same direction the
rest of ``components`` runs in.
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


# bf16 elements in the 32 bytes FlashInfer's bf16 GDN decode kernel wants a
# tensor to start on (its fp32 kernel is happy with 16, but the bf16 one is
# the default whenever K = V = 128, and it checks on every call).
_ALIGN = 16


def gate_pad(num_v_heads: int) -> int:
    """Padding between the ``a`` and ``b`` blocks, in elements.

    The decode kernel takes ``a`` and ``b`` as raw slices of the fused
    projection, and the kernel checks each one's data pointer for 32-byte
    alignment. ``a`` is fine — every block before it is a whole number of
    128-wide heads — but ``b`` starts one v-head count later, so it lands
    mid-word whenever that count is not a multiple of 16. This block is what
    pushes it back onto one; it is never loaded and never read. Under tensor
    parallelism the count is the rank's share, which is why the pad is sized
    per rank (16 v-heads at TP2 leaves 8, and 8 x 2 bytes is only half a
    32-byte line).
    """
    return -num_v_heads % _ALIGN


# The blocks of the fused SPLIT projection, in order: [q|k|v|z|a|pad|b]. Each
# checkpoint tensor maps to the blocks it covers — ``in_proj_qkv`` arrives as
# one flat [q|k|v], the other three are a block each. Shared with the
# tensor-parallel subclass, which shards block by block.
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
            # One GEMM rather than four. At decode the two gate projections are
            # 2560x32 each — a single threadblock apiece, ~19 GB/s — and
            # absorbing them into a wide GEMM costs 0.3% more bytes.
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

        # per head: the checkpoint's weight is [head_v_dim], so it is the one
        # thing here that does not shard — a head dim, not a head count
        self.norm = RMSNormGated(head_v_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=proj_bias)
        self._attach_weight_loaders()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _attach_weight_loaders(self) -> None:
        """Bind the loaders that are not a parameter's own.

        The fused projection is four checkpoint tensors in one, so it needs a
        loader even unsharded. ``_apply`` re-runs this because ``.to(...)``
        re-allocates Parameters and drops attribute attachments, and that
        happens before weights load.
        """
        proj = getattr(self, "in_proj_fused", None)
        if proj is not None:
            proj.weight.weight_loader = self._fused_in_proj_loader
        self._zero_gate_pad()

    def _zero_gate_pad(self) -> None:
        """The pad block is never loaded, so give it zeros rather than whatever
        ``torch.empty`` left there: its output column is discarded, but NaNs
        in a weight trip any finiteness check or quantization pass."""
        proj = getattr(self, "in_proj_fused", None)
        pad = self.in_proj_blocks[5]
        if proj is None or pad == 0:
            return
        offset = sum(self.in_proj_blocks[:5])
        with torch.no_grad():
            proj.weight[offset:offset + pad].zero_()

    def _apply(self, fn, recurse: bool = True):
        """Keep the fp32 parameters fp32 through any ``.to(dtype)``.

        FlashInfer's GDN kernels assert on these three, and the checkpoint
        stores them fp32. A whole-module cast to bf16 — which the engine does
        once at load, after the model is built — would otherwise silently
        downcast them and fail at the first decode. Owning the invariant here
        means it survives whoever calls ``.to``.
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
        """Qwen3.5: one GEMM over [q|k|v|z|a|b], then split.

        The splits are column views, so they are strided rather than
        contiguous. Both conv kernels take explicit strides and ``z``'s
        reshape only splits a unit-stride dim, so nothing here copies.
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

        The rows group by k-head; within a group come that head's q, its k, and
        the ``v_per_k`` heads' worth of v and z that belong to it. Reshaping to
        ``[tokens, num_k_heads, group]`` and splitting the last dim undoes it.
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

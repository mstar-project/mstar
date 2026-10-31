# Copyright (c) 2026 OpenBMB.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: the MiniCPM-4 stacks rebuilt on
# M* components over a paged KV cache.
#
# Original file (OpenBMB/VoxCPM, src/voxcpm/modules/minicpm4/model.py) was
# released under Apache-2.0, with the full license text available at
# https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""MiniCPM-4 transformer stacks of VoxCPM2, built from the shared components.

MiniCPM-4 (with ``use_mup`` off, as in VoxCPM2) is a Llama-style pre-norm
decoder: GQA attention without biases or QK-norm, SwiGLU MLP, RMSNorm. Its one
twist is LongRoPE: every rotary frequency is divided by its own factor from
``rope_scaling``, which FlashInfer's RoPE kernels cannot express, so the
rotation here reads a precomputed cos/sin table, the way the reference
``MiniCPMLongRoPE`` does.

* :class:`MiniCPMLM` is a causal stack over the node's paged KV cache, used
  for the text-semantic base LM (with RoPE) and the residual acoustic LM
  (without).
* :class:`MiniCPMLocalTransformer` is a bidirectional stack over a short,
  fixed-length sequence per row (5 tokens for the local encoder, 11 for the
  local DiT), attended densely with SDPA and kept out of the KV resources.

Parameter paths follow the checkpoint (``layers.N.self_attn.q_proj`` ...) once
``LLAMA_STACKED_PARAMS`` routes q/k/v and gate/up into the fused projections.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.decoder_layer import DecoderLayer
from mstar.model.components.distributed import (
    ParallelAttention,
    ParallelGatedMLP,
    VocabParallelEmbedding,
)
from mstar.model.components.norm import RMSNorm
from mstar.model.voxcpm2.config import MiniCPM4Config


def longrope_cos_sin(config: MiniCPM4Config, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``[max_len, head_dim]`` fp32 cos/sin, as ``MiniCPMLongRoPE`` builds them (on the CPU)."""
    dim = config.head_dim
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
    factors = config.rope_factors
    ext = torch.tensor(factors, dtype=torch.float32) if factors else torch.ones(dim // 2)
    if ext.numel() != dim // 2:
        raise ValueError(f"LongRoPE needs {dim // 2} factors, the checkpoint has {ext.numel()}")
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.mul(torch.outer(t, 1.0 / ext), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    scale = config.rope_attention_scale
    return emb.cos() * scale, emb.sin() * scale


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``q``/``k`` (``[..., heads, head_dim]``) in fp32; ``cos``/``sin``
    broadcast over the heads axis (``[..., 1, head_dim]``)."""
    dtype = q.dtype
    qf, kf = q.float(), k.float()
    q = qf * cos + _rotate_half(qf) * sin
    k = kf * cos + _rotate_half(kf) * sin
    return q.to(dtype), k.to(dtype)


class RopeTable(nn.Module):
    """cos/sin buffers shared by every attention layer of one stack.

    Non-persistent and absent from the checkpoint, so the module is built
    empty on ``meta`` and :meth:`materialize` fills it once on the device.
    The reference casts its fp32 tables along with the model, so a bf16 model
    rotates by bf16-rounded values: the tables hold fp32, rounded through the
    dtype the stack was cast to before materializing.
    """

    def __init__(self, config: MiniCPM4Config, max_len: int):
        super().__init__()
        self.config = config
        self.max_len = max_len
        self.register_buffer("cos", torch.empty(max_len, config.head_dim), persistent=False)
        self.register_buffer("sin", torch.empty(max_len, config.head_dim), persistent=False)

    def materialize(self) -> None:
        cos, sin = longrope_cos_sin(self.config, self.max_len)
        self.cos = cos.to(self.cos.dtype).to(device=self.cos.device, dtype=torch.float32)
        self.sin = sin.to(self.sin.dtype).to(device=self.sin.device, dtype=torch.float32)

    def _apply(self, fn, recurse=True):
        # The worker casts each submodule to the autocast dtype; once
        # materialized, the tables follow device moves but stay fp32.
        before = dict(self._buffers)
        super()._apply(fn, recurse)
        for name, old in before.items():
            new = self._buffers[name]
            if old is not None and not old.is_meta and new is not None:
                self._buffers[name] = old.to(device=new.device)
        return self


class MiniCPMPagedAttention(ParallelAttention):
    """``ParallelAttention`` whose RoPE is LongRoPE, read from a table by the
    position resource's planned ids (``pos_key=None`` for the residual LM)."""

    def __init__(self, *, rope: RopeTable | None, **kwargs):
        super().__init__(**kwargs)
        # a plain attribute: the table is registered once, on the owning stack
        self.__dict__["_rope"] = rope

    def _apply_rope(self, q, k, label):
        if self.pos is None or self._rope is None:
            return q, k
        pos_ids = self.pos.pos_ids(label)[: q.shape[0]].long()
        cos = self._rope.cos.index_select(0, pos_ids).unsqueeze(1)
        sin = self._rope.sin.index_select(0, pos_ids).unsqueeze(1)
        return apply_rope(q, k, cos, sin)


def _mlp(config: MiniCPM4Config, comm_group: CommGroup) -> ParallelGatedMLP:
    return ParallelGatedMLP(
        comm_group=comm_group,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        activation="silu",
    )


class MiniCPMLM(nn.Module):
    """Causal MiniCPM-4 stack over the node's paged KV cache.

    ``forward`` maps packed ``[tokens, hidden]`` embeddings to normed hidden
    states. The label and layer index are resource cursors, bound once and
    advanced per layer (see ``Qwen3DenseLM.forward``).
    """

    def __init__(
        self,
        config: MiniCPM4Config,
        *,
        attn_key: str,
        kv_key: str,
        pos_key: str | None,
        max_len: int,
        comm_group: CommGroup | None = None,
    ):
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.config = config
        use_rope = not config.no_rope and pos_key is not None
        self.rope = RopeTable(config, max_len) if use_rope else None
        if config.vocab_size > 0:
            self.embed_tokens = VocabParallelEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                comm_group=comm_group,
            )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    self_attn=MiniCPMPagedAttention(
                        rope=self.rope,
                        comm_group=comm_group,
                        hidden_size=config.num_attention_heads * config.head_dim,
                        num_heads=config.num_attention_heads,
                        num_kv_heads=config.num_key_value_heads,
                        head_dim=config.head_dim,
                        input_hidden_size=config.hidden_size,
                        attn_key=attn_key,
                        kv_key=kv_key,
                        pos_key=pos_key if use_rope else None,
                    ),
                    mlp=_mlp(config, comm_group),
                    input_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                    post_attention_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def materialize_buffers(self) -> None:
        if self.rope is not None:
            self.rope.materialize()

    def forward(self, input_embeds: torch.Tensor, *, label: str) -> torch.Tensor:
        hidden_states = input_embeds
        self.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.attend.set_layer_idx(layer_idx)
            hidden_states = layer(hidden_states=hidden_states)
        return self.norm(hidden_states)


class MiniCPMLocalAttention(ParallelAttention):
    """Bidirectional attention over ``[rows, seq, hidden]``, dense SDPA.

    Reuses the fused QKV / output projections of ``ParallelAttention`` but no
    resource: every row is a complete, short sequence that nothing caches.
    """

    def __init__(self, *, rope: RopeTable, **kwargs):
        super().__init__(pos_key=None, **kwargs)
        self.__dict__["_rope"] = rope

    def bind_resources(self, resources: dict) -> None:
        del resources

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        rows, seq, _ = hidden_states.shape
        q, k, v = self._project_qkv(hidden_states.reshape(rows * seq, -1))
        q = q.view(rows, seq, self.num_heads, self.head_dim)
        k = k.view(rows, seq, self.num_kv_heads, self.head_dim)
        v = v.view(rows, seq, self.num_kv_heads, self.head_dim)
        cos = self._rope.cos[:seq].unsqueeze(1)
        sin = self._rope.sin[:seq].unsqueeze(1)
        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        out = out.transpose(1, 2).reshape(rows * seq, self.num_heads * self.head_dim)
        return self.o_proj(out).view(rows, seq, -1)


class MiniCPMLocalTransformer(nn.Module):
    """Bidirectional MiniCPM-4 stack (no embedding) over fixed-length rows."""

    def __init__(self, config: MiniCPM4Config, max_len: int):
        super().__init__()
        comm_group = CommGroup.trivial()
        self.config = config
        self.rope = RopeTable(config, max_len)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    self_attn=MiniCPMLocalAttention(
                        rope=self.rope,
                        comm_group=comm_group,
                        hidden_size=config.num_attention_heads * config.head_dim,
                        num_heads=config.num_attention_heads,
                        num_kv_heads=config.num_key_value_heads,
                        head_dim=config.head_dim,
                        input_hidden_size=config.hidden_size,
                    ),
                    mlp=_mlp(config, comm_group),
                    input_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                    post_attention_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def materialize_buffers(self) -> None:
        self.rope.materialize()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states)
        return self.norm(hidden_states)

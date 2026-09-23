"""Gated NoPE Multi-head Latent Attention reference (spec D).

Two mathematically identical forms are provided:

* :func:`mla_forward_dense` mirrors the HF module: expand the latent to per-head K/V and
  run causal softmax attention with head dim 192 (128 nope + 64 shared "rope" part that
  is never rotated).
* :func:`mla_forward_absorbed` is the serving form: fold ``W_UK`` into the query and
  ``W_UV`` into the output so attention runs against the 576-wide per-token latent
  ``[kv_a_layernorm(c) | k_rot]`` that the paged cache stores.

Both take the same ``MLAWeights`` and return ``[T, hidden]``; tests assert they agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``KimiRMSNorm``: fp32 normalization, weight multiply in the activation dtype."""
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(dtype)


@dataclass
class MLAWeights:
    q_a_proj: torch.Tensor  # [q_lora, H]
    q_a_layernorm: torch.Tensor  # [q_lora]
    q_b_proj: torch.Tensor  # [heads*192, q_lora]
    kv_a_proj_with_mqa: torch.Tensor  # [512+64, H]
    kv_a_layernorm: torch.Tensor  # [512]
    kv_b_proj: torch.Tensor  # [heads*(128+128), 512]
    g_proj: torch.Tensor | None  # [heads*128, H]
    o_proj: torch.Tensor  # [H, heads*128]
    num_heads: int
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    kv_lora_rank: int = 512
    norm_eps: float = 1e-6

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def scale(self) -> float:
        return self.qk_head_dim ** -0.5

    @classmethod
    def from_hf_module(cls, m) -> MLAWeights:
        return cls(
            q_a_proj=m.q_a_proj.weight,
            q_a_layernorm=m.q_a_layernorm.weight,
            q_b_proj=m.q_b_proj.weight,
            kv_a_proj_with_mqa=m.kv_a_proj_with_mqa.weight,
            kv_a_layernorm=m.kv_a_layernorm.weight,
            kv_b_proj=m.kv_b_proj.weight,
            g_proj=m.g_proj.weight if getattr(m, "use_output_gate", False) else None,
            o_proj=m.o_proj.weight,
            num_heads=m.num_heads,
            qk_nope_head_dim=m.qk_nope_head_dim,
            qk_rope_head_dim=m.qk_rope_head_dim,
            v_head_dim=m.v_head_dim,
            kv_lora_rank=m.kv_lora_rank,
            norm_eps=m.q_a_layernorm.variance_epsilon,
        )


def mla_query(w: MLAWeights, x: torch.Tensor) -> torch.Tensor:
    """``[T, H] -> [T, heads, 192]``."""
    q = F.linear(rms_norm(F.linear(x, w.q_a_proj), w.q_a_layernorm, w.norm_eps), w.q_b_proj)
    return q.view(x.shape[0], w.num_heads, w.qk_head_dim)


def mla_latent(w: MLAWeights, x: torch.Tensor) -> torch.Tensor:
    """The 576-wide cache entry per token: ``[kv_a_layernorm(c) | k_rot]``."""
    ckv = F.linear(x, w.kv_a_proj_with_mqa)
    c, k_rot = ckv.split([w.kv_lora_rank, w.qk_rope_head_dim], dim=-1)
    return torch.cat([rms_norm(c, w.kv_a_layernorm, w.norm_eps), k_rot], dim=-1)


def _causal_mask(t_q: int, t_kv: int, device) -> torch.Tensor:
    # query i (0-based within the new tokens) may attend keys 0 .. (t_kv - t_q + i)
    offset = t_kv - t_q
    i = torch.arange(t_q, device=device)[:, None]
    j = torch.arange(t_kv, device=device)[None, :]
    return j <= (i + offset)


def mla_output_gate_and_proj(w: MLAWeights, attn: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``attn [T, heads, v_head]`` -> gated, projected ``[T, hidden]``."""
    attn = attn.reshape(attn.shape[0], w.num_heads * w.v_head_dim)
    if w.g_proj is not None:
        attn = attn * torch.sigmoid(F.linear(x, w.g_proj))
    return F.linear(attn, w.o_proj)


def mla_forward_dense(
    w: MLAWeights,
    x: torch.Tensor,
    latent_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF-form attention. ``x: [T, hidden]`` are the new tokens; ``latent_cache
    [T_past, 576]`` holds earlier tokens' cache entries. Returns ``(out [T, hidden],
    latent [T, 576])`` where ``latent`` is what a cache would store for the new tokens."""
    t = x.shape[0]
    q = mla_query(w, x)  # [T, heads, 192]
    latent_new = mla_latent(w, x)
    latent = latent_new if latent_cache is None else torch.cat([latent_cache, latent_new], 0)
    c, k_rot = latent.split([w.kv_lora_rank, w.qk_rope_head_dim], dim=-1)
    kv = F.linear(c, w.kv_b_proj).view(latent.shape[0], w.num_heads, w.qk_nope_head_dim + w.v_head_dim)
    k_nope, v = kv.split([w.qk_nope_head_dim, w.v_head_dim], dim=-1)
    k = torch.cat([k_nope, k_rot[:, None, :].expand(-1, w.num_heads, -1)], dim=-1)  # [Tkv, heads, 192]
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * w.scale
    mask = _causal_mask(t, latent.shape[0], x.device)
    scores = scores.masked_fill(~mask[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    attn = torch.einsum("hqk,khd->qhd", probs, v.float()).to(x.dtype)
    return mla_output_gate_and_proj(w, attn, x), latent_new


def absorb_kv_b_proj(w: MLAWeights) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``kv_b_proj`` into ``W_UK [heads, nope, latent]`` and ``W_UV [heads, latent, v]``."""
    kv_b = w.kv_b_proj.view(w.num_heads, w.qk_nope_head_dim + w.v_head_dim, w.kv_lora_rank)
    w_uk = kv_b[:, : w.qk_nope_head_dim, :]  # [heads, 128, 512]
    w_uv = kv_b[:, w.qk_nope_head_dim :, :].transpose(1, 2)  # [heads, 512, 128]
    return w_uk.contiguous(), w_uv.contiguous()


def mla_forward_absorbed(
    w: MLAWeights,
    x: torch.Tensor,
    latent_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Serving-form attention over the 576-wide latent (what FlashInfer's MLA wrapper
    computes): ``q_latent = q_nope @ W_UK`` (512) + ``q_rot`` (64) against
    ``[c | k_rot]``; output ``(probs @ c) @ W_UV``."""
    t = x.shape[0]
    q = mla_query(w, x)
    q_nope, q_rot = q.split([w.qk_nope_head_dim, w.qk_rope_head_dim], dim=-1)
    w_uk, w_uv = absorb_kv_b_proj(w)
    q_lat = torch.einsum("qhn,hnl->qhl", q_nope.float(), w_uk.float())  # [T, heads, 512]
    latent_new = mla_latent(w, x)
    latent = latent_new if latent_cache is None else torch.cat([latent_cache, latent_new], 0)
    c, k_rot = latent.split([w.kv_lora_rank, w.qk_rope_head_dim], dim=-1)
    scores = (
        torch.einsum("qhl,kl->hqk", q_lat, c.float())
        + torch.einsum("qhr,kr->hqk", q_rot.float(), k_rot.float())
    ) * w.scale
    mask = _causal_mask(t, latent.shape[0], x.device)
    scores = scores.masked_fill(~mask[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    o_lat = torch.einsum("hqk,kl->qhl", probs, c.float())  # [T, heads, 512]
    attn = torch.einsum("qhl,hlv->qhv", o_lat, w_uv.float()).to(x.dtype)
    return mla_output_gate_and_proj(w, attn, x), latent_new

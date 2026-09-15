"""Variable-length (packed) self-attention primitives, shared across encoders.

q/k/v are ``(total_tokens, num_heads, head_dim)`` segmented by ``cu_seqlens``. Every
backend computes the same block-diagonal bidirectional attention and is pinned
against the others by ``test_qwen3_omni_varlen_backend_parity``. ``varlen_attention``
is the entry point; under capture it goes through the engine's ragged attention
resource, which the runner plans outside the graph (see ``capture_legal_backend``).
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from flash_attn import flash_attn_varlen_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:  # pragma: no cover
    flash_attn_varlen_func = None
    _FLASH_ATTN_AVAILABLE = False
    logger.warning("flash_attn unavailable; native AuT falls back to SDPA varlen (slow).")

# --------------------------------------------------------------------------- #
# varlen attention primitive (mirrors bagel vit_encoder.run_attention)
# --------------------------------------------------------------------------- #
def _sdpa_varlen_dense(q, k, v, cu_seqlens, scale):
    # Block-diagonal mask O(total_tokens^2). Large-batch TTFT is SDPA-pessimistic.
    # Kept for A/B + parity.
    total_len = q.shape[0]
    seg_ids = torch.zeros(total_len, dtype=torch.int32, device=q.device)
    seg_ids[cu_seqlens[1:-1].long()] = 1
    seg_ids = torch.cumsum(seg_ids, dim=0)
    attn_mask = seg_ids[:, None] == seg_ids[None, :]
    q_b = q.transpose(0, 1).unsqueeze(0)
    k_b = k.transpose(0, 1).unsqueeze(0)
    v_b = v.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=attn_mask, scale=scale)
    return out.squeeze(0).transpose(0, 1).contiguous()


def _sdpa_varlen_per_segment(q, k, v, cu_seqlens, scale):
    # One SDPA kernel per segment: O(sum L_i^2) not O((sum L_i)^2).
    # Mathematically identical to block-diagonal; avoids quadratic cross-segment cost.
    cu = cu_seqlens.tolist()
    out = torch.empty_like(q)
    for a, b in zip(cu[:-1], cu[1:], strict=False):
        qs = q[a:b].transpose(0, 1).unsqueeze(0)
        ks = k[a:b].transpose(0, 1).unsqueeze(0)
        vs = v[a:b].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(qs, ks, vs, scale=scale)
        out[a:b] = o.squeeze(0).transpose(0, 1)
    return out


def _sdpa_varlen_padded(q, k, v, cu_seqlens, scale):
    # Pad-to-max + batched SDPA: one kernel over (n_seg, heads, max_len, head_dim).
    # Best when segments are similar length; wastes work when lengths vary widely.
    cu = cu_seqlens.tolist()
    lens = [b - a for a, b in zip(cu[:-1], cu[1:], strict=False)]
    nseg, max_len = len(lens), max(lens)
    h, d = q.shape[1], q.shape[2]
    qb = q.new_zeros(nseg, max_len, h, d)
    kb = q.new_zeros(nseg, max_len, h, d)
    vb = q.new_zeros(nseg, max_len, h, d)
    mask = torch.zeros(nseg, 1, 1, max_len, device=q.device, dtype=torch.bool)
    for i, (a, b) in enumerate(zip(cu[:-1], cu[1:], strict=False)):
        n = b - a
        qb[i, :n] = q[a:b]
        kb[i, :n] = k[a:b]
        vb[i, :n] = v[a:b]
        mask[i, 0, 0, :n] = True
    qb, kb, vb = (t.permute(0, 2, 1, 3) for t in (qb, kb, vb))
    o = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask, scale=scale).permute(0, 2, 1, 3)
    out = torch.empty_like(q)
    for i, (a, b) in enumerate(zip(cu[:-1], cu[1:], strict=False)):
        out[a:b] = o[i, : b - a]
    return out


def _sdpa_varlen_adaptive(q, k, v, cu_seqlens, scale):
    # Selects dense vs per_segment by mean segment length (no GPU sync, shapes only).
    # Small segs (audio ~100 tok): dense wins — many tiny per-segment launches are overhead-bound.
    # Large segs (vision ~728 tok): per_segment wins — avoids O(total^2) cross-segment mask.
    # Threshold=350 splits audio(104) from vision(728); total cap limits dense memory at extreme batch.
    _DENSE_MEAN_SEG = 350
    _DENSE_TOTAL_CAP = 16384
    total = q.shape[0]
    n_seg = max(cu_seqlens.shape[0] - 1, 1)
    mean_seg = total / n_seg
    if mean_seg < _DENSE_MEAN_SEG and total <= _DENSE_TOTAL_CAP:
        return _sdpa_varlen_dense(q, k, v, cu_seqlens, scale)
    return _sdpa_varlen_per_segment(q, k, v, cu_seqlens, scale)


# --------------------------------------------------------------------------- #
# Cacheless (ragged) varlen self-attention, through the engine resource.
# --------------------------------------------------------------------------- #
try:
    import flashinfer as _flashinfer  # noqa: F401
    _FLASHINFER_AVAILABLE = True
except Exception:  # pragma: no cover
    _flashinfer = None
    _FLASHINFER_AVAILABLE = False

# Set for the duration of a captured region: the RaggedAttnManager the runner
# planned outside the graph for this bucket. The layout lives in that plan, so
# cu_seqlens goes unread on this path.
_fi_override = None


def set_fi_override(state):
    global _fi_override
    _fi_override = state


def _flashinfer_varlen(q, k, v, cu_seqlens, scale):
    """q/k/v: (total_tokens, num_heads, head_dim), packed by cu_seqlens."""
    if _fi_override is None:
        return _sdpa_varlen_adaptive(q, k, v, cu_seqlens, scale)
    return _fi_override.run(q, k, v)


_VARLEN_BACKEND = "flashinfer"
_VARLEN_FALLBACKS = {"adaptive": _sdpa_varlen_adaptive,
                     "per_segment": _sdpa_varlen_per_segment, "dense": _sdpa_varlen_dense,
                     "padded": _sdpa_varlen_padded, "flashinfer": _flashinfer_varlen}


def capture_legal_backend() -> bool:
    """Whether the active backend can run inside a captured graph. Only FlashInfer
    qualifies; the SDPA variants build their mask from host-side segment lengths,
    which a replay cannot re-derive. False means the encoder runs eager."""
    return _FLASHINFER_AVAILABLE and _VARLEN_BACKEND == "flashinfer"


def _sdpa_varlen(q, k, v, cu_seqlens, scale):
    return _VARLEN_FALLBACKS.get(_VARLEN_BACKEND, _sdpa_varlen_per_segment)(
        q, k, v, cu_seqlens, scale)


@torch.compiler.disable
def varlen_attention(q, k, v, cu_seqlens, max_seqlen, scale):
    """q/k/v: (total_tokens, num_heads, head_dim). Bidirectional, packed by cu_seqlens."""
    # During graph capture _fi_override is set: flash-attn's varlen op is not reliably
    # capture-safe for production head dims, so we must use the resource.
    if _fi_override is not None:
        return _flashinfer_varlen(q, k, v, cu_seqlens, scale)
    if _FLASH_ATTN_AVAILABLE:
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=False, softmax_scale=scale,
        )
    return _sdpa_varlen(q, k, v, cu_seqlens, scale)



"""Variable-length (packed) self-attention primitives, shared across encoders.

q/k/v are ``(total_tokens, num_heads, head_dim)`` segmented by ``cu_seqlens``. Every
backend computes the same block-diagonal bidirectional attention and is pinned
against the others by ``test_qwen3_omni_varlen_backend_parity``. ``varlen_attention``
is the entry point; under capture it must use FlashInfer (see ``capture_legal_backend``).

TODO: BAGEL's ViT carries a near-duplicate ``_sdpa_varlen`` that belongs here.
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
# FlashInfer ragged varlen self-attention (capture-legal; plan once, run once).
# --------------------------------------------------------------------------- #
try:
    import flashinfer as _flashinfer
    _FLASHINFER_AVAILABLE = True
except Exception:  # pragma: no cover
    _flashinfer = None
    _FLASHINFER_AVAILABLE = False

# Per-device {workspace, wrapper, last cu_seqlens} — plan once per forward, re-plan only on layout change.
_FI_STATE: dict = {}


def _fi_pad_hd(t, target):
    # FlashInfer Hopper kernel requires head_dim in {64,128,256}; Qwen3-Omni uses 72.
    # Zero-padding to 128 is exact: padded dims contribute 0 to QK^T and 0 to output.
    return t if t.shape[-1] == target else F.pad(t, (0, target - t.shape[-1]))


def make_fi_state(device):
    """Build isolated FlashInfer ragged-prefill wrapper for one CUDA-graph capture key.
    Each key owns its own wrapper planned exactly once — re-planning a shared wrapper
    mutates buffers recorded by a prior capture, silently corrupting replay."""
    if not _FLASHINFER_AVAILABLE:
        return None
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = _flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD")
    return {"ws": ws, "wrapper": wrapper, "cu_obj": None}


def make_fi_graph_state(device, num_segments):
    """Capture-safe ragged-prefill wrapper for PiecewiseCudaGraphRunner.

    Fixed [num_segments+1] indptr buffers + use_cuda_graph=True make FlashInfer
    replan in place, so replanning between replays reallocates nothing the graph
    captured. Shape is a superset of make_fi_state so one override slot drives
    both paths."""
    if not _FLASHINFER_AVAILABLE:
        return None
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    qo_buf = torch.zeros(num_segments + 1, dtype=torch.int32, device=device)
    kv_buf = torch.zeros(num_segments + 1, dtype=torch.int32, device=device)
    wrapper = _flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws, kv_layout="NHD", use_cuda_graph=True,
        qo_indptr_buf=qo_buf, kv_indptr_buf=kv_buf,
    )
    return {"ws": ws, "wrapper": wrapper, "cu_obj": None,
            "external_plan": True, "qo_buf": qo_buf, "kv_buf": kv_buf}


def plan_fi_graph_state(state, seq_lens, num_heads, head_dim, scale, q_dtype):
    """Replan the wrapper with the real per-segment lengths.

    seq_lens arrives zero-padded to the bucket's segment count, so the indptr
    plateaus over the pad tail and those tokens belong to no segment. Bidirectional
    self-attention, hence qo_indptr == kv_indptr."""
    dev = state["qo_buf"].device
    indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=dev)
    indptr[1:] = torch.tensor(seq_lens, dtype=torch.int32, device=dev).cumsum(0)
    Dp = 64 if head_dim <= 64 else (128 if head_dim <= 128 else 256)
    state["wrapper"].plan(indptr, indptr, num_heads, num_heads, Dp,
                          causal=False, sm_scale=float(scale), q_data_type=q_dtype)


# Capture-path accounting: one dict increment, always on, because "ran and did
# During CUDA-graph capture, routes _flashinfer_varlen through a dedicated state
# instead of _FI_STATE so the capture records a wrapper that is never re-planned.
_fi_override: dict | None = None


def set_fi_override(state):
    global _fi_override
    _fi_override = state


def _flashinfer_varlen(q, k, v, cu_seqlens, scale):
    """q/k/v: (total_tokens, num_heads, head_dim), packed/segmented by cu_seqlens.
    Bidirectional self-attention (qo_indptr == kv_indptr == cu_seqlens)."""
    if not _FLASHINFER_AVAILABLE:
        return _sdpa_varlen_adaptive(q, k, v, cu_seqlens, scale)
    dev = q.device
    if _fi_override is not None:
        st = _fi_override
    else:
        st = _FI_STATE.get(dev)
        if st is None:
            st = make_fi_state(dev)
            _FI_STATE[dev] = st
    wrapper = st["wrapper"]
    H, D = q.shape[1], q.shape[2]
    Dp = 64 if D <= 64 else (128 if D <= 128 else 256)   # FlashInfer-supported head_dim
    qp, kp, vp = (_fi_pad_hd(t, Dp).contiguous() for t in (q, k, v))
    # external_plan: the PiecewiseCudaGraphRunner planned this wrapper's fixed
    # index buffers OUTSIDE the graph (plan_attn_fn) before each replay, so a
    # host-side plan() here — illegal inside a captured/replayed region — must be
    # skipped. Without the flag (eager / legacy self-capture) behavior is
    # unchanged: plan once per forward, then run.
    if not st.get("external_plan") and st["cu_obj"] is not cu_seqlens:
        cu = cu_seqlens.to(torch.int32)
        wrapper.plan(cu, cu, H, H, Dp, causal=False, sm_scale=float(scale),
                     q_data_type=q.dtype)
        st["cu_obj"] = cu_seqlens
    out = wrapper.run(qp, kp, vp)
    return out[..., :D].contiguous()


# The only capture-legal backend; SDPA variants remain as fallback + test reference.
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
    # capture-safe for production head dims, so we must use the flashinfer path.
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



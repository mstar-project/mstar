"""Native Qwen3-Omni audio encoder (AuT, Whisper-style).

Mstar reimplementation of HF's Qwen3OmniMoeAudioEncoder, decoupled from
transformers at inference time. Weight names mirror HF exactly so
load_weights_from_hf_shards(..., prefix="thinker.audio_tower") loads with
no remapping. Attention via varlen_attention (flash-attn / FlashInfer /
SDPA fallback). Frontend helpers replicate HF bit-for-bit (parity-tested).
"""
from __future__ import annotations

import logging
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

logger = logging.getLogger(__name__)

# Mirrors HF's BaseModelOutput.last_hidden_state; defined at module scope.
AudioEncoderOutput = namedtuple("AudioEncoderOutput", ["last_hidden_state"])

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
# not regress" and "never ran" look identical in results.json.
_ENCODER_PATH_COUNTS: dict[str, int] = {}
_SEEN_LAYOUTS: set[tuple] = set()
_SEEN_LAYOUTS_CAP = 512          # bounded: a long run must not leak keys
_WARNED_NO_BUCKET: set[str] = set()

# Measured buckets: s2t (1..7 segments, 36..426 tokens) and s2s (1..16, ~1057).
CAPTURE_TOKENS_AUDIO = (48, 64, 96, 128, 192, 256, 384, 512, 704, 896, 1088)
# One bs value: crossed with total_tokens, so extras multiply graphs and 128 MiB
# workspaces, while padding bs up is free.
CAPTURE_BATCH_SIZES_AUDIO = (32,)


def note_encoder_path(path: str) -> None:
    _ENCODER_PATH_COUNTS[path] = _ENCODER_PATH_COUNTS.get(path, 0) + 1


def encoder_path_counts() -> dict[str, int]:
    """Snapshot of {path: count}, e.g. {"vision.piecewise": 96, "vision.eager": 0}."""
    return dict(_ENCODER_PATH_COUNTS)


def note_encoder_layout(kind: str, n_seg: int, total_tokens: int, fitted: bool) -> None:
    """Log once per distinct layout; WARN the first time one fits no bucket —
    the only visible signal that buckets are mis-sized, since the fallback is
    otherwise silent."""
    if not fitted and kind not in _WARNED_NO_BUCKET:
        _WARNED_NO_BUCKET.add(kind)
        logger.warning(
            "%s encoder: NO capture bucket fits segments=%d total_tokens=%d — "
            "falling back to eager. Widen CAPTURE_BATCH_SIZES_%s / "
            "CAPTURE_TOKENS_%s; piecewise numbers from this run are "
            "NOT measuring the captured path.",
            kind, n_seg, total_tokens, kind.upper(), kind.upper(),
        )
    key = (kind, n_seg, total_tokens)
    if key in _SEEN_LAYOUTS:
        return
    if len(_SEEN_LAYOUTS) < _SEEN_LAYOUTS_CAP:
        _SEEN_LAYOUTS.add(key)
    logger.info("%s encoder layout: segments=%d total_tokens=%d bucket=%s",
                kind, n_seg, total_tokens, "HIT" if fitted else "MISS")


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


# --------------------------------------------------------------------------- #
# deterministic frontend helpers (replicate HF exactly)
# --------------------------------------------------------------------------- #
def _feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Post-CNN length per HF ``_get_feat_extract_output_lengths`` (module-level)."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def chunk_and_pad_features(input_features, feature_lens, n_window):
    # PARITY NOTE: pad_sequence pads to the longest chunk in THIS call, not a fixed n_window*2.
    # A short clip batched behind a longer one gets extra zero-padding; Conv2d bias makes the
    # boundary non-zero (~4e-4 fp32 batch-mate dependence). This matches HF exactly —
    # pinning to n_window*2 would be deterministic but diverge from the HF reference.
    chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
    chunk_lengths = torch.full((chunk_num.sum(),), n_window * 2, dtype=torch.long,
                               device=feature_lens.device)
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
    chunk_lengths = torch.where(chunk_lengths == 0, n_window * 2, chunk_lengths)
    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    return padded_feature, chunk_lengths


def get_valid_indices(chunk_lengths: torch.Tensor) -> torch.Tensor:
    feature_lens_after_cnn = _feat_extract_output_lengths(chunk_lengths)
    max_len_after_cnn = feature_lens_after_cnn.max().item()
    mask = torch.arange(max_len_after_cnn, device=chunk_lengths.device) < feature_lens_after_cnn.unsqueeze(1)
    return mask.flatten().nonzero().squeeze(-1)


def get_audio_cu_seqlens(chunk_lengths, feature_lens, n_window_infer, n_window):
    aftercnn_lens = _feat_extract_output_lengths(feature_lens)
    feature_lens_after_cnn = _feat_extract_output_lengths(chunk_lengths)
    max_len_after_cnn = feature_lens_after_cnn.max().item()
    n_window_ratio = n_window_infer // (n_window * 2)
    window_aftercnn = max_len_after_cnn * n_window_ratio
    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens:
        cnn_len = int(cnn_len)
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]
    return torch.tensor(cu_chunk_lens, device=feature_lens.device).cumsum(-1, dtype=torch.int32)


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_inc = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_inc * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )


# --------------------------------------------------------------------------- #
# native modules (weight names == HF)
# --------------------------------------------------------------------------- #
class NativeAudioAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, hidden_states, cu_seqlens, max_seqlen):
        s = hidden_states.shape[0]
        q = self.q_proj(hidden_states).reshape(s, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(s, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(s, self.num_heads, self.head_dim)
        o = varlen_attention(q, k, v, cu_seqlens, max_seqlen, self.scaling)
        return self.out_proj(o.reshape(s, -1))


class NativeAudioEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, activation):
        super().__init__()
        self.self_attn = NativeAudioAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.activation_fn = ACT2FN[activation]

    def forward(self, hidden_states, cu_seqlens, max_seqlen):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens, max_seqlen)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states


class NativeQwen3OmniAudioEncoder(nn.Module):
    """Native AuT. Same I/O contract as HF: forward(input_features, feature_lens)
    -> object with ``.last_hidden_state`` of shape (num_audio_tokens, output_dim)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize
        self.num_heads = config.encoder_attention_heads

        self.positional_embedding = SinusoidsPositionEmbedding(config.max_source_positions, d_model)
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        mel_reduced = (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(config.downsample_hidden_size * mel_reduced, d_model, bias=False)
        self.layers = nn.ModuleList([
            NativeAudioEncoderLayer(d_model, config.encoder_attention_heads,
                                    config.encoder_ffn_dim, config.activation_function)
            for _ in range(config.encoder_layers)
        ])
        self.ln_post = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(d_model, config.output_dim)

    def _layer_loop_tail(self, hidden_states, cu_seqlens, max_seqlen):
        """The expensive, capture-legal region: encoder layer loop + post-norm /
        projection head. ``cu_seqlens``/``max_seqlen`` depend only on the audio
        length layout, so for a fixed layout this whole region replays as one graph."""
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, max_seqlen)
        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states

    def get_piecewise_cuda_graph_config(self, device, autocast_dtype):
        """Build the ``PiecewisePackedConfig`` that migrates the layer-loop CUDA
        graph onto ``PiecewiseCudaGraphRunner``.

        The captured region is ``_layer_loop_tail`` over a packed ``[total_tokens,
        d_model]`` hidden state whose windows are delimited by ``cu_seqlens``. The
        runner buckets on ``(num_segments, total_tokens)`` and pads; the FlashInfer
        ragged wrapper is re-planned per replay with the real per-segment lengths,
        which is what lets one captured bucket serve many audio layouts.

        Returns None (eager path) when FlashInfer isn't the active varlen backend.
        Buckets come from ``CAPTURE_BATCH_SIZES_AUDIO`` (segment counts) and
        ``CAPTURE_TOKENS_AUDIO`` (token counts); each ``(bs, tokens)`` bucket owns
        a persistent 128 MiB workspace, so keep the product modest."""
        from mstar.engine.cuda_graph_config import PiecewisePackedConfig
        if not (_FLASHINFER_AVAILABLE and _VARLEN_BACKEND == "flashinfer"):
            return None

        d_model = self.config.d_model
        num_heads = self.num_heads
        head_dim = d_model // num_heads
        scale = head_dim ** -0.5

        def make_static_inputs(shape):
            # x matches autocast dtype so the per-replay copy_ is a same-dtype
            # memcpy. cu_seqlens is a static input (not baked) even though the
            # FI-external path ignores it, so no freed layout tensor is captured.
            return {
                "x": torch.zeros(shape.total_tokens, d_model,
                                 dtype=autocast_dtype, device=device),
                "cu_seqlens": torch.zeros(shape.bs + 1, dtype=torch.int32,
                                          device=device),
            }

        def make_attn_state(shape):
            return make_fi_graph_state(device, shape.bs)

        def plan_attn_fn(state, shape, seq_lens):
            plan_fi_graph_state(state, seq_lens, num_heads, head_dim, scale,
                                autocast_dtype)

        def capture_fn(static_inputs, static_cm=None, attn_state=None, **kw):
            # Route the block loop's attention through the runner-owned wrapper.
            # max_seqlen is unused: the FI-external path ignores it and capture
            # never reaches the flash-attn branch (_fi_override is set).
            set_fi_override(attn_state)
            try:
                x = self._layer_loop_tail(
                    static_inputs["x"], static_inputs["cu_seqlens"], 0)
            finally:
                set_fi_override(None)
            return {"x": x}

        return PiecewisePackedConfig(
            capture_fn=capture_fn,
            make_static_inputs=make_static_inputs,
            make_attn_state=make_attn_state,
            plan_attn_fn=plan_attn_fn,
            uses_kv_cache=False,
            total_tokens=list(CAPTURE_TOKENS_AUDIO),
            capture_batch_sizes=list(CAPTURE_BATCH_SIZES_AUDIO),
        )

    @torch.no_grad()
    def forward(self, input_features, feature_lens=None, return_dict=True,
                piecewise_runner=None, **kwargs):
        # return_dict/**kwargs accepted for HF signature compatibility; always returns AudioEncoderOutput.
        assert feature_lens is not None, "native AuT requires feature_lens"
        param_dtype = self.conv2d1.weight.dtype
        input_features = input_features.to(param_dtype)
        padded_feature, chunk_lengths = chunk_and_pad_features(input_features, feature_lens, self.n_window)
        valid_indices = get_valid_indices(chunk_lengths)
        cu_seqlens = get_audio_cu_seqlens(chunk_lengths, feature_lens, self.n_window_infer, self.n_window)

        padded_feature = padded_feature.unsqueeze(1).to(param_dtype)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            x = F.gelu(self.conv2d1(chunk))
            x = F.gelu(self.conv2d2(x))
            x = F.gelu(self.conv2d3(x))
            padded_embeds.append(x)
        padded_embed = torch.cat(padded_embeds, dim=0)

        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        pos = self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        padded_embed = padded_embed + pos
        hidden_states = torch.index_select(padded_embed.reshape(-1, padded_embed.shape[-1]), 0, valid_indices)

        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())

        # Piecewise path (default): replay the layer loop on a bucketed graph,
        # re-planning ragged attention with the real per-segment lengths. The
        # submodule passes the runner in from engine_inputs; without one (or if
        # no bucket fits this layout) we fall through to eager.
        runner = piecewise_runner
        n_seg = cu_seqlens.shape[0] - 1
        total_tokens = hidden_states.shape[0]
        if runner is not None:
            fitted = runner.can_run(n_seg, total_tokens)
            note_encoder_layout("audio", n_seg, total_tokens, fitted)
            if fitted:
                seg_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
                out = runner.run(
                    static_inputs={
                        "x": hidden_states,
                        "cu_seqlens": cu_seqlens.to(torch.int32),
                    },
                    seq_lens=seg_lens,
                    real_bs=n_seg,
                )
                note_encoder_path("audio.piecewise")
                return AudioEncoderOutput(last_hidden_state=out["x"])

        note_encoder_path("audio.eager")
        hidden_states = self._layer_loop_tail(hidden_states, cu_seqlens, max_seqlen)
        return AudioEncoderOutput(last_hidden_state=hidden_states)

"""Native Qwen3-Omni audio encoder (AuT, Whisper-style).

Mstar reimplementation of HF's Qwen3OmniMoeAudioEncoder, decoupled from
transformers at inference time. Weight names mirror HF exactly so
load_weights_from_hf_shards(..., prefix="thinker.audio_tower") loads with
no remapping. Attention via varlen_attention (flash-attn / FlashInfer /
SDPA fallback). Frontend helpers replicate HF bit-for-bit (parity-tested).
"""
from __future__ import annotations

import logging
import os
from collections import namedtuple
from dataclasses import dataclass

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
    if st["cu_obj"] is not cu_seqlens:          # plan once per forward
        cu = cu_seqlens.to(torch.int32)
        wrapper.plan(cu, cu, H, H, Dp, causal=False, sm_scale=float(scale),
                     q_data_type=q.dtype)
        st["cu_obj"] = cu_seqlens
    out = wrapper.run(qp, kp, vp)
    return out[..., :D].contiguous()


# Default=flashinfer: the only capture-legal varlen backend (SDPA mask builds are not graph-safe).
# MSTAR_VARLEN_BACKEND in {adaptive, per_segment, dense, padded, flashinfer}.
_VARLEN_BACKEND = os.environ.get("MSTAR_VARLEN_BACKEND", "flashinfer")
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


@dataclass
class _AudioGraph:
    """Captured layer-loop graph for one audio-length layout.
    cu_seqlens and fi_state kept alive so captured kernels' storage outlives the graph."""
    graph: "torch.cuda.CUDAGraph"
    static_x: torch.Tensor
    out: torch.Tensor
    cu_seqlens: torch.Tensor
    fi_state: dict


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

        # CUDA-graph cache for the encoder layer-loop tail, keyed on audio-length layout.
        # Enabled via MSTAR_ENCODER_CUDA_GRAPH=1; requires FlashInfer varlen backend.
        self._cg_cache: dict = {}
        self._cg_max_keys = int(os.environ.get("MSTAR_ENCODER_CG_MAX_KEYS", "16"))
        self._cg_warmed = False

    def _maybe_cg_warmup(self, input_features, feature_lens):
        """Pre-capture graphs for configured batch sizes before serving (MSTAR_ENCODER_CG_WARMUP).
        Triggered once on first forward; replicates clip-0 layout to each batch size."""
        if self._cg_warmed:
            return
        self._cg_warmed = True
        spec = os.environ.get("MSTAR_ENCODER_CG_WARMUP", "1,2,4,8")
        if not spec or not self._cuda_graph_enabled() or feature_lens is None:
            return
        try:
            batch_sizes = sorted({int(b) for b in spec.split(",") if b.strip()})
        except ValueError:
            return
        fl = feature_lens.reshape(-1)
        L0 = int(fl[0])                                   # frames in clip 0
        mel = input_features.shape[0]
        dev, dt = input_features.device, input_features.dtype
        logger.info("audio encoder CUDA-graph warmup: pre-capturing batch sizes %s", batch_sizes)
        for k in batch_sizes:
            try:
                lens = torch.full((k,), L0, dtype=torch.long, device=dev)
                feats = input_features[:, :L0].repeat(1, k) if input_features.shape[1] >= L0 \
                    else torch.zeros(mel, L0 * k, device=dev, dtype=dt)
                self.forward(feats, feature_lens=lens)       # triggers capture for key_k
            except Exception:
                logger.warning("audio CG warmup failed for bs=%d", k, exc_info=True)

    def _cuda_graph_enabled(self) -> bool:
        if os.environ.get("MSTAR_ENCODER_CUDA_GRAPH", "1") not in ("1", "true", "True"):
            return False
        return _FLASHINFER_AVAILABLE and _VARLEN_BACKEND == "flashinfer"

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

    def _capture_graph(self, key, hidden_states, cu_seqlens, max_seqlen):
        """Capture _layer_loop_tail for one audio-length key with a dedicated FlashInfer
        wrapper (planned once, never re-planned). Returns _AudioGraph or None on failure."""
        dev = hidden_states.device
        fi_state = make_fi_state(dev)
        if fi_state is None:
            return None
        static_x = hidden_states.clone()
        set_fi_override(fi_state)
        graph = torch.cuda.CUDAGraph()
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._layer_loop_tail(static_x, cu_seqlens, max_seqlen)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph):
                out = self._layer_loop_tail(static_x, cu_seqlens, max_seqlen)
        except Exception:
            logger.warning("audio encoder CUDA-graph capture failed for key=%s; "
                           "falling back to eager", key, exc_info=True)
            try:
                torch.cuda.synchronize(dev)
            except Exception:
                pass
            return None
        finally:
            set_fi_override(None)
        return _AudioGraph(graph=graph, static_x=static_x, out=out,
                           cu_seqlens=cu_seqlens, fi_state=fi_state)

    @torch.no_grad()
    def forward(self, input_features, feature_lens=None, return_dict=True, **kwargs):
        # return_dict/**kwargs accepted for HF signature compatibility; always returns AudioEncoderOutput.
        assert feature_lens is not None, "native AuT requires feature_lens"
        self._maybe_cg_warmup(input_features, feature_lens)
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

        if self._cuda_graph_enabled():
            # Key on varlen layout: same key => identical cu_seqlens => graph replay valid.
            key = (int(hidden_states.shape[0]), tuple(cu_seqlens.tolist()))
            ag = self._cg_cache.get(key)
            if ag is None and len(self._cg_cache) < self._cg_max_keys:
                ag = self._capture_graph(key, hidden_states, cu_seqlens, max_seqlen)
                if ag is not None:
                    self._cg_cache[key] = ag
            if ag is not None:
                ag.static_x.copy_(hidden_states)
                ag.graph.replay()
                return AudioEncoderOutput(last_hidden_state=ag.out.clone())

        hidden_states = self._layer_loop_tail(hidden_states, cu_seqlens, max_seqlen)
        return AudioEncoderOutput(last_hidden_state=hidden_states)

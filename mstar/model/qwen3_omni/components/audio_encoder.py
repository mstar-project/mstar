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

from mstar.model.components.encoder_telemetry import (
    note_encoder_layout,
    note_encoder_path,
)
from mstar.model.components.varlen_attention import (
    capture_legal_backend,
    make_fi_graph_state,
    plan_fi_graph_state,
    set_fi_override,
    varlen_attention,
)

logger = logging.getLogger(__name__)

# Mirrors HF's BaseModelOutput.last_hidden_state; defined at module scope.
AudioEncoderOutput = namedtuple("AudioEncoderOutput", ["last_hidden_state"])

# Measured buckets: s2t (1..7 segments, 36..426 tokens) and s2s (1..16, ~1057).
CAPTURE_TOKENS_AUDIO = (48, 64, 96, 128, 192, 256, 384, 512, 704, 896, 1088)
# One bs value: crossed with total_tokens, so extras multiply graphs and 128 MiB
# workspaces, while padding bs up is free.
CAPTURE_BATCH_SIZES_AUDIO = (32,)


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
        if not capture_legal_backend():
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

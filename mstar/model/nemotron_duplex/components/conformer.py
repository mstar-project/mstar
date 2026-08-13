"""Fast-Conformer STT perception stack (``stt_model.perception.*``).

NeMo Fast-Conformer: mel preprocessor -> dw-striding conv subsampling ->
24 Conformer layers (Macaron FF, rel-pos MHA, conv module) -> projection into
the nano LLM hidden space (1024 -> 4480).

Ported to exact checkpoint parity **and** numerically verified against the NeMo
``nemotron-labs-voicechat`` reference. Config (from the model's ``config.json``):
``att_context_size=[70, 0]`` chunked-limited (left window 70, causal), causal
``dw_striding`` subsampling factor 8, ``conv_context_size='causal'`` depthwise
conv, ``conv_norm_type='layer_norm'`` (so the ``conv.batch_norm`` tensor is a
LayerNorm over channels -- which is why the checkpoint has only weight+bias and
no running stats), Macaron 0.5 FF scaling, ``xscaling=False``, ``use_bias=False``
on all FF/attn/conv linears, Swish (SiLU) activations, mel power spectrum with
preemphasis 0.97.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.nemotron_duplex.components._util import param

D = 1024          # encoder d_model
N_HEADS = 8
HEAD_DIM = 128
FF = 4096
NUM_LAYERS = 24
SUB_CH = 256      # subsampling channels
LLM_HIDDEN = 4480
CONV_KERNEL = 9
SUB_FLAT = 4352   # SUB_CH * 17 (freq after 3 causal stride-2 stages)
ATT_LEFT = 70     # att_context_size = [70, 0]  (chunked_limited, chunk_size 1)
INF_VAL = 10000.0

# mel featurizer
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 128
PREEMPH = 0.97
MAG_POWER = 2.0
LOG_GUARD = 2.0 ** -24


class _CausalConv2d(nn.Conv2d):
    """nn.Conv2d with NeMo ``CausalConv2D`` padding (left=k-1, right=stride-1 on
    both time and freq axes). Keeps ``weight``/``bias`` param names for loading."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, groups=groups)

    def forward(self, x):
        x = F.pad(x, (self._left_padding, self._right_padding, self._left_padding, self._right_padding))
        return super().forward(x)


class _Featurizer(nn.Module):
    """Log-mel power spectrogram (``AudioToMelSpectrogramPreprocessor.featurizer``).

    ``fb`` (mel filterbank) and ``window`` (Hann) are loaded from the checkpoint.
    """

    def __init__(self):
        super().__init__()
        self.fb = param(1, N_MELS, N_FFT // 2 + 1)   # mel filterbank
        self.window = param(WIN_LENGTH)               # STFT window (Hann, periodic=False)

    def get_seq_len(self, seq_len: torch.Tensor) -> torch.Tensor:
        pad_amount = N_FFT // 2 * 2
        return torch.floor_divide(seq_len + pad_amount - N_FFT, HOP_LENGTH) + 1

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor):
        out_len = self.get_seq_len(seq_len)
        # preemphasis
        x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - PREEMPH * x[:, :-1]), dim=1)
        spec = torch.stft(
            x, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
            center=True, window=self.window.to(torch.float32), return_complex=True,
        )
        spec = torch.view_as_real(spec)
        mag = torch.sqrt(spec.pow(2).sum(-1))
        if MAG_POWER != 1.0:
            mag = mag.pow(MAG_POWER)
        mel = torch.matmul(self.fb.to(mag.dtype), mag)     # (B, n_mels, T)
        mel = torch.log(mel + LOG_GUARD)                    # normalize="NA" -> no-op
        max_len = mel.size(-1)
        mask = torch.arange(max_len, device=mel.device).expand(mel.size(0), -1) >= out_len.unsqueeze(1)
        mel = mel.masked_fill(mask.unsqueeze(1), 0.0)
        return mel, out_len


class Preprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.featurizer = _Featurizer()

    def forward(self, x, seq_len):
        return self.featurizer(x, seq_len)


class ConvSubsampling(nn.Module):
    """Causal dw-striding subsampling (factor 8): CausalConv2d, ReLU,
    (depthwise CausalConv2d, pointwise, ReLU) x2 -> Linear(4352 -> d_model)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            _CausalConv2d(1, SUB_CH, 3, stride=2),                       # 0
            nn.ReLU(),                                                    # 1
            _CausalConv2d(SUB_CH, SUB_CH, 3, stride=2, groups=SUB_CH),   # 2 depthwise
            nn.Conv2d(SUB_CH, SUB_CH, kernel_size=1),                    # 3 pointwise
            nn.ReLU(),                                                    # 4
            _CausalConv2d(SUB_CH, SUB_CH, 3, stride=2, groups=SUB_CH),   # 5 depthwise
            nn.Conv2d(SUB_CH, SUB_CH, kernel_size=1),                    # 6 pointwise
            nn.ReLU(),                                                    # 7
        )
        self.out = nn.Linear(SUB_FLAT, D)

    @staticmethod
    def calc_length(lengths: torch.Tensor) -> torch.Tensor:
        # 3 stride-2 causal stages: all_paddings=(k-1)+(s-1)=3, kernel=3, stride=2
        add_pad = 3 - 3
        for _ in range(3):
            lengths = torch.floor(torch.div(lengths.to(torch.float32) + add_pad, 2) + 1.0)
        return lengths.to(torch.int64)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        lengths = self.calc_length(lengths)
        x = x.unsqueeze(1)                       # (B, 1, T, feat)
        x = self.conv(x)                          # (B, C, T', F')
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        return x, lengths


class _RelPosMHA(nn.Module):
    """Transformer-XL relative-position multi-head attention (rel_pos)."""

    def __init__(self):
        super().__init__()
        self.linear_q = nn.Linear(D, D, bias=False)
        self.linear_k = nn.Linear(D, D, bias=False)
        self.linear_v = nn.Linear(D, D, bias=False)
        self.linear_out = nn.Linear(D, D, bias=False)
        self.linear_pos = nn.Linear(D, D, bias=False)
        self.pos_bias_u = param(N_HEADS, HEAD_DIM)
        self.pos_bias_v = param(N_HEADS, HEAD_DIM)
        self.s_d_k = math.sqrt(HEAD_DIM)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        b, h, qlen, pos_len = x.size()
        x = F.pad(x, (1, 0))
        x = x.view(b, h, -1, qlen)
        return x[:, :, 1:].view(b, h, qlen, pos_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        n_batch = x.size(0)
        q = self.linear_q(x).view(n_batch, -1, N_HEADS, HEAD_DIM)
        k = self.linear_k(x).view(n_batch, -1, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.linear_v(x).view(n_batch, -1, N_HEADS, HEAD_DIM).transpose(1, 2)
        # q kept as (B, t1, h, d_k) to add per-head biases
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, N_HEADS, HEAD_DIM).transpose(1, 2)
        q_u = (q + self.pos_bias_u).transpose(1, 2)   # (B, h, t1, d_k)
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))
        matrix_bd = self._rel_shift(matrix_bd)
        matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
        scores = (matrix_ac + matrix_bd) / self.s_d_k
        if mask is not None:
            m = mask.unsqueeze(1)
            scores = scores.masked_fill(m, -INF_VAL)
            attn = torch.softmax(scores, dim=-1).masked_fill(m, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(n_batch, -1, D)
        return self.linear_out(out)


class _FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(D, FF, bias=False)
        self.linear2 = nn.Linear(FF, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.silu(self.linear1(x)))


class _ConvModule(nn.Module):
    """Conformer conv module: pointwise -> GLU -> causal depthwise -> LayerNorm
    (checkpoint's ``batch_norm`` is a LayerNorm) -> SiLU -> pointwise."""

    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(D, 2 * D, kernel_size=1, bias=False)
        # causal depthwise: left pad k-1, right pad 0
        self.depthwise_conv = nn.Conv1d(D, D, kernel_size=CONV_KERNEL, padding=0, groups=D, bias=False)
        self.batch_norm = nn.LayerNorm(D)
        self.pointwise_conv2 = nn.Conv1d(D, D, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        x = x.transpose(1, 2)                    # (B, D, T)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = F.pad(x, (CONV_KERNEL - 1, 0))       # causal padding
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)                     # (B, T, D)
        x = self.batch_norm(x)                    # LayerNorm over channels
        x = x.transpose(1, 2)                     # (B, D, T)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)                  # (B, T, D)


class ConformerLayer(nn.Module):
    FC_FACTOR = 0.5

    def __init__(self):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(D)
        self.feed_forward1 = _FeedForward()
        self.norm_self_att = nn.LayerNorm(D)
        self.self_attn = _RelPosMHA()
        self.norm_conv = nn.LayerNorm(D)
        self.conv = _ConvModule()
        self.norm_feed_forward2 = nn.LayerNorm(D)
        self.feed_forward2 = _FeedForward()
        self.norm_out = nn.LayerNorm(D)

    def forward(self, x, att_mask, pos_emb, pad_mask):
        residual = x + self.FC_FACTOR * self.feed_forward1(self.norm_feed_forward1(x))
        residual = residual + self.self_attn(self.norm_self_att(residual), att_mask, pos_emb)
        residual = residual + self.conv(self.norm_conv(residual), pad_mask)
        residual = residual + self.FC_FACTOR * self.feed_forward2(self.norm_feed_forward2(residual))
        return self.norm_out(residual)


class ConformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_encode = ConvSubsampling()
        self.layers = nn.ModuleList([ConformerLayer() for _ in range(NUM_LAYERS)])

    @staticmethod
    def _rel_pos_emb(length: int, dim: int, device, dtype) -> torch.Tensor:
        """RelPositionalEncoding: positions (L-1 .. -(L-1)), sinusoidal, (1, 2L-1, dim)."""
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        pe = torch.zeros(positions.size(0), dim, device=device)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(math.log(INF_VAL) / dim)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0).to(dtype)

    @staticmethod
    def _create_masks(length: torch.Tensor, max_len: int, device):
        # chunked_limited, att_context_size=[70, 0] -> chunk_size=1, left window 70
        idx = torch.arange(0, max_len, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        att_ok = (diff <= ATT_LEFT) & (diff >= 0)                       # (T, T)
        pad_ok = torch.arange(0, max_len, device=device).expand(length.size(0), -1) < length.unsqueeze(-1)
        pad_att = pad_ok.unsqueeze(1).repeat(1, max_len, 1)
        pad_att = pad_att & pad_att.transpose(1, 2)
        att_mask = ~(pad_att & att_ok.unsqueeze(0))                     # True = masked
        pad_mask = ~pad_ok                                             # True = pad
        return pad_mask, att_mask

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        # audio_signal: (B, feat, T)
        audio_signal = audio_signal.transpose(1, 2)                   # (B, T, feat)
        audio_signal, length = self.pre_encode(audio_signal, length)  # (B, T', D)
        max_audio_length = audio_signal.size(1)
        pos_emb = self._rel_pos_emb(max_audio_length, D, audio_signal.device, audio_signal.dtype)
        pad_mask, att_mask = self._create_masks(length, max_audio_length, audio_signal.device)
        for layer in self.layers:
            audio_signal = layer(audio_signal, att_mask, pos_emb, pad_mask)
        return audio_signal.transpose(1, 2), length                   # (B, D, T'), length


class Perception(nn.Module):
    """Full perception stack: preprocessor + conformer encoder + LLM projection."""

    def __init__(self):
        super().__init__()
        self.preprocessor = Preprocessor()
        self.encoder = ConformerEncoder()
        self.proj = nn.Linear(D, LLM_HIDDEN)  # 1024 -> 4480, into nano hidden

    def forward(
        self,
        input_signal: torch.Tensor | None = None,
        input_signal_lens: torch.Tensor | None = None,
        processed_signal: torch.Tensor | None = None,
        processed_signal_lens: torch.Tensor | None = None,
    ):
        """(B, samples) waveform -> (B, T', 4480) LLM-space embeddings, lengths.

        Pass ``processed_signal`` (B, n_mels, T) to bypass the mel featurizer.
        """
        if processed_signal is None:
            processed_signal, processed_signal_lens = self.preprocessor(input_signal, input_signal_lens)
        encoder_emb, encoded_len = self.encoder(processed_signal, processed_signal_lens)
        encoded = self.proj(encoder_emb.transpose(1, 2))              # (B, T', 4480)
        return encoded, encoded_len

    @staticmethod
    def remap(name: str) -> str | None:
        prefix = "stt_model.perception."
        return name[len(prefix):] if name.startswith(prefix) else None

    def load_weights(self, weights):
        from mstar.model.loader import load_hf_weights

        return load_hf_weights(self, weights, stacked_params=[], name_remapper=self.remap)

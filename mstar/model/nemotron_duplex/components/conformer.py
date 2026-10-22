"""Fast-Conformer STT perception stack (``stt_model.perception.*``).

NeMo Fast-Conformer: mel preprocessor -> dw-striding conv subsampling ->
24 Conformer layers (Macaron FF, rel-pos MHA, conv module) -> projection into
the nano LLM hidden space (1024 -> 4480).

Ported to exact checkpoint parity **and** numerically verified against the NeMo
``nemotron-labs-voicechat`` reference. Every architecture dim comes from
``ConformerSTTConfig`` (populated from the checkpoint's ``config.json`` ->
``model.stt.model.perception``); nothing is hardcoded here. Config highlights:
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
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.resources.kv.bounded import BoundedKVConfig, SinkWindow
from mstar.engine.resources.recurrent.config import RecurrentBlockConfig
from mstar.model.nemotron_duplex.components._util import param
from mstar.model.nemotron_duplex.components.dft import stft_as_real, stft_basis
from mstar.model.nemotron_duplex.config import ConformerSTTConfig

if TYPE_CHECKING:
    from mstar.engine.resources.kv.bounded.manager import BoundedKVManager
    from mstar.engine.resources.recurrent.pool import RecurrentAddressing, RecurrentStatePool

# Pure numerical constants (not architecture dims / not in config.json).
INF_VAL = 10000.0        # attention mask fill value (NeMo RelPos MHA)
LOG_GUARD = 2.0 ** -24   # log-mel zero guard (NeMo FilterbankFeatures log_zero_guard_value)


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

    def __init__(self, config: ConformerSTTConfig):
        super().__init__()
        self.config = config
        self.fb = param(1, config.n_mels, config.n_fft // 2 + 1)   # mel filterbank
        self.window = param(config.win_length)                     # STFT window (Hann, periodic=False)

    def get_seq_len(self, seq_len: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        pad_amount = cfg.n_fft // 2 * 2
        return torch.floor_divide(seq_len + pad_amount - cfg.n_fft, cfg.hop_length) + 1

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor):
        cfg = self.config
        out_len = self.get_seq_len(seq_len)
        # preemphasis
        x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - cfg.preemph * x[:, :-1]), dim=1)
        spec = torch.stft(
            x, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length,
            center=True, window=self.window.to(torch.float32), return_complex=True,
        )
        spec = torch.view_as_real(spec)
        mag = torch.sqrt(spec.pow(2).sum(-1))
        if cfg.mag_power != 1.0:
            mag = mag.pow(cfg.mag_power)
        mel = torch.matmul(self.fb.to(mag.dtype), mag)     # (B, n_mels, T)
        mel = torch.log(mel + LOG_GUARD)                    # normalize="NA" -> no-op
        max_len = mel.size(-1)
        mask = torch.arange(max_len, device=mel.device).expand(mel.size(0), -1) >= out_len.unsqueeze(1)
        mel = mel.masked_fill(mask.unsqueeze(1), 0.0)
        return mel, out_len


class Preprocessor(nn.Module):
    def __init__(self, config: ConformerSTTConfig):
        super().__init__()
        self.featurizer = _Featurizer(config)

    def forward(self, x, seq_len):
        return self.featurizer(x, seq_len)


class ConvSubsampling(nn.Module):
    """Causal dw-striding subsampling (factor 8): CausalConv2d, ReLU,
    (depthwise CausalConv2d, pointwise, ReLU) x2 -> Linear(4352 -> d_model)."""

    def __init__(self, config: ConformerSTTConfig):
        super().__init__()
        self.config = config
        ch = config.subsampling_conv_channels
        self._num_stages = int(math.log2(config.subsampling_factor))
        layers: list[nn.Module] = [_CausalConv2d(1, ch, 3, stride=2), nn.ReLU()]
        for _ in range(self._num_stages - 1):
            layers.append(_CausalConv2d(ch, ch, 3, stride=2, groups=ch))   # depthwise
            layers.append(nn.Conv2d(ch, ch, kernel_size=1))                # pointwise
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(config.subsampling_flat, config.d_model)

    def calc_length(self, lengths: torch.Tensor) -> torch.Tensor:
        # stride-2 causal stages: all_paddings=(k-1)+(s-1)=3, kernel=3, stride=2 -> add_pad=0
        add_pad = 3 - 3
        for _ in range(self._num_stages):
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

    def __init__(self, config: ConformerSTTConfig):
        super().__init__()
        d = config.d_model
        self.n_heads = config.num_heads
        self.head_dim = config.head_dim
        self.linear_q = nn.Linear(d, d, bias=False)
        self.linear_k = nn.Linear(d, d, bias=False)
        self.linear_v = nn.Linear(d, d, bias=False)
        self.linear_out = nn.Linear(d, d, bias=False)
        self.linear_pos = nn.Linear(d, d, bias=False)
        self.pos_bias_u = param(self.n_heads, self.head_dim)
        self.pos_bias_v = param(self.n_heads, self.head_dim)
        self.s_d_k = math.sqrt(self.head_dim)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        b, h, qlen, pos_len = x.size()
        x = F.pad(x, (1, 0))
        x = x.view(b, h, -1, qlen)
        return x[:, :, 1:].view(b, h, qlen, pos_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        n_batch = x.size(0)
        h, dk, d = self.n_heads, self.head_dim, self.n_heads * self.head_dim
        q = self.linear_q(x).view(n_batch, -1, h, dk)
        k = self.linear_k(x).view(n_batch, -1, h, dk).transpose(1, 2)
        v = self.linear_v(x).view(n_batch, -1, h, dk).transpose(1, 2)
        # q kept as (B, t1, h, d_k) to add per-head biases
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, h, dk).transpose(1, 2)
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
        out = torch.matmul(attn, v).transpose(1, 2).reshape(n_batch, -1, d)
        return self.linear_out(out)


class _FeedForward(nn.Module):
    def __init__(self, config: ConformerSTTConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.ff_dim, bias=False)
        self.linear2 = nn.Linear(config.ff_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.silu(self.linear1(x)))


class _ConvModule(nn.Module):
    """Conformer conv module: pointwise -> GLU -> causal depthwise -> LayerNorm
    (checkpoint's ``batch_norm`` is a LayerNorm) -> SiLU -> pointwise."""

    def __init__(self, config: ConformerSTTConfig):
        super().__init__()
        d = config.d_model
        self.kernel_size = config.conv_kernel_size
        self.pointwise_conv1 = nn.Conv1d(d, 2 * d, kernel_size=1, bias=False)
        # causal depthwise: left pad k-1, right pad 0
        self.depthwise_conv = nn.Conv1d(d, d, kernel_size=self.kernel_size, padding=0, groups=d, bias=False)
        self.batch_norm = nn.LayerNorm(d)
        self.pointwise_conv2 = nn.Conv1d(d, d, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        x = x.transpose(1, 2)                    # (B, D, T)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = F.pad(x, (self.kernel_size - 1, 0))  # causal padding
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)                     # (B, T, D)
        x = self.batch_norm(x)                    # LayerNorm over channels
        x = x.transpose(1, 2)                     # (B, D, T)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)                  # (B, T, D)


class ConformerLayer(nn.Module):
    FC_FACTOR = 0.5

    def __init__(self, config: ConformerSTTConfig):
        super().__init__()
        d = config.d_model
        self.norm_feed_forward1 = nn.LayerNorm(d)
        self.feed_forward1 = _FeedForward(config)
        self.norm_self_att = nn.LayerNorm(d)
        self.self_attn = _RelPosMHA(config)
        self.norm_conv = nn.LayerNorm(d)
        self.conv = _ConvModule(config)
        self.norm_feed_forward2 = nn.LayerNorm(d)
        self.feed_forward2 = _FeedForward(config)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, x, att_mask, pos_emb, pad_mask):
        residual = x + self.FC_FACTOR * self.feed_forward1(self.norm_feed_forward1(x))
        residual = residual + self.self_attn(self.norm_self_att(residual), att_mask, pos_emb)
        residual = residual + self.conv(self.norm_conv(residual), pad_mask)
        residual = residual + self.FC_FACTOR * self.feed_forward2(self.norm_feed_forward2(residual))
        return self.norm_out(residual)


class ConformerEncoder(nn.Module):
    def __init__(self, config: ConformerSTTConfig):
        super().__init__()
        self.config = config
        self.pre_encode = ConvSubsampling(config)
        self.layers = nn.ModuleList([ConformerLayer(config) for _ in range(config.num_layers)])

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

    def _create_masks(self, length: torch.Tensor, max_len: int, device):
        # chunked_limited, att_context_size=(left, 0) -> chunk_size=1, left window = left
        att_left = self.config.att_context_size[0]
        idx = torch.arange(0, max_len, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        att_ok = (diff <= att_left) & (diff >= 0)                       # (T, T)
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
        pos_emb = self._rel_pos_emb(max_audio_length, self.config.d_model,
                                    audio_signal.device, audio_signal.dtype)
        pad_mask, att_mask = self._create_masks(length, max_audio_length, audio_signal.device)
        for layer in self.layers:
            audio_signal = layer(audio_signal, att_mask, pos_emb, pad_mask)
        return audio_signal.transpose(1, 2), length                   # (B, D, T'), length


class Perception(nn.Module):
    """Full perception stack: preprocessor + conformer encoder + LLM projection."""

    def __init__(self, config: ConformerSTTConfig | None = None):
        super().__init__()
        self.config = config or ConformerSTTConfig()
        self.preprocessor = Preprocessor(self.config)
        self.encoder = ConformerEncoder(self.config)
        self.proj = nn.Linear(self.config.d_model, self.config.output_dim)  # 1024 -> 4480, into nano hidden

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


# ---------------------------------------------------------------------------
# Streaming: the same encoder, one chunk of a call at a time
# ---------------------------------------------------------------------------

class StreamState:
    """One call's place in its audio stream: host counters, which drive every
    slice so a step needs no sync. The audio and mel frames a later step still
    reads live in the ``front`` pool (``front_state_blocks``)."""

    __slots__ = ("samples", "mels", "frames")

    def __init__(self):
        self.samples = 0        # samples of the call so far
        self.mels = 0           # mel frames computed
        self.frames = 0         # encoder frames emitted (= the call's position in the bounded KV)


def front_state_blocks(config: ConformerSTTConfig) -> dict[str, RecurrentBlockConfig]:
    """A call's front-end state, all a later step reads: its newest
    preemphasized samples (``sig``; every sample until its first mel frame,
    then the window of the next one), its newest mel frames (``mel``; the
    subsampling window of the next encoder frame) and its last raw sample, for
    the preemphasis. Right-aligned: the newest sample / frame is last."""
    return {
        "sig": RecurrentBlockConfig(shape=(config.hop_length * config.subsampling_factor,), dtype=torch.float32),
        "mel": RecurrentBlockConfig(shape=(2 * config.subsampling_factor, config.n_mels), dtype=torch.float32),
        "last": RecurrentBlockConfig(shape=(1,), dtype=torch.float32),
    }


def encoder_state_blocks(config: ConformerSTTConfig,
                         dtype: torch.dtype | None = None) -> dict[str, RecurrentBlockConfig]:
    """A call's conv state, per layer: the last ``kernel - 1`` inputs of its
    depthwise conv (its attention window is the bounded KV, ``ENCODER_KV``).
    By default in the engine's compute dtype, which the layers compute it in:
    stored in another, a call's frames drift from the offline ones."""
    return {
        "conv": RecurrentBlockConfig(shape=(config.d_model, config.conv_kernel_size - 1), dtype=dtype),
    }


def encoder_kv_config(config: ConformerSTTConfig, max_slots: int) -> BoundedKVConfig:
    """A call's attention keys and values, per layer: each new frame attends the
    ``att_context_size[0]`` frames before it and itself, so the KV keeps that
    many. In the engine's compute dtype, as ``encoder_state_blocks``."""
    return BoundedKVConfig(
        num_layers=config.num_layers, num_heads=config.num_heads, head_dim=config.head_dim, rows_per_request=1,
        max_source_len=0, retention=SinkWindow(0, config.att_context_size[0]),
        query_window=config.att_context_size[0], max_slots=max_slots, dtype=None,
    )


def final_mel_frames(config: ConformerSTTConfig, samples: int, final: bool) -> int:
    """Mel frames whose 512-sample window is complete, for a call of more than
    ``n_fft/2`` samples: frame m spans ``[m * hop - n_fft/2, m * hop + n_fft/2)``.
    At the end of the call every frame counts, the last ones over the reflected
    right edge, as offline."""
    if final:
        return samples // config.hop_length + 1
    return (samples - config.n_fft // 2) // config.hop_length + 1


def final_encoder_frames(config: ConformerSTTConfig, mels: int, samples: int, final: bool) -> int:
    """Encoder frames that no later audio can change, given at least one mel
    frame: frame ``j`` reads mel frames up to ``8 * j``. At the end, every frame
    offline would produce."""
    if final:
        return config.num_frames(samples)
    return (mels - 1) // config.subsampling_factor + 1


def device_counts(values, device: torch.device) -> torch.Tensor:
    """Host counts on the device without a blocking copy, which would wait for
    every kernel queued ahead of this step."""
    t = torch.tensor(values, dtype=torch.long)
    return (t.pin_memory() if device.type == "cuda" else t).to(device, non_blocking=True)


class EncoderInputs(NamedTuple):
    """A step's new encoder-input frames, ``x (B, t, D)`` zero-padded to the
    most any call got, and per call its new ones."""
    x: torch.Tensor
    n: list[int]


class ConformerStreamer:
    """Runs a ``Perception`` incrementally over a batch of calls, matching what it
    computes over each whole call up to float rounding: about 1e-6 in fp32, about
    1% relative in bf16, where each chunking takes different kernel shapes.

    - Mel: frames are computed once their window is complete, from the
      preemphasized signal, with the featurizer's own ops.
    - Subsampling: the strided convs are recomputed over a window that starts
      16 mel frames before the first new output frame. An output frame reads
      15 mel frames, so the first two outputs of the window are discarded and
      the rest are exact; the window starts on a multiple of 8 so the strides
      line up with the offline ones.
    - Layers: each new frame attends the 70 frames before it and itself in the
      bounded KV, which keeps a call's last 70 keys/values per layer; the
      depthwise conv reads its cached last 8 inputs.

    A call's state is its host counters (``StreamState``), two pool slots, the
    front end's (``front_state_blocks``) and the conv tails
    (``encoder_state_blocks``), and its bounded-KV slot. The resources' plans
    for a step come from the positions its declaration gives
    (``StreamState.frames``). ``steady_step`` is a whole step for calls that
    append exactly one frame of audio, all device work over fixed shapes, so it
    can replay as a CUDA graph; ``front`` + ``layers`` take any step.
    """

    def __init__(self, perception: "Perception"):
        self.p = perception
        self.c = perception.config
        enc = perception.encoder
        self._ctx = self.c.att_context_size[0]
        self.frame_samples = self.c.hop_length * self.c.subsampling_factor
        # steady_step's slices take a featurizer whose half window spans more
        # than one hop and at most two (n_fft 512, hop 160); any other runs
        # every step through front + layers
        self.has_steady_step = -(-(self.c.n_fft // 2) // self.c.hop_length) == 2
        # sinusoids of relative positions 0..ctx, row d = position d: all a query can see
        device = enc.layers[0].self_attn.linear_pos.weight.device
        pe = ConformerEncoder._rel_pos_emb(self._ctx + 1, self.c.d_model, device, torch.float32)
        self._pe = pe[0, : self._ctx + 1].flip(0)
        self._pos: dict = {}    # each layer's linear_pos(_pe), by layer, dtype and autocast: fixed weights
        feat = perception.preprocessor.featurizer
        self._stft_basis = stft_basis(feat.window.detach(), self.c.n_fft)    # steady_step's STFT, as a matmul

    # -- front end -----------------------------------------------------------

    def _mel(self, batch: torch.Tensor, by_matmul: bool = False) -> torch.Tensor:
        """``(B, samples)`` preemphasized windows -> ``(B, T, n_mels)`` log-mel frames.
        ``by_matmul`` takes the STFT as a matmul, for a captured graph (see ``dft``);
        otherwise it is the featurizer's ``torch.stft``."""
        c, feat = self.c, self.p.preprocessor.featurizer
        if by_matmul:
            spec = stft_as_real(batch, self._stft_basis, c.n_fft, c.hop_length)
        else:
            spec = torch.view_as_real(torch.stft(
                batch, n_fft=c.n_fft, hop_length=c.hop_length, win_length=c.win_length, center=False,
                window=feat.window.to(torch.float32), return_complex=True))
        mag = torch.sqrt(spec.pow(2).sum(-1))
        if c.mag_power != 1.0:
            mag = mag.pow(c.mag_power)
        mel = torch.log(torch.matmul(feat.fb.to(mag.dtype), mag) + LOG_GUARD)      # (B, n_mels, T)
        return mel.transpose(1, 2).to(torch.float32)

    def _subsample(self, windows: torch.Tensor, lens: list[int] | None = None) -> torch.Tensor:
        """``(B, T, n_mels)`` mel windows -> ``(B, T', D)`` encoder-input frames;
        ``lens`` gives each row's length when the rows are zero-padded to ``T``."""
        out = windows.unsqueeze(1)                                                          # (B, 1, T, n_mels)
        if lens is None:
            out = self.p.encoder.pre_encode.conv(out)                                       # (B, C, T', F')
        else:
            # A strided stage reads one frame past a row's end, where a row on its own
            # reads the conv's zero padding: zero each stage's input past each row.
            n = device_counts(lens, out.device).view(-1, 1, 1, 1)
            for m in self.p.encoder.pre_encode.conv:
                if isinstance(m, _CausalConv2d):
                    t = torch.arange(out.shape[2], device=out.device).view(1, 1, -1, 1)
                    out = out * (t < n).to(out.dtype)
                    n = n // 2 + 1
                out = m(out)
        b, ch, t, fr = out.shape
        return self.p.encoder.pre_encode.out(out.transpose(1, 2).reshape(b, t, -1))

    def is_steady(self, st: StreamState, chunk: torch.Tensor | None, final: bool) -> bool:
        """Whether ``steady_step`` serves this call's step: one frame of audio
        onto a call already two frames in, on the frame grid."""
        fs = self.frame_samples
        return (self.has_steady_step and not final and chunk is not None and chunk.numel() == fs
                and st.samples >= 2 * fs and st.samples % fs == 0)

    def steady_step(self, chunk: torch.Tensor, front: "RecurrentStatePool", front_rows: "RecurrentAddressing",
                    kv: "BoundedKVManager", pool: "RecurrentStatePool", rows: "RecurrentAddressing") -> torch.Tensor:
        """One frame for each call, from ``chunk (B, frame_samples)``: the new
        mel frames, the subsampling window and the layers, all on the device.
        It is captured, so its STFT is a matmul: a graph must not hold a cuFFT
        plan, which the plan cache can destroy (see ``dft``).

        A call ``S = (k - 1) fs`` samples in has ``8(k - 1) - 1`` mel frames and
        ``k - 1`` encoder frames. This frame of audio adds mel frames
        ``8k - 9 .. 8k - 2`` and encoder frame ``k - 1``, whose subsampling
        window is the 15 mel frames before them plus them (the third output)."""
        c, fs, f = self.c, self.frame_samples, self.c.subsampling_factor
        last, sig = front.gather("last", front_rows), front.gather("sig", front_rows)
        mel_tail = front.gather("mel", front_rows)
        x = chunk.to(torch.float32)
        y = x - c.preemph * torch.cat([last, x[:, :-1]], dim=1)
        win = torch.cat([sig, y], dim=1)                                    # samples S - fs .. S + fs
        hop, pad = c.hop_length, c.n_fft // 2
        # mel frames f(k-1)-1 .. fk-2 span samples [S - hop - pad, S + fs - 2 hop + pad)
        mel_new = self._mel(win[:, fs - hop - pad: 2 * fs - 2 * hop + pad], by_matmul=True)
        window = torch.cat([mel_tail[:, 1:], mel_new], dim=1)               # (B, 23, n_mels)
        new = self._subsample(window)[:, 2:3]
        front.scatter_("sig", front_rows, y)
        front.scatter_("last", front_rows, x[:, -1:])
        front.scatter_("mel", front_rows, torch.cat([mel_tail, mel_new], dim=1)[:, -2 * f:])
        return self.layers(new, kv, pool, rows, chunk.new_ones(chunk.shape[0], dtype=torch.long))

    def front(self, sts: list[StreamState], chunks: list[torch.Tensor | None], final: list[bool], n: list[int],
              front: "RecurrentStatePool", front_rows: "RecurrentAddressing") -> EncoderInputs | None:
        """Append each call's chunk; the ``n`` encoder-input frames it made final
        (``new_frames``, counted before the step), or None when no call has one.
        A call with no chunk and not at its end is left as it is."""
        c, pad, hop, f = self.c, self.c.n_fft // 2, self.c.hop_length, self.c.subsampling_factor
        keep_sig, keep_mel = self.frame_samples, 2 * f
        busy = [i for i, (ch, fin) in enumerate(zip(chunks, final, strict=True))
                if fin or (ch is not None and ch.numel())]
        if not busy:
            return None
        last, sig = front.gather("last", front_rows), front.gather("sig", front_rows)
        mel_tail = front.gather("mel", front_rows)
        new_sig, new_last, new_mel = list(sig), list(last), list(mel_tail)
        rows, lo_hi, mel_rows = [], [], []
        for i in busy:
            st, chunk = sts[i], chunks[i]
            held = min(st.samples, keep_sig)
            win, base = sig[i, keep_sig - held:], st.samples - held          # win[0] is sample ``base``
            if chunk is not None and chunk.numel():
                xr = chunk.to(torch.float32).reshape(-1)
                # ``last`` reads 0 in a fresh slot: the featurizer's preemphasis keeps the first sample
                y = xr - c.preemph * torch.cat([last[i], xr[:-1]])
                win = torch.cat([win, y])
                new_last[i] = xr[-1:]
                st.samples += xr.numel()
            new_sig[i] = F.pad(win[-keep_sig:], (max(0, keep_sig - win.numel()), 0))
            # a call's first mel frame reflects n_fft/2 samples past its first, as
            # the featurizer's padding does, so it needs more than that to start
            if st.samples <= pad:
                continue
            lo, hi = st.mels, final_mel_frames(c, st.samples, final[i])
            if hi <= lo:
                continue
            if lo * hop < pad:  # frame lo reads the reflected left edge, as torch.stft pads it
                win, base = torch.cat([win[1: pad + 1].flip(0), win]), base - pad   # (base is 0: all held)
            if final[i]:        # and the right edge at the end of the call
                win = torch.cat([win, win[-pad - 1: -1].flip(0)])
            rows.append(win[lo * hop - pad - base: (hi - 1) * hop + pad - base])
            lo_hi.append((i, lo, hi))
        added = {}
        if rows:
            width = max(r.numel() for r in rows)
            mel = self._mel(torch.stack([F.pad(r, (0, width - r.numel())) for r in rows]))
            added = {i: mel[row, : hi - lo] for row, (i, lo, hi) in enumerate(lo_hi)}
        for i in busy:
            st = sts[i]
            if st.samples <= pad:       # no mel frame yet, so no encoder frame
                continue
            held = min(st.mels, keep_mel)
            mels, first = mel_tail[i, keep_mel - held:], st.mels - held      # mels[0] is mel frame ``first``
            if i in added:
                mels = torch.cat([mels, added[i]])
                st.mels += added[i].shape[0]
                new_mel[i] = F.pad(mels[-keep_mel:], (0, 0, max(0, keep_mel - mels.shape[0]), 0))
            if n[i]:
                start = max(0, f * st.frames - 2 * f)     # the subsampling window's first mel frame
                mel_rows.append((i, mels[start - first:], (st.frames - start // f, n[i])))
        front.scatter_("sig", front_rows, torch.stack(new_sig))
        front.scatter_("last", front_rows, torch.stack(new_last))
        front.scatter_("mel", front_rows, torch.stack(new_mel))
        if not mel_rows:
            return None
        lens = [w.shape[0] for _, w, _ in mel_rows]
        out = self._subsample(torch.stack([F.pad(w, (0, 0, 0, max(lens) - w.shape[0])) for _, w, _ in mel_rows]),
                              lens)
        x = out.new_zeros(len(sts), max(n), out.shape[-1])
        for row, (i, _, (at, k)) in enumerate(mel_rows):
            x[i, :k] = out[row, at: at + k]
        return EncoderInputs(x=x, n=n)

    def new_frames(self, st: StreamState, samples: int, final: bool) -> int:
        """The encoder frames a step makes final for a call at ``st`` that then
        has ``samples`` of audio, counted on the host before the step runs, so
        its declaration can name them and ``front`` make them. Also without a
        new mel frame: the end of the call can make its last frame final."""
        c = self.c
        if samples <= c.n_fft // 2:
            return 0
        mels = max(st.mels, final_mel_frames(c, samples, final))
        return max(0, final_encoder_frames(c, mels, samples, final) - st.frames)

    @staticmethod
    def advance(sts: list[StreamState], enc: EncoderInputs) -> None:
        """Count the step's frames into each call's state, once its layers ran."""
        for st, n in zip(sts, enc.n, strict=True):
            st.frames += n

    # -- layers --------------------------------------------------------------

    def _attend(self, layer: "ConformerLayer", li: int, x: torch.Tensor, kv: "BoundedKVManager") -> torch.Tensor:
        """Self-attention of the new frames ``x (B, t, D)``: each attends the
        ``ctx`` frames before it and itself, the older ones from ``kv``, which
        then keeps this step's keys and values. Transformer-XL's position term
        depends on the distance alone, so it goes in as the band of distances
        ``0..ctx`` (``bd_all``), unscaled: the kernel scales the sum."""
        attn = layer.self_attn
        b, t, _ = x.shape
        h, dk, ctx = attn.n_heads, attn.head_dim, self._ctx
        q = attn.linear_q(x).view(b, t, h, dk)
        k = attn.linear_k(x).view(b, t, h, dk).transpose(1, 2)
        v = attn.linear_v(x).view(b, t, h, dk).transpose(1, 2)
        key = (attn, x.dtype, torch.is_autocast_enabled(x.device.type))
        p = self._pos.get(key)
        if p is None:                                                         # (h, ctx+1, dk)
            p = self._pos[key] = attn.linear_pos(self._pe.to(x.dtype)).view(ctx + 1, h, dk).transpose(0, 1)
        # in k's dtype: under autocast the bias add promotes q, and the kernel takes one dtype
        q_u = (q + attn.pos_bias_u).to(k.dtype).transpose(1, 2)
        q_v = (q + attn.pos_bias_v).transpose(1, 2)
        bd_all = torch.matmul(q_v, p.transpose(-2, -1))                       # (B, h, t, ctx+1)
        out = kv.attend(li, q_u, k, v, rel_bias=bd_all, ieee=k.dtype == torch.float32)
        return attn.linear_out(out.reshape(b, t, h * dk))

    def _conv(self, layer: "ConformerLayer", x: torch.Tensor,
              conv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv = layer.conv
        z = conv.pointwise_conv1(x.transpose(1, 2))
        z = F.glu(z, dim=1)                                                     # (B, D, t)
        z = torch.cat([conv_cache.to(z.dtype), z], dim=2)                       # (B, D, kernel-1+t)
        tails = z
        z = conv.depthwise_conv(z)
        z = conv.batch_norm(z.transpose(1, 2)).transpose(1, 2)
        z = F.silu(z)
        return conv.pointwise_conv2(z).transpose(1, 2), tails

    def layers(self, x: torch.Tensor, kv: "BoundedKVManager", pool: "RecurrentStatePool",
               rows: "RecurrentAddressing", n: torch.Tensor) -> torch.Tensor:
        """The layer stack over ``x (B, t, D)`` and the projection. Attention
        runs on ``kv`` under the plan its step declared; the conv tails are read
        and written in ``pool`` at ``rows``. ``n`` is each call's new frames,
        a device ``(B,)`` count (``n <= t``; a call with none keeps its conv
        tails as they were)."""
        b, t, _ = x.shape
        keep = self.c.conv_kernel_size - 1
        # each call keeps its newest kernel-1 conv inputs: positions n .. n + keep of the concatenation
        conv_idx = (n.view(b, 1, 1) + torch.arange(keep, device=x.device).view(1, 1, keep)).expand(
            b, self.c.d_model, keep)
        for li, layer in enumerate(self.p.encoder.layers):
            conv_cache = pool.gather("conv", rows, li)
            r = x + layer.FC_FACTOR * layer.feed_forward1(layer.norm_feed_forward1(x))
            r = r + self._attend(layer, li, layer.norm_self_att(r), kv)
            cv, tails = self._conv(layer, layer.norm_conv(r), conv_cache)
            r = r + cv
            r = r + layer.FC_FACTOR * layer.feed_forward2(layer.norm_feed_forward2(r))
            x = layer.norm_out(r)
            pool.scatter_("conv", rows, tails.gather(2, conv_idx).to(conv_cache.dtype), li)
        return self.p.proj(x)

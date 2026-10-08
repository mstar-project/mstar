"""S3 speech tokenizer (S3TokenizerV2, 25 Hz) for reference-audio prompts.

Whisper-style 128-bin log-mel at 16 kHz, two strided convolutions (x4 in
time), six pre-LN attention blocks with rotary positions and an FSMN memory
branch on the values, then a finite-scalar quantizer: eight tanh channels
rounded to {-1, 0, 1} form a base-3 code in ``[0, 6560]``.

Read from the ``s3tokenizer`` package (``model.py``, ``model_v2.py``,
``utils.py``) and ``chatterbox/models/s3tokenizer/s3tokenizer.py``, which
wraps it with the per-utterance log-mel and the ``max_len`` cap. Only the
short-audio path (<= 30 s) exists here: a prompt is at most 15 s.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.chatterbox.components.audio_frontend import mel_filter_bank, stft_power
from mstar.model.chatterbox.config import S3TokenizerConfig
from mstar.model.chatterbox.loader import DEFAULT_SKIP_FRAGMENTS, WeightStream, load_component


def _rotary_cos_sin(head_dim: int, max_positions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``[max_positions, head_dim]`` cos/sin tables with the frequency vector
    repeated over both halves (rotate-half convention)."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_positions).float(), inv_freq)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class FSMNAttention(nn.Module):
    """Multi-head attention whose value path also feeds a depthwise FSMN
    memory block; the two are summed at the output."""

    def __init__(self, n_state: int, n_head: int, kernel_size: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_state // n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
        self.fsmn_block = nn.Conv1d(
            n_state, n_state, kernel_size, stride=1, padding=0, groups=n_state, bias=False,
        )
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding

    def _fsmn(self, v: torch.Tensor, mask_pad: torch.Tensor) -> torch.Tensor:
        """``v [B, T, n_state]`` masked, filtered along time, residual, masked."""
        v = v * mask_pad
        x = F.pad(v.transpose(1, 2), (self.left_padding, self.right_padding))
        x = self.fsmn_block(x).transpose(1, 2)
        return (x + v) * mask_pad

    def forward(
        self, x: torch.Tensor, mask_bias: torch.Tensor, mask_pad: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = x.shape
        q = self.query(x).view(B, T, self.n_head, self.head_dim)
        k = self.key(x).view(B, T, self.n_head, self.head_dim)
        v = self.value(x)
        # rotary positions on q and k; cos/sin are [T, head_dim]
        cos = cos[:T].unsqueeze(0).unsqueeze(2)
        sin = sin[:T].unsqueeze(0).unsqueeze(2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        memory = self._fsmn(v, mask_pad)
        v = v.view(B, T, self.n_head, self.head_dim)

        # the reference scales q and k each by dim**-0.25 before the product
        scale = self.head_dim ** -0.25
        q = q.permute(0, 2, 1, 3) * scale
        k = k.permute(0, 2, 3, 1) * scale
        v = v.permute(0, 2, 1, 3)
        qk = q @ k + mask_bias
        w = torch.softmax(qk.float(), dim=-1).to(q.dtype)
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.out(out) + memory


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, kernel_size: int):
        super().__init__()
        self.attn = FSMNAttention(n_state, n_head, kernel_size)
        self.attn_ln = nn.LayerNorm(n_state, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(n_state, 4 * n_state), nn.GELU(), nn.Linear(4 * n_state, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x, mask_bias, mask_pad, cos, sin):
        x = x + self.attn(self.attn_ln(x), mask_bias, mask_pad, cos, sin)
        return x + self.mlp(self.mlp_ln(x))


class AudioEncoderV2(nn.Module):
    def __init__(self, config: S3TokenizerConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.n_mels, config.n_state, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(config.n_state, config.n_state, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(config.n_state, config.n_head, config.fsmn_kernel_size)
            for _ in range(config.n_layer)
        ])
        head_dim = config.n_state // config.n_head
        cos, sin = _rotary_cos_sin(head_dim, 2 * 1024)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def _non_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]

    @staticmethod
    def _conv_out_len(length: torch.Tensor | int, stride: int) -> torch.Tensor | int:
        # kernel 3, padding 1: floor((L + 2 - 3) / stride) + 1
        return (length - 1) // stride + 1

    def forward(self, mel: torch.Tensor, mel_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``mel [B, n_mels, T]`` (padded) + lengths -> ``[B, T // 4, n_state]``
        hidden states and their lengths."""
        T = mel.shape[-1]
        mask = self._non_pad_mask(mel_len, T).unsqueeze(1)
        x = F.gelu(self.conv1(mel * mask))
        x_len = self._conv_out_len(mel_len, 2)
        x_slen = self._conv_out_len(T, 2)
        mask = self._non_pad_mask(x_len, x_slen).unsqueeze(1)
        x = F.gelu(self.conv2(x * mask))
        x_len = self._conv_out_len(x_len, 2)
        x_slen = self._conv_out_len(x_slen, 2)
        mask = self._non_pad_mask(x_len, x_slen).unsqueeze(1)  # [B, 1, T']
        x = x.permute(0, 2, 1)
        mask_pad = mask.transpose(1, 2).to(x.dtype)  # [B, T', 1]
        mask_bias = ((1.0 - mask.to(x.dtype)) * -1.0e10).unsqueeze(1)  # [B, 1, 1, T']
        cos = self.rope_cos.to(x.dtype)
        sin = self.rope_sin.to(x.dtype)
        for block in self.blocks:
            x = block(x, mask_bias, mask_pad, cos, sin)
        return x, x_len


class FSQuantizer(nn.Module):
    """Finite scalar quantization to a base-``levels`` code."""

    def __init__(self, dim: int, fsq_dim: int = 8, levels: int = 3):
        super().__init__()
        self.project_down = nn.Linear(dim, fsq_dim)
        self.levels = levels
        self.register_buffer(
            "powers", torch.pow(levels, torch.arange(fsq_dim)).float(), persistent=False,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.project_down(hidden).float().tanh()
        h = (h * 0.9990000128746033).round() + 1
        return (h * self.powers).sum(dim=-1).long()


class S3Tokenizer(nn.Module):
    def __init__(self, config: S3TokenizerConfig | None = None):
        super().__init__()
        self.config = config or S3TokenizerConfig()
        cfg = self.config
        self.encoder = AudioEncoderV2(cfg)
        self.quantizer = FSQuantizer(cfg.n_state, cfg.fsq_dim, cfg.fsq_levels)
        self.register_buffer(
            "mel_filters", mel_filter_bank(cfg.sample_rate, cfg.n_fft, cfg.n_mels), persistent=False,
        )
        self.register_buffer("window", torch.hann_window(cfg.n_fft, periodic=True), persistent=False)

    # ---------------------------------------------------------------- front end

    def log_mel(self, wav16: torch.Tensor) -> torch.Tensor:
        """``[T]`` or ``[B, T]`` 16 kHz waveform -> ``[B, n_mels, T // hop]``
        Whisper log-mel; the dynamic-range floor is per utterance."""
        cfg = self.config
        wav = wav16.float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        power = stft_power(wav, cfg.n_fft, cfg.hop_size, cfg.n_fft, self.window)[..., :-1]
        mel = self.mel_filters @ power
        log_spec = torch.clamp(mel, min=1e-10).log10()
        floor = log_spec.amax(dim=(1, 2), keepdim=True) - 8.0
        log_spec = torch.maximum(log_spec, floor)
        return (log_spec + 4.0) / 4.0

    # ---------------------------------------------------------------- tokens

    def encode_mel(self, mel: torch.Tensor, mel_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Padded log-mel batch -> ``(tokens [B, N], lens [B])``; entries past a
        row's length are padding and must be ignored."""
        if int(mel_len.max()) > self.config.max_frames_per_pass:
            raise ValueError(
                f"S3 tokenizer prompts are limited to {self.config.max_frames_per_pass} mel "
                f"frames ({self.config.max_frames_per_pass * self.config.hop_size / self.config.sample_rate:.0f} s)"
            )
        hidden, lens = self.encoder(mel, mel_len)
        return self.quantizer(hidden), lens.long()

    @torch.no_grad()
    def forward(
        self, wavs16: list[torch.Tensor] | torch.Tensor, max_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize 16 kHz waveforms (a list of ``[T_i]``, or one ``[B, T]`` /
        ``[T]`` tensor). ``max_len`` caps the token count per utterance by
        truncating its log-mel to ``4 * max_len`` frames, as the reference
        does for T3's speech prompt."""
        if isinstance(wavs16, torch.Tensor):
            wavs16 = [wavs16] if wavs16.dim() == 1 else list(wavs16)
        mels = []
        for wav in wavs16:
            mel = self.log_mel(wav)[0]
            if max_len is not None:
                mel = mel[:, : max_len * 4]
            mels.append(mel)
        lens = torch.tensor([m.shape[1] for m in mels], dtype=torch.long, device=mels[0].device)
        width = int(lens.max())
        batch = torch.stack([F.pad(m, (0, width - m.shape[1])) for m in mels])
        return self.encode_mel(batch, lens)

    # ---------------------------------------------------------------- weights

    @staticmethod
    def _remap(name: str) -> str:
        return name.replace("quantizer._codebook.", "quantizer.")

    def load_weights(self, weights: WeightStream) -> set[str]:
        """Load the ``tokenizer.`` subtree of ``s3gen.safetensors`` (prefix
        already stripped). The stored mel filter and window are recomputed."""
        return load_component(
            self, weights, component="S3 tokenizer", name_remapper=self._remap,
            skip_fragments=DEFAULT_SKIP_FRAGMENTS + ("_mel_filters", "window"),
        )

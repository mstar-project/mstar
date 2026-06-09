"""Whisper-style audio encoder for Ming-flash-omni-2.0.

Self-contained port of vllm-omni's
``vllm_omni/model_executor/models/ming_flash_omni/audio_encoder.py`` (247
LOC) — itself a re-implementation of the OpenAI Whisper encoder that
supports packed variable-length inputs (the Ming source's
``modeling_whisper_encoder.py`` uses padded batches and depends on
``openai-whisper``; we avoid that runtime dep entirely).

Weight-key parity with the upstream Whisper encoder:
  - ``conv1.{weight,bias}``                  (kernel=3, stride=1, pad=1)
  - ``conv2.{weight,bias}``                  (kernel=3, stride=2, pad=1)
  - ``positional_embedding``                 buffer (sinusoidal, not loaded)
  - ``blocks.{N}.attn.{query,key,value,out}.{weight,bias}``
  - ``blocks.{N}.attn_ln.{weight,bias}``
  - ``blocks.{N}.mlp.{0,2}.{weight,bias}``   (Linear, GELU, Linear)
  - ``blocks.{N}.mlp_ln.{weight,bias}``
  - ``ln_post.{weight,bias}``

The released Ming checkpoint stores these under the top-level prefix
``audio.*`` (see ``model.safetensors.index.json``); the loader strips
that prefix before applying state_dict here.
"""

from __future__ import annotations

import logging
import operator
from itertools import accumulate

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Whisper primitives (auto-dtype-casting layers + sinusoidal embedding)
# ---------------------------------------------------------------------------


def _sinusoids(length: int, channels: int, max_timescale: int = 10000) -> torch.Tensor:
    """Sinusoidal positional embedding from Whisper.

    Args:
        length:   positions.
        channels: must be even.
        max_timescale: matches OpenAI Whisper's default (10_000).
    """
    if channels % 2 != 0:
        raise ValueError(f"channels must be even, got {channels}")
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class _AutoCastConv1d(nn.Conv1d):
    """Conv1d that casts its weight/bias to the input dtype on every forward.

    Lets the encoder keep bf16 weights while taking fp32 mel inputs
    without an explicit ``.to(bf16)`` at the call site (Whisper does
    this too).
    """

    def _conv_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype),
        )


class _AutoCastLinear(nn.Linear):
    """Linear with the same auto-cast trick."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype),
        )


# ---------------------------------------------------------------------------
# Multi-head attention (packed sequence with optional FA2 fast path)
# ---------------------------------------------------------------------------


def _try_import_flash_attn():
    """Return flash_attn_varlen_func if importable, else None.

    Wrapped so test boxes without flash-attn keep green via the manual
    PyTorch fallback. Audio encoder forward shape is identical either way.
    """
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore
        return flash_attn_varlen_func
    except ImportError:
        return None


_FLASH_ATTN_VARLEN = _try_import_flash_attn()


class _PackedMultiHeadAttention(nn.Module):
    """Whisper-style MHA with variable-length packed sequences.

    Param naming matches OpenAI Whisper (``query`` / ``key`` / ``value`` /
    ``out`` — not ``q_proj`` / ``k_proj`` / etc.) so the checkpoint keys
    load directly.
    """

    def __init__(self, n_state: int, n_head: int, use_flash_attn: bool = True) -> None:
        super().__init__()
        if n_state % n_head != 0:
            raise ValueError(f"n_state={n_state} not divisible by n_head={n_head}")
        self.n_head = n_head
        self.query = _AutoCastLinear(n_state, n_state)
        self.key = _AutoCastLinear(n_state, n_state, bias=False)
        self.value = _AutoCastLinear(n_state, n_state)
        self.out = _AutoCastLinear(n_state, n_state)

        if use_flash_attn and _FLASH_ATTN_VARLEN is None:
            logger.warning("flash-attn not available — falling back to manual attention.")
        self.use_flash_attn = use_flash_attn and _FLASH_ATTN_VARLEN is not None

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Packed-sequence attention.

        Args:
            x:          (total_tokens, n_state) packed tensor.
            cu_seqlens: (num_seqs + 1,) cumulative seq lengths,
                        e.g. [0, len1, len1+len2, ...]. int32.
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        n_tokens, n_state = q.shape
        head_dim = n_state // self.n_head
        q = q.view(n_tokens, self.n_head, head_dim)
        k = k.view(n_tokens, self.n_head, head_dim)
        v = v.view(n_tokens, self.n_head, head_dim)

        if self.use_flash_attn and q.dtype in (torch.float16, torch.bfloat16):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            attn_output = _FLASH_ATTN_VARLEN(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
            )
        else:
            attn_output = self._manual_packed_attention(q, k, v, cu_seqlens)

        attn_output = attn_output.contiguous().view(n_tokens, n_state)
        return self.out(attn_output)

    @staticmethod
    def _manual_packed_attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Pad-attention-unpack fallback for the packed format."""
        _, n_head, head_dim = q.shape
        scale = head_dim ** -0.5

        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        batch = len(seqlens)
        max_len = max(seqlens)

        # Pad each sequence to max_len so we can run a single batched matmul.
        q_pad = torch.zeros(batch, max_len, n_head, head_dim, dtype=q.dtype, device=q.device)
        k_pad = torch.zeros_like(q_pad)
        v_pad = torch.zeros_like(q_pad)
        for i, ln in enumerate(seqlens):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            q_pad[i, :ln] = q[start:end]
            k_pad[i, :ln] = k[start:end]
            v_pad[i, :ln] = v[start:end]

        # (B, H, T, D)
        q_pad = q_pad.transpose(1, 2)
        k_pad = k_pad.transpose(1, 2)
        v_pad = v_pad.transpose(1, 2)

        # Mask padding columns out of softmax.
        padding_mask = (
            torch.arange(max_len, device=q.device)[None, :]
            >= torch.tensor(seqlens, device=q.device)[:, None]
        )
        attn_mask = torch.zeros(batch, 1, 1, max_len, dtype=q.dtype, device=q.device)
        attn_mask = attn_mask.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(2), -torch.finfo(q.dtype).max,
        )

        scores = torch.matmul(q_pad, k_pad.transpose(-2, -1)) * scale + attn_mask
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, v_pad)  # (B, H, T, D)
        context = context.transpose(1, 2).contiguous()  # (B, T, H, D)

        # Unpack back to packed.
        return torch.cat([context[i, :ln] for i, ln in enumerate(seqlens)], dim=0)


# ---------------------------------------------------------------------------
# Residual block (Whisper attn + FFN)
# ---------------------------------------------------------------------------


class _ResidualAttentionBlock(nn.Module):
    """Whisper-style attn + FFN residual block (param names match upstream)."""

    def __init__(self, n_state: int, n_head: int, use_flash_attn: bool = True) -> None:
        super().__init__()
        self.attn = _PackedMultiHeadAttention(n_state, n_head, use_flash_attn=use_flash_attn)
        self.attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        # Sequential layout (Linear, GELU, Linear) so checkpoint keys
        # blocks.{N}.mlp.0.* / .2.* hit the right module by integer index.
        self.mlp = nn.Sequential(
            _AutoCastLinear(n_state, n_mlp),
            nn.GELU(),
            _AutoCastLinear(n_mlp, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x), cu_seqlens=cu_seqlens)
        x = x + self.mlp(self.mlp_ln(x))
        return x


# ---------------------------------------------------------------------------
# Encoder — public API
# ---------------------------------------------------------------------------


class MingAudioEncoder(nn.Module):
    """Whisper audio encoder with packed-sequence support.

    Loadable from the released Ming-flash-omni-2.0 checkpoint's
    ``audio.*`` weight subtree (caller strips the prefix). Defaults
    match the released ckpt's ``audio_config.whisper_encoder_config``.

    Note the deviation from the openai-whisper original: the
    ``positional_embedding`` is a *buffer* with a fixed sinusoidal
    table sized to ``n_ctx`` (15000 on the released ckpt — enough for
    ~150 s of audio at the post-conv frame rate). The Ming source's
    ``modeling_whisper_encoder.py`` notes the same change — they drop
    the trainable parameter so they can shrink the sequence length
    below the original 30 s pad.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_ctx: int = 15000,
        n_state: int = 1280,
        n_head: int = 20,
        n_layer: int = 32,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        self.n_layer = n_layer
        self.n_mels = n_mels
        self.use_flash_attn = use_flash_attn
        self.audio_emb_dim = n_state

        self.conv1 = _AutoCastConv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = _AutoCastConv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        # Buffer (not Parameter) — checkpoint doesn't ship this; we
        # recompute it. Keeps load_state_dict happy with the snapshot.
        self.register_buffer("positional_embedding", _sinusoids(n_ctx, n_state))
        self.blocks = nn.ModuleList(
            [_ResidualAttentionBlock(n_state, n_head, use_flash_attn=use_flash_attn) for _ in range(n_layer)]
        )
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, x_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder on a list of variable-length mel spectrograms.

        Args:
            x_list: list of (n_mels, T_i) mel features per audio clip.

        Returns:
            (packed, cu_seqlens):
              - packed:     (total_T', n_state) all clips concatenated
                            along time.
              - cu_seqlens: (len(x_list) + 1,) int32 cumulative encoded
                            lengths suitable for re-segmenting / feeding
                            into the projector.
        """
        target_dtype = self.conv1.weight.dtype

        encoded = []
        encoded_lens: list[int] = []
        for mel in x_list:
            mel = mel.to(target_dtype)
            x = mel.unsqueeze(0)                          # (1, n_mels, T)
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.squeeze(0).transpose(0, 1)              # (T', n_state)

            seq_len = x.shape[0]
            x = (x + self.positional_embedding[:seq_len, :]).to(x.dtype)
            encoded.append(x)
            encoded_lens.append(seq_len)

        packed = torch.cat(encoded, dim=0)                # (sum T', n_state)
        cu_seqlens = torch.tensor(
            list(accumulate(encoded_lens, func=operator.add, initial=0)),
            device=packed.device, dtype=torch.int32,
        )
        for block in self.blocks:
            packed = block(packed, cu_seqlens=cu_seqlens)
        packed = self.ln_post(packed)
        return packed, cu_seqlens


def build_audio_encoder(
    audio_config,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    use_flash_attn: bool = True,
) -> MingAudioEncoder:
    """Construct :class:`MingAudioEncoder` from an ``AudioEncoderConfig``.

    Matches ``build_vision_encoder``'s factory shape so the model class
    treats both modalities symmetrically when wiring submodules.
    """
    whisper_cfg = audio_config.whisper_encoder_config
    encoder = MingAudioEncoder(
        n_mels=int(whisper_cfg["n_mels"]),
        n_ctx=int(whisper_cfg["n_ctx"]),
        n_state=int(whisper_cfg["n_state"]),
        n_head=int(whisper_cfg["n_head"]),
        n_layer=int(whisper_cfg["n_layer"]),
        use_flash_attn=use_flash_attn,
    )
    encoder = encoder.to(dtype=dtype, device=device)
    encoder.eval()
    return encoder


__all__ = ["MingAudioEncoder", "build_audio_encoder"]

"""Phi-4-multimodal's speech encoder: the ``cascades`` conformer and its two
projections into the LLM's embedding space.

Global CMVN on 80 log-mel bins, NeMo's dw_striding subsampling (8x, the
shared FastConformer stack), then 24 conformer blocks over windows of at most
500 frames, each window attending in full within itself with a learned T5
bias per distance. The bias is dense, so attention runs through SDPA in the
layer, as FastConformer's does. A projection MLP maps each frame to an LLM
row: ``speech`` for audio-only prompts, ``vision`` when the prompt also holds
an image.

The encoder is cut where its batching changes: ``subsample`` runs per clip
(its convs see across window boundaries), ``blocks`` per window, and the
projection per output row. Every length comes in as a device tensor, so no
part reads the host.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.fastconformer import DwStridingSubsampling, FastConformerConfig
from mstar.model.components.masking import length_mask
from mstar.utils.h2d import to_device_nonblocking

SPEECH = "speech"
VISION = "vision"
CHECKPOINT_PREFIX = "model.embed_tokens_extend.audio_embed."


@dataclass(frozen=True)
class Phi4MMAudioConfig:
    num_mel_bins: int = 80
    hidden_size: int = 1024
    num_heads: int = 16
    intermediate_size: int = 1536
    num_layers: int = 24
    conv_kernel_size: int = 3
    subsampling_channels: int = 1024
    time_reduction: int = 8
    # the T5 bias table holds distances -max_distance .. max_distance - 1
    max_distance: int = 500
    # encoder frames per attention window; the reference's ``max_seq_len``, a constant
    window: int = 500
    llm_hidden_size: int = 3072

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def tokens(self, mel_frames: int) -> int:
        """Encoder frames, one LLM row each, for ``mel_frames`` mel frames."""
        return -(-mel_frames // self.time_reduction)

    def windows(self, tokens: int) -> int:
        return -(-tokens // self.window)

    def subsampling_config(self) -> FastConformerConfig:
        return FastConformerConfig(
            hidden_size=self.hidden_size,
            num_mel_bins=self.num_mel_bins,
            subsampling_factor=self.time_reduction,
            subsampling_channels=self.subsampling_channels,
            causal=False,
        )

    @classmethod
    def from_hf(cls, hf: dict) -> Phi4MMAudioConfig:
        """From the checkpoint's ``config.json``. The settings this port
        implements are fixed; a checkpoint that changes one is refused."""
        processor = hf["audio_processor"]
        enc = processor["config"]
        embd = hf["embd_layer"]["audio_embd_layer"]
        dim = enc["attention_dim"]
        bias = enc.get("relative_attention_bias_args") or {}
        expected = {
            "name": (processor.get("name"), "cascades"),
            "input_layer": (enc.get("input_layer"), "nemo_conv"),
            "nemo_conv_settings": (sorted(enc.get("nemo_conv_settings") or {}), ["conv_channels"]),
            "chunk_size": (enc.get("chunk_size"), -1),
            "causal": (enc.get("causal"), True),
            "batch_norm": (enc.get("batch_norm"), False),
            "ext_pw_kernel_size": (enc.get("ext_pw_kernel_size"), 1),
            "ext_pw_out_channel": (enc.get("ext_pw_out_channel"), dim),
            "depthwise_seperable_out_channel": (enc.get("depthwise_seperable_out_channel"), dim),
            "depthwise_multiplier": (enc.get("depthwise_multiplier"), 1),
            "activation": (enc.get("activation"), "swish"),
            "conv_activation": (enc.get("conv_activation"), "swish"),
            "conv_glu_type": (enc.get("conv_glu_type"), "swish"),
            "bias_in_glu": (enc.get("bias_in_glu"), True),
            "linear_glu_in_convm": (enc.get("linear_glu_in_convm", False), False),
            "attention_group_size": (enc.get("attention_group_size", 1), 1),
            "relative_attention_bias_args.type": (bias.get("type"), "t5"),
            "t5_bias_symmetric": (bias.get("t5_bias_symmetric", False), False),
            "projection_cls": (embd.get("projection_cls"), "mlp"),
            "downsample_rate": (embd.get("downsample_rate", 1), 1),
            "use_qformer": (embd.get("use_qformer", False), False),
            "use_conv_downsample": (embd.get("use_conv_downsample", False), False),
            "compression_rate": (embd.get("compression_rate"), enc["time_reduction"]),
        }
        wrong = {k: got for k, (got, want) in expected.items() if got != want}
        if wrong:
            raise ValueError(f"unsupported Phi-4-multimodal audio settings: {wrong}")
        return cls(
            num_mel_bins=enc["input_size"],
            hidden_size=dim,
            num_heads=enc["attention_heads"],
            intermediate_size=enc["linear_units"],
            num_layers=enc["num_blocks"],
            conv_kernel_size=enc["kernel_size"],
            subsampling_channels=enc["nemo_conv_settings"]["conv_channels"],
            time_reduction=enc["time_reduction"],
            max_distance=bias.get("t5_bias_max_distance", 1000),
            llm_hidden_size=hf["hidden_size"],
        )


class _FeedForward(nn.Module):
    """Pre-norm GLU feed-forward: ``down(a * silu(g))``, ``a, g = up(norm(x))``."""

    def __init__(self, config: Phi4MMAudioConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.up = nn.Linear(config.hidden_size, 2 * config.intermediate_size)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.up(self.norm(x)).chunk(2, dim=-1)
        return self.down(value * F.silu(gate))


class _Attention(nn.Module):
    def __init__(self, config: Phi4MMAudioConfig):
        super().__init__()
        self.num_heads, self.head_dim = config.num_heads, config.head_dim
        # the checkpoint's linear_q/k/v, fused at load
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """``x (B, T, hidden)``; ``bias`` ``(B or 1, H, T, T)``, the T5 bias with
        masked keys at the dtype's minimum."""
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out(out.transpose(1, 2).reshape(b, t, -1))


class _ConvModule(nn.Module):
    """Norm, GLU with biases on both halves (``(a + b1) * silu(g + b2)``),
    causal depthwise conv, pointwise, SiLU, pointwise. The checkpoint's 1x1
    convs are linears here. Causal, so padding after a window's last real
    frame never reaches one."""

    def __init__(self, config: Phi4MMAudioConfig):
        super().__init__()
        c, k = config.hidden_size, config.conv_kernel_size
        self.kernel_size = k
        self.norm = nn.LayerNorm(c)
        self.glu = nn.Linear(c, 2 * c)
        self.b1 = nn.Parameter(torch.zeros(c))
        self.b2 = nn.Parameter(torch.zeros(c))
        self.depthwise = nn.Conv1d(c, c, k, groups=c)
        self.pointwise = nn.Linear(c, c)
        self.out = nn.Linear(c, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.glu(self.norm(x)).chunk(2, dim=-1)
        x = (value + self.b1) * F.silu(gate + self.b2)
        x = self.depthwise(F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))).transpose(1, 2)
        return self.out(F.silu(self.pointwise(x)))


class _Block(nn.Module):
    def __init__(self, config: Phi4MMAudioConfig):
        super().__init__()
        self.ff_in = _FeedForward(config)
        self.norm_attn = nn.LayerNorm(config.hidden_size)
        self.attn = _Attention(config)
        self.conv = _ConvModule(config)
        self.ff_out = _FeedForward(config)
        self.norm_out = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff_in(x)
        x = x + self.attn(self.norm_attn(x), bias)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff_out(x)
        return self.norm_out(x)


class Phi4MMAudioEncoder(nn.Module):
    def __init__(self, config: Phi4MMAudioConfig):
        super().__init__()
        self.config = config
        self.register_buffer("global_mean", torch.zeros(config.num_mel_bins))
        self.register_buffer("global_invstd", torch.ones(config.num_mel_bins))
        self.subsampling = DwStridingSubsampling(config.subsampling_config())
        # one learned bias per head and distance, shared by every layer
        self.rel_bias = nn.Embedding(2 * config.max_distance, config.num_heads)
        self.layers = nn.ModuleList(_Block(config) for _ in range(config.num_layers))

    def subsample(self, mel: torch.Tensor, mel_lengths: torch.Tensor) -> torch.Tensor:
        """``mel (C, T, mel bins)`` clips padded to the longest, ``mel_lengths
        (C,)`` -> ``(C, ceil(T / 8), hidden)``, zero past each clip's frames.

        Padding is zeroed after CMVN and after every conv stage
        (`DwStridingSubsampling`), so the convs, which see both sides, give a
        clip's frames what they give it alone. Zero is also what the reference
        pads a window with after subsampling."""
        x = (mel - self.global_mean) * self.global_invstd
        x = x * length_mask(mel_lengths, x.shape[1])[:, :, None]
        x = self.subsampling(x, mel_lengths)
        tokens = -(-mel_lengths // self.config.time_reduction)
        return x * length_mask(tokens, x.shape[1])[:, :, None]

    def attention_bias(self, width: int, key_lengths: torch.Tensor | None) -> torch.Tensor:
        """The T5 bias over a ``width``-frame window, ``(1, H, W, W)``, or
        ``(N, H, W, W)`` with keys at or past ``key_lengths (N,)`` masked."""
        pos = torch.arange(width, device=self.rel_bias.weight.device)
        distance = (pos[None, :] - pos[:, None]).clamp(-self.config.max_distance, self.config.max_distance - 1)
        # contiguous: SDPA's memory-efficient kernel refuses a permuted bias and falls to math
        bias = self.rel_bias(distance + self.config.max_distance).permute(2, 0, 1).contiguous()[None]
        if key_lengths is None:
            return bias
        keys = length_mask(key_lengths, width)[:, None, None, :]
        return bias.masked_fill(~keys, torch.finfo(bias.dtype).min)

    def blocks(self, windows: torch.Tensor, key_lengths: torch.Tensor | None = None) -> torch.Tensor:
        """The conformer blocks over ``windows (N, W, hidden)``, each its own
        sequence; ``key_lengths (N,)`` masks the keys past each window's own,
        None when no window is padded."""
        bias = self.attention_bias(windows.shape[1], key_lengths).to(windows.dtype)
        x = windows
        for layer in self.layers:
            x = layer(x, bias)
        return x


class AudioWindows(NamedTuple):
    """A step's clips cut into attention windows, on the host.

    Each clip's encoder frames, padded to ``per_clip * width``, form
    ``per_clip`` slots of ``width`` frames. Window ``w`` is slot
    ``source[w]`` (of clip ``source[w] // per_clip``) and attends its first
    ``keys[w]`` frames; ``rows`` is where each output row sits among the
    windows' frames, clip by clip."""

    width: int
    per_clip: int
    source: list[int]
    keys: list[int]
    rows: list[int]


def plan_windows(tokens: Sequence[int], window: int, width: int | None = None) -> AudioWindows:
    """The windows of clips of ``tokens`` encoder frames each, ``width``
    frames wide (a capture bucket), or as wide as they need.

    Clips that fit one window run at the longest clip's width, each masked
    past its own frames. Past one window every row is ``window`` wide and a
    long clip's windows attend in full, the last one's zero tail included:
    the reference unfolds a clip alone that way, with no mask."""
    longest = max(tokens)
    needed, per_clip = (longest, 1) if longest <= window else (window, -(-longest // window))
    if width is None:
        width = needed
    elif not needed <= width <= window:
        raise ValueError(f"windows of {needed} frames don't run {width} wide")
    source: list[int] = []
    keys: list[int] = []
    rows: list[int] = []
    for c, n in enumerate(tokens):
        for k in range(-(-n // window)):
            start = len(source) * width
            source.append(c * per_clip + k)
            keys.append(window if n > window else n)
            rows.extend(range(start, start + min(window, n - k * window)))
    return AudioWindows(width, per_clip, source, keys, rows)


class AudioBatch(NamedTuple):
    """A step's clips on the device, as `Phi4MMAudioTower.encode` reads them."""

    mel: torch.Tensor  # (clips, longest mel frames, mel bins), the tower's dtype
    mel_lengths: torch.Tensor  # (clips,)
    width: int
    per_clip: int
    source: torch.Tensor  # (windows,)
    key_lengths: torch.Tensor | None  # (windows,), None when no window is masked and no width bucketed
    rows: torch.Tensor  # (output rows,)


def audio_batch(
    clips: Sequence[torch.Tensor], config: Phi4MMAudioConfig, device: torch.device, dtype: torch.dtype,
    mel_frames: int | None = None, width: int | None = None,
) -> AudioBatch:
    """``clips``, each ``(mel frames, mel bins)``, as one `AudioBatch`, its
    mel padded to ``mel_frames`` and its windows ``width`` wide when given (a
    capture's buckets). The copies are queued without a stall, and the
    padding is done on the device: a host copy of that size runs on torch's
    intra-op pool, which stalls on a busy host (~75 ms for 8 clips at a load
    of 40)."""
    frames = [int(c.shape[0]) for c in clips]
    plan = plan_windows([config.tokens(f) for f in frames], config.window, width)
    mel = torch.nn.utils.rnn.pad_sequence([to_device_nonblocking(c, device) for c in clips], batch_first=True)
    if mel_frames is not None:
        mel = F.pad(mel, (0, 0, 0, mel_frames - mel.shape[1]))
    # a capture's key lengths are a static buffer, so a bucketed batch always carries them
    masked = width is not None or any(k != plan.width for k in plan.keys)
    return AudioBatch(
        # the reference rounds the mel to the weights' dtype before CMVN
        mel=mel.to(dtype),
        mel_lengths=to_device_nonblocking(frames, device, dtype=torch.long),
        width=plan.width,
        per_clip=plan.per_clip,
        source=to_device_nonblocking(plan.source, device, dtype=torch.long),
        key_lengths=to_device_nonblocking(plan.keys, device, dtype=torch.long) if masked else None,
        rows=to_device_nonblocking(plan.rows, device, dtype=torch.long),
    )


def _mlp(config: Phi4MMAudioConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(config.hidden_size, config.llm_hidden_size),
        nn.GELU(),
        nn.Linear(config.llm_hidden_size, config.llm_hidden_size),
    )


class Phi4MMAudioTower(nn.Module):
    """The encoder and both projections, loaded from the checkpoint."""

    def __init__(self, config: Phi4MMAudioConfig):
        super().__init__()
        self.config = config
        self.encoder = Phi4MMAudioEncoder(config)
        self.projection = nn.ModuleDict({SPEECH: _mlp(config), VISION: _mlp(config)})

    def encode(
        self, batch: AudioBatch,
        subsample: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        blocks: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The conformer's output rows, ``(rows, hidden)``, clip by clip.
        ``subsample`` and ``blocks`` stand in for the encoder's own (a
        captured replay of each)."""
        x = (subsample or self.encoder.subsample)(batch.mel, batch.mel_lengths)
        clips, frames, hidden = x.shape
        # a bucketed mel can subsample past the windows' frames, all zero there
        x = F.pad(x, (0, 0, 0, batch.per_clip * batch.width - frames))
        windows = x.reshape(clips * batch.per_clip, batch.width, hidden).index_select(0, batch.source)
        out = (blocks or self.encoder.blocks)(windows, batch.key_lengths)
        return out.reshape(-1, hidden).index_select(0, batch.rows)

    def project(self, rows: torch.Tensor, mode: str) -> torch.Tensor:
        """``rows`` into the LLM's embedding space, by the ``speech`` or
        ``vision`` MLP."""
        return self.projection[mode](rows)

    @classmethod
    def load(
        cls, config: Phi4MMAudioConfig, repo_dir: str | Path, *,
        device: torch.device | str, dtype: torch.dtype | None = None,
    ) -> Phi4MMAudioTower:
        """Built on ``device`` in ``dtype`` from the checkpoint, every
        parameter and buffer loaded, in eval mode."""
        from mstar.model.loader.iterators import iter_safetensors_shards

        return cls.load_weights(
            config, iter_safetensors_shards(repo_dir, prefix=CHECKPOINT_PREFIX), device=device, dtype=dtype,
        )

    @classmethod
    def load_weights(
        cls, config: Phi4MMAudioConfig, weights: Iterable[tuple[str, torch.Tensor]], *,
        device: torch.device | str, dtype: torch.dtype | None = None,
    ) -> Phi4MMAudioTower:
        with torch.device("meta"):
            tower = cls(config)
        if dtype is not None:
            tower = tower.to(dtype)
        tower = tower.to_empty(device=device)
        state = tower.state_dict()
        loaded: set[str] = set()
        unused: list[str] = []
        for name, tensor in _checkpoint_names(weights):
            if name not in state:
                unused.append(name)
                continue
            target = state[name]
            if target.shape != tensor.shape:
                raise ValueError(f"Phi-4-multimodal audio: {name} is {tuple(tensor.shape)}, want {tuple(target.shape)}")
            with torch.no_grad():
                target.copy_(tensor)
            loaded.add(name)
        missing = sorted(set(state) - loaded)
        if missing or unused:
            raise KeyError(
                f"Phi-4-multimodal audio: {len(missing)} unloaded (e.g. {missing[:4]}), "
                f"{len(unused)} checkpoint tensors unused (e.g. {unused[:4]})"
            )
        return tower.eval()


_RENAMES = (
    (r"^encoder\.encoder_embedding\.", "encoder."),
    (r"^encoder\.embed\.conv\.", "encoder.subsampling.layers."),
    (r"^encoder\.embed\.out\.", "encoder.subsampling.linear."),
    (r"^encoder\.relative_attention_bias_layer\.bias_values\.", "encoder.rel_bias."),
    (r"^encoder\.encoders\.", "encoder.layers."),
    (r"\.feed_forward_(in|out)\.layer_norm\.", r".ff_\1.norm."),
    (r"\.feed_forward_(in|out)\.net\.0\.linear\.", r".ff_\1.up."),
    (r"\.feed_forward_(in|out)\.net\.2\.", r".ff_\1.down."),
    (r"\.self_attn\.linear_out\.", ".attn.out."),
    (r"\.layer_norm_att\.", ".norm_attn."),
    (r"\.conv\.layer_norm\.", ".conv.norm."),
    (r"\.conv\.glu\.ext_pw_conv_1d\.", ".conv.glu."),
    (r"\.conv\.glu\.b([12])$", r".conv.b\1"),
    (r"\.conv\.dw_sep_conv_1d\.dw_conv\.", ".conv.depthwise."),
    (r"\.conv\.dw_sep_conv_1d\.pw_conv\.", ".conv.pointwise."),
    (r"\.conv\.ext_pw_conv_1d\.", ".conv.out."),
    (r"^(encoder\.layers\.\d+)\.layer_norm\.", r"\1.norm_out."),
    (r"^audio_projection\.", "projection."),
)
_QKV = re.compile(r"^(encoder\.layers\.\d+)\.self_attn\.linear_([qkv])\.(weight|bias)$")


def _checkpoint_names(weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[tuple[str, torch.Tensor]]:
    """``(checkpoint name, tensor)`` under `CHECKPOINT_PREFIX` -> this
    module's ``(name, tensor)``: 1x1 conv kernels squeezed to linear weights,
    the GLU biases to vectors, and each layer's q, k, v fused."""
    pending: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for full, tensor in weights:
        name = full.removeprefix(CHECKPOINT_PREFIX)
        for pattern, repl in _RENAMES:
            name = re.sub(pattern, repl, name)
        qkv = _QKV.match(name)
        if qkv is not None:
            layer, part, kind = qkv.groups()
            parts = pending.setdefault((layer, kind), {})
            parts[part] = tensor
            if len(parts) == 3:
                del pending[(layer, kind)]
                yield f"{layer}.attn.qkv.{kind}", torch.cat([parts["q"], parts["k"], parts["v"]])
            continue
        if re.search(r"\.conv\.b[12]$", name):
            tensor = tensor.reshape(-1)
        elif re.search(r"\.conv\.(glu|pointwise|out)\.weight$", name):
            tensor = tensor.squeeze(-1)
        yield name, tensor
    if pending:
        raise KeyError(f"Phi-4-multimodal audio: incomplete q/k/v for {sorted(pending)}")

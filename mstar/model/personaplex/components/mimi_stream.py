"""Incremental Mimi decode: a chunk's new frames only, from per-stream state.

``MimiCodec.decode`` reruns the transformer over everything it is given, so a
stream that decodes in blocks must either pay for its history on every block or
cut it. Every module of Mimi's decoder is causal, so a stream's audio equals one
whole-track decode when each chunk carries, from the chunks before it:

* each causal conv's last ``padding_total`` input samples (zeros are a fresh
  stream's state: the conv's own left padding);
* each transposed conv's last input sample, whose second half overlaps the next
  chunk's first output;
* the transformer's last ``context - 1`` keys and values per layer, and its step
  count (RoPE positions and the ring's visibility continue from it).

Rows can differ in length: they are zero-padded on the right to the batch's width.
The decoder is causal, so padding never reaches a row's real audio (the audio past
it is junk the caller drops), and each state block is read from the row's own last
real samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache

import torch
import torch.nn.functional as F
from torch import nn

from ..config import AUDIO_CODEBOOKS_PER_STREAM, FRAME_SIZE
from .mimi import MIMI_CONTEXT, MIMI_STEPS_PER_FRAME, MimiCodec, _mimi_modules, build_hf_mimi

STEPS = "steps"


def _rope_theta(config) -> float:
    # transformers 5 keeps the RoPE base in rope_parameters rather than rope_theta
    theta = getattr(config, "rope_theta", None)
    return float(theta if theta is not None else config.rope_parameters["rope_theta"])


def _tail(xs: torch.Tensor, valid: torch.Tensor, keep: int, dim: int) -> torch.Tensor:
    """Per row, the ``keep`` entries of ``xs`` along ``dim`` that end ``valid[row]``
    entries past its first ``keep`` (``xs`` is ``[keep of state | new]``)."""
    shape = [1] * xs.dim()
    shape[0], shape[dim] = -1, keep
    index = valid.view(-1, *[1] * (xs.dim() - 1)) + torch.arange(keep, device=xs.device).view(shape)
    size = list(xs.shape)
    size[dim] = keep
    return xs.gather(dim, index.expand(size))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class IncrementalMimiDecoder(nn.Module):
    """``MimiCodec.decode`` over a chunk's frames, from and to each stream's state."""

    def __init__(self, codec: MimiCodec, context: int = MIMI_CONTEXT):
        super().__init__()
        self.quantizer = codec.model.quantizer
        self.seanet, self.transformer, self.upsample = _mimi_modules(codec.model, "decoder")
        config = self.transformer.config
        self.heads, self.head_dim = config.num_attention_heads, config.head_dim
        if config.num_key_value_heads != self.heads:
            raise ValueError("Mimi's transformer is expected to have as many KV heads as query heads")
        self.context = context
        self.cache = context - 1
        self.register_buffer(
            "inv_freq",
            1.0 / _rope_theta(config) ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim),
            persistent=False,
        )
        self.convs: list[tuple[str, nn.Module]] = []
        # each stride-1 conv's left padding, read once here: Mimi keeps it in a buffer, and reading
        # that per step would be a host sync on the GPU
        self.keep: dict[str, int] = {}
        self._collect("upsample", self.upsample)
        self._collect("seanet", self.seanet.layers)

    def _collect(self, prefix: str, module) -> None:
        """Name every module with left state, in execution order, checking the shapes the decode assumes."""
        if isinstance(module, nn.ModuleList):
            for i, layer in enumerate(module):
                self._collect(f"{prefix}.{i}", layer)
        elif hasattr(module, "block"):  # a residual block
            if not isinstance(module.shortcut, nn.Identity):
                raise TypeError("incremental decode assumes Mimi's residual blocks have no conv shortcut")
            self._collect(prefix, module.block)
        elif isinstance(getattr(module, "conv", None), nn.ConvTranspose1d):
            conv = module.conv
            if conv.kernel_size[0] != 2 * conv.stride[0]:
                raise TypeError("incremental decode assumes transposed convs of kernel 2 * stride")
            self.convs.append((prefix, module))
        elif isinstance(getattr(module, "conv", None), nn.Conv1d):
            conv = module.conv
            if conv.stride[0] != 1 or module.pad_mode != "constant":
                raise TypeError("incremental decode assumes stride-1 convs with zero padding")
            # transformers' MimiConv1d.padding_total for stride 1
            self.keep[prefix] = (conv.kernel_size[0] - 1) * conv.dilation[0]
            if self.keep[prefix]:
                self.convs.append((prefix, module))

    def state_blocks(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Each state block's per-stream shape and dtype."""
        dtype = next(self.seanet.parameters()).dtype
        blocks: dict[str, tuple[tuple[int, ...], torch.dtype]] = {STEPS: ((1,), torch.long)}
        for layer in range(len(self.transformer.layers)):
            for kv in "kv":
                blocks[f"{kv}{layer}"] = ((self.heads, self.cache, self.head_dim), dtype)
        for name, module in self.convs:
            transposed = isinstance(module.conv, nn.ConvTranspose1d)
            width = 1 if transposed else self.keep[name]
            blocks[name] = ((module.conv.in_channels, width), dtype)
        return blocks

    def fresh_state(self, batch: int, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """Zeros: a stream that has decoded nothing."""
        return {name: torch.zeros(batch, *shape, dtype=dtype, device=device)
                for name, (shape, dtype) in self.state_blocks().items()}

    def visibility(self, steps: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Query positions ``[B, width]`` and which of ``[cache | new]`` keys each sees, ``[B, 1, width, cache +
        width]``: NVIDIA's ring (``ring_attention_mask``) for rows ``steps [B, 1]`` steps in."""
        qpos = steps + torch.arange(width, device=steps.device)
        kpos = torch.cat([steps - self.cache + torch.arange(self.cache, device=steps.device), qpos], 1)[:, None]
        # a position is processed in a call of two steps; the ring is judged at that call's end
        oldest = (qpos // MIMI_STEPS_PER_FRAME + 1) * MIMI_STEPS_PER_FRAME - self.context + 1
        # the cached keys a stream has not produced yet sit at negative positions
        visible = (kpos >= 0) & (kpos <= qpos[:, :, None]) & (kpos >= oldest[:, :, None])
        return qpos, visible[:, None]

    def _attend(self, h: torch.Tensor, state, new, valid: torch.Tensor) -> torch.Tensor:
        """The transformer over ``h [B, T, D]`` from each row's keys and values."""
        batch, width, _ = h.shape
        qpos, mask = self.visibility(state[STEPS].view(batch, 1), width)
        angle = qpos[..., None].to(self.inv_freq.dtype) * self.inv_freq
        angle = torch.cat([angle, angle], -1)
        cos, sin = angle.cos()[:, None].to(h.dtype), angle.sin()[:, None].to(h.dtype)
        for i, layer in enumerate(self.transformer.layers):
            attn = layer.self_attn
            hn = layer.input_layernorm(h)
            q, k, v = (proj(hn).view(batch, width, self.heads, self.head_dim).transpose(1, 2)
                       for proj in (attn.q_proj, attn.k_proj, attn.v_proj))
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            keys = torch.cat([state[f"k{i}"], k], 2)
            values = torch.cat([state[f"v{i}"], v], 2)
            out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
            h = h + layer.self_attn_layer_scale(attn.o_proj(out.transpose(1, 2).reshape(batch, width, -1)))
            h = h + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(h)))
            new[f"k{i}"], new[f"v{i}"] = _tail(keys, valid, self.cache, 2), _tail(values, valid, self.cache, 2)
        return h

    def _conv(self, name: str, module: nn.Module, x: torch.Tensor, state, new, valid: torch.Tensor) -> torch.Tensor:
        keep = self.keep[name]
        if not keep:
            return module.conv(x)
        xs = torch.cat([state[name], x], -1)
        new[name] = _tail(xs, valid, keep, 2)
        return module.conv(xs)

    @staticmethod
    def _transposed(name: str, module: nn.Module, x: torch.Tensor, state, new, valid: torch.Tensor) -> torch.Tensor:
        # Kernel 2 * stride: input i writes outputs i*stride .. i*stride + 2*stride - 1, so
        # each output block is its input's first half plus the previous input's second.
        stride = module.conv.stride[0]
        xs = torch.cat([state[name], x], -1)
        new[name] = _tail(xs, valid, 1, 2)
        return module.conv(xs)[..., stride : stride * (1 + x.shape[-1])]

    def _seanet(self, prefix: str, layers, x: torch.Tensor, state, new, valid: torch.Tensor) -> tuple:
        for i, layer in enumerate(layers):
            name = f"{prefix}.{i}"
            if hasattr(layer, "block"):
                residual, valid = self._seanet(name, layer.block, x, state, new, valid)
                x = x + residual
            elif isinstance(getattr(layer, "conv", None), nn.ConvTranspose1d):
                x = self._transposed(name, layer, x, state, new, valid)
                valid = valid * layer.conv.stride[0]
            elif isinstance(getattr(layer, "conv", None), nn.Conv1d):
                x = self._conv(name, layer, x, state, new, valid)
            else:
                x = layer(x)
        return x, valid

    def forward(
        self, codes: torch.Tensor, state: Mapping[str, torch.Tensor], lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``codes [B, 8, width]``, each row's state and its real frame count ``lengths [B]`` ->
        PCM ``[B, 1, width * 1920]`` and the state after each row's real frames."""
        if codes.dim() != 3 or codes.shape[1] != AUDIO_CODEBOOKS_PER_STREAM:
            raise ValueError(f"expected [B, {AUDIO_CODEBOOKS_PER_STREAM}, N] codes, got {tuple(codes.shape)}")
        new: dict[str, torch.Tensor] = {}
        x = self._transposed("upsample", self.upsample, self.quantizer.decode(codes), state, new, lengths)
        valid = lengths * MIMI_STEPS_PER_FRAME
        h = self._attend(x.transpose(1, 2), state, new, valid)
        new[STEPS] = state[STEPS] + valid.view(-1, 1)
        pcm, _ = self._seanet("seanet", self.seanet.layers, h.transpose(1, 2), state, new, valid)
        if pcm.shape[-1] != codes.shape[-1] * FRAME_SIZE:
            raise RuntimeError(f"decoded {pcm.shape[-1]} samples for {codes.shape[-1]} frames")
        return pcm, new


@cache
def mimi_state_blocks() -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """The served Mimi's (``MimiConfig`` defaults, fp32) per-stream state, without its weights: the decoder
    built on the meta device, so the pool can be declared before the checkpoint loads."""
    with torch.device("meta"):
        model = build_hf_mimi().float()
    return IncrementalMimiDecoder(MimiCodec(model)).state_blocks()

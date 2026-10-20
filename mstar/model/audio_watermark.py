"""Opt-in AudioSeal watermark for generated speech.

A deployment turns it on in its YAML ``model_kwargs``::

    model_kwargs:
      audio_watermark:
        enabled: true        # default false
        message: 0xBEEF      # 16-bit payload: int 0..65535 or a 16-char bit string
        alpha: 1.0           # watermark strength (AudioSeal's tune-down ratio)

The watermark is Meta's AudioSeal (MIT code and weights, ``pip install
'mstar-ai[audioseal]'``): a 14.7M-parameter causal conv/LSTM generator whose
output, the same length as its input, is added to the waveform. The matching
detector scores a clip by the fraction of its samples it finds marked
(``python -m mstar.model.audio_watermark detect out.wav``). Embedding is
streaming-exact chunk by chunk; detection on one chunk is not guaranteed. On
generated speech even 80 ms chunks scored as marked, but on a synthetic
harmonic tone 200 ms slices scored about 0.07 and 1 s slices about 0.8, so
detect on whole utterances, or at least a few seconds of audio. The mark
is found only at the rate it was added at (24 kHz), so the detector
resamples a file at another rate to 24 kHz first.

Streaming. Audio leaves the engine in chunks, and a chunk-by-chunk watermark
must equal the watermark of the whole utterance, or the boundaries click.
AudioSeal 0.2.0's ``model.streaming(batch_size)`` does not deliver that: its
state propagation walks ``named_children()`` and stops at the
``nn.Sequential`` holding the layers, so no conv caches anything and every
chunk is watermarked as if it began the stream (upstream issue #105). This
module keeps each request's stream state itself, as one flat row of tensors,
and runs the generator's layers as a function of it: the same ops AudioSeal's
streaming layers run, with the state passed in and out. That lets requests
at the same point share one batched call, and lets the engine capture that
call per (batch, chunk length) bucket.

The generator works on whole 320-sample frames (its total stride). A chunk
that is not a whole number of frames leaves its tail pending until the next
chunk, or until the final chunk zero-pads it, so streamed output equals the
offline watermark of the whole utterance to float rounding (on a GPU, to a
few int16 steps: cuDNN runs the convolutions in TF32). AudioSeal's own
whole-utterance call pads inside each layer instead, so for an utterance
that ends mid-frame the two differ in that last partial frame only (a few
tens of int16 steps). Codecs whose chunks are whole frames (Qwen3-TTS: 1920
samples per codec frame) never hold anything back and never pad.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

logger = logging.getLogger(__name__)

MESSAGE_BITS = 16
DEFAULT_GENERATOR = "audioseal_wm_streaming"
DEFAULT_DETECTOR = "audioseal_detector_streaming"
# the rate the mark is added at: Qwen3-TTS's output, the only codec that adds it
MARK_SAMPLE_RATE = 24_000
_INSTALL_HINT = "pip install 'mstar-ai[audioseal]' (or pip install 'audioseal>=0.2,<0.3')"


@dataclass(frozen=True)
class AudioWatermarkConfig:
    """``model_kwargs.audio_watermark``: per deployment, off by default."""

    enabled: bool = False
    message: int = 0
    alpha: float = 1.0
    generator: str = DEFAULT_GENERATOR

    @classmethod
    def parse(cls, value: Any) -> AudioWatermarkConfig:
        if value is None:
            return cls()
        if isinstance(value, AudioWatermarkConfig):
            return value
        if not isinstance(value, dict):
            raise ValueError(f"audio_watermark must be a mapping, got {type(value).__name__}")
        unknown = set(value) - {"enabled", "message", "alpha", "generator"}
        if unknown:
            raise ValueError(
                f"audio_watermark has unknown keys {sorted(unknown)}; "
                "expected enabled, message, alpha, generator"
            )
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"audio_watermark.enabled must be true or false, got {enabled!r}")
        alpha = value.get("alpha", 1.0)
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha <= 4:
            raise ValueError(f"audio_watermark.alpha must be a number in (0, 4], got {alpha!r}")
        generator = value.get("generator", DEFAULT_GENERATOR)
        if not isinstance(generator, str) or not generator:
            raise ValueError("audio_watermark.generator must be an AudioSeal card name or checkpoint path")
        return cls(
            enabled=enabled,
            message=_parse_message(value.get("message", 0)),
            alpha=float(alpha),
            generator=generator,
        )

    def message_bits(self) -> list[int]:
        """The payload, most significant bit first."""
        return [(self.message >> (MESSAGE_BITS - 1 - i)) & 1 for i in range(MESSAGE_BITS)]


def _parse_message(value: Any) -> int:
    if isinstance(value, str):
        if len(value) != MESSAGE_BITS or set(value) - {"0", "1"}:
            raise ValueError(
                f"audio_watermark.message as a string must be {MESSAGE_BITS} characters of 0/1, got {value!r}"
            )
        return int(value, 2)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**MESSAGE_BITS:
        raise ValueError(
            f"audio_watermark.message must be an int in [0, {2**MESSAGE_BITS - 1}] "
            f"or a {MESSAGE_BITS}-bit string, got {value!r}"
        )
    return value


def _cli_message(value: str) -> int:
    """A message on the command line: 16 characters of 0/1, or an int (``0xBEEF``, ``48879``)."""
    if len(value) == MESSAGE_BITS and not set(value) - {"0", "1"}:
        return _parse_message(value)
    return _parse_message(int(value, 0))


def import_audioseal():
    try:
        import audioseal
        from audioseal.libs.moshi.utils.compile import no_compile
    except ImportError as exc:
        raise ImportError(
            f"audio_watermark.enabled is true but the 'audioseal' package is not installed: {_INSTALL_HINT}"
        ) from exc
    return audioseal, no_compile


@contextmanager
def _watermark_precision():
    """fp32 matmuls (the engine turns TF32 matmuls on process-wide) and TF32
    cuDNN. With cuDNN TF32 off, cuDNN on H100 picks a conv about 18x slower
    at batch 1 and 2; with it on, a chunk and the whole utterance, which
    convolve over different lengths, round apart by a few int16 steps.

    The matmul precision goes through ``set_float32_matmul_precision``, as
    the engine sets it: flipping ``cuda.matmul.allow_tf32`` alone leaves the
    CPU matmul setting behind, and ``get_float32_matmul_precision()`` raises
    on that mix from any thread while the window is open."""
    matmul, cudnn = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(matmul)
        torch.backends.cudnn.allow_tf32 = cudnn


@dataclass
class WatermarkStream:
    """One request's watermark state: the generator's layer caches as one
    flat row (``AudioSealWatermarker.cache_size`` floats) and the samples of
    a partial frame not yet watermarked."""

    caches: torch.Tensor | None = None
    pending: torch.Tensor | None = None
    finished: bool = False


@dataclass(frozen=True)
class _Conv:
    """A causal streaming conv: the input is the cached tail plus the new
    samples, and the new cache is the last ``keep`` input samples."""

    conv: torch.nn.Conv1d
    slot: int
    keep: int


@dataclass(frozen=True)
class _ConvTranspose:
    """A streaming transposed conv: the cached partial output (bias
    included) is added to the next output's head."""

    conv: torch.nn.ConvTranspose1d
    slot: int
    keep: int


@dataclass(frozen=True)
class _LSTM:
    lstm: torch.nn.LSTM
    h_slot: int
    c_slot: int
    skip: bool


@dataclass(frozen=True)
class _Residual:
    block: tuple
    shortcut: tuple | None   # None: identity


@dataclass(frozen=True)
class _Layout:
    """Where each layer's cache sits in a stream's flat row."""

    offsets: tuple[int, ...]
    shapes: tuple[tuple[int, ...], ...]   # per stream, without the batch dim
    lstm_slots: frozenset[int]            # [layers, hidden] here, [layers, B, hidden] in the LSTM

    @property
    def size(self) -> int:
        return self.offsets[-1]


def _walk(ops: tuple):
    """Every op, residual blocks flattened."""
    for op in ops:
        if isinstance(op, _Residual):
            yield from _walk(op.shortcut or ())
            yield from _walk(op.block)
        else:
            yield op


class AudioSealWatermarker:
    """AudioSeal generator plus per-request streaming, on one device.

    The generator runs as a plain function of the audio and every stream's
    caches (``_mark``): the same convolutions, LSTM and adds AudioSeal's
    streaming layers would run, with each layer's state passed in and out
    instead of kept on the module. Its shapes depend on the batch and chunk
    length only, so the engine can capture it per (batch, length) bucket
    (``piecewise_configs``) and ``apply_pcm16`` replays those captures."""

    def __init__(self, generator: torch.nn.Module, config: AudioWatermarkConfig, no_compile):
        self.generator = generator
        self.config = config
        self._no_compile = no_compile
        self.device = next(generator.parameters()).device
        self.hop = int(generator.encoder.hop_length)
        shapes: list[tuple[int, ...]] = []
        lstm_slots: set[int] = set()
        self._encoder = self._program(generator.encoder, shapes, lstm_slots)
        self._decoder = self._program(generator.decoder, shapes, lstm_slots)
        offsets = [0]
        for shape in shapes:
            offsets.append(offsets[-1] + math.prod(shape))
        self._layout = _Layout(tuple(offsets), tuple(shapes), frozenset(lstm_slots))
        self._message_embedding = self._embed_message(generator, config)
        self._fresh = self._fresh_caches()
        self._capture_lengths: list[int] = []
        self._eager_logged: set[tuple[str, int]] = set()

    @property
    def cache_size(self) -> int:
        """Floats of state per stream."""
        return self._layout.size

    @torch.no_grad()
    def _embed_message(self, generator: torch.nn.Module, config: AudioWatermarkConfig) -> torch.Tensor:
        """What AudioSeal's ``MsgProcessor`` adds to every frame, ``[1, hidden, 1]``.
        The message is fixed per deployment, and the processor would rebuild
        it from a host ``arange`` on every call, a copy that syncs the GPU."""
        embedding = generator.msg_processor.msg_processor
        bits = torch.tensor(config.message_bits(), dtype=torch.long, device=self.device)
        indices = 2 * torch.arange(MESSAGE_BITS, device=self.device) + bits
        return embedding(indices.unsqueeze(0)).sum(dim=-2).unsqueeze(-1)

    @classmethod
    def build(cls, config: AudioWatermarkConfig, device: str | torch.device) -> AudioSealWatermarker:
        audioseal, no_compile = import_audioseal()
        # the loader's model-card parsing must not move the serving RNG
        with torch.random.fork_rng(devices=[]):
            generator = audioseal.AudioSeal.load_generator(config.generator, nbits=MESSAGE_BITS)
        generator = generator.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        logger.info(
            "audio watermark: AudioSeal %s on %s, message=0x%04x alpha=%.2f",
            config.generator, device, config.message, config.alpha,
        )
        return cls(generator, config, no_compile)

    # -- the generator as a function of its streaming state ------------------

    @classmethod
    def _program(cls, container: torch.nn.Module, shapes: list, lstm_slots: set) -> tuple:
        """The ops of a SEANet encoder or decoder, each stateful one given the
        next cache slots (``shapes`` grows by the per-stream cache shape)."""
        from audioseal.libs.moshi.modules.conv import StreamingConv1d, StreamingConvTranspose1d
        from audioseal.libs.moshi.modules.seanet import SEANetDecoder, SEANetEncoder

        for module in container.modules():
            if isinstance(module, (StreamingConv1d, StreamingConvTranspose1d)) and not module.causal:
                raise ValueError(
                    "audio_watermark needs a causal (streaming) AudioSeal generator such as "
                    f"{DEFAULT_GENERATOR!r}; this checkpoint's convolutions are not causal"
                )
        if type(container) not in (SEANetEncoder, SEANetDecoder):
            raise ValueError(f"audio_watermark: unsupported AudioSeal {type(container).__name__}")
        return cls._sequence(container.model, shapes, lstm_slots)

    @classmethod
    def _sequence(cls, modules, shapes: list, lstm_slots: set) -> tuple:
        from audioseal.libs.moshi.modules.conv import StreamingConv1d, StreamingConvTranspose1d
        from audioseal.libs.moshi.modules.lstm import RawStreamingLSTM
        from audioseal.libs.moshi.modules.seanet import SEANetResnetBlock
        from audioseal.libs.moshi.modules.streaming import StreamingModule

        ops = []
        for module in modules:
            if isinstance(module, StreamingConv1d):
                conv = module.conv.conv.inner_conv
                keep = (conv.kernel_size[0] - 1) * conv.dilation[0] + 1 - conv.stride[0]
                shapes.append((conv.in_channels, keep))
                ops.append(_Conv(conv, len(shapes) - 1, keep))
            elif isinstance(module, StreamingConvTranspose1d):
                conv = module.convtr.convtr.inner_conv
                shapes.append((conv.out_channels, conv.kernel_size[0] - conv.stride[0]))
                ops.append(_ConvTranspose(conv, len(shapes) - 1, shapes[-1][1]))
            elif isinstance(module, RawStreamingLSTM):
                lstm = module.lstm
                for _ in range(2):
                    shapes.append((lstm.num_layers, lstm.hidden_size))
                    lstm_slots.add(len(shapes) - 1)
                ops.append(_LSTM(lstm, len(shapes) - 2, len(shapes) - 1, module.skip))
            elif isinstance(module, SEANetResnetBlock):
                identity = isinstance(module.shortcut, torch.nn.Identity)
                shortcut = None if identity else cls._sequence([module.shortcut], shapes, lstm_slots)
                ops.append(_Residual(cls._sequence(module.block, shapes, lstm_slots), shortcut))
            elif isinstance(module, StreamingModule) or any(True for _ in module.children()):
                raise ValueError(f"audio_watermark: unsupported AudioSeal layer {type(module).__name__}")
            else:
                ops.append(module)   # a stateless activation
        return tuple(ops)

    def _fresh_caches(self) -> torch.Tensor:
        """A new stream's flat cache row: what the layers would build from
        their causal zero padding on the first chunk, made explicit so a new
        stream batches with ones already under way. Never written in place,
        so every new stream shares it."""
        pieces = [torch.zeros(math.prod(shape), device=self.device) for shape in self._layout.shapes]
        for op in _walk((*self._encoder, *self._decoder)):
            if isinstance(op, _ConvTranspose) and op.conv.bias is not None:
                # the layer adds ``partial - bias`` to the next output; bias makes that 0
                pieces[op.slot] = op.conv.bias.detach().view(-1, 1).expand(-1, op.keep).reshape(-1)
        return torch.cat(pieces)

    def _unpack(self, flat: torch.Tensor) -> list[torch.Tensor]:
        """``[B, cache_size]`` -> each layer's cache, batch first (the LSTM's
        ``[layers, B, hidden]``, contiguous as cuDNN wants it)."""
        batch, layout = flat.shape[0], self._layout
        caches = []
        for slot, shape in enumerate(layout.shapes):
            view = flat[:, layout.offsets[slot]:layout.offsets[slot + 1]].view(batch, *shape)
            caches.append(view.transpose(0, 1).contiguous() if slot in layout.lstm_slots else view)
        return caches

    def _pack(self, caches: list[torch.Tensor | None], batch: int) -> torch.Tensor:
        pieces = []
        offsets = self._layout.offsets
        for slot, cache in enumerate(caches):
            if offsets[slot + 1] == offsets[slot]:   # a layer that keeps nothing
                continue
            pieces.append((cache.transpose(0, 1) if slot in self._layout.lstm_slots else cache).reshape(batch, -1))
        return torch.cat(pieces, dim=1)

    def _run(self, ops: tuple, x: torch.Tensor, caches: list, new: list) -> torch.Tensor:
        """AudioSeal's streaming layers, op for op (``moshi.modules.streaming``),
        with the state in ``caches`` and the next state written to ``new``.
        Every input holds whole frames, so each conv consumes all of it but
        its last ``keep`` samples. The kept tails are copied out at once: a
        view would hold the layer's whole activation until the end of the
        call, which doubles a 32-row, 2 s capture's memory (5 GB, not 2.5)."""
        for op in ops:
            if isinstance(op, _Conv):
                if (x.shape[-1] % op.conv.stride[0]) != 0:
                    raise RuntimeError("AudioSeal conv input is not a whole number of strides")
                if op.keep:
                    x = torch.cat([caches[op.slot], x], dim=-1)
                    new[op.slot] = x[..., x.shape[-1] - op.keep:].clone()
                x = op.conv(x)
            elif isinstance(op, _ConvTranspose):
                out = op.conv(x)
                total = out.shape[-1]
                if op.conv.bias is not None:
                    out[..., :op.keep] += caches[op.slot] - op.conv.bias[:, None]
                else:
                    out[..., :op.keep] += caches[op.slot]
                new[op.slot] = out[..., total - op.keep:].clone()
                x = out[..., :total - op.keep]
            elif isinstance(op, _LSTM):
                seq = x.permute(2, 0, 1)
                y, (h, c) = op.lstm(seq, (caches[op.h_slot], caches[op.c_slot]))
                new[op.h_slot], new[op.c_slot] = h, c
                if op.skip:
                    y = y + seq
                x = y.permute(1, 2, 0)
            elif isinstance(op, _Residual):
                shortcut = x if op.shortcut is None else self._run(op.shortcut, x, caches, new)
                x = shortcut + self._run(op.block, x, caches, new)
            else:
                x = op(x)
        return x

    # -- watermarking -------------------------------------------------------

    @torch.no_grad()
    def _mark(self, audio: torch.Tensor, caches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``audio`` [B, T] float, T a multiple of the hop, one stream per row,
        and the streams' caches [B, cache_size]. Returns the watermark [B, T]
        and the caches after it. Shapes depend on B and T only."""
        batch = audio.shape[0]
        old = self._unpack(caches)
        new: list[torch.Tensor | None] = [None] * len(old)
        with _watermark_precision():
            hidden = self._run(self._encoder, audio.unsqueeze(1), old, new) + self._message_embedding
            wm = self._run(self._decoder, hidden, old, new)
        return wm.squeeze(1), self._pack(new, batch)

    @torch.no_grad()
    def _mark_pcm16(self, pcm: torch.Tensor, caches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``_mark`` for PCM16 rows [B, T], quantized back the way the codecs do."""
        audio = pcm.to(torch.float32) / 32767
        wm, new = self._mark(audio, caches)
        marked = audio + self.config.alpha * wm
        return (marked.clamp(-1, 1) * 32767).to(torch.int16), new

    def _stacked_caches(self, streams: Sequence[WatermarkStream]) -> torch.Tensor:
        return torch.stack([self._fresh if s.caches is None else s.caches for s in streams])

    @torch.no_grad()
    def apply(
        self,
        chunks: list[torch.Tensor],
        streams: list[WatermarkStream],
        finals: list[bool] | None = None,
    ) -> list[torch.Tensor]:
        """Watermark the next chunk of each stream (float [-1, 1], 1-D).

        Returns one tensor per chunk: the watermarked audio of every whole
        frame available so far, which for frame-aligned chunks is the whole
        chunk. A final chunk flushes the stream's pending partial frame.
        Streams at the same frame count run as one batch.
        """
        finals = finals or [False] * len(chunks)
        results: list[torch.Tensor | None] = [None] * len(chunks)
        groups: dict[int, list[tuple[int, torch.Tensor, int]]] = defaultdict(list)
        for i, (chunk, stream, final) in enumerate(zip(chunks, streams, finals, strict=True)):
            if stream.finished:
                raise RuntimeError("watermark stream got a chunk after its final one")
            x = chunk.reshape(-1).to(self.device, torch.float32)
            if stream.pending is not None and stream.pending.numel():
                x = torch.cat([stream.pending, x])
            keep = x.shape[0]
            whole = keep - keep % self.hop
            if final and whole < keep:
                x = torch.nn.functional.pad(x, (0, self.hop - keep % self.hop))
                whole = x.shape[0]
            stream.pending = x[whole:] if whole < x.shape[0] else None
            stream.finished = final
            if whole == 0:
                results[i] = x[:0]
                continue
            groups[whole].append((i, x[:whole], keep if final else whole))
        for members in groups.values():
            audio = torch.stack([x for _, x, _ in members])
            wm, caches = self._mark(audio, self._stacked_caches([streams[i] for i, _, _ in members]))
            marked = audio + self.config.alpha * wm
            for row, (i, _, length) in enumerate(members):
                streams[i].caches = None if streams[i].finished else caches[row]
                results[i] = marked[row, :length]
        return results  # type: ignore[return-value]

    @torch.no_grad()
    def apply_pcm16(
        self,
        chunks: list[torch.Tensor],
        streams: list[WatermarkStream],
        finals: list[bool] | None = None,
        runners: Mapping[str, Any] | None = None,
    ) -> list[torch.Tensor]:
        """``apply`` for PCM16 chunks, quantized back the way the codecs do.

        Chunks of whole frames with nothing pending (every Qwen3-TTS chunk)
        run as one call per length, the conversion to float and back
        included; with ``runners`` (the engine's captures of
        ``piecewise_configs``) a call whose length was captured replays it,
        and a final chunk is zero-padded to the next captured length (the
        generator is causal and the stream's caches are dropped, so its
        samples change by float rounding only). Anything else goes through
        ``apply``."""
        finals = finals or [False] * len(chunks)
        results: list[torch.Tensor | None] = [None] * len(chunks)
        groups: dict[int, list[int]] = defaultdict(list)
        other = []
        for i, (chunk, stream, final) in enumerate(zip(chunks, streams, finals, strict=True)):
            if stream.finished:
                raise RuntimeError("watermark stream got a chunk after its final one")
            samples = chunk.numel()
            if samples == 0 or samples % self.hop or (stream.pending is not None and stream.pending.numel()):
                other.append(i)
                continue
            groups[self._captured_length(samples, runners) if final else samples].append(i)
        for length, members in groups.items():
            rows = [chunks[i].reshape(-1).to(self.device, torch.int16) for i in members]
            pcm = torch.stack([
                row if row.shape[0] == length else torch.nn.functional.pad(row, (0, length - row.shape[0]))
                for row in rows
            ])
            caches = self._stacked_caches([streams[i] for i in members])
            runner = runners.get(self.capture_label(length)) if runners else None
            if runner is not None and runner.can_run(len(members)):
                out = runner.run({"pcm": pcm, "caches": caches}, real_bs=len(members))
                marked, caches = out["pcm"], out["caches"]
            else:
                if runners:
                    self._log_eager("length" if runner is None else "batch", length, len(members))
                marked, caches = self._mark_pcm16(pcm, caches)
            for row, i in enumerate(members):
                stream = streams[i]
                stream.finished = finals[i]
                stream.caches = None if finals[i] else caches[row]
                results[i] = marked[row, :rows[row].shape[0]]
        if other:
            if runners:
                self._log_eager("partial", 0, len(other))
            floats = [chunks[i].to(self.device, torch.float32) / 32767 for i in other]
            marked = self.apply(floats, [streams[i] for i in other], [finals[i] for i in other])
            for i, m in zip(other, marked, strict=True):
                results[i] = (m.clamp(-1, 1) * 32767).to(torch.int16)
        return results  # type: ignore[return-value]

    # -- capture --------------------------------------------------------------

    def _log_eager(self, reason: str, length: int, rows: int) -> None:
        """Say once per reason and length that captures exist but a call runs
        eagerly, so a config whose chunks miss every capture is visible."""
        if (reason, length) in self._eager_logged:
            return
        self._eager_logged.add((reason, length))
        what = {
            "length": f"a chunk of {length} samples, a length with no capture",
            "batch": f"{rows} chunks of {length} samples, more than its largest captured batch",
            "partial": "a chunk that is not whole frames or has a partial frame pending",
        }[reason]
        logger.warning("audio watermark: %s runs eagerly (logged once)", what)

    @staticmethod
    def capture_label(samples: int) -> str:
        return f"watermark_{samples}"

    def _captured_length(self, samples: int, runners: Mapping[str, Any] | None) -> int:
        """The shortest captured chunk length that holds ``samples``, else ``samples``."""
        if runners:
            for length in self._capture_lengths:
                if length >= samples and self.capture_label(length) in runners:
                    return length
        return samples

    def piecewise_configs(self, lengths: Sequence[int], batch_sizes: Sequence[int]) -> dict[str, Any]:
        """One ``PiecewiseBatchedConfig`` per chunk length (samples, whole
        frames), for ``NodeSubmodule.get_piecewise_accelerator_graph_configs``: the
        engine captures ``_mark_pcm16`` per (batch, length) bucket, with the
        PCM rows and the streams' caches as static inputs."""
        from mstar.engine.accelerator_graph_config import PiecewiseBatchedConfig

        lengths = sorted(set(lengths))
        if not lengths or any(length <= 0 or length % self.hop for length in lengths):
            raise ValueError(f"audio_watermark capture lengths must be whole {self.hop}-sample frames, got {lengths}")
        self._capture_lengths = lengths

        def static_inputs(length, shape):
            return {
                "pcm": torch.zeros(shape.bs, length, dtype=torch.int16, device=self.device),
                "caches": self._fresh.expand(shape.bs, -1).clone(),
            }

        def capture(call):
            pcm, caches = self._mark_pcm16(call.static_inputs["pcm"], call.static_inputs["caches"])
            return {"pcm": pcm, "caches": caches}

        return {
            self.capture_label(length): PiecewiseBatchedConfig(
                capture_fn=capture,
                make_static_inputs=partial(static_inputs, length),
                seq_len=length // self.hop,
                capture_batch_sizes=sorted(set(batch_sizes)),
            )
            for length in lengths
        }

    @torch.no_grad()
    def offline(self, audio: torch.Tensor) -> torch.Tensor:
        """The whole-utterance watermark of 1-D float audio: the reference a
        stream must equal. Zero-pads to whole frames like a final chunk."""
        stream = WatermarkStream()
        return self.apply([audio], [stream], [True])[0]


class AudioSealDetector:
    """AudioSeal's detector: P(watermarked) and the decoded 16-bit message."""

    def __init__(self, detector: torch.nn.Module, no_compile):
        self.detector = detector
        self._no_compile = no_compile
        self.device = next(detector.parameters()).device

    @classmethod
    def build(cls, device: str | torch.device = "cpu", card: str = DEFAULT_DETECTOR) -> AudioSealDetector:
        audioseal, no_compile = import_audioseal()
        with torch.random.fork_rng(devices=[]):
            detector = audioseal.AudioSeal.load_detector(card, nbits=MESSAGE_BITS)
        return cls(detector.to(device=device, dtype=torch.float32).eval().requires_grad_(False), no_compile)

    @torch.no_grad()
    def detect(self, audio: torch.Tensor, sample_rate: int = MARK_SAMPLE_RATE) -> tuple[float, int]:
        """1-D float audio at ``sample_rate`` -> (fraction of samples scored
        watermarked, message). The mark is found only at the rate it was
        added at, so other rates are resampled to ``MARK_SAMPLE_RATE`` first."""
        x = audio.reshape(1, 1, -1).to(self.device, torch.float32)
        if sample_rate != MARK_SAMPLE_RATE:
            import torchaudio

            x = torchaudio.functional.resample(x, sample_rate, MARK_SAMPLE_RATE)
        with self._no_compile():
            prob, bits = self.detector.detect_watermark(x)
        message = 0
        for bit in bits[0].tolist():
            message = (message << 1) | int(bit)
        return float(prob[0]), message


def _main(argv: list[str] | None = None) -> None:
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(
        prog="python -m mstar.model.audio_watermark", description="AudioSeal watermark detection",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    detect = sub.add_parser("detect", help="score audio files for the AudioSeal watermark")
    detect.add_argument("files", nargs="+", help="whole utterances; a clip under a few seconds can miss the mark")
    detect.add_argument("--device", default="cpu")
    detect.add_argument("--threshold", type=float, default=0.5, help="score above which a file counts as marked")
    detect.add_argument(
        "--message", type=_cli_message, default=None,
        help="the deployment's message (0xBEEF, 48879 or 16 bits): a file counts as marked only if it decodes to it",
    )
    args = parser.parse_args(argv)
    detector = AudioSealDetector.build(args.device)
    for path in args.files:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        score, message = detector.detect(torch.from_numpy(audio.mean(axis=1)), sample_rate)
        # MP3 coding noise alone can push an unmarked file's score past 0.5;
        # it does not decode to the deployment's message
        marked = score > args.threshold and (args.message is None or message == args.message)
        verdict = "watermarked" if marked else "not watermarked"
        print(f"{path}: {verdict} (score {score:.3f}, message 0x{message:04x}, {sample_rate} Hz)")


if __name__ == "__main__":
    _main()

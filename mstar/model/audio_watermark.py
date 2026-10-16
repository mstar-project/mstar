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
detect on whole utterances, or at least a few seconds of audio.

Streaming. Audio leaves the engine in chunks, and a chunk-by-chunk watermark
must equal the watermark of the whole utterance, or the boundaries click.
AudioSeal 0.2.0's ``model.streaming(batch_size)`` does not deliver that: its
state propagation walks ``named_children()`` and stops at the
``nn.Sequential`` holding the layers, so no conv caches anything and every
chunk is watermarked as if it began the stream (upstream issue #105). This
module keeps each request's stream state itself, as plain tensors, and
installs it in the generator's streaming layers around each call. That also
lets requests at the same point share one batched call.

The generator works on whole 320-sample frames (its total stride). A chunk
that is not a whole number of frames leaves its tail pending until the next
chunk, or until the final chunk zero-pads it, so streamed output equals the
offline watermark of the whole utterance to float rounding. AudioSeal's own
whole-utterance call pads inside each layer instead, so for an utterance
that ends mid-frame the two differ in that last partial frame only (a few
tens of int16 steps). Codecs whose chunks are whole frames (Qwen3-TTS: 1920
samples per codec frame) never hold anything back and never pad.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)

MESSAGE_BITS = 16
DEFAULT_GENERATOR = "audioseal_wm_streaming"
DEFAULT_DETECTOR = "audioseal_detector_streaming"
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
def _no_tf32():
    """Full fp32 convolutions and matmuls. A chunk and the whole utterance
    convolve over different lengths, and TF32 rounds them apart by a few
    int16 steps (the engine turns TF32 matmuls on process-wide)."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


@dataclass
class WatermarkStream:
    """One request's watermark state: the generator's layer caches (batch 1)
    and the samples of a partial frame not yet watermarked."""

    caches: list[torch.Tensor] | None = None
    pending: torch.Tensor | None = None
    finished: bool = False


@dataclass
class _Slots:
    """The generator's stateful layers, in a fixed order."""

    convs: list = field(default_factory=list)        # (StreamingConv1d, RawStreamingConv1d)
    convtrs: list = field(default_factory=list)      # RawStreamingConvTranspose1d
    lstms: list = field(default_factory=list)        # RawStreamingLSTM
    adds: list = field(default_factory=list)         # StreamingAdd
    others: list = field(default_factory=list)       # stateless streaming containers


class AudioSealWatermarker:
    """AudioSeal generator plus per-request streaming, on one device."""

    def __init__(self, generator: torch.nn.Module, config: AudioWatermarkConfig, no_compile):
        self.generator = generator
        self.config = config
        self._no_compile = no_compile
        self.device = next(generator.parameters()).device
        self.hop = int(generator.encoder.hop_length)
        self._slots = self._collect_slots(generator)
        self._message = torch.tensor([config.message_bits()], dtype=torch.long, device=self.device)
        self._fresh = self._fresh_caches()

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

    # -- streaming state ----------------------------------------------------

    @staticmethod
    def _collect_slots(generator: torch.nn.Module) -> _Slots:
        from audioseal.libs.moshi.modules.conv import StreamingConv1d, StreamingConvTranspose1d
        from audioseal.libs.moshi.modules.lstm import RawStreamingLSTM
        from audioseal.libs.moshi.modules.streaming import (
            RawStreamingConv1d,
            RawStreamingConvTranspose1d,
            StreamingAdd,
            StreamingModule,
        )

        slots = _Slots()
        raw_owned = set()
        for module in generator.modules():
            if isinstance(module, StreamingConv1d):
                if not module.causal:
                    raise ValueError(
                        "audio_watermark needs a causal (streaming) AudioSeal generator such as "
                        f"{DEFAULT_GENERATOR!r}; this checkpoint's convolutions are not causal"
                    )
                raw = module.conv.conv
                slots.convs.append((module, raw))
                raw_owned.add(id(raw))
            elif isinstance(module, StreamingConvTranspose1d) and not module.causal:
                raise ValueError(f"audio_watermark needs a causal AudioSeal generator such as {DEFAULT_GENERATOR!r}")
        for module in generator.modules():
            if isinstance(module, RawStreamingConvTranspose1d):
                slots.convtrs.append(module)
            elif isinstance(module, RawStreamingLSTM):
                slots.lstms.append(module)
            elif isinstance(module, StreamingAdd):
                slots.adds.append(module)
            elif isinstance(module, RawStreamingConv1d):
                if id(module) not in raw_owned:
                    raise ValueError("unexpected bare streaming conv in the AudioSeal generator")
            elif isinstance(module, StreamingModule) and not isinstance(module, StreamingConv1d):
                slots.others.append(module)
        return slots

    def _fresh_caches(self) -> list[torch.Tensor]:
        """A new stream's caches, batch 1: what the layers would build from
        their causal zero padding on the first chunk, made explicit so a new
        stream batches with ones already under way. Never written in place,
        so every new stream shares them."""
        caches = []
        dev = self.device
        for outer, raw in self._slots.convs:
            conv = raw.inner_conv
            caches.append(torch.zeros(1, conv.in_channels, outer._padding_total, device=dev))
        for raw in self._slots.convtrs:
            conv = raw.inner_conv
            width = conv.kernel_size[0] - conv.stride[0]
            # the layer adds ``partial - bias`` to the next output; bias makes that 0
            bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=dev)
            caches.append(bias.detach().view(1, -1, 1).expand(1, -1, width).contiguous())
        for raw in self._slots.lstms:
            lstm = raw.lstm
            zeros = torch.zeros(lstm.num_layers, 1, lstm.hidden_size, device=dev)
            caches.extend([zeros, zeros])
        return caches

    def _install(self, caches: list[torch.Tensor], batch: int) -> None:
        it = iter(caches)
        for outer, raw in self._slots.convs:
            outer._streaming_state = outer._init_streaming_state(batch)
            outer._streaming_state.init_stream()
            outer._streaming_state.padding_to_add = 0
            raw._streaming_state = raw._init_streaming_state(batch)
            raw._streaming_state.init_stream()
            raw._streaming_state.previous = next(it)
        for raw in self._slots.convtrs:
            raw._streaming_state = raw._init_streaming_state(batch)
            raw._streaming_state.init_stream()
            raw._streaming_state.partial = next(it)
        for raw in self._slots.lstms:
            raw._streaming_state = raw._init_streaming_state(batch)
            raw._streaming_state.init_stream()
            raw._streaming_state.previous_h = next(it)
            raw._streaming_state.previous_c = next(it)
        for module in (*self._slots.adds, *self._slots.others):
            module._streaming_state = module._init_streaming_state(batch)
            module._streaming_state.init_stream()

    def _harvest(self) -> list[torch.Tensor]:
        # the conv caches are views into whole-chunk activations: copy them out
        # so a stream between chunks holds a few KB, not the activations
        caches = [raw._streaming_state.previous.contiguous() for _, raw in self._slots.convs]
        caches += [raw._streaming_state.partial.contiguous() for raw in self._slots.convtrs]
        for raw in self._slots.lstms:
            caches += [raw._streaming_state.previous_h, raw._streaming_state.previous_c]
        for add in self._slots.adds:
            # whole frames in means both branches of every residual add stay in step
            for leftover in (add._streaming_state.previous_x, add._streaming_state.previous_y):
                if leftover is not None and leftover.shape[-1]:
                    raise RuntimeError("AudioSeal residual branches fell out of step")
        return caches

    def _uninstall(self) -> None:
        for outer, raw in self._slots.convs:
            outer._streaming_state = outer._empty_state()
            raw._streaming_state = raw._empty_state()
        for module in (*self._slots.convtrs, *self._slots.lstms, *self._slots.adds, *self._slots.others):
            module._streaming_state = module._empty_state()

    @property
    def _lstm_offset(self) -> int:
        return len(self._slots.convs) + len(self._slots.convtrs)

    def _stack(self, per_stream: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        if len(per_stream) == 1:
            return per_stream[0]
        lstm_from = self._lstm_offset
        return [
            torch.cat(column, dim=1 if i >= lstm_from else 0)
            for i, column in enumerate(zip(*per_stream, strict=True))
        ]

    def _unstack(self, caches: list[torch.Tensor], batch: int) -> list[list[torch.Tensor]]:
        if batch == 1:
            return [caches]
        lstm_from = self._lstm_offset
        out: list[list[torch.Tensor]] = [[] for _ in range(batch)]
        for i, tensor in enumerate(caches):
            lstm = i >= lstm_from
            for b, part in enumerate(tensor.split(1, dim=1 if lstm else 0)):
                # cuDNN's LSTM refuses a non-contiguous hx, which a batch-dim
                # slice of [layers, batch, hidden] is, once the stream runs alone
                out[b].append(part.contiguous() if lstm else part)
        return out

    # -- watermarking -------------------------------------------------------

    @torch.no_grad()
    def _watermark_frames(self, audio: torch.Tensor, caches: list[list[torch.Tensor]]) -> torch.Tensor:
        """``audio`` [B, T] float, T a multiple of the hop, one stream per row;
        ``caches`` is updated in place. Returns the watermark [B, T]."""
        batch = audio.shape[0]
        self._install(self._stack(caches), batch)
        try:
            with self._no_compile(), _no_tf32():
                gen = self.generator
                hidden = gen.encoder(audio.unsqueeze(1))
                hidden = gen.msg_processor(hidden, self._message.expand(batch, -1))
                wm = gen.decoder(hidden)
            new = self._unstack(self._harvest(), batch)
        finally:
            self._uninstall()
        caches[:] = new
        return wm.squeeze(1)

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
            caches = [streams[i].caches or self._fresh for i, _, _ in members]
            wm = self._watermark_frames(audio, caches)
            marked = audio + self.config.alpha * wm
            for row, ((i, _, length), cache) in enumerate(zip(members, caches, strict=True)):
                streams[i].caches = cache
                results[i] = marked[row, :length]
        return results  # type: ignore[return-value]

    def apply_pcm16(
        self,
        chunks: list[torch.Tensor],
        streams: list[WatermarkStream],
        finals: list[bool] | None = None,
    ) -> list[torch.Tensor]:
        """``apply`` for PCM16 chunks, quantized back the way the codecs do."""
        floats = [chunk.to(self.device, torch.float32) / 32767 for chunk in chunks]
        marked = self.apply(floats, streams, finals)
        return [(m.clamp(-1, 1) * 32767).to(torch.int16) for m in marked]

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
    def detect(self, audio: torch.Tensor) -> tuple[float, int]:
        """1-D float audio -> (fraction of samples scored watermarked, message)."""
        x = audio.reshape(1, 1, -1).to(self.device, torch.float32)
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
    args = parser.parse_args(argv)
    detector = AudioSealDetector.build(args.device)
    for path in args.files:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        score, message = detector.detect(torch.from_numpy(audio.mean(axis=1)))
        verdict = "watermarked" if score > args.threshold else "not watermarked"
        print(f"{path}: {verdict} (score {score:.3f}, message 0x{message:04x}, {sample_rate} Hz)")


if __name__ == "__main__":
    _main()

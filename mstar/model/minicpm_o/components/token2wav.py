"""MiniCPM-o's token2wav: streamed s3 speech codes -> 24 kHz waveform.

Ported from Step-Audio2's ``Token2wav`` streaming path (``set_stream_cache`` + ``stream``,
``minicpmo-utils``, Apache-2.0). A voice's caches are prepared once by running the flow over
its prompt; each window of 28 codes (25 new, 3 lookahead; ``stream_windows``) then becomes mel
(``token2wav_flow``) and audio (HiFT), cross-faded with the previous window's tail.

Two paths. The per-request one (``stream`` over a ``Token2WavState``) is bit-exact against the
reference at fp32; serving does not use it, the tests compare against it. The batched one runs
a window in two captured stages: ``window_flow``, whose attention caches are on bounded KV
resources (``CACHE_FAMILIES``) so windows at any position batch, and ``window_vocode`` (HiFT),
which batches a request's first window apart from later ones. It matches the reference to
summation order and TF32.

The flow's initial noise is a slice of a fixed buffer (``ChunkCFM.rand_noise``); HiFT draws its
excitation from the global RNG unless given a generator or the draws themselves.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.resources.kv.bounded import SinkWindow
from mstar.engine.resources.recurrent.pool import RecurrentAddressing, RecurrentStatePool
from mstar.model.chatterbox.components.s3gen_hift import HiFTGenerator, HiFTNoise
from mstar.model.chatterbox.config import S3GenHiFTConfig
from mstar.model.minicpm_o.components.token2wav_flow import (
    DIT_DEPTH,
    DIT_HEAD_DIM,
    DIT_HEADS,
    DIT_HIDDEN,
    ENC_BLOCKS,
    ENC_HEAD_DIM,
    ENC_HEADS,
    ENC_UP_BLOCKS,
    MEL_BINS,
    N_TIMESTEPS,
    PRE_LOOKAHEAD,
    TOKEN_DIM,
    UP_RATE,
    BoundedLayerKV,
    DenseKV,
    FlowKV,
    Token2WavFlow,
)
from mstar.model.minicpm_o.components.voice_prompt import (
    VoicePrompt,
    VoicePromptEncoder,
    campplus_state,
    read_onnx_parameters,
    tokenizer_state,
)

SAMPLE_RATE = 24000
SILENCE_CODE = 4218
LEAD_SILENCE = 3
WINDOW = 28
HOP = 25
# frames of generated context the reference keeps after the prompt's (``stream``'s truncation)
KEEP_RECENT = 100
HIFT_MEL_CACHE = 8
SAMPLES_PER_FRAME = 480
HIFT_TAIL = HIFT_MEL_CACHE * SAMPLES_PER_FRAME
MAX_LAST_TOKENS = WINDOW - 1


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def stream_windows(codes: list[int]) -> Iterator[tuple[list[int], bool]]:
    """``(window, last)`` per ``stream`` call for a whole utterance's codes."""
    buf = [SILENCE_CODE] * LEAD_SILENCE + list(codes)
    while len(buf) >= WINDOW:
        yield buf[:WINDOW], False
        buf = buf[HOP:]
    yield buf, True


def num_windows(num_codes: int) -> int:
    n = num_codes + LEAD_SILENCE
    return 1 + (0 if n < WINDOW else (n - WINDOW) // HOP + 1)


def window_frames(num_tokens: int, last: bool) -> int:
    """50 Hz mel frames a window yields: the lookahead only counts on the last one."""
    return UP_RATE * (num_tokens if last else num_tokens - PRE_LOOKAHEAD)


def window_samples(num_tokens: int, last: bool, first: bool) -> int:
    """Samples ``stream`` returns for a window."""
    if first:
        return window_frames(num_tokens, last) * SAMPLES_PER_FRAME
    total = (HIFT_MEL_CACHE + window_frames(num_tokens, last)) * SAMPLES_PER_FRAME
    return total if last else total - HIFT_TAIL


# ---------------------------------------------------------------------------
# Per-request state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheCapacity:
    """Slot sizes for voice prompts of up to ``prompt_tokens`` tokens (``P``). The 50 Hz
    caches peak at ``2P + KEEP_RECENT`` plus a last window of ``2 * 27`` frames; the 25 Hz
    one at half the kept frames plus 27 tokens."""

    prompt_tokens: int

    @property
    def enc_25hz(self) -> int:
        return self.prompt_tokens + KEEP_RECENT // UP_RATE + MAX_LAST_TOKENS

    @property
    def frames_50hz(self) -> int:
        return UP_RATE * self.prompt_tokens + KEEP_RECENT + UP_RATE * MAX_LAST_TOKENS


@dataclass(slots=True)
class Token2WavState:
    """A request's state on the per-request path (and a voice's prepared caches), float32,
    batch dims of 1 kept. The batched path keeps the attention caches on bounded KV
    resources and the rest in a slot (``slot_layout``).

    - ``enc_cnn``, ``dit_cnn``: the conformer's and DiT's conv left context;
    - ``enc_kv1`` / ``enc_kv2``: the 25 Hz / 50 Hz conformer keys|values, oldest first;
    - ``dit_kv``: per Euler step and block, keys|values, newest window first;
    - ``hift_mel``, ``hift_source``, ``hift_speech``: the previous window's last mel frames,
      excitation and un-faded samples.
    """

    enc_cnn: torch.Tensor
    enc_kv1: torch.Tensor
    enc_kv2: torch.Tensor
    dit_cnn: torch.Tensor
    dit_kv: torch.Tensor
    hift_mel: torch.Tensor
    hift_source: torch.Tensor
    hift_speech: torch.Tensor
    prompt_frames: int  # 2P
    enc_len1: int = 0
    enc_len2: int = 0
    dit_len: int = 0
    calls: int = 0

    @classmethod
    def allocate(cls, capacity: CacheCapacity, device: torch.device | str, dtype=torch.float32) -> Token2WavState:
        z = lambda *shape: torch.zeros(shape, device=device, dtype=dtype)  # noqa: E731
        c1, c2 = capacity.enc_25hz, capacity.frames_50hz
        return cls(
            enc_cnn=z(1, TOKEN_DIM, 2 + 2 * UP_RATE),
            enc_kv1=z(ENC_BLOCKS, 1, ENC_HEADS, c1, 2 * ENC_HEAD_DIM),
            enc_kv2=z(ENC_UP_BLOCKS, 1, ENC_HEADS, c2, 2 * ENC_HEAD_DIM),
            dit_cnn=z(N_TIMESTEPS, DIT_DEPTH, 2, 2 * DIT_HIDDEN, 2),
            dit_kv=z(N_TIMESTEPS, DIT_DEPTH, 2, DIT_HEADS, c2, 2 * DIT_HEAD_DIM),
            hift_mel=z(1, MEL_BINS, HIFT_MEL_CACHE),
            hift_source=z(1, 1, HIFT_TAIL),
            hift_speech=z(1, HIFT_TAIL),
            prompt_frames=0,
        )

    def copy_from(self, other: Token2WavState) -> None:
        """Load ``other``'s contents (e.g. a voice's initial state) into this slot, whose
        capacity may be larger."""
        for f in fields(self):
            src, dst = getattr(other, f.name), getattr(self, f.name)
            if isinstance(dst, torch.Tensor):
                dst[tuple(slice(0, n) for n in src.shape)].copy_(src)
            else:
                setattr(self, f.name, src)

    def kv_caches(self) -> FlowKV:
        return FlowKV(
            [DenseKV(kv, self.enc_len1) for kv in self.enc_kv1],
            [DenseKV(kv, self.enc_len2) for kv in self.enc_kv2],
            [[DenseKV(kv, self.dit_len) for kv in step] for step in self.dit_kv],
        )

    def truncate(self) -> None:
        """The reference's bound: past ``2P + 100`` frames the 50 Hz caches keep their first
        ``2P`` entries and last 100. The 25 Hz cache follows the reference's tiled layout,
        which in effect keeps its first ``P + 50``."""
        limit = self.prompt_frames + KEEP_RECENT
        if self.dit_len > limit:
            n = self.dit_len
            self.dit_kv[..., self.prompt_frames:limit, :] = self.dit_kv[..., n - KEEP_RECENT:n, :].clone()
            self.dit_len = limit
        if self.enc_len2 > limit:
            n2, n1 = self.enc_len2, self.enc_len1
            assert n2 == UP_RATE * n1
            kept = list(range(self.prompt_frames)) + list(range(n2 - KEEP_RECENT, n2))
            src = [i % n1 for i in kept[: limit // UP_RATE]]
            if src != list(range(len(src))):
                idx = torch.tensor(src, device=self.enc_kv1.device)
                self.enc_kv1[..., : len(src), :] = self.enc_kv1.index_select(3, idx)
            self.enc_len1 = len(src)
            self.enc_kv2[..., self.prompt_frames:limit, :] = self.enc_kv2[..., n2 - KEEP_RECENT:n2, :].clone()
            self.enc_len2 = limit


# Token2WavState's tensors as slot blocks: the batch axis of 1 each tensor keeps for the
# modules is dropped in the pool and re-added (as a view) by `state_from_slot`.
_BATCH_AXIS = {"enc_cnn": 0, "enc_kv1": 1, "enc_kv2": 1, "hift_mel": 0, "hift_source": 0, "hift_speech": 0}


class SlotBlock(NamedTuple):
    shape: tuple[int, ...]
    dtype: torch.dtype


def slot_layout(capacity: CacheCapacity) -> dict[str, SlotBlock]:
    """Per-request slot of the batched path, for voices of up to ``capacity.prompt_tokens``
    tokens: ``Token2WavState``'s tensors without their batch axes, except the attention
    caches, which live in bounded KV resources (``CACHE_FAMILIES``)."""
    meta = Token2WavState.allocate(capacity, "meta")
    layout = {}
    for f in fields(Token2WavState):
        t = getattr(meta, f.name)
        if isinstance(t, torch.Tensor) and f.name not in ("enc_kv1", "enc_kv2", "dit_kv"):
            shape = list(t.shape)
            if f.name in _BATCH_AXIS:
                del shape[_BATCH_AXIS[f.name]]
            layout[f.name] = SlotBlock(tuple(shape), torch.float32)
    return layout


# Each attention cache keeps a sink and a window of a stream that starts with the voice's
# own cache (its source) and continues with the request's windows, as upstream's
# truncation does (``Token2WavState.truncate``): past ``2P + 100`` frames the 50 Hz caches
# keep their first ``2P`` entries and their last 100, and the 25 Hz one its first
# ``P + 50``. The DiT attends ``[newest window, ..., voice]`` without positions, so its
# stream is the voice reordered and each window last-first, which makes upstream's
# "first 2P" a window.


def _enc1_retention(source_len: int) -> SinkWindow:
    return SinkWindow(source_len + KEEP_RECENT // UP_RATE, 0)


def _enc2_retention(source_len: int) -> SinkWindow:
    return SinkWindow(source_len, KEEP_RECENT)


def _dit_retention(source_len: int) -> SinkWindow:
    return SinkWindow(KEEP_RECENT, source_len)


def _enc1_source(voice: Token2WavVoice) -> torch.Tensor:
    return voice.initial.enc_kv1[..., : voice.initial.enc_len1, :].contiguous()


def _enc2_source(voice: Token2WavVoice) -> torch.Tensor:
    return voice.initial.enc_kv2[..., : voice.initial.enc_len2, :].contiguous()


def _dit_source(voice: Token2WavVoice) -> torch.Tensor:
    """Its last 100 frames, then the rest last-first, per Euler step x block."""
    p2 = voice.initial.prompt_frames
    kv = voice.initial.dit_kv[..., :p2, :]
    out = torch.cat([kv[..., p2 - KEEP_RECENT:, :], kv[..., :p2 - KEEP_RECENT, :].flip(-2)], dim=-2)
    return out.flatten(0, 1).contiguous()


class CacheFamily(NamedTuple):
    """One kind of token2wav attention cache, as a bounded KV resource declares it."""

    num_layers: int
    rows: int  # 2 for the DiT's guidance rows
    num_heads: int
    head_dim: int
    # stream entries per token: 1 at 25 Hz, 2 at 50 Hz
    rate: int
    retention: Callable[[int], SinkWindow]
    # the voice's cache in stream order, ``[layers, rows, H, L, 2d]``
    source: Callable[[Token2WavVoice], torch.Tensor]
    reverse_step_order: bool = False

    def position(self, voice_tokens: int, written_tokens: int) -> tuple[int, int]:
        """``(source_len, written)`` once a request's windows have added ``written_tokens``
        (``window_tokens`` each)."""
        return self.rate * voice_tokens, self.rate * written_tokens

    def span(self, num_tokens: int, last: bool) -> int:
        return self.rate * window_tokens(num_tokens, last)


def window_tokens(num_tokens: int, last: bool) -> int:
    """25 Hz tokens a window adds to its caches: its own, without the lookahead (a window
    is 28 codes, or fewer when it carried the TTS's stop code); none for a last window."""
    return 0 if last else num_tokens - PRE_LOOKAHEAD


CACHE_FAMILIES = {
    "enc1": CacheFamily(ENC_BLOCKS, 1, ENC_HEADS, ENC_HEAD_DIM, 1, _enc1_retention, _enc1_source),
    "enc2": CacheFamily(ENC_UP_BLOCKS, 1, ENC_HEADS, ENC_HEAD_DIM, UP_RATE, _enc2_retention, _enc2_source),
    "dit": CacheFamily(N_TIMESTEPS * DIT_DEPTH, 2, DIT_HEADS, DIT_HEAD_DIM, UP_RATE, _dit_retention,
                       _dit_source, reverse_step_order=True),
}


class SlotRows(NamedTuple):
    """A batch's rows of the vocoder's state pool (``slot_layout``): the pool and the step's
    addressing of them, the same in a captured and an eager forward."""

    pool: RecurrentStatePool
    rows: RecurrentAddressing

    def get(self, name: str) -> torch.Tensor:
        return self.pool.gather(name, self.rows)

    def set(self, name: str, value: torch.Tensor) -> None:
        self.pool.scatter_(name, self.rows, value)

    @property
    def fresh(self) -> torch.Tensor:
        """``[B]`` rows whose slot holds no state yet: their request's first window."""
        return ~self.rows.has_state[: self.rows.num_rows]


class AttentionCache(NamedTuple):
    """Where a batch's caches of one family are: the bounded KV resource, the voice's
    source and the batch's rows of the step's plan."""

    resource: object
    source: torch.Tensor
    plan: object

    def layer(self, idx: int) -> BoundedLayerKV:
        return BoundedLayerKV(self.resource, idx, self.source[idx], self.plan)


class WindowCaches(NamedTuple):
    enc1: AttentionCache
    enc2: AttentionCache
    dit: AttentionCache

    def flow_kv(self) -> FlowKV:
        return FlowKV(
            [self.enc1.layer(i) for i in range(ENC_BLOCKS)],
            [self.enc2.layer(i) for i in range(ENC_UP_BLOCKS)],
            [[self.dit.layer(step * DIT_DEPTH + i) for i in range(DIT_DEPTH)] for step in range(N_TIMESTEPS)],
        )


class Token2WavVoice(NamedTuple):
    """A voice ready to stream: its prompt, its projected speaker condition and the state a
    request starts from."""

    prompt: VoicePrompt
    spk: torch.Tensor  # [1, 80]
    initial: Token2WavState


# ---------------------------------------------------------------------------
# Vocoder
# ---------------------------------------------------------------------------


class SineGen2Source(nn.Module):
    """Step-Audio2's 24 kHz harmonic source (``SineGen2``): unlike Chatterbox's
    ``HarmonicSource``, the phase is integrated at the mel rate and upsampled. Its random
    draws, in the reference's order, travel as ``HiFTNoise(phase, harmonic)``."""

    def __init__(self, config: S3GenHiFTConfig):
        super().__init__()
        self.sampling_rate = config.sampling_rate
        self.harmonic_num = config.nb_harmonics
        self.sine_amp = config.nsf_alpha
        self.noise_std = config.nsf_sigma
        self.voiced_threshold = config.nsf_voiced_threshold
        self.upsample_scale = config.upsample_factor
        self.l_linear = nn.Linear(config.nb_harmonics + 1, 1)
        self.register_buffer(
            "harmonics", torch.arange(1, config.nb_harmonics + 2, dtype=torch.float32).view(1, 1, -1), persistent=False,
        )

    def draw_noise(
        self, batch: int, length: int, dtype: torch.dtype, device, generator: torch.Generator | None,
    ) -> HiFTNoise:
        h = self.harmonic_num + 1
        phase = torch.rand((batch, h), dtype=dtype, device=device, generator=generator)
        # the reference draws ``randn_like`` of harmonics laid out time-minor, so the
        # stream fills ``[B, H+1, L]`` in memory order
        harmonic = torch.randn((batch, h, length), dtype=dtype, device=device, generator=generator).transpose(1, 2)
        torch.randn((batch, length, 1), dtype=dtype, device=device, generator=generator)
        return HiFTNoise(phase=phase, harmonic=harmonic)

    def _sines(self, f0: torch.Tensor, noise: HiFTNoise) -> torch.Tensor:
        """``f0 [B, L, 1]`` -> noisy harmonics ``[B, L, H+1]``."""
        fn = torch.multiply(f0, self.harmonics)
        rad_values = (fn / self.sampling_rate) % 1
        rand_ini = noise.phase.clone()
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        rad_values = F.interpolate(
            rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear",
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * math.pi
        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear",
        ).transpose(1, 2)
        sine_waves = torch.sin(phase) * self.sine_amp
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        return sine_waves * uv + noise_amp * noise.harmonic

    def forward(self, f0_upsampled: torch.Tensor, noise: HiFTNoise) -> torch.Tensor:
        with torch.no_grad():
            sine_waves = self._sines(f0_upsampled, noise)
        return torch.tanh(self.l_linear(sine_waves))


class StepAudioHiFT(HiFTGenerator):
    """Chatterbox's HiFT (same network and config) with Step-Audio2's 24 kHz source."""

    def __init__(self, config: S3GenHiFTConfig | None = None):
        config = config or S3GenHiFTConfig()
        super().__init__(config)
        self.m_source = SineGen2Source(config)


def fade_in_out(fade_in: torch.Tensor, fade_out: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Cross-fade the head of ``fade_in`` with the tail of ``fade_out`` (float64 window, as
    the reference's numpy Hamming window, so the blend is computed in float64)."""
    n = window.shape[0] // 2
    out = fade_in.clone()
    out[..., :n] = fade_in[..., :n] * window[:n] + fade_out[..., -n:] * window[n:]
    return out


# ---------------------------------------------------------------------------
# The whole thing
# ---------------------------------------------------------------------------


class Token2Wav(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow = Token2WavFlow()
        self.hift = StepAudioHiFT()
        self.voice_encoder = VoicePromptEncoder()
        self.register_buffer("speech_window", torch.from_numpy(np.hamming(2 * HIFT_TAIL)), persistent=False)

    @torch.inference_mode()
    def prepare_voice(self, prompt: VoicePrompt) -> Token2WavVoice:
        """``set_stream_cache``: the flow over the prompt tokens + 3 silence tokens with the
        prompt mel as condition, keeping only the caches. Runs once per bundled voice at
        load, on ``DenseKV`` (see there for why that is not a resource)."""
        device = self.speech_window.device
        tokens = prompt.tokens.to(device)
        p = tokens.shape[1]
        spk = self.flow.project_speaker(prompt.spk_emb.to(device))
        state = Token2WavState.allocate(CacheCapacity(p), device)
        state.prompt_frames = UP_RATE * p
        silence = torch.full((1, LEAD_SILENCE), SILENCE_CODE, dtype=tokens.dtype, device=device)
        self.flow(
            torch.cat([tokens, silence], dim=1), spk, prompt.mel.to(device).transpose(1, 2).contiguous(), False,
            state.enc_cnn, state.dit_cnn, state.kv_caches(),
        )
        state.enc_len1, state.enc_len2, state.dit_len = p, UP_RATE * p, UP_RATE * p
        return Token2WavVoice(prompt=prompt, spk=spk, initial=state)

    # -- the per-request path: not used in serving; the tests' reference, and where voice
    # cloning (per-request voices) would start ------------------------------------------

    def new_state(self, voice: Token2WavVoice, capacity: CacheCapacity | None = None) -> Token2WavState:
        state = Token2WavState.allocate(
            capacity or CacheCapacity(voice.prompt.tokens.shape[1]), self.speech_window.device,
        )
        state.copy_from(voice.initial)
        return state

    @torch.inference_mode()
    def flow_chunk(
        self, state: Token2WavState, voice: Token2WavVoice, tokens: torch.Tensor, last: bool,
    ) -> torch.Tensor:
        """``tokens [1, n]`` -> mel ``[1, 80, window_frames(n, last)]``; advances the caches."""
        mel = self.flow(tokens, voice.spk, None, last, state.enc_cnn, state.dit_cnn, state.kv_caches())
        frames = mel.shape[2]
        state.enc_len1 += frames // UP_RATE
        state.enc_len2 += frames
        state.dit_len += frames
        state.truncate()
        return mel

    @torch.inference_mode()
    def vocode_chunk(
        self,
        state: Token2WavState,
        chunk_mel: torch.Tensor,
        last: bool,
        noise: HiFTNoise | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Mel chunk -> this call's samples ``[1, N]``, with the previous call's last 8 mel
        frames re-vocoded in front and cross-faded with its held-back tail."""
        first = state.calls == 0
        mel = chunk_mel if first else torch.cat([state.hift_mel, chunk_mel], dim=2)
        if noise is None:
            noise = self.hift.draw_noise(mel, generator)
        speech, source = self.hift.vocode(mel, cache_source=None if first else state.hift_source, noise=noise)
        if not first:
            speech = fade_in_out(speech, state.hift_speech, self.speech_window)
        state.hift_mel.copy_(mel[..., -HIFT_MEL_CACHE:])
        state.hift_source.copy_(source[:, :, -HIFT_TAIL:])
        state.hift_speech.copy_(speech[:, -HIFT_TAIL:])
        state.calls += 1
        if last:
            return speech
        if first:
            return torch.cat([speech.new_zeros(1, HIFT_TAIL), speech[:, :-HIFT_TAIL]], dim=1)
        return speech[:, :-HIFT_TAIL]

    def stream(
        self,
        state: Token2WavState,
        voice: Token2WavVoice,
        tokens: list[int] | torch.Tensor,
        last: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """One ``Token2wav.stream`` call."""
        if not torch.is_tensor(tokens):
            tokens = torch.tensor([tokens], dtype=torch.int32, device=self.speech_window.device)
        mel = self.flow_chunk(state, voice, tokens, last)
        return self.vocode_chunk(state, mel, last, generator=generator)


    # -- windows batched over pool slots ---------------------------------------------------

    @torch.inference_mode()
    def window_flow(
        self,
        state: SlotRows,
        voice: Token2WavVoice,
        tokens: torch.Tensor,
        caches: WindowCaches,
        last: bool = False,
    ) -> torch.Tensor:
        """The flow for ``B`` windows of one voice at any positions: ``tokens [B, n]`` -> mel
        ``[B, 80, frames]``. ``caches`` know each row's position; rows on their first window
        (``state.fresh``) read the voice's conv caches rather than their slot's. A last
        window writes no caches. Capturable."""
        b = tokens.shape[0]
        init = voice_blocks(voice)
        fresh = state.fresh

        def conv_cache(name):
            current = state.get(name)
            return torch.where(fresh.view(-1, *([1] * (current.dim() - 1))), init[name], current)

        enc_cnn = conv_cache("enc_cnn")
        # guidance rows of a request side by side, as the flow runs them
        dit_cnn = conv_cache("dit_cnn").permute(1, 2, 0, 3, 4, 5).contiguous().flatten(2, 3)
        mel = self.flow(tokens, voice.spk.expand(b, -1), None, last, enc_cnn, dit_cnn, caches.flow_kv())
        if not last:
            state.set("enc_cnn", enc_cnn)
            state.set("dit_cnn", dit_cnn.unflatten(2, (b, 2)).permute(2, 0, 1, 3, 4, 5))
        return mel

    @torch.inference_mode()
    def window_vocode(
        self,
        state: SlotRows,
        mel: torch.Tensor,
        first: bool,
        last: bool = False,
        noise: HiFTNoise | None = None,
    ) -> dict[str, torch.Tensor]:
        """HiFT up to its output spectrum for ``B`` windows' mel, prefixed by the slot's
        held-back frames unless they are their requests' ``first`` (a shorter input, so the
        two do not batch). Finish with ``window_finish``."""
        if first:
            hift_mel, cache_source = mel, None
        else:
            hift_mel = torch.cat([state.get("hift_mel"), mel], dim=2)
            cache_source = state.get("hift_source")
        if noise is None:
            noise = self.hift.draw_noise(hift_mel, None)
        magnitude, phase, source = self.hift.spectrum(hift_mel, noise.phase, noise.harmonic, cache_source)
        if not last:
            state.set("hift_mel", hift_mel[..., -HIFT_MEL_CACHE:])
            state.set("hift_source", source[:, :, -HIFT_TAIL:])
        return {"magnitude": magnitude, "phase": phase}

    @torch.inference_mode()
    def window_spectrum(
        self,
        state: SlotRows,
        voice: Token2WavVoice,
        first: bool,
        tokens: torch.Tensor,
        caches: WindowCaches,
        last: bool = False,
        noise: HiFTNoise | None = None,
    ) -> dict[str, torch.Tensor]:
        """``window_flow`` and ``window_vocode`` for ``B`` windows that are all, or all not,
        their request's ``first``."""
        mel = self.window_flow(state, voice, tokens, caches, last)
        return {"mel": mel, **self.window_vocode(state, mel, first, last, noise)}

    @torch.inference_mode()
    def window_finish(
        self,
        state: SlotRows,
        magnitude: torch.Tensor,
        phase: torch.Tensor,
        first: bool,
        last: bool = False,
    ) -> torch.Tensor:
        """``window_spectrum``'s spectrum -> each row's samples ``[B, window_samples(...)]``:
        the inverse STFT (kept out of the capture: it syncs), the cross-fade with the slot's
        held-back tail, and the new tail, as ``vocode_chunk`` does them."""
        speech = self.hift.waveform(magnitude, phase)
        if not first:
            speech = fade_in_out(speech, state.get("hift_speech"), self.speech_window)
        if last:
            return speech
        state.set("hift_speech", speech[:, -HIFT_TAIL:])
        if first:
            return torch.cat([speech.new_zeros(speech.shape[0], HIFT_TAIL), speech[:, :-HIFT_TAIL]], dim=1)
        return speech[:, :-HIFT_TAIL]

    def stream_batched(
        self,
        state: SlotRows,
        voice: Token2WavVoice,
        first: bool,
        tokens: torch.Tensor,
        caches: WindowCaches,
        last: bool = False,
        noise: HiFTNoise | None = None,
    ) -> torch.Tensor:
        """``B`` ``stream`` calls of one voice, all or none of them ``first``, eagerly."""
        out = self.window_spectrum(state, voice, first, tokens, caches, last, noise)
        return self.window_finish(state, out["magnitude"], out["phase"], first, last)


# The token counts a last window after a full one can have: the left context plus 0 to
# HOP - 1 new codes (a full window's worth would have made a full window).
LAST_WINDOW_TOKENS = range(WINDOW - HOP, WINDOW)


def voice_blocks(voice: Token2WavVoice) -> dict[str, torch.Tensor]:
    """A voice's initial state laid out as a one-slot pool (views)."""
    out = {}
    for f in fields(Token2WavState):
        t = getattr(voice.initial, f.name)
        if isinstance(t, torch.Tensor):
            out[f.name] = (t.squeeze(_BATCH_AXIS[f.name]) if f.name in _BATCH_AXIS else t).unsqueeze(0)
    return out


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def _remap_flow(name: str) -> str:
    for old, new in ((".conv.block.1.", ".conv.conv1."), (".conv.block.3.", ".conv.norm."),
                     (".conv.block.6.", ".conv.conv2.")):
        name = name.replace(old, new)
    return name


def _load_exact(module: nn.Module, state: dict[str, torch.Tensor], what: str) -> None:
    result = module.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"{what}: missing {result.missing_keys[:8]} unexpected {result.unexpected_keys[:8]}")


def _fold_weight_norm_on(state: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Fold ``parametrizations.weight.original{0,1}`` pairs into plain weights on ``device``,
    where the reference's parametrization computes them, so the folded weights carry the
    same bits it uses."""
    out: dict[str, torch.Tensor] = {}
    marker = ".parametrizations.weight.original"
    for name, tensor in state.items():
        if marker not in name:
            out[name] = tensor
        elif name.endswith("original0"):
            base = name[: name.index(marker)]
            g = tensor.to(device)
            v = state[f"{base}{marker}1"].to(device)
            out[f"{base}.weight"] = torch._weight_norm(v, g, dim=0)
    return out


def load_token2wav(asset_dir: str, device: torch.device | str = "cuda", noise: torch.Tensor | None = None) -> Token2Wav:
    """Build ``Token2Wav`` from MiniCPM-o's ``assets/token2wav`` (``flow.pt``, ``hift.pt``,
    ``speech_tokenizer_v2_25hz.onnx``, ``campplus.onnx``); raises if any parameter is left
    unfilled. ``noise`` is the flow's ``[1, 80, 30000]`` noise buffer; without it one is
    drawn from a fixed seed (the reference draws it from the global RNG at construction)."""
    device = torch.device(device)
    model = Token2Wav()
    flow = torch.load(f"{asset_dir}/flow.pt", map_location="cpu", weights_only=True)
    _load_exact(model.flow, {_remap_flow(k): v for k, v in flow.items()}, "flow.pt")
    model.to(device)
    hift = torch.load(f"{asset_dir}/hift.pt", map_location="cpu", weights_only=True)
    hift = {k.removeprefix("generator."): v for k, v in hift.items()}
    _load_exact(model.hift, _fold_weight_norm_on(hift, device), "hift.pt")
    tok = read_onnx_parameters(f"{asset_dir}/speech_tokenizer_v2_25hz.onnx")
    _load_exact(model.voice_encoder.tokenizer, tokenizer_state(tok), "speech_tokenizer_v2_25hz.onnx")
    spk = read_onnx_parameters(f"{asset_dir}/campplus.onnx")
    _load_exact(model.voice_encoder.speaker, campplus_state(spk, model.voice_encoder.speaker), "campplus.onnx")
    if noise is None:
        noise = torch.randn(model.flow.decoder.rand_noise.shape, generator=torch.Generator().manual_seed(0))
    model.flow.decoder.rand_noise.copy_(noise)
    model = model.to(device).eval()
    model.flow.decoder.reset_constants()
    return model

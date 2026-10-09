"""MiniCPM-o's token2wav: streamed s3 speech codes -> 24 kHz waveform.

Ported from Step-Audio2's ``Token2wav`` (``stepaudio2/token2wav.py`` in ``minicpmo-utils``,
Apache-2.0), streaming path (``set_stream_cache`` + ``stream``). Per voice, ``setup`` runs the
flow over the voice prompt's tokens (plus three silence tokens of lookahead) and mel and keeps
the resulting caches; a request starts from a copy of them. Each ``stream`` call then turns a
window of codes into mel (``token2wav_flow``) and the mel into audio (HiFT), cross-fading with
the previous call's tail.

The caller drives the windows as the reference's ``streaming_generate`` does: three silence
codes first, 28-code windows advancing by 25 (the last 3 are lookahead), then one final window
with whatever is left (``stream_windows``).

All of a request's state is one ``Token2WavState``: fixed-capacity cache tensors plus their
valid lengths. The lengths only depend on the prompt length and how many windows came before,
so they are host integers, and every shape an eager call sees is a slice of the slot.

Randomness: the flow's initial noise is a slice of a fixed buffer (``ChunkCFM.rand_noise``),
and HiFT draws its excitation per call (``StepAudioHiFT``) from the global RNG unless given a
generator or the draws themselves.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

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
    """Everything one request carries between ``stream`` calls (batch dims of 1 kept so the
    tensors feed the modules as they are). With ``C1 = P + 77`` and ``C2 = 2P + 154`` for a
    ``P``-token voice prompt (``CacheCapacity``), float32:

    - ``enc_cnn [1, 512, 6]``: the lookahead conv's and the upsampler's left context;
    - ``enc_kv1 [6, 1, 8, C1, 128]`` / ``enc_kv2 [4, 1, 8, C2, 128]``: the 25 Hz / 50 Hz
      conformer keys|values, oldest first;
    - ``dit_cnn [10, 16, 2, 1024, 2]``: per Euler step, per DiT block, for the conditional
      and unconditional rows, both causal convs' left context;
    - ``dit_kv [10, 16, 2, 8, C2, 128]``: per Euler step and block, keys|values, newest chunk
      first (the reference's order). This is ~1.3 MB per frame: ~600 MB at ``P = 151``;
    - ``hift_mel [1, 80, 8]``, ``hift_source [1, 1, 3840]``, ``hift_speech [1, 3840]``: the
      previous call's last mel frames, excitation and (un-faded) samples.
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

    def truncate(self) -> None:
        """The reference's bound on the 50 Hz caches: past ``2P + 100`` frames keep the first
        ``2P`` entries and the last 100. The 25 Hz conformer cache is stored by the
        reference tiled twice along time (so it lines up with the 50 Hz one) and read back as
        the first half of the truncated tiling; the gather reproduces that."""
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


def state_block_shapes(capacity: CacheCapacity) -> dict[str, tuple[int, ...]]:
    """Per-request slot layout for voices of up to ``capacity.prompt_tokens`` tokens."""
    meta = Token2WavState.allocate(capacity, "meta")
    shapes = {}
    for f in fields(Token2WavState):
        t = getattr(meta, f.name)
        if isinstance(t, torch.Tensor):
            shape = list(t.shape)
            if f.name in _BATCH_AXIS:
                del shape[_BATCH_AXIS[f.name]]
            shapes[f.name] = tuple(shape)
    return shapes


def state_from_slot(blocks: dict[str, torch.Tensor], lengths: dict[str, int]) -> Token2WavState:
    """A ``Token2WavState`` over one slot's block views, so every in-place update lands in
    the slot; ``lengths`` holds the host-side fields (``prompt_frames``, ``enc_len1``, ...)."""
    tensors = {
        name: view.unsqueeze(_BATCH_AXIS[name]) if name in _BATCH_AXIS else view
        for name, view in blocks.items()
    }
    return Token2WavState(**tensors, **lengths)


def state_lengths(state: Token2WavState) -> dict[str, int]:
    return {
        f.name: getattr(state, f.name) for f in fields(state)
        if not isinstance(getattr(state, f.name), torch.Tensor)
    }


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
    """Step-Audio2's 24 kHz harmonic source (``SourceModuleHnNSF2`` / ``SineGen2``): the phase
    is integrated at the mel rate and linearly upsampled, unlike Chatterbox's per-sample
    ``HarmonicSource``. Its draws, in the reference's order, are an initial phase per
    harmonic ``[B, H+1]``, Gaussian noise ``[B, L, H+1]`` (drawn as ``[B, H+1, L]``) and a
    ``[B, L, 1]`` field it discards; they travel as ``HiFTNoise(phase, harmonic)`` in this
    layout."""

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
        prompt mel as condition, keeping only the caches."""
        device = self.speech_window.device
        tokens = prompt.tokens.to(device)
        p = tokens.shape[1]
        spk = self.flow.project_speaker(prompt.spk_emb.to(device))
        state = Token2WavState.allocate(CacheCapacity(p), device)
        state.prompt_frames = UP_RATE * p
        silence = torch.full((1, LEAD_SILENCE), SILENCE_CODE, dtype=tokens.dtype, device=device)
        self.flow(
            torch.cat([tokens, silence], dim=1), spk, prompt.mel.to(device).transpose(1, 2).contiguous(), False,
            state.enc_cnn, state.enc_kv1, 0, state.enc_kv2, 0, state.dit_cnn, state.dit_kv, 0,
        )
        state.enc_len1, state.enc_len2, state.dit_len = p, UP_RATE * p, UP_RATE * p
        return Token2WavVoice(prompt=prompt, spk=spk, initial=state)

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
        mel = self.flow(
            tokens, voice.spk, None, last,
            state.enc_cnn, state.enc_kv1, state.enc_len1, state.enc_kv2, state.enc_len2,
            state.dit_cnn, state.dit_kv, state.dit_len,
        )
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
    return model.to(device).eval()

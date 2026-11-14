#!/usr/bin/env python3
"""Step-Audio2 token2wav parity against the reference's streamed windows (GPU).

The oracle (``token2wav_stream.pt``) is Step-Audio2's ``Token2wav.stream`` over the greedy
spoken reply's first 160 codes, fed as ``examples-vllm-stream.py`` feeds it (28-code windows,
hop 25, a last flush), fp32, with the CUDA RNG seeded 0 once before the stream; voice
``default_female.wav``. Both sides start from the oracle's own voice prompt pieces (tokens,
x-vector, mel) and the flow's recorded noise buffer.

* **one-request driver**: the shared token2wav modules on a request's own ``Token2WavState``
  (dense caches), cut as the reference cuts them (the DiT's cache only), HiFT's draws from a
  ``torch.Generator('cuda')`` seeded 0 once: each window's mel and PCM bytes must be
  bit-exact.
  Mels are bit-exact; PCM is held to ``HIFT_NONDET_LSB``, since the vocoder (the reference's
  too) is not deterministic run to run, and under deterministic cuDNN on both sides (the reference's
  replay, ``hift_float.py``) the vocoder is bit-exact too.
* **served path**: ``window_flow`` / ``window_vocode`` / ``window_finish`` on the caches as
  declared (``cache_arrays``), through their state handles, with Step-Audio2's
  ``Token2WavStreaming``, the DiT in float16 as served. The yardstick is the reference's own
  float16 token2wav (``record_t2w_fp16.py``): the flow's mels must be no further from the
  fp32 oracle than its, the audio's log-spectral distance within ``SPREAD`` times its
  (sample-level error is meaningless once HiFT's source phase drifts; another HiFT seed
  is printed for scale), and the served vocoder fed the recorded mels must be within
  ``HIFT_NONDET_LSB``.
* **negative control**: the one-request driver with MiniCPM-o's knobs (silence lookahead,
  conformer caches cut), which must diverge from the oracle.
* the M* voice prompt encoder on the voice clip against the oracle's prompt pieces (reported).

Run: ``python test/step_audio2/parity_token2wav.py``; exits nonzero on any failed check.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import torch
from _common import ASSETS, CKPT, DEVICE, Verdict, oracle, setup_torch

from mstar.engine.resources import Segment, StepContext, StepRunner, SubmoduleStep
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.state import ArrayStep
from mstar.engine.state.adapters import build_specs, node_handles, state_steps
from mstar.model.components.token2wav.keys import Token2WavKeys
from mstar.model.components.token2wav.model import (
    DIT_COMPUTE_DTYPE,
    HIFT_MEL_CACHE,
    HIFT_TAIL,
    KEEP_RECENT,
    CacheCapacity,
    Token2Wav,
    Token2WavState,
    Token2WavStreaming,
    Token2WavVoice,
    VoiceBank,
    WindowCaches,
    fade_in_out,
    load_token2wav,
    slot_layout,
)
from mstar.model.components.token2wav.resources import cache_arrays
from mstar.model.components.token2wav.voice_prompt import VoicePrompt
from mstar.model.step_audio2.token2wav import CONFORMER_REPLY_TOKENS, T2W_STREAMING

KEYS = Token2WavKeys(codes="speech_codes", state="t2w_state",
                     kv={"enc1": "t2w_enc1_kv", "enc2": "t2w_enc2_kv", "dit": "t2w_dit_kv"})
# The served flow runs the DiT in float16 and attends its caches in another order
# (``Token2WavSubmodule``); max abs over mel values of about -12..3.
SPREAD = 1.5
# HiFT's cuDNN convolutions are not deterministic run to run, in the reference too
# (``hift_float.py``): up to ~1.7e-4 (~6 LSB) on a raw sample between runs on the same
# mels and seed. 16-bit PCM units; bit-exactness is checked under deterministic cuDNN.
HIFT_NONDET_LSB = 8
CONTROL_MIN = 1000
# default_female's token 85 sits 3e-5 from a rounding boundary of the tokenizer's quantizer,
# on the CPU and on CUDA alike
VOICE_TOKEN_FLIPS = 1


def pcm_bytes(speech: torch.Tensor) -> bytes:
    """``token2wav.py:164-168``."""
    wav = np.clip(speech.cpu().numpy(), -1.0, 1.0)
    return (wav * 32767.0).astype("<i2").tobytes()


def pcm(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.int64)


# -- the reference's one-request driver over the shared modules -------------------------

def new_state(model: Token2Wav, voice: Token2WavVoice) -> Token2WavState:
    state = Token2WavState.allocate(model.streaming.capacity(voice.prompt.tokens.shape[1]), DEVICE)
    for f in dataclasses.fields(state):
        src, dst = getattr(voice.initial, f.name), getattr(state, f.name)
        if isinstance(dst, torch.Tensor):
            dst[tuple(slice(0, n) for n in src.shape)].copy_(src)
        else:
            setattr(state, f.name, src)
    return state


def cut(state: Token2WavState, conformer: bool) -> None:
    """The reference's bound past ``2P + 100`` frames: the DiT's cache keeps its first 2P
    entries and last 100 (``token2wav.py:137-141``); with ``conformer`` (MiniCPM-o) its
    caches too, the 25 Hz one in the reference's tiled layout."""
    limit = state.prompt_frames + KEEP_RECENT
    if state.dit_len > limit:
        n = state.dit_len
        state.dit_kv[..., state.prompt_frames:limit, :] = state.dit_kv[..., n - KEEP_RECENT:n, :].clone()
        state.dit_len = limit
    if conformer and state.enc_len2 > limit:
        n2, n1 = state.enc_len2, state.enc_len1
        kept = list(range(state.prompt_frames)) + list(range(n2 - KEEP_RECENT, n2))
        src = [i % n1 for i in kept[: limit // 2]]
        idx = torch.tensor(src, device=state.enc_kv1.device)
        state.enc_kv1[..., : len(src), :] = state.enc_kv1.index_select(3, idx)
        state.enc_len1 = len(src)
        state.enc_kv2[..., state.prompt_frames:limit, :] = state.enc_kv2[..., n2 - KEEP_RECENT:n2, :].clone()
        state.enc_len2 = limit


@torch.inference_mode()
def stream(model: Token2Wav, state: Token2WavState, voice: Token2WavVoice, codes: list[int], last: bool,
           gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """One ``Token2wav.stream`` call: (mel, samples)."""
    tokens = torch.tensor([codes], dtype=torch.int32, device=DEVICE)
    mel = model.flow(tokens, voice.spk, None, last, state.enc_cnn, state.dit_cnn, state.kv_caches())
    frames = mel.shape[2]
    state.enc_len1 += frames // 2
    state.enc_len2 += frames
    state.dit_len += frames
    cut(state, conformer=model.streaming.conformer_reply_tokens is None)
    first = state.calls == 0
    hift_mel = mel if first else torch.cat([state.hift_mel, mel], dim=2)
    speech, source = model.hift.vocode(hift_mel, gen, cache_source=None if first else state.hift_source)
    if not first:
        speech = fade_in_out(speech, state.hift_speech, model.speech_window)
    state.hift_mel.copy_(hift_mel[..., -HIFT_MEL_CACHE:])
    state.hift_source.copy_(source[:, :, -HIFT_TAIL:])
    state.hift_speech.copy_(speech[:, -HIFT_TAIL:])
    state.calls += 1
    if not last:
        if first and model.streaming.first_window_zeros:
            speech = torch.cat([speech.new_zeros(1, HIFT_TAIL), speech[:, :-HIFT_TAIL]], dim=1)
        else:
            speech = speech[:, :-HIFT_TAIL]
    return mel, speech


def one_request(model: Token2Wav, voice: Token2WavVoice, windows, seed: int = 0) -> tuple[list, list[bytes]]:
    state = new_state(model, voice)
    gen = torch.Generator(DEVICE).manual_seed(seed)
    mels, out = [], []
    for codes, last in windows:
        mel, speech = stream(model, state, voice, codes, last, gen)
        mels.append(mel.cpu())
        out.append(pcm_bytes(speech))
    return mels, out


# -- the served path on the engine's resources -------------------------------------------

def served(model: Token2Wav, voice: Token2WavVoice, windows, vocode_mels=None) -> tuple[list, list[bytes]]:
    """Each window's flow mel and PCM; with ``vocode_mels``, HiFT vocodes those (the
    reference's) in place of the flow's, to check the vocoder stage alone."""
    p = voice.prompt.tokens.shape[1]
    arrays = cache_arrays(KEYS, "token2wav", (p, p), CONFORMER_REPLY_TOKENS, streaming=model.streaming)
    specs = build_specs(arrays, kernel_settings={array.name: {"max_slots": 2} for array in arrays})
    resources = {s.resource_key: build_resource(s, EngineResourceInfo(device=DEVICE, kv_dtype=torch.float32))
                 for s in specs}
    handles = node_handles(SimpleNamespace(declare_state=lambda: arrays), "token2wav", resources)
    runner = StepRunner(resources)
    families = model.streaming.cache_families()
    bank = VoiceBank.build({"v": voice}, dit_dtype=DIT_COMPUTE_DTYPE)
    for name, fam in families.items():
        handles[KEYS.kv[name]].set_sources(bank.source_fields(name, fam.head_dim), [fam.rate * p])
    caches = WindowCaches.bind(handles, KEYS.kv, families, {
        name: handles[KEYS.kv[name]].array.source.max_len for name in families
    })
    rid = "r0"
    runner.ingest_request(rid, {})
    # the request's Value fields, as its regions read and write them; it starts with none
    fields: dict[str, torch.Tensor] = {"started": torch.zeros(1, dtype=torch.bool, device=DEVICE)}
    for name, block in slot_layout(CacheCapacity(1), dit_dtype=DIT_COMPUTE_DTYPE).items():
        fields[name] = torch.zeros(1, *block.shape, dtype=block.dtype, device=DEVICE)
    mels, out = [], []
    with torch.inference_mode():
        for k, (codes, last) in enumerate(windows):
            n, first = len(codes), k == 0
            step = state_steps(SubmoduleStep(state={
                KEYS.kv[name]: ArrayStep(segments=(Segment(request_id=rid, label="main", span=fam.span(n, last)),),
                                         sources={rid: 0})
                for name, fam in families.items()
            }), handles)
            step.set_ctx(StepContext(request_ids=(rid,), graph_walk="t2w", slot=0, capture=False))
            assert runner.admit(step).ok
            runner.plan(step)
            step.begin_calls()
            tokens = torch.tensor([codes], dtype=torch.int32, device=DEVICE)
            voices = torch.zeros(1, dtype=torch.long, device=DEVICE)
            mel, conv = model.window_flow(fields, bank, voices, tokens, caches, last)
            fields.update(conv)
            if vocode_mels is not None:
                mels.append(mel.cpu())
                mel = vocode_mels[k].to(DEVICE)
            else:
                mels.append(mel.cpu())
            # the request's seed is 0, as the one-request reference's generator
            noise = model.window_noise(mel.shape[-1], first, [(0, k)])
            spec, held = model.window_vocode(fields, mel, first, noise)
            fields.update(held)
            speech, held = model.window_finish(fields, spec["magnitude"], spec["phase"], first, last)
            fields.update(held)
            step.check_calls()
            step.end_calls()
            runner.commit(step)
            out.append(pcm_bytes(speech))
    runner.remove_request(rid)
    return mels, out


def log_spectral_distance(a: bytes, b: bytes) -> float:
    """dB RMS over frequency, averaged over frames, of two PCM streams' 1024-point STFTs:
    sample-level differences are meaningless once HiFT's source phase drifts."""
    def db(x: bytes) -> torch.Tensor:
        wav = torch.from_numpy(pcm(x).astype(np.float32) / 32768.0)
        spec = torch.stft(wav, 1024, 256, window=torch.hann_window(1024), return_complex=True).abs()
        return 20 * torch.log10(spec.clamp_min(1e-5))
    da, dbb = db(a), db(b)
    return ((da - dbb) ** 2).mean(0).sqrt().mean().item()


def deterministic_hift(model: Token2Wav, rec: dict) -> bool | None:
    """With cuDNN's deterministic algorithms, the vocoder on the recorded mels against the
    reference's own replay of them under the same setting (``hift_float.py``)."""
    try:
        want = oracle("hift_float_det")
    except FileNotFoundError:
        return None
    torch.backends.cudnn.deterministic = True
    try:
        gen = torch.Generator(DEVICE).manual_seed(0)
        state = Token2WavState.allocate(model.streaming.capacity(1), DEVICE)
        ok = True
        with torch.inference_mode():
            for k, mel in enumerate(rec["window_mels"]):
                mel = mel.to(DEVICE)
                first = k == 0
                hift_mel = mel if first else torch.cat([state.hift_mel, mel], dim=2)
                speech, source = model.hift.vocode(hift_mel, gen, cache_source=None if first else state.hift_source)
                ok &= torch.equal(speech.cpu(), want[k]["raw"]) and torch.equal(source.cpu(), want[k]["source"])
                state.hift_mel.copy_(hift_mel[..., -HIFT_MEL_CACHE:])
                state.hift_source.copy_(source[:, :, -HIFT_TAIL:])
        return ok
    finally:
        torch.backends.cudnn.deterministic = False


def main() -> None:
    setup_torch()
    # as the recording ran: PyTorch's defaults, under which cuDNN convolutions take TF32
    torch.backends.cudnn.allow_tf32 = True
    verdict = Verdict()
    rec = oracle("token2wav_stream")
    windows = [(list(c), bool(last)) for c, last in rec["windows"]]
    want_mels, want_pcm = rec["window_mels"], rec["window_pcm"]
    prompt = VoicePrompt(tokens=rec["prompt_tokens"].to(DEVICE, torch.int32), spk_emb=rec["spk_emb"].to(DEVICE),
                         mel=rec["prompt_mels"].to(DEVICE))
    print(f"{len(windows)} windows of {[len(c) for c, _ in windows]} codes; voice of {prompt.tokens.shape[1]} tokens")

    model = load_token2wav(str(CKPT / "token2wav"), DEVICE, noise=rec["rand_noise"], streaming=T2W_STREAMING)
    voice = model.prepare_voice(prompt)

    mels, got = one_request(model, voice, windows)
    for k in range(len(windows)):
        mel_err = (mels[k] - want_mels[k]).abs().max().item()
        d = np.abs(pcm(got[k]) - pcm(want_pcm[k])).max() if len(got[k]) == len(want_pcm[k]) else None
        print(f"window {k}: mel max abs {mel_err:.3e}; pcm {len(got[k])} vs {len(want_pcm[k])} bytes, max diff {d}")
    verdict.check(all(torch.equal(a, b) for a, b in zip(mels, want_mels, strict=True)),
                  "one-request driver: every window's mel is bit-exact")
    off = [int((pcm(a) != pcm(b)).sum()) for a, b in zip(got, want_pcm, strict=True)]
    worst_lsb = max(np.abs(pcm(a) - pcm(b)).max() for a, b in zip(got, want_pcm, strict=True))
    replay = oracle("hift_float")  # the reference's vocoder replayed on its recorded mels and seed
    self_lsb = max(np.abs(pcm(pcm_bytes(r["speech"])) - pcm(b)).max() for r, b in zip(replay, want_pcm, strict=True))
    self_off = [int((pcm(pcm_bytes(r["speech"])) != pcm(b)).sum()) for r, b in zip(replay, want_pcm, strict=True)]
    verdict.check([len(a) for a in got] == [len(b) for b in want_pcm] and worst_lsb <= HIFT_NONDET_LSB,
                  f"one-request driver: PCM within {HIFT_NONDET_LSB} LSB of the recording (max {worst_lsb}; samples "
                  f"off per window {off}); the reference's own replay is {self_lsb} LSB off ({self_off})")
    det = deterministic_hift(model, rec)
    if det is None:
        print("hift_float_det.pt missing: run hift_float.py with HIFT_DETERMINISTIC=1 for the bit-exact vocoder check")
    else:
        verdict.check(det, "under deterministic cuDNN the vocoder (speech and source, every window) is bit-exact "
                      "against the reference's replay")
    scale = log_spectral_distance(b"".join(one_request(model, voice, windows, seed=1)[1]), b"".join(want_pcm))
    print(f"yardstick: the one-request driver with HiFT drawing from seed 1, log-spectral distance {scale:.3f} dB")

    s_mels, s_pcm = served(model, voice, windows)
    worst_mel = max((a - b).abs().max().item() for a, b in zip(s_mels, want_mels, strict=True))
    lens_ok = [len(a) for a in s_pcm] == [len(b) for b in want_pcm]
    lsd = log_spectral_distance(b"".join(s_pcm), b"".join(want_pcm))
    ref16 = oracle("token2wav_stream_fp16")  # the reference's own float16 token2wav
    ref16_mel = max((a - b).abs().max().item() for a, b in zip(ref16["window_mels"], want_mels, strict=True))
    ref16_lsd = log_spectral_distance(b"".join(ref16["window_pcm"]), b"".join(want_pcm))
    print(f"yardstick: the reference in float16, mel max abs {ref16_mel:.3e}, log-spectral distance {ref16_lsd:.3f} dB")
    print(f"served (fp16 DiT): mel max abs {worst_mel:.3e}, log-spectral distance {lsd:.3f} dB")
    verdict.check(worst_mel <= ref16_mel, f"served path: flow mel no further from the oracle than the reference's "
                  f"own float16 run ({worst_mel:.3e} vs {ref16_mel:.3e})")
    verdict.check(lens_ok and lsd <= SPREAD * ref16_lsd,
                  f"served path: window lengths match and the audio is within {SPREAD}x the reference's float16 "
                  f"run's log-spectral distance ({lsd:.3f} vs {ref16_lsd:.3f} dB)")
    _, v_pcm = served(model, voice, windows, vocode_mels=want_mels)
    v_worst = max(np.abs(pcm(a) - pcm(b)).max() for a, b in zip(v_pcm, want_pcm, strict=True))
    verdict.check(v_worst <= HIFT_NONDET_LSB, f"served vocoder on the recorded mels: PCM within {HIFT_NONDET_LSB} "
                  f"LSB (max {v_worst})")

    # negative control: MiniCPM-o's knobs on the same codes
    control = load_token2wav(str(CKPT / "token2wav"), DEVICE, noise=rec["rand_noise"], streaming=Token2WavStreaming(
        first_window_zeros=False))
    c_mels, c_pcm = one_request(control, control.prepare_voice(prompt), windows)
    c_worst = max((a - b).abs().max().item() for a, b in zip(c_mels, want_mels, strict=True))
    c_pcm_worst = max(np.abs(pcm(a) - pcm(b)).max() for a, b in zip(c_pcm, want_pcm, strict=True))
    first_bad = next(k for k, (a, b) in enumerate(zip(c_mels, want_mels, strict=True)) if not torch.equal(a, b))
    print(f"control (silence lookahead, conformer cut): mel max abs {c_worst:.3e}, pcm max diff {c_pcm_worst}, "
          f"first differing window {first_bad}")
    verdict.check(c_pcm_worst > CONTROL_MIN,
                  f"negative control (MiniCPM-o's knobs) diverges: pcm max diff {c_pcm_worst}")

    # the M* voice prompt encoder on the clip as the model serves it, against the reference's
    # prompt pieces: the mel from the clip's own rate must be exact; a token past a rounding
    # near-tie may flip
    from mstar.model.step_audio2.step_audio2_model import _load_voice_clip

    model.voice_encoder.cpu()
    mine = model.voice_encoder(*_load_voice_clip(ASSETS / "default_female.wav"))
    differ = int((mine.tokens.cpu() != prompt.tokens.cpu()).sum()) if mine.tokens.shape == prompt.tokens.shape else -1
    xvec = (mine.spk_emb.cpu() - prompt.spk_emb.cpu()).abs().max().item()
    print(f"voice encoder: tokens {tuple(mine.tokens.shape)} vs {tuple(prompt.tokens.shape)}, {differ} differ; "
          f"x-vector max abs {xvec:.3e}")
    verdict.check(mine.mel.shape == prompt.mel.shape and torch.equal(mine.mel.cpu(), prompt.mel.cpu()),
                  "voice encoder: the prompt mel, taken from the clip's own rate, is bit-exact")
    verdict.check(0 <= differ <= VOICE_TOKEN_FLIPS and xvec < 1e-4,
                  f"voice encoder: tokens agree but for {differ} (at most {VOICE_TOKEN_FLIPS}), x-vector to 1e-4")
    verdict.finish()


if __name__ == "__main__":
    main()

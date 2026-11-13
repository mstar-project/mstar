"""MiniCPM-o token2wav: window arithmetic, per-request state layout and truncation, and the
streaming flow/vocoder on random weights (CPU).

The oracle for the batched path is the reference's one-request driver (upstream's
``Token2wav.stream``) over the same modules, defined here. Its truncation test replays the
reference's own cache bookkeeping (the 25 Hz conformer cache tiled twice, both 50 Hz caches
cut to the prompt's frames plus the last 100) on labelled entries and checks the state holds
the same entries. Loading the real assets skips unless MINICPM_O_CKPT points at a MiniCPM-o 4.5
snapshot; bit-exactness against the reference lives in the parity harness, not here.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from collections.abc import Callable
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import StepContext
from mstar.engine.state import ArrayStep, Segment, SubmoduleStep
from mstar.engine.state.adapters import StateSetup, _resolve, build_handle, build_specs, state_steps
from mstar.model.components.token2wav.flow import FlowKV
from mstar.model.components.token2wav.keys import Token2WavKeys
from mstar.model.components.token2wav.model import (
    CACHE_FAMILIES,
    DIT_COMPUTE_DTYPE,
    HIFT_MEL_CACHE,
    HIFT_TAIL,
    KEEP_RECENT,
    LAST_WINDOW_TOKENS,
    SAMPLES_PER_FRAME,
    SLOT_PARTS,
    UP_RATE,
    WINDOW,
    CacheCapacity,
    SineGen2Source,
    Token2Wav,
    Token2WavState,
    Token2WavVoice,
    VoiceBank,
    excitation_samples,
    fade_in_out,
    window_frames,
    window_tokens,
)
from mstar.model.components.token2wav.state import check_token2wav_slots, token2wav_kernel_settings, token2wav_state
from mstar.model.components.token2wav.submodule import Token2WavSubmodule
from mstar.model.components.token2wav.voice_prompt import VoicePrompt
from mstar.model.submodule_base import InputMetadata, ModelInputsFromEngine, engine_step

T2W_STATE = {"conv": "t2w_conv", "vocoder": "t2w_vocoder", "tail": "t2w_tail"}
T2W_KV = {"enc1": "t2w_enc1_kv", "enc2": "t2w_enc2_kv", "dit": "t2w_dit_kv"}
T2W_ATTN = {"enc1": "t2w_enc1_attn", "enc2": "t2w_enc2_attn", "dit": "t2w_dit_attn"}
T2W_KEYS = Token2WavKeys(codes="tts_code", state=T2W_STATE, request="t2w_request", kv=T2W_KV, attn=T2W_ATTN)
NODE = "Token2Wav"
EOS = 6561

# ---------------------------------------------------------------------------
# The reference's one-request driver (upstream's ``Token2wav.stream``) over the served
# modules, on a request's own ``Token2WavState``
# ---------------------------------------------------------------------------


def reference_windows(codes: list[int]) -> list[tuple[list[int], bool]]:
    """The recorder's driver loop, verbatim: ``(window, last)`` per ``stream`` call."""
    buf = [4218] * 3 + codes
    out = []
    while len(buf) >= 28:
        out.append((buf[:28], False))
        buf = buf[25:]
    out.append((buf, True))
    return out


def window_samples(num_tokens: int, last: bool, first: bool) -> int:
    """Samples ``stream`` returns for a window."""
    if first:
        return window_frames(num_tokens, last) * SAMPLES_PER_FRAME
    total = (HIFT_MEL_CACHE + window_frames(num_tokens, last)) * SAMPLES_PER_FRAME
    return total if last else total - HIFT_TAIL


def truncate(state: Token2WavState) -> None:
    """The reference's bound: past ``2P + 100`` frames the 50 Hz caches keep their first
    ``2P`` entries and last 100. The 25 Hz cache follows the reference's tiled layout,
    which in effect keeps its first ``P + 50``."""
    limit = state.prompt_frames + KEEP_RECENT
    if state.dit_len > limit:
        n = state.dit_len
        state.dit_kv[..., state.prompt_frames:limit, :] = state.dit_kv[..., n - KEEP_RECENT:n, :].clone()
        state.dit_len = limit
    if state.enc_len2 > limit:
        n2, n1 = state.enc_len2, state.enc_len1
        assert n2 == UP_RATE * n1
        kept = list(range(state.prompt_frames)) + list(range(n2 - KEEP_RECENT, n2))
        src = [i % n1 for i in kept[: limit // UP_RATE]]
        if src != list(range(len(src))):
            idx = torch.tensor(src, device=state.enc_kv1.device)
            state.enc_kv1[..., : len(src), :] = state.enc_kv1.index_select(3, idx)
        state.enc_len1 = len(src)
        state.enc_kv2[..., state.prompt_frames:limit, :] = state.enc_kv2[..., n2 - KEEP_RECENT:n2, :].clone()
        state.enc_len2 = limit


def new_state(model: Token2Wav, voice: Token2WavVoice, capacity: CacheCapacity | None = None) -> Token2WavState:
    """A request's state, starting from its voice's, in a slot that may be larger."""
    state = Token2WavState.allocate(capacity or CacheCapacity(voice.prompt.tokens.shape[1]), model.speech_window.device)
    for f in dataclasses.fields(state):
        src, dst = getattr(voice.initial, f.name), getattr(state, f.name)
        if isinstance(dst, torch.Tensor):
            dst[tuple(slice(0, n) for n in src.shape)].copy_(src)
        else:
            setattr(state, f.name, src)
    return state


@torch.inference_mode()
def flow_chunk(model: Token2Wav, state: Token2WavState, voice: Token2WavVoice, tokens: torch.Tensor,
               last: bool) -> torch.Tensor:
    """``tokens [1, n]`` -> mel ``[1, 80, window_frames(n, last)]``; advances the caches."""
    mel = model.flow(tokens, voice.spk, None, last, state.enc_cnn, state.dit_cnn, state.kv_caches())
    frames = mel.shape[2]
    state.enc_len1 += frames // UP_RATE
    state.enc_len2 += frames
    state.dit_len += frames
    truncate(state)
    return mel


@torch.inference_mode()
def vocode_chunk(model: Token2Wav, state: Token2WavState, chunk_mel: torch.Tensor, last: bool,
                 noise=None, generator: torch.Generator | None = None) -> torch.Tensor:
    """Mel chunk -> this call's samples ``[1, N]``, with the previous call's last 8 mel
    frames re-vocoded in front and cross-faded with its held-back tail."""
    first = state.calls == 0
    mel = chunk_mel if first else torch.cat([state.hift_mel, chunk_mel], dim=2)
    if noise is None:
        noise = model.hift.draw_noise(mel, generator)
    speech, source = model.hift.vocode(mel, cache_source=None if first else state.hift_source, noise=noise)
    if not first:
        speech = fade_in_out(speech, state.hift_speech, model.speech_window)
    state.hift_mel.copy_(mel[..., -HIFT_MEL_CACHE:])
    state.hift_source.copy_(source[:, :, -HIFT_TAIL:])
    state.hift_speech.copy_(speech[:, -HIFT_TAIL:])
    state.calls += 1
    if last:
        return speech
    if first:
        return torch.cat([speech.new_zeros(1, HIFT_TAIL), speech[:, :-HIFT_TAIL]], dim=1)
    return speech[:, :-HIFT_TAIL]


def stream(model: Token2Wav, state: Token2WavState, voice: Token2WavVoice, tokens: list[int], last: bool,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """One ``Token2wav.stream`` call."""
    mel = flow_chunk(model, state, voice, torch.tensor([tokens], dtype=torch.int32, device=model.speech_window.device),
                     last)
    return vocode_chunk(model, state, mel, last, generator=generator)


class _Window(NamedTuple):
    """One window's open step: what ``window_flow`` and the stages after it take, and its commit."""

    slots: dict
    kv: FlowKV
    voices: torch.Tensor
    fresh: torch.Tensor
    noise_offsets: torch.Tensor
    commit: Callable[[], None]


def _flow(model: Token2Wav, bank: VoiceBank, w: _Window, tokens: torch.Tensor, last: bool = False) -> torch.Tensor:
    return model.window_flow(w.slots["conv"], bank, w.voices, w.fresh, tokens, w.kv, w.noise_offsets, last)


def _declared(voices: dict, dit_dtype: torch.dtype) -> list:
    """Token2wav's state for ``voices``, with the DiT's fields in ``dit_dtype``."""
    tokens = [voice.prompt.tokens.shape[1] for voice in voices.values()]

    def cast(array):
        tensors = {name: dataclasses.replace(t, dtype=dit_dtype) if t.dtype == DIT_COMPUTE_DTYPE else t
                   for name, t in array.entry.tensors.items()}
        return dataclasses.replace(array, entry=dataclasses.replace(array.entry, tensors=tensors))

    return [cast(array) for array in token2wav_state(T2W_KEYS, NODE, (min(tokens) / 25, max(tokens) / 25))]


class _Served:
    """A token2wav node over the state it declares for ``voices``, built as the engine builds it. ``window`` opens
    one window's step for the model's stages, and ``step`` runs one as the engine does, keeping it as ``last_step``.
    ``dit_dtype``: the DiT's dtype, float32 unless a test runs it as served."""

    def __init__(self, model, voices: dict, slots: int, device="cpu", dit_dtype: torch.dtype = torch.float32):
        arrays = _declared(voices, dit_dtype)
        info = EngineResourceInfo(device=torch.device(device), kv_dtype=torch.float32)
        specs = build_specs(arrays, kernel_settings=token2wav_kernel_settings(T2W_KEYS, slots + 1))
        self.resources = {spec.resource_key: build_resource(spec, info) for spec in specs}
        self.setup = StateSetup()
        resolved = _resolve(arrays, self.setup)
        self.handles = {array.name: build_handle(array, self.resources, resolved, self.setup) for array in arrays}
        self.node = Token2WavSubmodule(model, T2W_KEYS, voices, eos_code=EOS, dit_dtype=dit_dtype)
        self.node.bind_node_state(self.handles)
        self.bank = self.node.bank
        self.runner = StepRunner(self.resources)
        self.written: dict = {}
        self.voice: dict = {}

    def open(self, rid, voice: int = 0) -> None:
        """The engine's add_request, once per request."""
        if rid not in self.voice:
            self.voice[rid], self.written[rid] = voice, 0
            self.runner.ingest_request(rid)
            self.setup.host_fields.open(rid)

    def slot(self, rid) -> dict[str, torch.Tensor]:
        """``rid``'s slot of each slot Value, block by block."""
        out = {}
        for key in T2W_STATE.values():
            pool = self.resources[key]
            index = pool._slots[rid]["main"].index
            out.update({name: pool.block(name, 0)[index] for name in pool.config.blocks})
        return out

    def window(self, rids, num_tokens: int, last: bool, voices: list[int] | None = None, parts=("conv",)) -> _Window:
        """One window of ``rids`` (each in ``voices``, the bank's first by default), opened as a step that lists the
        slot ``parts`` its stages write and the caches."""
        for rid, voice in zip(rids, voices or [0] * len(rids), strict=True):
            self.open(rid, voice)
        caches = self.node._cache_steps(
            list(rids), [num_tokens] * len(rids), [last] * len(rids), keep=not last,
            voices={rid: self.voice[rid] for rid in rids},
        )
        step = state_steps(SubmoduleStep(
            segments=[Segment(request_id=rid, label="main", span=1) for rid in rids],
            state={**{T2W_STATE[part]: ArrayStep() for part in parts}, **caches},
        ), self.handles)
        step.set_ctx(StepContext(request_ids=list(rids), graph_walk="t2w", slot=0, capture=False))
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        step.begin_calls()

        def commit():
            step.check_calls()
            self.runner.commit(step)
            step.commit_calls()
            step.end_calls()
            for rid in rids:
                self.written[rid] += window_tokens(num_tokens, last)

        dit, device = CACHE_FAMILIES["dit"], self.bank.spk.device
        offsets = [dit.retained(dit.rate * self.bank.prompt_tokens[self.voice[rid]], dit.rate * self.written[rid])
                   for rid in rids]
        return _Window(
            slots=self.node.slots, kv=self.node._flow_kv(),
            voices=torch.tensor([self.voice[rid] for rid in rids], device=device),
            fresh=torch.tensor([self.written[rid] == 0 for rid in rids], device=device),
            noise_offsets=torch.tensor(offsets, device=device), commit=commit,
        )

    def retained(self, name: str, rid) -> torch.Tensor:
        """The keys|values ``rid`` would attend next in ``name``'s cache, ``[layers, rows, H, n, 2d]``."""
        from mstar.engine.resources.kv.bounded.layout import READ_FROM_SOURCE, row_layout

        family, kv, handle = CACHE_FAMILIES[name], self.resources[T2W_KV[name]], self.handles[T2W_KV[name]]
        source_len = family.rate * self.bank.prompt_tokens[self.voice[rid]]
        layout = row_layout(source_len, family.rate * self.written[rid], 0, kv.config.retention_for(source_len),
                            kv.config.sink_capacity, family.reverse_step_order)
        slot = kv._requests[rid].slot
        parts = []
        for from_source, (start, count) in zip(READ_FROM_SOURCE, layout.reads, strict=True):
            buf = handle._starts[self.voice[rid]] if from_source else kv._cache[:, slot]
            parts.append(buf[..., start:start + count, :])
        # the array holds the DiT's guidance rows as heads
        return torch.cat(parts, dim=-2).unflatten(2, (family.rows, family.num_heads)).flatten(1, 2)

    def prepare(self, rid, codes: torch.Tensor, start_offset: int, final: bool = False, seed: int = 0):
        """What the stream hands the node for ``rid``'s chunk at ``start_offset``."""
        self.open(rid)
        info = SimpleNamespace(rid_handle=rid, step_metadata={"voice": self.bank.names[self.voice[rid]]},
                               random_seed=seed)
        chunk = InputMetadata(stream_chunks={"tts_code": SimpleNamespace(start_offset=start_offset)})
        return self.node.prepare_inputs("t2w_chunk", info, {"tts_code": [codes]}, is_final_stream_chunk=final,
                                        input_metadata=chunk)

    def leave(self, rid, inputs) -> None:
        """Apply the host fields ``rid``'s window leaves at its commit, without running it."""
        declared = self.node.declare_step("t2w_chunk", [rid], [inputs]).state[T2W_KEYS.request]
        self.setup.host_fields.commit(self.handles[T2W_KEYS.request].array, declared, [rid])

    def regions(self, captured=()) -> dict:
        """The node's piecewise regions as the engine runs them: its eager ones, and ``captured``, warmed up and
        captured on CPU at one and two rows."""
        from test_fastconformer import _captured_on_cpu

        from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner

        configs = self.node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)
        runners = {
            label: PiecewiseAcceleratorGraphRunner(
                label=label, config=config, resources=self.resources, step_runner=self.runner,
                device=torch.device("cpu"), autocast_dtype=None, num_slots=1, node_name=label,
                state_handles=self.node.state_handles,
            )
            for label, config in configs.items() if not config.capture_batch_sizes
        }
        for label in captured:
            sizes = [n for n in (1, 2) if n in configs[label].capture_batch_sizes]
            runners[label] = _captured_on_cpu(configs[label], label, self.resources, self.runner, bs=sizes,
                                              node=self.node)
        return runners

    def step(self, rids, inputs, regions) -> dict:
        """One engine step over ``rids``: declare, admit and plan, open the handles, forward, check, commit."""
        step = self.last_step = engine_step(self.node, "t2w_chunk", list(rids), inputs)
        step.set_ctx(StepContext(request_ids=list(rids), graph_walk="t2w_chunk", slot=0, capture=False))
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        step.begin_calls()
        try:
            engine_inputs = ModelInputsFromEngine(
                request_ids=list(rids), per_request_info={}, resources={}, piecewise_runners=regions, step=step,
            )
            out = self.node.forward_batched("t2w_chunk", engine_inputs,
                                            **self.node.preprocess("t2w_chunk", engine_inputs, inputs))
            step.check_calls()
            self.runner.commit(step)
            step.commit_calls()
        finally:
            step.end_calls()
        for rid, inp in zip(rids, inputs, strict=True):
            codes = inp.tensor_inputs["codes"].shape[1]
            self.written[rid] += window_tokens(codes, inp.kwargs["last"]) if codes else 0
        return out


def _tiny_voice(model: Token2Wav, tokens: int = 6) -> Token2WavVoice:
    return model.prepare_voice(VoicePrompt(
        tokens=torch.randint(0, 6561, (1, tokens), dtype=torch.int32), spk_emb=torch.randn(1, 192),
        mel=torch.randn(1, 2 * tokens, 80),
    ))


CKPT = os.environ.get("MINICPM_O_CKPT")
needs_ckpt = pytest.mark.skipif(not CKPT, reason="set MINICPM_O_CKPT to a MiniCPM-o 4.5 snapshot")


@pytest.mark.parametrize("n", [0, 1, 24, 25, 26, 50, 127, 269, 300])
def test_windows_voice_every_code_once(n):
    windows = reference_windows(list(range(n)))
    # every code is synthesised exactly once: non-last windows emit 25 tokens' worth
    frames = sum(window_frames(len(w), last) for w, last in windows)
    assert frames == 2 * (n + 3)
    # the first window is delayed by the held-back tail, which the last one releases
    samples = sum(window_samples(len(w), last, i == 0) for i, (w, last) in enumerate(windows))
    assert samples == frames * 480 + (HIFT_TAIL if len(windows) > 1 else 0)


def test_capacity_and_allocation():
    cap = CacheCapacity(151)
    assert (cap.enc_25hz, cap.frames_50hz) == (151 + 77, 302 + 154)
    s = Token2WavState.allocate(cap, "cpu")
    assert s.enc_cnn.shape == (1, 512, 6)
    assert s.enc_kv1.shape == (6, 1, 8, 228, 128)
    assert s.enc_kv2.shape == (4, 1, 8, 456, 128)
    assert s.dit_cnn.shape == (10, 16, 2, 1024, 2)
    assert s.dit_kv.shape == (10, 16, 2, 8, 456, 128)
    assert s.hift_mel.shape == (1, 80, 8) and s.hift_source.shape == (1, 1, HIFT_TAIL)
    assert s.hift_speech.shape == (1, HIFT_TAIL)


def _labels(start: int, n: int) -> torch.Tensor:
    return torch.arange(start, start + n, dtype=torch.float32)


@pytest.mark.parametrize("p", [10, 30, 50, 151])
def test_truncation_matches_reference_bookkeeping(p):
    """Entries are labelled by when they were written; after every window the slot's valid
    regions must hold the labels the reference's tiled/concatenated caches hold."""
    two_p = 2 * p
    windows = [(28, False)] * 9 + [(19, True)]
    cap = CacheCapacity(p)
    # the reference's view: 1-D label tensors along time
    ref_conf1 = _labels(0, p)
    ref_tiled = torch.cat([ref_conf1, ref_conf1])  # conformer_att_cache[:6] along time
    ref_conf2 = _labels(10_000, two_p)
    ref_est = _labels(20_000, two_p)
    # ours: one channel carries the label
    s = Token2WavState.allocate(cap, "cpu")
    s.prompt_frames, s.enc_len1, s.enc_len2, s.dit_len = two_p, p, two_p, two_p
    s.enc_kv1[..., :p, :] = ref_conf1[:, None]
    s.enc_kv2[..., :two_p, :] = ref_conf2[:, None]
    s.dit_kv[..., :two_p, :] = ref_est[:, None]
    label = 100_000
    for n_tok, last in windows:
        t = n_tok if last else n_tok - 3
        new1, new2, new_est = _labels(label, t), _labels(label + 1000, 2 * t), _labels(label + 2000, 2 * t)
        label += 10_000
        # reference: encoder.forward_chunk + solve_euler_chunk + stream's truncation
        c1 = ref_tiled[: ref_tiled.shape[0] // 2]
        new_c1 = torch.cat([c1, new1])
        ref_conf2 = torch.cat([ref_conf2, new2])
        ref_tiled = torch.cat([new_c1, new_c1])
        ref_est = torch.cat([new_est, ref_est])
        if ref_est.shape[0] > two_p + KEEP_RECENT:
            ref_est = torch.cat([ref_est[:two_p], ref_est[-KEEP_RECENT:]])
        if ref_tiled.shape[0] > two_p + KEEP_RECENT:
            ref_tiled = torch.cat([ref_tiled[:two_p], ref_tiled[-KEEP_RECENT:]])
            ref_conf2 = torch.cat([ref_conf2[:two_p], ref_conf2[-KEEP_RECENT:]])
        # ours: what the attention layers write, then the state's truncation
        s.enc_kv1[..., s.enc_len1:s.enc_len1 + t, :] = new1[:, None]
        s.enc_kv2[..., s.enc_len2:s.enc_len2 + 2 * t, :] = new2[:, None]
        old = s.dit_kv[..., : s.dit_len, :].clone()
        s.dit_kv[..., : s.dit_len + 2 * t, :] = torch.cat([new_est[:, None].expand(*old.shape[:-2], -1, 128), old], -2)
        s.enc_len1 += t
        s.enc_len2 += 2 * t
        s.dit_len += 2 * t
        truncate(s)
        assert s.enc_len2 == ref_conf2.shape[0] == ref_tiled.shape[0]
        assert torch.equal(s.enc_kv1[0, 0, 0, : s.enc_len1, 0], ref_tiled[: ref_tiled.shape[0] // 2])
        assert torch.equal(s.enc_kv2[0, 0, 0, : s.enc_len2, 0], ref_conf2)
        assert torch.equal(s.dit_kv[3, 5, 1, 2, : s.dit_len, 7], ref_est)
        assert s.enc_len1 <= cap.enc_25hz and s.enc_len2 <= cap.frames_50hz and s.dit_len <= cap.frames_50hz


def test_sine_source_draw_order():
    """The excitation's draws come in the reference's order (phase, noise laid out
    harmonic-major, then a discarded field), so a seeded generator reproduces them and ends
    in the same state."""
    from mstar.model.chatterbox.config import S3GenHiFTConfig

    src = SineGen2Source(S3GenHiFTConfig())
    g = torch.Generator().manual_seed(7)
    noise = src.draw_noise(2, 960, torch.float32, "cpu", g)
    ref = torch.Generator().manual_seed(7)
    phase = torch.rand(2, 9, generator=ref)
    sines = torch.randn(2, 9, 960, generator=ref)
    torch.randn(2, 960, 1, generator=ref)
    assert torch.equal(noise.phase, phase)
    assert torch.equal(noise.harmonic, sines.transpose(1, 2))
    assert torch.equal(g.get_state(), ref.get_state())


def test_fade_in_out_blends_in_float64():
    window = torch.from_numpy(__import__("numpy").hamming(8))
    a, b = torch.ones(1, 6), torch.full((1, 6), 2.0)
    out = fade_in_out(a, b, window)
    assert out.dtype == torch.float32
    want = (a[..., :4].double() * window[:4] + b[..., -4:].double() * window[4:]).float()
    assert torch.equal(out[..., :4], want) and torch.equal(out[..., 4:], a[..., 4:])


@pytest.fixture(scope="module")
def random_token2wav():
    torch.manual_seed(0)
    model = Token2Wav().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.02)
        model.flow.decoder.rand_noise.normal_()
    return model


def test_stream_shapes_and_capacity_independence(random_token2wav):
    """A request's output depends only on its valid cache entries: a larger slot (as a
    fixed-size pool would hand out) gives identical results, and lengths follow the windows."""
    model = random_token2wav
    p = 6
    prompt = VoicePrompt(
        tokens=torch.randint(0, 6561, (1, p), dtype=torch.int32),
        spk_emb=torch.randn(1, 192),
        mel=torch.randn(1, 2 * p, 80),
    )
    voice = model.prepare_voice(prompt)
    assert (voice.initial.enc_len1, voice.initial.enc_len2, voice.initial.dit_len) == (p, 2 * p, 2 * p)
    windows = reference_windows(list(range(100, 130)))
    outs = []
    for extra in (0, 9):
        state = new_state(model, voice, CacheCapacity(p + extra))
        gen = torch.Generator().manual_seed(3)
        wavs = []
        for i, (win, last) in enumerate(windows):
            wavs.append(stream(model, state, voice, win, last, generator=gen))
            assert wavs[-1].shape == (1, window_samples(len(win), last, i == 0))
            assert state.dit_len <= 2 * p + KEEP_RECENT or last
        outs.append(torch.cat(wavs, dim=1))
    assert torch.equal(outs[0], outs[1])


def _assert_same_frames(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-2) -> None:
    """``[..., L, 2d]`` caches holding the same L frames in any order: every frame of each
    has a match in the other (frames are ~10 apart; tol is summation-order drift)."""
    assert a.shape == b.shape
    a, b = a.flatten(0, -3), b.flatten(0, -3)
    for i in range(a.shape[0]):
        d = torch.cdist(a[i], b[i])
        worst = max(d.min(dim=1).values.max().item(), d.min(dim=0).values.max().item())
        assert worst < tol, (i, worst)


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
])
def test_batched_windows_on_pool(random_token2wav, device):
    """Every window of an utterance batched on pool slots, the DiT cache wrapping several
    times: one row matches ``stream`` (samples; the slot against the reference state after
    every window, the DiT caches as a multiset since only that enters attention), to the
    summation order of attending the same keys in another order; two rows agree with it to
    the batch-size dependence of the GEMMs."""
    import copy

    from mstar.model.chatterbox.components.s3gen_hift import HiFTNoise

    model = copy.deepcopy(random_token2wav).to(device)
    # The CUDA path's attention kernel multiplies in TF32, as served: the DiT's conv caches
    # drift from the reference state by up to ~1.1e-3 absolute, near-zero entries included.
    tol = dict(rtol=1e-4, atol=1e-4) if device == "cpu" else dict(rtol=1e-3, atol=2e-3)
    p = 50
    torch.manual_seed(1)
    voice = model.prepare_voice(VoicePrompt(
        tokens=torch.randint(0, 6561, (1, p), dtype=torch.int32).to(device),
        spk_emb=torch.randn(1, 192).to(device), mel=torch.randn(1, 2 * p, 80).to(device),
    ))
    kv_steps = _Served(model, {"v": voice}, 3, device)
    bank = kv_steps.bank
    codes = [torch.randint(0, 6561, (215,)).tolist() for _ in range(2)]
    wins = [reference_windows(c) for c in codes]  # 9 windows each, the last of 18 tokens
    gen = torch.Generator().manual_seed(9)

    def noise(rows, calls, win, last):
        frames = window_frames(len(win), last) + (0 if calls == 0 else 8)
        nz = model.hift.m_source.draw_noise(rows, frames * 480, torch.float32, "cpu", gen)
        return HiFTNoise(phase=nz.phase.to(device), harmonic=nz.harmonic.to(device))

    def check_slot(rid, st):
        from mstar.model.components.token2wav.flow import DIT_DEPTH, N_TIMESTEPS

        # the conformer attends its caches in order; the DiT's only as a set
        torch.testing.assert_close(kv_steps.retained("enc1", "r0"), st.enc_kv1[..., :st.enc_len1, :], **tol)
        torch.testing.assert_close(kv_steps.retained("enc2", "r0"), st.enc_kv2[..., :st.enc_len2, :], **tol)
        _assert_same_frames(kv_steps.retained("dit", "r0").unflatten(0, (N_TIMESTEPS, DIT_DEPTH)),
                            st.dit_kv[..., :st.dit_len, :], tol=1e-2 if device == "cpu" else 0.5)
        # the slot holds the rest without the state's batch axes
        for name, t in kv_steps.slot(rid).items():
            torch.testing.assert_close(t, getattr(st, name).reshape(t.shape), **tol, msg=name)

    ref = new_state(model, voice)
    mels, noises = [], []
    for i, (win, last) in enumerate(wins[0]):
        first = i == 0
        nz = noise(2, 0 if first else 1, win, last)
        row0 = HiFTNoise(phase=nz.phase[:1], harmonic=nz.harmonic[:1])
        mel = flow_chunk(model, ref, voice, torch.tensor([win], dtype=torch.int32, device=device), last)
        want = vocode_chunk(model, ref, mel, last, noise=row0)
        tokens = torch.tensor([win], dtype=torch.int32, device=device)
        w = kv_steps.window(["r0"], len(win), last, parts=SLOT_PARTS)
        got_mel = _flow(model, bank, w, tokens, last)
        out = model.window_vocode(w.slots["vocoder"], got_mel, first, last, row0)
        got = model.window_finish(w.slots["tail"], out["magnitude"], out["phase"], first, last)
        w.commit()
        torch.testing.assert_close(got_mel, mel, **tol)
        torch.testing.assert_close(got, want, **tol)
        if not last:
            check_slot("r0", ref)
        mels.append(mel)
        noises.append(nz)

    if device == "cuda":
        # as served: the DiT computes in DIT_COMPUTE_DTYPE (its caches' dtype) on its
        # fused kernels; its mel tracks the fp32 run above
        kv16 = _Served(model, {"v": voice}, 3, device, dit_dtype=DIT_COMPUTE_DTYPE)
        bank16 = kv16.bank
        for (win, last), want in zip(wins[0], mels, strict=True):
            w = kv16.window(["r0"], len(win), last)
            got = _flow(model, bank16, w, torch.tensor([win], dtype=torch.int32, device=device), last)
            w.commit()
            rel = ((got.float() - want).norm() / want.norm()).item()
            assert rel < 1e-3, rel  # 1.4e-4 measured; a broken kernel is far past it

    # The same utterance in two rows two windows apart, their full windows' flow in one
    # batch: rows at different positions (a first window beside a later one, then
    # rows on either side of the caches' bound) each match the one-row run.
    kv_steps = _Served(model, {"v": voice}, 3, device)
    bank = kv_steps.bank
    full = [win for win, last in wins[0] if not last]
    lag = 2
    for k in range(len(full) + lag):
        rows = [(r, k - off) for r, off in (("a", 0), ("b", lag)) if 0 <= k - off < len(full)]
        rids = [r for r, _ in rows]
        tokens = torch.tensor([full[w] for _, w in rows], dtype=torch.int32, device=device)
        w = kv_steps.window(rids, len(full[0]), False)
        mel = _flow(model, bank, w, tokens)
        w.commit()
        for i, (_, w) in enumerate(rows):
            torch.testing.assert_close(mel[i:i + 1], mels[w], **tol)


def test_batched_relative_positions_match_the_dense_path(random_token2wav):
    """The conformer's relative-position term on the bounded caches, at a magnitude that
    matters: ``random_token2wav``'s small weights leave it near zero, where a band read
    backwards (relative position ``-k`` for ``k``) still passes the test above."""
    import copy

    from mstar.model.components.token2wav.flow import RelPositionAttention

    model = copy.deepcopy(random_token2wav)
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for mod in model.flow.encoder.modules():
            if isinstance(mod, RelPositionAttention):
                mod._bands.clear()
                for t in (mod.linear_pos.weight, mod.pos_bias_u, mod.pos_bias_v):
                    t.copy_(torch.randn(t.shape, generator=gen))
    p = 12
    voice = model.prepare_voice(VoicePrompt(
        tokens=torch.randint(0, 6561, (1, p), dtype=torch.int32, generator=gen),
        spk_emb=torch.randn(1, 192, generator=gen), mel=torch.randn(1, 2 * p, 80, generator=gen),
    ))
    kv_steps = _Served(model, {"v": voice}, 2, "cpu")
    bank = kv_steps.bank
    ref = new_state(model, voice)
    # the token encoder's output: the DiT's small weights would hide it in the mel
    encoded = []
    hook = model.flow.encoder.register_forward_hook(lambda _m, _i, out: encoded.append(out.clone()))
    try:
        for win, last in reference_windows(torch.randint(0, 6561, (60,), generator=gen).tolist()):
            tokens = torch.tensor([win], dtype=torch.int32)
            flow_chunk(model, ref, voice, tokens, last)
            w = kv_steps.window(["r0"], len(win), last)
            _flow(model, bank, w, tokens, last)
            w.commit()
            want, got = encoded[-2:]
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    finally:
        hook.remove()


@needs_ckpt
def test_assets_load_completely():
    from mstar.model.components.token2wav.model import load_token2wav

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_token2wav(f"{CKPT}/assets/token2wav", device)
    import soundfile as sf

    wav, sr = sf.read(f"{CKPT}/assets/HT_ref_audio.wav", dtype="float32")
    assert sr == 16000
    prompt = model.voice_encoder(torch.from_numpy(wav))
    p = prompt.tokens.shape[1]
    assert prompt.mel.shape == (1, 2 * p, 80) and prompt.spk_emb.shape == (1, 192)
    voice = model.prepare_voice(prompt)
    state = new_state(model, voice)
    wav = stream(model, state, voice, [4218] * 3 + [100] * 25, last=False)
    assert wav.shape == (1, 50 * 480) and torch.isfinite(wav).all()


@needs_ckpt
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_served_float16_dit_tracks_float32_on_the_checkpoint():
    """The DiT as served, in DIT_COMPUTE_DTYPE on its fused kernels, against the fp32 DiT
    over an utterance's windows in three rows, on the real weights: the mel stays within
    0.2% of fp32's (0.05-0.12% measured here, 0.07-0.14% at batch 24; a 0.1% random
    perturbation of the fp32 mel gives 0.1%)."""
    import soundfile as sf

    from mstar.model.components.token2wav.model import DIT_COMPUTE_DTYPE, load_token2wav

    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_token2wav(f"{CKPT}/assets/token2wav", "cuda").requires_grad_(False)
    wav, _ = sf.read(f"{CKPT}/assets/HT_ref_audio.wav", dtype="float32")
    voice = model.prepare_voice(model.voice_encoder(torch.from_numpy(wav)))
    torch.manual_seed(0)
    codes = torch.randint(0, 6561, (3, 215)).tolist()
    wins = [reference_windows(c) for c in codes]
    rids = ["a", "b", "c"]
    mels = {}
    with torch.inference_mode():
        for dtype in (None, DIT_COMPUTE_DTYPE):
            kv_steps = _Served(model, {"v": voice}, 3, "cuda", dit_dtype=dtype or torch.float32)
            bank = kv_steps.bank
            mels[dtype] = []
            for w in range(len(wins[0])):
                last = wins[0][w][1]
                tokens = torch.tensor([ws[w][0] for ws in wins], dtype=torch.int32, device="cuda")
                w = kv_steps.window(rids, tokens.shape[1], last)
                mels[dtype].append(_flow(model, bank, w, tokens, last).float())
                w.commit()
    for w, (want, got) in enumerate(zip(mels[None], mels[DIT_COMPUTE_DTYPE], strict=True)):
        rel = ((got - want).norm(dim=(1, 2)) / want.norm(dim=(1, 2))).max().item()
        assert rel < 2e-3, (w, rel)


@needs_ckpt
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_batched_flow_matches_the_driver_on_the_real_weights():
    """The served (bounded-cache) flow against the one-request driver on MiniCPM-o's own
    weights and voice, where the conformer's relative-position term is large enough to
    show in the mel."""
    import soundfile as sf

    from mstar.model.components.token2wav.model import load_token2wav

    model = load_token2wav(f"{CKPT}/assets/token2wav", "cuda")
    wav, _ = sf.read(f"{CKPT}/assets/HT_ref_audio.wav", dtype="float32")
    voice = model.prepare_voice(model.voice_encoder(torch.from_numpy(wav)))
    kv_steps = _Served(model, {"v": voice}, 2, "cuda")
    bank = kv_steps.bank
    ref = new_state(model, voice)
    gen = torch.Generator().manual_seed(0)
    with torch.inference_mode():
        for win, last in reference_windows(torch.randint(0, 6561, (160,), generator=gen).tolist()):
            tokens = torch.tensor([win], dtype=torch.int32, device="cuda")
            want = flow_chunk(model, ref, voice, tokens, last)
            w = kv_steps.window(["r0"], len(win), last)
            got = _flow(model, bank, w, tokens, last)
            w.commit()
            # TF32 in the served attention kernel; a misplaced position term moves mels by > 1
            assert (got.float() - want.float()).abs().max().item() < 0.15


@pytest.mark.parametrize(("n", "stop"), [
    *((n, True) for n in [1, 21, 22, 23, 24, 25, 49, 99, 100, 230]), (0, False), (25, False), (50, False),
])
def test_served_windows_are_upstreams(random_token2wav, n, stop):
    """The TTS streams ``n`` codes and then its stop code, or none when it ends at its cap; the
    policy's chunks, through ``prepare_inputs``, are the reference's windows of the codes: every
    code voiced once and the same last window, also when the stop code ends a full window
    (``n = 25j + 24``) or a capped reply's codes do (``n = 25j``). Every
    chunk still runs, voicing nothing past the last window: a skipped step never completes the
    node, which then refuses the stream's final chunk. ``n = 0`` is a reply that ended at the
    LLM's prefill: the TTS never ran, and the stream's one chunk is an empty, last step that
    declares no row (so leases no slot)."""
    from mstar.model.components.token2wav.model import HOP, LEAD_SILENCE
    from mstar.streaming.chunk_policy import LeftContextChunkPolicy
    from mstar.streaming.stream_buffer import StreamBuffer

    eos = EOS
    served_node = _Served(random_token2wav, {"v": _tiny_voice(random_token2wav)}, 2)
    served_node.open(0)
    node = served_node.node
    info = SimpleNamespace(rid_handle=0, step_metadata={"voice": "v"}, random_seed=0)
    codes = torch.randint(0, eos, (n,)).tolist()

    buffer = StreamBuffer(request_id=0, edge_name="tts_code", from_partition="main",
                          policy=LeftContextChunkPolicy(chunk=HOP, left_context=LEAD_SILENCE))
    served = []
    for i, code in enumerate((codes + [eos] if stop else codes) + [None]):
        if code is None:
            buffer.signal_done()
        else:
            buffer.pre_read_register(f"t{i}")
            buffer.put(f"t{i}", torch.tensor([code], dtype=torch.int32))
        while buffer.has_chunk_ready():
            chunk = buffer.pop_chunk()
            data = chunk.data["data"]
            got = node.prepare_inputs(
                "t2w_chunk", info, {"tts_code": [data]} if data is not None else {},
                is_final_stream_chunk=chunk.is_final,
                input_metadata=InputMetadata(stream_chunks={"tts_code": chunk.info} if data is not None else {}),
            )
            assert got is not None
            if got.tensor_inputs["codes"].numel():
                served.append((got.tensor_inputs["codes"][0].tolist(), got.kwargs["last"]))
                assert got.kwargs["draw"] == (0, len(served) - 1), "a window's noise would be another window's"
                # what its commit leaves for the next chunk
                served_node.leave(0, got)
    assert got.kwargs["last"]
    if n:
        assert served == reference_windows(codes)
    else:
        step = node.declare_step("t2w_chunk", [0], [got])
        assert served == [] and not step.segments and not any(declared.segments for declared in step.state.values())


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
])
def test_rows_of_different_voices_share_a_batch(random_token2wav, device):
    """Two voices with prompts of different lengths, one row of each in every flow batch,
    a window apart: each row matches its own voice's one-request run."""
    import copy

    model = copy.deepcopy(random_token2wav).to(device)
    tol = dict(rtol=1e-4, atol=1e-4) if device == "cpu" else dict(rtol=1e-3, atol=1e-3)
    torch.manual_seed(3)
    voices = {
        name: model.prepare_voice(VoicePrompt(
            tokens=torch.randint(0, 6561, (1, p), dtype=torch.int32).to(device),
            spk_emb=torch.randn(1, 192).to(device), mel=torch.randn(1, 2 * p, 80).to(device),
        ))
        for name, p in (("short", 50), ("long", 64))
    }
    bank = VoiceBank.build(voices)
    full = [w for w, last in reference_windows(torch.randint(0, 6561, (180,)).tolist()) if not last]
    want = {}
    for name, voice in voices.items():
        ref = new_state(model, voice)
        want[name] = [flow_chunk(model, ref, voice, torch.tensor([w], dtype=torch.int32, device=device), False)
                      for w in full]
    kv_steps = _Served(model, voices, 3, device)
    lag = 1
    for k in range(len(full) + lag):
        rows = [(name, k - off) for name, off in (("short", 0), ("long", lag)) if 0 <= k - off < len(full)]
        tokens = torch.tensor([full[w] for _, w in rows], dtype=torch.int32, device=device)
        w = kv_steps.window(
            [name for name, _ in rows], len(full[0]), False, [bank.index(name) for name, _ in rows])
        mel = _flow(model, bank, w, tokens)
        w.commit()
        for i, (name, w) in enumerate(rows):
            torch.testing.assert_close(mel[i:i + 1], want[name][w], **tol)


def test_hift_noise_is_the_requests_own(random_token2wav):
    """A spoken reply's HiFT excitation comes from its own seed: a window vocodes the same
    alone and beside another request, whatever the global RNG drew in between."""
    model = random_token2wav
    torch.manual_seed(5)
    served = _Served(model, {"v": _tiny_voice(model)}, 3)
    regions = served.regions()
    codes = torch.randint(0, EOS, (25,), dtype=torch.int32)

    def first_window(seeds: dict[str, int]) -> torch.Tensor:
        rids = list(seeds)
        served.step(rids, [served.prepare(rid, codes, 0, seed=seed) for rid, seed in seeds.items()], regions)
        # the excitation it held back for the next window: the noise, nearly alone
        return torch.stack([served.slot(rid)["hift_source"] for rid in rids])

    alone = first_window({"a": 7})
    torch.randn(1000)
    batched = first_window({"b": 8, "c": 7})
    scale = alone.abs().max()
    assert (batched[1:] - alone).abs().max() < 1e-4 * scale
    assert (batched[:1] - alone).abs().max() > 0.1 * scale


def test_state_declares_a_value_per_slot_part_and_an_array_per_cache_family():
    """Every cache family gets its bounded KV under the model's key, one slot short of
    the slot Values, which also keep their sink slot."""
    from mstar.engine.resources.kv.bounded import BoundedKVSpec
    from mstar.engine.resources.recurrent.config import RecurrentStateSpec

    specs = build_specs(token2wav_state(T2W_KEYS, NODE, (3.0, 10.0)),
                        kernel_settings=token2wav_kernel_settings(T2W_KEYS, 9))
    state = {s.resource_key: s for s in specs if isinstance(s, RecurrentStateSpec)}
    kv = {s.resource_key: s for s in specs if isinstance(s, BoundedKVSpec)}
    assert set(state) == set(T2W_STATE.values()) and all(s.config.max_slots == 9 for s in state.values())
    assert set(kv) == {T2W_KV[name] for name in CACHE_FAMILIES}
    assert all(s.config.max_slots == 8 and s.nodes == {NODE} for s in kv.values())


@pytest.mark.parametrize(
    ("state", "kv", "ok"), [(None, None, True), (17, 16, True), (16, 16, False), (17, None, False)],
)
def test_slot_check_wants_one_more_slot_in_each_value_than_in_each_cache(state, kv, ok):
    resources = {}
    if state is not None:
        resources.update({key: {"max_slots": state} for key in T2W_STATE.values()})
    if kv is not None:
        resources.update({key: {"max_slots": kv} for key in T2W_KV.values()})
    if ok:
        check_token2wav_slots(T2W_KEYS, resources, 9, "cfg.yaml")
    else:
        with pytest.raises(ValueError, match="be one more than"):
            check_token2wav_slots(T2W_KEYS, resources, 9, "cfg.yaml")


def test_slot_values_of_different_sizes_are_refused():
    resources = {T2W_STATE["conv"]: {"max_slots": 10}}
    with pytest.raises(ValueError, match="must agree"):
        check_token2wav_slots(T2W_KEYS, resources, 9, "cfg.yaml")


# HiFT's excitation, from the request's seed and the window's index


def _kept_stream(model, seed: int, device) -> list:
    """Upstream's draws for one request: a generator kept from its seed draws a first window, two full ones, then
    each length a last window may have, every last one from the state the full ones left."""
    source, full = model.hift.m_source, window_frames(WINDOW, False)
    kept = torch.Generator(device=device).manual_seed(seed)
    draws = [((full, k == 0, k), source.draw_noise(1, excitation_samples(full, k == 0), torch.float32, device, kept))
             for k in range(3)]
    state = kept.get_state()
    for n in LAST_WINDOW_TOKENS:
        kept.set_state(state)
        frames = window_frames(n, True)
        draws.append(((frames, False, 3), source.draw_noise(1, excitation_samples(frames, False), torch.float32,
                                                            device, kept)))
    return draws


def _assert_window_noise_is_the_kept_streams(model, device) -> None:
    for seed in (0, 1234, 2**40 + 7):
        for (frames, first, k), want in _kept_stream(model, seed, device):
            got = model.window_noise(frames, first, [(seed, k)])
            assert torch.equal(got.phase, want.phase) and torch.equal(got.harmonic, want.harmonic), (
                f"window {k} of seed {seed} ({frames} frames) would draw from another place in its stream"
            )


def test_noise_from_the_seed_and_window_is_the_kept_generators(random_token2wav):
    _assert_window_noise_is_the_kept_streams(random_token2wav, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA generator's offset is the derivation on the device")
def test_noise_from_the_seed_and_window_is_the_kept_generators_on_the_device(random_token2wav):
    """On the device the draw moves the generator's offset, measured there, in place of drawing the earlier windows."""
    _assert_window_noise_is_the_kept_streams(copy.deepcopy(random_token2wav).cuda(), "cuda")


def test_a_window_drawn_again_draws_the_same_noise(random_token2wav):
    """A refused admit or an uncommitted forward runs a window again, which draws what it drew the first time."""
    model = random_token2wav
    first = model.window_noise(50, False, [(7, 4)])
    model.window_noise(50, False, [(8, 2)])
    again = model.window_noise(50, False, [(7, 4)])
    assert torch.equal(first.harmonic, again.harmonic), "another request's draw would have moved this one's"


def test_a_later_window_takes_its_voice_from_the_step_metadata(random_token2wav):
    """The worker keeps the step metadata of the partition's one step, the voice included, for every chunk the stream
    triggers, so a later window reads the voice there and the node keeps no host field for it."""
    model = random_token2wav
    served = _Served(model, {"short": _tiny_voice(model), "long": _tiny_voice(model, 9)}, 2)
    served.open("a", voice=1)
    codes = torch.randint(0, EOS, (60,), dtype=torch.int32)
    assert served.prepare("a", codes[22:50], 22).kwargs["voice"] == 1, "a later window would take another voice"


def test_a_short_window_before_the_last_is_refused(random_token2wav):
    """Window k's noise counts the draws of k full windows before it, so a shorter one would shift every later draw."""
    served = _Served(random_token2wav, {"v": _tiny_voice(random_token2wav)}, 2)
    with pytest.raises(RuntimeError, match="a window before the last holds 23 codes"):
        served.prepare("a", torch.arange(20), 0)


# The node over its declared state, as the engine runs it


def _reply_chunks(codes: torch.Tensor) -> list[tuple[torch.Tensor, int, bool]]:
    """The stream's chunks of a 60-code reply, ``(codes, start_offset, final)``: 25 new codes each after 3 of
    look-back, and what is left."""
    return [(codes[:25], 0, False), (codes[22:50], 22, False), (codes[47:], 47, True)]


def test_the_declared_cache_steps_are_the_hand_written_ones(random_token2wav):
    """A first, a full and a last window in two voices: each cache's step, as the engine builds it from what the
    node declares, has the spans and stream positions the node wrote by hand before it declared its state."""
    from mstar.engine.resources.kv.bounded import StreamPosition
    from mstar.model.components.token2wav.model import HOP, LEAD_SILENCE

    model = random_token2wav
    torch.manual_seed(2)
    served = _Served(model, {"short": _tiny_voice(model), "long": _tiny_voice(model, 9)}, 3)
    regions = served.regions()
    voices = {"a": 0, "b": 1}
    for rid, voice in voices.items():
        served.open(rid, voice)
    codes = torch.randint(0, EOS, (60,), dtype=torch.int32)
    for k, (chunk, start, final) in enumerate(_reply_chunks(codes)):
        served.step(list(voices), [served.prepare(rid, chunk, start, final) for rid in voices], regions)
        num_tokens = len(chunk) + (LEAD_SILENCE if k == 0 else 0)
        for name, family in CACHE_FAMILIES.items():
            step = served.last_step.steps[T2W_KV[name]]
            span = family.rate * window_tokens(num_tokens, final)
            assert list(step.segments) == [Segment(request_id=rid, label="main", span=span) for rid in voices], (
                f"{name}'s step would add another number of entries for window {k}"
            )
            assert step.positions == {
                rid: StreamPosition(family.rate * served.bank.prompt_tokens[voice], family.rate * HOP * k, source=voice)
                for rid, voice in voices.items()
            }, f"{name}'s rows would stand elsewhere in their voices' streams at window {k}"


def test_the_voices_caches_are_each_array_s_starts(random_token2wav):
    model = random_token2wav
    torch.manual_seed(2)
    voices = {"short": _tiny_voice(model), "long": _tiny_voice(model, 9)}
    served = _Served(model, voices, 2)
    for name, family in CACHE_FAMILIES.items():
        starts = served.handles[T2W_KV[name]]._starts
        for i, voice in enumerate(voices.values()):
            # [layers, rows, H, L, 2d], the guidance rows as heads of the array's one row
            source = family.source(voice).flatten(1, 2).unsqueeze(1)
            assert torch.equal(starts[i][..., :source.shape[-2], :], source), (
                f"{name}'s rows in voice {i} would begin from other entries than the voice's cache"
            )


def test_a_dit_cache_whose_count_moved_without_the_stream_is_refused(random_token2wav):
    """The flow's noise offset counts the DiT's kept entries from the window's stream offset. A cache whose committed
    count moved while the stream offset didn't (``reset_request`` without freeing zeroes it) would take another
    window's noise."""
    model = random_token2wav
    torch.manual_seed(4)
    served = _Served(model, {"v": _tiny_voice(model)}, 2)
    regions = served.regions()
    chunks = _reply_chunks(torch.randint(0, EOS, (60,), dtype=torch.int32))
    served.step(["a"], [served.prepare("a", *chunks[0])], regions)
    served.resources[T2W_KV["dit"]].reset_request("a")
    with pytest.raises(RuntimeError, match="would index another window's noise"):
        served.step(["a"], [served.prepare("a", *chunks[1])], regions)


def test_a_reply_through_the_node_s_regions_is_the_one_request_driver_s(random_token2wav):
    """Two replies a window apart through the node as the engine runs it, the full windows' flow and HiFT regions
    warmed up and captured and the rest eager. A step replays a region for its rows or runs two eagerly (a full
    window beside a last one), and each region steps the Values it writes inside the node's step. Each reply's
    samples are the one-request driver's on the same noise."""
    model = random_token2wav
    torch.manual_seed(4)
    voice = _tiny_voice(model)
    served = _Served(model, {"v": voice}, 3)
    full = window_frames(WINDOW, False)
    captured = [served.node.flow_region(WINDOW, False), served.node.hift_region(True, full, False),
                served.node.hift_region(False, full, False)]
    regions = served.regions(captured)
    seeds = {"a": 3, "b": 2**40 + 7}
    codes = {rid: torch.randint(0, EOS, (60,), dtype=torch.int32) for rid in seeds}
    got = {rid: [] for rid in seeds}
    for step in range(4):
        rows = [(rid, step - lag) for rid, lag in (("a", 0), ("b", 1)) if 0 <= step - lag < 3]
        inputs = [served.prepare(rid, *_reply_chunks(codes[rid])[k], seed=seeds[rid]) for rid, k in rows]
        out = served.step([rid for rid, _ in rows], inputs, regions)
        for rid, _ in rows:
            got[rid].append(out[rid][T2W_KEYS.audio][0])
    assert all(regions[label].calls for label in captured), "a full window would never have reached its region"
    for rid, seed in seeds.items():
        ref = new_state(model, voice)
        for k, (win, last) in enumerate(reference_windows(codes[rid].tolist())):
            mel = flow_chunk(model, ref, voice, torch.tensor([win], dtype=torch.int32), last)
            want = vocode_chunk(model, ref, mel, last, noise=model.window_noise(mel.shape[-1], k == 0, [(seed, k)]))
            torch.testing.assert_close(got[rid][k][None], want, rtol=1e-4, atol=1e-4,
                                       msg=f"reply {rid}'s window {k} isn't the one-request driver's")

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

import dataclasses
import os

import pytest
import torch

from mstar.model.components.token2wav.keys import Token2WavKeys
from mstar.model.components.token2wav.model import (
    HIFT_MEL_CACHE,
    HIFT_TAIL,
    KEEP_RECENT,
    SAMPLES_PER_FRAME,
    UP_RATE,
    CacheCapacity,
    SineGen2Source,
    Token2Wav,
    Token2WavState,
    Token2WavVoice,
    VoiceBank,
    fade_in_out,
    slot_layout,
    window_frames,
    window_tokens,
)
from mstar.model.components.token2wav.voice_prompt import VoicePrompt

T2W_STATE = "t2w_state"
T2W_KV = {"enc1": "t2w_enc1_kv", "enc2": "t2w_enc2_kv", "dit": "t2w_dit_kv"}
T2W_KEYS = Token2WavKeys(codes="tts_code", state=T2W_STATE, kv=T2W_KV)

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


class _CacheSteps:
    """Token2wav's resources (its state pool and its three attention caches) for requests
    in the voices of ``bank``, driven through admit / plan / commit one window at a time,
    as the engine does. ``dit_dtype``: the DiT caches' dtype, which it computes in (the
    bank's must match)."""

    def __init__(self, bank, slots: int, device, dit_dtype: torch.dtype | None = None):
        from mstar.engine.resources.kv.bounded.config import BoundedKVConfig
        from mstar.engine.resources.kv.bounded.manager import BoundedKVManager
        from mstar.engine.resources.recurrent.config import RecurrentBlockConfig, RecurrentStateConfig
        from mstar.engine.resources.recurrent.pool import RecurrentStatePool
        from mstar.model.components.token2wav.model import CACHE_FAMILIES, CacheCapacity

        self.bank = bank
        self.families = CACHE_FAMILIES
        p = max(bank.prompt_tokens)
        self.resources = {
            name: BoundedKVManager(BoundedKVConfig(
                num_layers=f.num_layers, num_heads=f.num_heads, head_dim=f.head_dim, rows_per_request=f.rows,
                max_source_len=f.rate * p, retention=f.retention, max_slots=slots,
                reverse_step_order=f.reverse_step_order,
                dtype=dit_dtype if name == "dit" and dit_dtype else torch.float32,
            ), device)
            for name, f in CACHE_FAMILIES.items()
        }
        self.pool = RecurrentStatePool(device, RecurrentStateConfig(
            num_layers=1, max_slots=slots + 1,
            blocks={n: RecurrentBlockConfig(shape=b.shape, dtype=b.dtype)
                    for n, b in slot_layout(CacheCapacity(p), dit_dtype=dit_dtype).items()},
        ))
        self.written: dict[str, int] = {}
        self.voice: dict[str, int] = {}

    def slot(self, rid) -> dict[str, torch.Tensor]:
        """``rid``'s slot of the state pool, block by block."""
        index = self.pool._slots[rid]["main"].index
        return {name: self.pool.block(name, 0)[index] for name in self.pool.config.blocks}

    def position(self, family, rid):
        voice = self.voice[rid]
        return family.position(self.bank.prompt_tokens[voice], self.written.get(rid, 0)), voice

    def window(self, rids, num_tokens: int, last: bool, voices: list[int] | None = None):
        """The state rows, caches and per-row voice ids for one window of ``rids`` (each in
        ``voices``, the bank's first by default), and a commit to call after it."""
        from mstar.engine.resources.kv.bounded import BoundedKVStep, StreamPosition
        from mstar.engine.resources.recurrent.config import RecurrentStep
        from mstar.engine.resources.step import Segment, StepContext
        from mstar.model.components.token2wav.model import AttentionCache, SlotRows, WindowCaches, window_tokens

        for r, v in zip(rids, voices or [0] * len(rids), strict=True):
            self.voice.setdefault(r, v)
        ctx = StepContext(request_ids=list(rids), graph_walk="t2w", slot=0, capture=False)
        state_step = RecurrentStep(segments=[Segment(request_id=r, label="main", span=1) for r in rids])
        assert self.pool.admit(state_step, ctx).ok
        state = SlotRows(self.pool, self.pool.plan(state_step, ctx)["main"])
        caches, steps = {}, {}
        for name, family in self.families.items():
            positions = {}
            for r in rids:
                (source_len, written), voice = self.position(family, r)
                positions[r] = StreamPosition(source_len, written, source=voice)
            step = steps[name] = BoundedKVStep(
                segments=[Segment(request_id=r, label="main", span=family.span(num_tokens, last)) for r in rids],
                positions=positions,
            )
            resource = self.resources[name]
            assert resource.admit(step, ctx).ok
            resource.plan(step, ctx)
            caches[name] = AttentionCache(resource, self.bank.sources[name])

        def commit():
            self.pool.commit(state_step, ctx)
            for name, step in steps.items():
                self.resources[name].commit(step, ctx)
            for r in rids:
                self.written[r] = self.written.get(r, 0) + window_tokens(num_tokens, last)
        ids = torch.tensor([self.voice[r] for r in rids], device=self.bank.spk.device)
        return state, WindowCaches(**caches), ids, commit

    def retained(self, name: str, rid) -> torch.Tensor:
        """The keys|values ``rid`` would attend next in ``name``'s caches, ``[layers, rows, H, n, 2d]``."""
        from mstar.engine.resources.kv.bounded.layout import READ_FROM_SOURCE, row_layout

        family, resource = self.families[name], self.resources[name]
        (source_len, written), voice = self.position(family, rid)
        layout = row_layout(source_len, written, 0, family.retention(source_len),
                            resource.config.sink_capacity, family.reverse_step_order)
        slot = resource._requests[rid].slot
        parts = []
        for from_source, (start, count) in zip(READ_FROM_SOURCE, layout.reads, strict=True):
            buf = self.bank.sources[name][voice] if from_source else resource._cache[:, slot]
            parts.append(buf[..., start:start + count, :])
        return torch.cat(parts, dim=-2)


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


def test_the_dit_refuses_voice_sources_in_another_dtype_than_its_caches():
    """The fused DiT computes in its caches' dtype; a voice bank built in another one
    is refused rather than attended in mixed precision."""
    from types import SimpleNamespace

    pytest.importorskip("triton")
    from mstar.model.components.token2wav.kernels import dit_forward

    cnn_cache = torch.zeros(1, 1, 4, 2, dtype=torch.float16)
    kv = [SimpleNamespace(source=torch.zeros(1, 1, dtype=torch.float32))]
    with pytest.raises(ValueError, match="build the VoiceBank in the caches' dtype"):
        dit_forward(None, torch.zeros(1, 80, 4), None, None, None, None, None, cnn_cache, kv)

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
    # drift from the reference state by up to ~1.1e-3 (values ~4) over the utterance, either
    # way round the conformer's relative-position band is read.
    tol = dict(rtol=1e-4, atol=1e-4) if device == "cpu" else dict(rtol=1e-3, atol=2e-3)
    p = 50
    torch.manual_seed(1)
    voice = model.prepare_voice(VoicePrompt(
        tokens=torch.randint(0, 6561, (1, p), dtype=torch.int32).to(device),
        spk_emb=torch.randn(1, 192).to(device), mel=torch.randn(1, 2 * p, 80).to(device),
    ))
    bank = VoiceBank.build({"v": voice})
    kv_steps = _CacheSteps(bank, 3, device)
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
        state, kv, ids, commit = kv_steps.window(["r0"], len(win), last)
        got_mel = model.window_flow(state, bank, ids, tokens, kv, last)
        out = model.window_vocode(state, got_mel, first, last, row0)
        got = model.window_finish(state, out["magnitude"], out["phase"], first, last)
        commit()
        torch.testing.assert_close(got_mel, mel, **tol)
        torch.testing.assert_close(got, want, **tol)
        if not last:
            check_slot("r0", ref)
        mels.append(mel)
        noises.append(nz)

    if device == "cuda":
        # as served: the DiT computes in DIT_COMPUTE_DTYPE (its caches' dtype) on its
        # fused kernels; its mel tracks the fp32 run above
        from mstar.model.components.token2wav.model import DIT_COMPUTE_DTYPE

        bank16 = VoiceBank.build({"v": voice}, dit_dtype=DIT_COMPUTE_DTYPE)
        kv16 = _CacheSteps(bank16, 3, device, dit_dtype=DIT_COMPUTE_DTYPE)
        for (win, last), want in zip(wins[0], mels, strict=True):
            state, kv, ids, commit = kv16.window(["r0"], len(win), last)
            got = model.window_flow(state, bank16, ids, torch.tensor([win], dtype=torch.int32, device=device), kv, last)
            commit()
            rel = ((got.float() - want).norm() / want.norm()).item()
            assert rel < 1e-3, rel  # 1.4e-4 measured; a broken kernel is far past it

    # The same utterance in two rows two windows apart, their full windows' flow in one
    # batch: rows at different positions (a first window beside a later one, then
    # rows on either side of the caches' bound) each match the one-row run.
    bank = VoiceBank.build({"v": voice})
    kv_steps = _CacheSteps(bank, 3, device)
    full = [win for win, last in wins[0] if not last]
    lag = 2
    for k in range(len(full) + lag):
        rows = [(r, k - off) for r, off in (("a", 0), ("b", lag)) if 0 <= k - off < len(full)]
        rids = [r for r, _ in rows]
        tokens = torch.tensor([full[w] for _, w in rows], dtype=torch.int32, device=device)
        state, kv, ids, commit = kv_steps.window(rids, len(full[0]), False)
        mel = model.window_flow(state, bank, ids, tokens, kv)
        commit()
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
    bank = VoiceBank.build({"v": voice})
    kv_steps = _CacheSteps(bank, 2, "cpu")
    ref = new_state(model, voice)
    # the token encoder's output: the DiT's small weights would hide it in the mel
    encoded = []
    hook = model.flow.encoder.register_forward_hook(lambda _m, _i, out: encoded.append(out.clone()))
    try:
        for win, last in reference_windows(torch.randint(0, 6561, (60,), generator=gen).tolist()):
            tokens = torch.tensor([win], dtype=torch.int32)
            flow_chunk(model, ref, voice, tokens, last)
            state, kv, ids, commit = kv_steps.window(["r0"], len(win), last)
            model.window_flow(state, bank, ids, tokens, kv, last)
            commit()
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
            bank = VoiceBank.build({"v": voice}, dit_dtype=dtype)
            kv_steps = _CacheSteps(bank, 3, "cuda", dit_dtype=dtype)
            mels[dtype] = []
            for w in range(len(wins[0])):
                last = wins[0][w][1]
                tokens = torch.tensor([ws[w][0] for ws in wins], dtype=torch.int32, device="cuda")
                state, kv, ids, commit = kv_steps.window(rids, tokens.shape[1], last)
                mels[dtype].append(model.window_flow(state, bank, ids, tokens, kv, last).float())
                commit()
    for w, (want, got) in enumerate(zip(mels[None], mels[DIT_COMPUTE_DTYPE], strict=True)):
        rel = ((got - want).norm(dim=(1, 2)) / want.norm(dim=(1, 2))).max().item()
        assert rel < 2e-3, (w, rel)


@pytest.mark.parametrize(("n", "stop"), [
    *((n, True) for n in [1, 21, 22, 23, 24, 25, 49, 99, 100, 230]), (0, False), (25, False), (50, False),
])
def test_served_windows_are_upstreams(n, stop):
    """The TTS streams ``n`` codes and then its stop code, or none when it ends at its cap; the
    policy's chunks, through ``prepare_inputs``, are the reference's windows of the codes: every
    code voiced once and the same last window, also when the stop code ends a full window
    (``n = 25j + 24``) or a capped reply's codes do (``n = 25j``). Every
    chunk still runs, voicing nothing past the last window: a skipped step never completes the
    node, which then refuses the stream's final chunk. ``n = 0`` is a reply that ended at the
    LLM's prefill: the TTS never ran, and the stream's one chunk is an empty, last step that
    declares no row (so leases no slot)."""
    from types import SimpleNamespace

    from mstar.model.components.token2wav.model import HOP, LEAD_SILENCE, SILENCE_CODE
    from mstar.model.components.token2wav.submodule import Token2WavSubmodule
    from mstar.model.submodule_base import InputMetadata
    from mstar.streaming.chunk_policy import LeftContextChunkPolicy
    from mstar.streaming.stream_buffer import StreamBuffer

    eos = 6561
    node = Token2WavSubmodule.__new__(Token2WavSubmodule)
    torch.nn.Module.__init__(node)
    node.request_states, node.eos_code = {}, eos
    node.keys = T2W_KEYS
    node._lead_silence = torch.full((LEAD_SILENCE,), SILENCE_CODE, dtype=torch.int32)
    node.model = torch.nn.Linear(1, 1)  # its device
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
                # each window's caches hold what the windows before it wrote
                assert node.request_state(0)["written"] == sum(window_tokens(len(w), last) for w, last in served[:-1])
    assert got.kwargs["last"]
    if n:
        assert served == reference_windows(codes)
    else:
        assert served == [] and node.request_states == {}
        step = node.declare_step("t2w_chunk", [0], [got])
        assert all(not s.segments for s in step.steps.values())


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
    kv_steps = _CacheSteps(bank, 3, device)
    lag = 1
    for k in range(len(full) + lag):
        rows = [(name, k - off) for name, off in (("short", 0), ("long", lag)) if 0 <= k - off < len(full)]
        tokens = torch.tensor([full[w] for _, w in rows], dtype=torch.int32, device=device)
        state, kv, ids, commit = kv_steps.window(
            [name for name, _ in rows], len(full[0]), False, [bank.index(name) for name, _ in rows])
        mel = model.window_flow(state, bank, ids, tokens, kv)
        commit()
        for i, (name, w) in enumerate(rows):
            torch.testing.assert_close(mel[i:i + 1], want[name][w], **tol)


def test_hift_noise_is_the_requests_own(random_token2wav):
    """A spoken reply's HiFT excitation comes from its own seed: a window vocodes the same
    alone and beside another request, whatever the global RNG drew in between."""
    from types import SimpleNamespace

    from mstar.engine.accelerator_graph_config import PiecewiseCallInputs
    from mstar.model.components.token2wav.submodule import Token2WavSubmodule
    from mstar.model.submodule_base import InputMetadata, ModelInputsFromEngine

    model = random_token2wav
    torch.manual_seed(5)
    voice = model.prepare_voice(VoicePrompt(
        tokens=torch.randint(0, 6561, (1, 6), dtype=torch.int32), spk_emb=torch.randn(1, 192),
        mel=torch.randn(1, 12, 80),
    ))
    node = Token2WavSubmodule(model, T2W_KEYS, {"v": voice}, eos_code=6561, dit_dtype=torch.float32)
    frames = window_frames(28, False)
    region = node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)[
        node.hift_region(True, frames, False)]
    kv_steps = _CacheSteps(node.bank, 3, "cpu")
    mel = torch.randn(1, 80, frames)
    chunk = InputMetadata(stream_chunks={"tts_code": SimpleNamespace(start_offset=0, context_items=0, num_items=25)})

    def first_window(seeds: dict[str, int]) -> torch.Tensor:
        rids = list(seeds)
        for rid, seed in seeds.items():
            info = SimpleNamespace(rid_handle=rid, step_metadata={"voice": "v"}, random_seed=seed)
            node.prepare_inputs("t2w_chunk", info, {"tts_code": [torch.arange(25)]}, input_metadata=chunk)
        _, _, _, commit = kv_steps.window(rids, 28, False)
        resources = {T2W_STATE: kv_steps.pool, **{T2W_KV[n]: kv_steps.resources[n] for n in T2W_KV}}
        engine_inputs = ModelInputsFromEngine(request_ids=rids, per_request_info={}, resources=resources)

        class Runner:  # the region, run eagerly on the rows the step planned
            def run(self, static_inputs, **kwargs):
                out = region.capture_fn(PiecewiseCallInputs(static_inputs=static_inputs, engine_inputs=engine_inputs))
                return SimpleNamespace(get_view=out.__getitem__)

        # the node draws the noise itself
        node._run_hift(engine_inputs, Runner(), rids, mel.expand(len(rids), -1, -1), True, False)
        commit()
        # the excitation it held back for the next window: the noise, nearly alone
        return torch.stack([kv_steps.slot(r)["hift_source"] for r in rids])

    alone = first_window({"a": 7})
    torch.randn(1000)
    batched = first_window({"b": 8, "c": 7})
    scale = alone.abs().max()
    assert (batched[1:] - alone).abs().max() < 1e-4 * scale
    assert (batched[:1] - alone).abs().max() > 0.1 * scale


def test_resources_declare_one_state_pool_and_a_cache_per_family():
    """Every cache family gets its bounded KV under the model's key, one slot short of
    the state pool, which also keeps its sink slot."""
    from mstar.engine.resources.kv.bounded import BoundedKVSpec
    from mstar.engine.resources.recurrent.config import RecurrentStateSpec
    from mstar.model.components.token2wav.model import CACHE_FAMILIES
    from mstar.model.components.token2wav.resources import token2wav_resources

    specs = token2wav_resources(T2W_KEYS, "Token2Wav", (3.0, 10.0), max_slots=9)
    state = [s for s in specs if isinstance(s, RecurrentStateSpec)]
    kv = {s.resource_key: s for s in specs if isinstance(s, BoundedKVSpec)}
    assert [s.resource_key for s in state] == [T2W_STATE] and state[0].config.max_slots == 9
    assert set(kv) == {T2W_KV[name] for name in CACHE_FAMILIES}
    assert all(s.config.max_slots == 8 and s.nodes == {"Token2Wav"} for s in kv.values())


@pytest.mark.parametrize(
    ("state", "kv", "ok"), [(None, None, True), (17, 16, True), (16, 16, False), (17, None, False)],
)
def test_slot_check_wants_one_more_state_slot_than_each_cache(state, kv, ok):
    from mstar.model.components.token2wav.resources import check_token2wav_slots

    resources = {}
    if state is not None:
        resources[T2W_STATE] = {"max_slots": state}
    if kv is not None:
        resources.update({key: {"max_slots": kv} for key in T2W_KV.values()})
    if ok:
        check_token2wav_slots(T2W_KEYS, resources, 9, "cfg.yaml")
    else:
        with pytest.raises(ValueError, match="must be one more than"):
            check_token2wav_slots(T2W_KEYS, resources, 9, "cfg.yaml")

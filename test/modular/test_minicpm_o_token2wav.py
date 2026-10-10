"""MiniCPM-o token2wav: window arithmetic, per-request state layout and truncation, and the
streaming flow/vocoder on random weights (CPU).

The truncation test replays the reference's own cache bookkeeping (``Token2wav.stream``: the
25 Hz conformer cache tiled twice, both 50 Hz caches cut to the prompt's frames plus the last
100) on labelled entries and checks the slot holds the same entries. Loading the real assets
skips unless MINICPM_O_CKPT points at a MiniCPM-o 4.5 snapshot; bit-exactness against the
reference lives in the parity harness, not here.
"""

from __future__ import annotations

import os

import pytest
import torch

from mstar.model.minicpm_o.components.token2wav import (
    HIFT_TAIL,
    KEEP_RECENT,
    CacheCapacity,
    SineGen2Source,
    Token2Wav,
    Token2WavState,
    fade_in_out,
    lengths_after,
    num_windows,
    slot_layout,
    state_from_slot,
    state_lengths,
    stream_windows,
    window_frames,
    window_samples,
)
from mstar.model.minicpm_o.components.voice_prompt import VoicePrompt, _module_path


class _CacheSteps:
    """Token2wav's three attention-cache resources for one voice, driven through admit /
    plan / commit one window at a time, as the engine does."""

    def __init__(self, voice, slots: int, device):
        from mstar.engine.resources.kv.bounded.config import BoundedKVConfig
        from mstar.engine.resources.kv.bounded.manager import BoundedKVManager
        from mstar.model.minicpm_o.components.token2wav import CACHE_FAMILIES

        self.p = voice.initial.prompt_frames // 2
        self.families = CACHE_FAMILIES
        self.sources = {name: f.source(voice) for name, f in CACHE_FAMILIES.items()}
        self.resources = {
            name: BoundedKVManager(BoundedKVConfig(
                num_layers=f.num_layers, num_heads=f.num_heads, head_dim=f.head_dim, rows_per_request=f.rows,
                max_source_len=f.rate * self.p, retention=f.retention, max_slots=slots,
                reverse_step_order=f.reverse_step_order,
            ), device)
            for name, f in CACHE_FAMILIES.items()
        }
        self.calls: dict[str, int] = {}

    def window(self, rids, num_tokens: int, last: bool):
        from mstar.engine.resources.kv.bounded import BoundedKVStep, StreamPosition
        from mstar.engine.resources.step import Segment, StepContext
        from mstar.model.minicpm_o.components.token2wav import AttentionCache, WindowCaches

        ctx = StepContext(request_ids=list(rids), graph_walk="t2w", slot=0, capture=False)
        caches, steps = {}, {}
        for name, family in self.families.items():
            step = steps[name] = BoundedKVStep(
                segments=[Segment(request_id=r, label="main", span=family.span(num_tokens, last)) for r in rids],
                positions={r: StreamPosition(*family.position(self.p, self.calls.get(r, 0))) for r in rids},
            )
            resource = self.resources[name]
            assert resource.admit(step, ctx).ok
            caches[name] = AttentionCache(resource, self.sources[name], resource.plan(step, ctx))

        def commit():
            for name, step in steps.items():
                self.resources[name].commit(step, ctx)
            if not last:
                for r in rids:
                    self.calls[r] = self.calls.get(r, 0) + 1
        return WindowCaches(**caches), commit

    def retained(self, name: str, rid) -> torch.Tensor:
        """The keys|values ``rid`` would attend next in ``name``'s caches, ``[layers, rows, H, n, 2d]``."""
        from mstar.engine.resources.kv.bounded.layout import READ_FROM_SOURCE, row_layout

        family, resource = self.families[name], self.resources[name]
        source_len, written = family.position(self.p, self.calls.get(rid, 0))
        layout = row_layout(source_len, written, 0, family.retention(source_len),
                            resource.config.sink_capacity, family.reverse_step_order)
        slot = resource._requests[rid].slot
        parts = []
        for from_source, (start, count) in zip(READ_FROM_SOURCE, layout.reads, strict=True):
            buf = self.sources[name] if from_source else resource._cache[:, slot]
            parts.append(buf[..., start:start + count, :])
        return torch.cat(parts, dim=-2)


CKPT = os.environ.get("MINICPM_O_CKPT")
needs_ckpt = pytest.mark.skipif(not CKPT, reason="set MINICPM_O_CKPT to a MiniCPM-o 4.5 snapshot")


def reference_windows(codes: list[int]) -> list[tuple[list[int], bool]]:
    """The recorder's driver loop, verbatim."""
    buf = [4218] * 3 + codes
    out = []
    while len(buf) >= 28:
        out.append((buf[:28], False))
        buf = buf[25:]
    out.append((buf, True))
    return out


@pytest.mark.parametrize("n", [0, 1, 24, 25, 26, 50, 127, 269, 300])
def test_windows_match_reference_driver(n):
    codes = list(range(n))
    windows = list(stream_windows(codes))
    assert windows == reference_windows(codes)
    assert num_windows(n) == len(windows)
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
        s.truncate()
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
    windows = list(stream_windows(list(range(100, 130))))
    outs = []
    for extra in (0, 9):
        state = model.new_state(voice, CacheCapacity(p + extra))
        gen = torch.Generator().manual_seed(3)
        wavs = []
        for i, (win, last) in enumerate(windows):
            wavs.append(model.stream(state, voice, win, last, generator=gen))
            assert wavs[-1].shape == (1, window_samples(len(win), last, i == 0))
            assert state.dit_len <= 2 * p + KEEP_RECENT or last
        outs.append(torch.cat(wavs, dim=1))
    assert torch.equal(outs[0], outs[1])


def test_lengths_after():
    p2 = 302
    lengths = {"prompt_frames": p2, "enc_len1": 151, "enc_len2": 302, "dit_len": 302, "calls": 0}
    for _ in range(5):
        lengths = lengths_after(lengths, 28, False)
    assert lengths == {"prompt_frames": p2, "enc_len1": 201, "enc_len2": 402, "dit_len": 402, "calls": 5}


def test_lengths_after_matches_state(random_token2wav):
    """The host bookkeeping agrees with what ``flow_chunk`` leaves in a state."""
    model = random_token2wav
    p = 50
    voice = model.prepare_voice(VoicePrompt(
        tokens=torch.randint(0, 6561, (1, p), dtype=torch.int32), spk_emb=torch.randn(1, 192),
        mel=torch.randn(1, 2 * p, 80),
    ))
    st = model.new_state(voice)
    for win, last in list(stream_windows(list(range(80))))[:3]:
        want = lengths_after(state_lengths(st), len(win), last)
        model.stream(st, voice, win, last)
        assert state_lengths(st) == want


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
    """Every window of an utterance batched on pool slots, the DiT ring wrapping several
    times: one row matches ``stream`` (samples; the slot against the reference state after
    every window, the DiT caches as a multiset since only that enters attention), to the
    summation order of attending the same keys in another order; two rows agree with it to
    the batch-size dependence of the GEMMs."""
    import copy

    from mstar.model.chatterbox.components.s3gen_hift import HiFTNoise

    model = copy.deepcopy(random_token2wav).to(device)
    # the CUDA path's attention kernel multiplies in TF32, as served
    tol = dict(rtol=1e-4, atol=1e-4) if device == "cpu" else dict(rtol=1e-3, atol=1e-3)
    p = 50
    torch.manual_seed(1)
    voice = model.prepare_voice(VoicePrompt(
        tokens=torch.randint(0, 6561, (1, p), dtype=torch.int32).to(device),
        spk_emb=torch.randn(1, 192).to(device), mel=torch.randn(1, 2 * p, 80).to(device),
    ))
    one, two = torch.tensor([1], device=device), torch.tensor([1, 2], device=device)
    kv_steps = _CacheSteps(voice, 3, device)
    layout = slot_layout(CacheCapacity(p))
    blocks = {name: torch.zeros((3, *blk.shape), dtype=blk.dtype, device=device) for name, blk in layout.items()}
    codes = [torch.randint(0, 6561, (215,)).tolist() for _ in range(2)]
    wins = [list(stream_windows(c)) for c in codes]  # 9 windows each, the last of 18 tokens
    gen = torch.Generator().manual_seed(9)

    def noise(rows, calls, win, last):
        frames = window_frames(len(win), last) + (0 if calls == 0 else 8)
        nz = model.hift.m_source.draw_noise(rows, frames * 480, torch.float32, "cpu", gen)
        return HiFTNoise(phase=nz.phase.to(device), harmonic=nz.harmonic.to(device))

    def check_slot(slot, st):
        from mstar.model.minicpm_o.components.token2wav_flow import DIT_DEPTH, N_TIMESTEPS

        fields_ = {n: t[slot] for n, t in blocks.items()}
        view = state_from_slot(
            {**fields_, "enc_kv1": st.enc_kv1, "enc_kv2": st.enc_kv2, "dit_kv": st.dit_kv}, state_lengths(st))
        # the conformer attends its caches in order; the DiT's only as a set
        torch.testing.assert_close(kv_steps.retained("enc1", "r0"), st.enc_kv1[..., :st.enc_len1, :], **tol)
        torch.testing.assert_close(kv_steps.retained("enc2", "r0"), st.enc_kv2[..., :st.enc_len2, :], **tol)
        _assert_same_frames(kv_steps.retained("dit", "r0").unflatten(0, (N_TIMESTEPS, DIT_DEPTH)),
                            st.dit_kv[..., :st.dit_len, :], tol=1e-2 if device == "cpu" else 0.5)
        for name in fields_:
            a, b = getattr(view, name), getattr(st, name)
            torch.testing.assert_close(a, b, **tol, msg=name)

    ref = model.new_state(voice)
    lengths = state_lengths(voice.initial)
    mels, noises = [], []
    for win, last in wins[0]:
        nz = noise(2, lengths["calls"], win, last)
        row0 = HiFTNoise(phase=nz.phase[:1], harmonic=nz.harmonic[:1])
        mel = model.flow_chunk(ref, voice, torch.tensor([win], dtype=torch.int32, device=device), last)
        want = model.vocode_chunk(ref, mel, last, noise=row0)
        tokens = torch.tensor([win], dtype=torch.int32, device=device)
        kv, commit = kv_steps.window(["r0"], len(win), last)
        first = lengths["calls"] == 0
        out = model.window_spectrum(blocks, one, voice, first, tokens, kv, last, row0)
        commit()
        got = model.window_finish(blocks, one, out["magnitude"], out["phase"], first, last)
        torch.testing.assert_close(out["mel"], mel, **tol)
        torch.testing.assert_close(got, want, **tol)
        if not last:
            check_slot(1, ref)
        lengths = lengths_after(lengths, len(win), last)
        mels.append(mel)
        noises.append(nz)

    # The same utterance in two rows two windows apart, their full windows' flow in one
    # batch: rows at different positions (a first window beside a later one, then
    # rows on either side of the caches' bound) each match the one-row run.
    for t in blocks.values():
        t.zero_()
    kv_steps = _CacheSteps(voice, 3, device)
    full = [win for win, last in wins[0] if not last]
    lag = 2
    for k in range(len(full) + lag):
        rows = [(r, k - off) for r, off in (("a", 0), ("b", lag)) if 0 <= k - off < len(full)]
        rids = [r for r, _ in rows]
        tokens = torch.tensor([full[w] for _, w in rows], dtype=torch.int32, device=device)
        fresh = torch.tensor([w == 0 for _, w in rows], device=device)
        slots = torch.tensor([1 if r == "a" else 2 for r in rids], device=device)
        kv, commit = kv_steps.window(rids, len(full[0]), False)
        mel = model.window_flow(blocks, slots, fresh, voice, tokens, kv)
        commit()
        for i, (_, w) in enumerate(rows):
            torch.testing.assert_close(mel[i:i + 1], mels[w], **tol)


def test_onnx_scope_paths():
    assert _module_path("/head/layer1/layer1.0/conv1/Conv") == "head.layer1.0.conv1"
    assert _module_path("/blocks.3/mlp/mlp.2/MatMul") == "blocks.3.mlp.2"
    assert _module_path("/xvector/block2/tdnnd7/cam_layer/linear1/Conv") == "xvector.block2.tdnnd7.cam_layer.linear1"


@needs_ckpt
def test_assets_load_completely():
    from mstar.model.minicpm_o.components.token2wav import load_token2wav

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_token2wav(f"{CKPT}/assets/token2wav", device)
    import soundfile as sf

    wav, sr = sf.read(f"{CKPT}/assets/HT_ref_audio.wav", dtype="float32")
    assert sr == 16000
    prompt = model.voice_encoder(torch.from_numpy(wav))
    p = prompt.tokens.shape[1]
    assert prompt.mel.shape == (1, 2 * p, 80) and prompt.spk_emb.shape == (1, 192)
    voice = model.prepare_voice(prompt)
    state = model.new_state(voice)
    wav = model.stream(state, voice, [4218] * 3 + [100] * 25, last=False)
    assert wav.shape == (1, 50 * 480) and torch.isfinite(wav).all()

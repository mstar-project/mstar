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
    PHASE_FIRST,
    PHASE_SECOND,
    PHASE_STEADY,
    CacheCapacity,
    SineGen2Source,
    Token2Wav,
    Token2WavState,
    fade_in_out,
    lengths_after,
    num_windows,
    state_block_shapes,
    state_from_slot,
    state_lengths,
    stream_windows,
    window_batch_key,
    window_frames,
    window_phase,
    window_samples,
)
from mstar.model.minicpm_o.components.voice_prompt import VoicePrompt, _module_path

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


def test_window_phase_and_lengths():
    p2 = 302
    fresh = {"prompt_frames": p2, "enc_len1": 151, "enc_len2": 302, "dit_len": 302, "calls": 0}
    phases, lengths = [], fresh
    for _ in range(5):
        phases.append(window_phase(lengths, 28, False))
        lengths = lengths_after(lengths, 28, False)
    assert phases == [PHASE_FIRST, PHASE_SECOND, PHASE_STEADY, PHASE_STEADY, PHASE_STEADY]
    assert lengths == {"prompt_frames": p2, "enc_len1": 201, "enc_len2": 402, "dit_len": 402, "calls": 5}
    assert window_phase(lengths, 28, True) is None and window_phase(lengths, 20, False) is None
    small = {"prompt_frames": 60, "enc_len1": 30, "enc_len2": 60, "dit_len": 60, "calls": 0}
    assert window_phase(small, 28, False) is None


def test_steady_windows_share_a_batch_key():
    p2 = 302
    lengths = {"prompt_frames": p2, "enc_len1": 151, "enc_len2": 302, "dit_len": 302, "calls": 0}
    keys = []
    for _ in range(6):
        keys.append(window_batch_key(lengths, 28, False))
        lengths = lengths_after(lengths, 28, False)
    assert len({keys[0], keys[1], keys[2]}) == 3
    assert keys[2] == keys[3] == keys[4] == keys[5]
    assert window_batch_key(lengths, 9, True) != window_batch_key(lengths, 10, True)


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


def test_batched_windows_on_pool(random_token2wav):
    """Every window of an utterance batched on pool slots: one row is bit-identical to
    ``stream`` (samples, and the slot against the state after every window: first window
    read from the voice, truncation, last window), and two rows agree with it to the
    batch-size dependence of the GEMMs."""
    from mstar.model.chatterbox.components.s3gen_hift import HiFTNoise

    model = random_token2wav
    p = 50
    torch.manual_seed(1)
    voice = model.prepare_voice(VoicePrompt(
        tokens=torch.randint(0, 6561, (1, p), dtype=torch.int32), spk_emb=torch.randn(1, 192),
        mel=torch.randn(1, 2 * p, 80),
    ))
    shapes = state_block_shapes(CacheCapacity(p))
    blocks = {name: torch.zeros((3, *shape)) for name, shape in shapes.items()}
    codes = [torch.randint(0, 6561, (90,)).tolist() for _ in range(2)]
    wins = [list(stream_windows(c)) for c in codes]  # 4 windows each, the last of 18 tokens
    gen = torch.Generator().manual_seed(9)

    def noise(rows, calls, win, last):
        frames = window_frames(len(win), last) + (0 if calls == 0 else 8)
        return model.hift.m_source.draw_noise(rows, frames * 480, torch.float32, "cpu", gen)

    def check_slot(slot, st):
        view = state_from_slot({n: t[slot] for n, t in blocks.items()}, state_lengths(st))
        for name in shapes:
            a, b = getattr(view, name), getattr(st, name)
            n = {"enc_kv1": st.enc_len1, "enc_kv2": st.enc_len2, "dit_kv": st.dit_len}.get(name)
            if n is not None:
                a, b = a[..., :n, :], b[..., :n, :]
            assert torch.equal(a, b), name

    ref = model.new_state(voice)
    lengths = state_lengths(voice.initial)
    mels, noises = [], []
    for win, last in wins[0]:
        nz = noise(2, lengths["calls"], win, last)
        row0 = HiFTNoise(phase=nz.phase[:1], harmonic=nz.harmonic[:1])
        mel = model.flow_chunk(ref, voice, torch.tensor([win], dtype=torch.int32), last)
        want = model.vocode_chunk(ref, mel, last, noise=row0)
        tokens = torch.tensor([win], dtype=torch.int32)
        out = model.window_spectrum(blocks, torch.tensor([1]), voice, lengths, tokens, last, row0)
        got = model.window_finish(blocks, torch.tensor([1]), out["magnitude"], out["phase"], lengths, last)
        assert torch.equal(got, want) and torch.equal(out["mel"], mel)
        if not last:
            check_slot(1, ref)
        lengths = lengths_after(lengths, len(win), last)
        mels.append(mel)
        noises.append(nz)

    for t in blocks.values():
        t.zero_()
    lengths = state_lengths(voice.initial)
    for k, ((w0, last), (w1, _)) in enumerate(zip(wins[0], wins[1], strict=True)):
        tokens = torch.tensor([w0, w1], dtype=torch.int32)
        out = model.window_spectrum(blocks, torch.tensor([1, 2]), voice, lengths, tokens, last, noises[k])
        torch.testing.assert_close(out["mel"][:1], mels[k], rtol=1e-4, atol=1e-4)
        lengths = lengths_after(lengths, len(w0), last)


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

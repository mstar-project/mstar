"""The vectorized silence trim against the reference's pydub one: sample-exact."""

import numpy as np
import pytest

from mstar.model.omnivoice.components import silence
from mstar.model.omnivoice.components.silence import remove_silence

RATE = 24000


@pytest.fixture
def reference():
    return pytest.importorskip("omnivoice.utils.audio")


def _speech(rng, seconds, rate=RATE):
    """Bursts of tone and noise separated by silences of varied length."""
    out = []
    while sum(len(x) for x in out) < seconds * rate:
        n = int(rng.uniform(0.05, 0.9) * rate)
        if rng.random() < 0.5:
            t = np.arange(n) / rate
            out.append((rng.uniform(0.05, 0.6) * np.sin(2 * np.pi * rng.uniform(80, 400) * t)).astype(np.float32))
        else:
            level = rng.choice([0.0, 1e-5, 2e-3, 3.2e-3, 1e-2])
            out.append((level * rng.standard_normal(n)).astype(np.float32))
    return np.concatenate(out)[None, : int(seconds * rate) + int(rng.integers(0, 24))]


@pytest.mark.parametrize("seed", range(40))
def test_matches_pydub(seed, reference):
    rng = np.random.default_rng(seed)
    audio = _speech(rng, rng.uniform(0.05, 12.0))
    for mid, lead, trail in ((500, 100, 100), (300, 100, 300), (0, 100, 300)):
        want = reference.remove_silence(audio, RATE, mid_sil=mid, lead_sil=lead, trail_sil=trail)
        got = remove_silence(audio, RATE, mid_sil=mid, lead_sil=lead, trail_sil=trail)
        assert got.dtype == want.dtype and np.array_equal(got, want), (seed, mid, got.shape, want.shape)


@pytest.mark.parametrize("audio", [
    np.zeros((1, 0), np.float32),
    np.zeros((1, 7), np.float32),
    np.zeros((1, RATE), np.float32),
    np.full((1, 13), 0.5, np.float32),
    # full scale both ways between silences: the int16 conversion clips
    np.concatenate([np.zeros(RATE // 5), np.ones(RATE // 10), -np.ones(RATE // 10), np.zeros(RATE // 2)])
    .astype(np.float32)[None],
])
def test_edge_cases_match_pydub(audio, reference):
    want = reference.remove_silence(audio, RATE, mid_sil=500, lead_sil=100, trail_sil=100)
    got = remove_silence(audio, RATE, mid_sil=500, lead_sil=100, trail_sil=100)
    assert np.array_equal(got, want)


def test_postprocess_shapes_the_waveform_and_refuses_a_masked_canvas(reference):
    """code2wav emits the raw float waveform and the API side shapes it the
    reference's way (here: peak-normalised to 0.5, faded and padded); a canvas
    that still held MASK cells arrives NaN-poisoned and is refused."""
    del reference  # the fade and pad are the reference package's
    import torch

    from mstar.model.omnivoice.components.codec import post_process
    from mstar.model.omnivoice.config import OmniVoiceConfig
    from mstar.model.omnivoice.omnivoice_model import OmniVoiceModel

    model = object.__new__(OmniVoiceModel)
    model.config = OmniVoiceConfig()
    rng = np.random.default_rng(0)
    wave = _speech(rng, 2.0)[0]
    pcm = np.frombuffer(model.postprocess(torch.from_numpy(wave), "audio", request_kwargs={}), dtype=np.int16)
    want = post_process(wave, config=model.config, ref_rms=None, pad_duration=0.1, fade_duration=0.1, enabled=True)
    assert np.array_equal(pcm, (np.clip(want, -1, 1) * 32767).astype(np.int16))
    poisoned = torch.from_numpy(wave.copy())
    poisoned[:] = float("nan")
    with pytest.raises(RuntimeError, match="masked cells"):
        model.postprocess(poisoned, "audio", request_kwargs={})


def test_code2wav_never_looks_up_the_mask_id():
    """A canvas that still holds MASK cells decodes with code 0 there and comes
    out NaN-poisoned: the MASK id is one past the codebooks, and on the GPU that
    lookup is a device-side assert that kills the worker and every request on it."""
    from types import SimpleNamespace

    import torch

    from mstar.model.omnivoice.config import OmniVoiceConfig
    from mstar.model.omnivoice.submodules import OmniVoiceCode2WavSubmodule

    config = OmniVoiceConfig()

    class Codec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.codebook = torch.nn.Embedding(config.audio_mask_id, 2)   # ids 0 .. mask - 1

        @property
        def device(self):
            return self.codebook.weight.device

        def decode(self, tokens):
            return SimpleNamespace(audio_values=self.codebook(tokens))

    code2wav = OmniVoiceCode2WavSubmodule(Codec(), config)
    canvas = torch.randint(0, config.audio_mask_id, (config.num_audio_codebook, 5))
    assert torch.isfinite(code2wav.forward("speech_gen", None, canvas)["audio_output"][0]).all()
    canvas[3, 2] = config.audio_mask_id
    assert torch.isnan(code2wav.forward("speech_gen", None, canvas)["audio_output"][0]).all()


def _trim_over_a_reversed_copy(audio, rate, mid_sil, lead_sil, trail_sil):
    """The edge trim as pydub runs it: the trailing trim reverses the lead-trimmed
    samples and measures them afresh."""
    samples = (audio[0] * 32768.0).clip(-32768, 32767).astype(np.int16)
    threshold = 10 ** (-50.0 / 20) * 32768.0
    if mid_sil > 0:
        samples = silence._split_on_silence(silence._Segment(samples, rate), mid_sil, threshold, mid_sil, 10)
    seg = silence._Segment(samples, rate)
    samples = seg.slice(max(0, silence._leading_silence(seg, -50.0) - lead_sil), len(seg))
    seg = silence._Segment(samples[::-1].copy(), rate)
    samples = seg.slice(max(0, silence._leading_silence(seg, -50.0) - trail_sil), len(seg))[::-1]
    return (samples.astype(np.float32) / 32768.0)[np.newaxis, :]


@pytest.mark.parametrize("rate", [24000, 22050, 44100, 16000])
@pytest.mark.parametrize("seed", range(25))
def test_trailing_trim_reads_the_shared_prefix_sums(seed, rate):
    """Guards the flipped-bounds path (``_Segment.reversed_tail``) where the pydub
    comparison skips, as on CI: the trailing trim reads the lead-trimmed segment
    backwards over its prefix sums, sample-exact against measuring a reversed
    copy, including where a rate's fractional samples per millisecond pad or
    drop the edge."""
    rng = np.random.default_rng(seed)
    audio = _speech(rng, rng.uniform(0.0, 4.0), rate)
    if seed % 5 == 0:  # loud to both edges
        audio = (0.3 * rng.standard_normal(audio.shape)).astype(np.float32)
    for args in ((500, 100, 100), (0, 100, 300), (0, 0, 0), (0, 500, 0)):
        want = _trim_over_a_reversed_copy(audio, rate, *args)
        got = remove_silence(audio, rate, *args)
        assert got.dtype == want.dtype and np.array_equal(got, want), (seed, rate, args)

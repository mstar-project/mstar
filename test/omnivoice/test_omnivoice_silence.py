"""The vectorized silence trim against the reference's pydub one: sample-exact."""

from types import SimpleNamespace

import numpy as np
import pytest

from mstar.model.omnivoice.components.silence import remove_silence

RATE = 24000


@pytest.fixture
def reference():
    """``omnivoice.utils.audio.remove_silence`` (Apache-2.0, k2-fsa/OmniVoice),
    restated over pydub itself so the oracle runs without the reference
    package. Mono only, as OmniVoice decodes; raw-sample segments need no ffmpeg."""
    pydub = pytest.importorskip("pydub")
    from pydub.silence import detect_leading_silence, split_on_silence

    def trim_edges(wave, lead_sil, trail_sil):
        wave = wave[max(0, detect_leading_silence(wave, silence_threshold=-50) - lead_sil):]
        wave = wave.reverse()
        wave = wave[max(0, detect_leading_silence(wave, silence_threshold=-50) - trail_sil):]
        return wave.reverse()

    def remove(audio, rate, mid_sil, lead_sil, trail_sil):
        samples = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
        wave = pydub.AudioSegment(data=samples.tobytes(), sample_width=2, frame_rate=rate, channels=1)
        if mid_sil > 0:
            segments = split_on_silence(
                wave, min_silence_len=mid_sil, silence_thresh=-50, keep_silence=mid_sil, seek_step=10,
            )
            wave = pydub.AudioSegment.silent(duration=0)
            for segment in segments:
                wave += segment
        wave = trim_edges(wave, lead_sil, trail_sil)
        return (np.array(wave.get_array_of_samples()).astype(np.float32) / 32768.0)[np.newaxis, :]

    return SimpleNamespace(remove_silence=remove)


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


@pytest.mark.parametrize("rate", [24000, 22050, 44100, 16000])
@pytest.mark.parametrize("seed", range(25))
def test_matches_pydub(seed, rate, reference):
    """Including the trailing trim's backwards read over the lead-trimmed
    segment's prefix sums, and rates whose fractional samples per millisecond
    pad or drop the edge."""
    rng = np.random.default_rng(seed)
    audio = _speech(rng, rng.uniform(0.05, 12.0), rate)
    if seed % 5 == 0:  # loud to both edges
        audio = (0.3 * rng.standard_normal(audio.shape)).astype(np.float32)
    for mid, lead, trail in ((500, 100, 100), (300, 100, 300), (0, 100, 300), (0, 0, 0), (0, 500, 0)):
        want = reference.remove_silence(audio, rate, mid_sil=mid, lead_sil=lead, trail_sil=trail)
        got = remove_silence(audio, rate, mid_sil=mid, lead_sil=lead, trail_sil=trail)
        assert got.dtype == want.dtype and np.array_equal(got, want), (seed, rate, mid, got.shape, want.shape)


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


def test_postprocess_shapes_the_waveform_and_refuses_a_masked_canvas():
    """code2wav emits the raw float waveform and the API side shapes it the
    reference's way (here: peak-normalised to 0.5, faded and padded); a canvas
    that still held MASK cells arrives NaN-poisoned and is refused."""
    pytest.importorskip("omnivoice.utils.audio")  # the fade and pad are the reference package's
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

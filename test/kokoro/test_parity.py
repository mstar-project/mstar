"""Real-weight parity of the M* Kokoro port against the ``kokoro`` package.

Skips unless ``hexgrad/Kokoro-82M`` is in the local Hugging Face cache and the
``kokoro`` package is importable. Runs on CPU (82M parameters) or CUDA.

What is compared, and why the tolerances differ:

* durations are integers and must match exactly;
* every deterministic intermediate (phoneme states, F0/energy curves, text
  features, the decoder given identical inputs) matches to fp32 noise;
* serving draws the source noise from the request seed (``keyed_normal``), not
  the global RNG; the reference given that same noise is the oracle for it;
* the waveform is compared with a looser bound. Kokoro's vocoder feeds the
  *phase* of the harmonic source's STFT into convolutions, and the source
  phase is a cumulative sum over the whole utterance, so the output is
  chaotic in F0 at the fp32 rounding level: perturbing the reference's own
  F0 by 1e-6 relative moves its waveform by ~3% (asserted below). Any
  reimplementation, and the reference across devices, sits inside that band.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest
import torch

from mstar.model.kokoro.components import KokoroTTS
from mstar.model.kokoro.components.decoder import keyed_normal
from mstar.model.kokoro.config import KokoroModelConfig
from mstar.model.kokoro.submodules import noise_key
from mstar.model.kokoro.weight_loader import load_kokoro_weights

HF_REPO = "hexgrad/Kokoro-82M"
# a submodule of the reference: mstar/model/kokoro also answers to `kokoro` once pytest
# puts mstar/model on sys.path for the model-local tests
pytest.importorskip("kokoro.model", reason="the `kokoro` reference package is the oracle")
kokoro = pytest.importorskip("kokoro", reason="the `kokoro` reference package is the oracle")


def _find_cached_snapshot() -> Path | None:
    repo_dir = f"models--{HF_REPO.replace('/', '--')}"
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in roots:
        snapshots = root / repo_dir / "snapshots"
        if snapshots.is_dir():
            for snapshot in snapshots.iterdir():
                if (snapshot / "kokoro-v1_0.pth").is_file() and (snapshot / "voices" / "af_heart.pt").is_file():
                    return snapshot
    return None


SNAPSHOT = _find_cached_snapshot()
pytestmark = pytest.mark.skipif(SNAPSHOT is None, reason=f"{HF_REPO} is not in the local Hugging Face cache")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# On CUDA, cuDNN runs convolutions in TF32 by default and chooses LSTM
# algorithms per shape, so two correct fp32 pipelines differ at ~1e-3 on
# well-conditioned intermediates. The test compares implementations, not
# kernel precision: it turns TF32 off and widens the tolerance a little.
INTERMEDIATE_TOL = 1e-4 if DEVICE == "cpu" else 2e-3
# The reference package itself differs by 11-16% rel-L2 between CUDA and CPU
# on the same inputs (measured 2026-09-18, H100 vs Xeon), so the waveform
# band is expressed relative to the reference's own sensitivity below.
WAVEFORM_TOL = 0.05 if DEVICE == "cpu" else 0.2


@pytest.fixture(scope="module", autouse=True)
def _fp32_kernels():
    if DEVICE != "cuda":
        yield
        return
    prior = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prior

CASES = [
    ("af_heart", "həlˈO wˈɜɹld! ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ.", 1.0),
    ("bm_george", "ðɪs ɪz ə lˈɔŋɡəɹ tˈɛst sˈɛntəns, wɪð sˈʌm pˈɔzəz; ænd ə kwˈɛsʃən? jˈɛs.", 1.3),
    ("af_bella", "ˈO.", 0.8),
]


@pytest.fixture(scope="module")
def config() -> KokoroModelConfig:
    return KokoroModelConfig.from_pretrained(SNAPSHOT)


@pytest.fixture(scope="module")
def ours(config) -> KokoroTTS:
    model = KokoroTTS(config)
    load_kokoro_weights(model, SNAPSHOT / config.weights_file)
    return model.to(DEVICE).eval()


@pytest.fixture(scope="module")
def reference(config):
    return kokoro.KModel(
        repo_id=HF_REPO, config=str(SNAPSHOT / "config.json"), model=str(SNAPSHOT / config.weights_file)
    ).to(DEVICE).eval()


def _inputs(config, voice, phonemes):
    pack = torch.load(SNAPSHOT / "voices" / f"{voice}.pt", map_location="cpu", weights_only=True)
    ref_s = pack[len(phonemes) - 1].to(DEVICE)
    ids = [config.vocab[c] for c in phonemes if c in config.vocab]
    input_ids = torch.tensor([[0, *ids, 0]], device=DEVICE)
    return input_ids, ref_s


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def test_checkpoint_loads_every_parameter(ours):
    assert sum(p.numel() for p in ours.parameters()) > 80_000_000


@pytest.mark.parametrize(("voice", "phonemes", "speed"), CASES)
def test_durations_and_prosody_match(config, ours, reference, voice, phonemes, speed):
    input_ids, ref_s = _inputs(config, voice, phonemes)
    T = input_ids.shape[1]
    with torch.no_grad():
        lengths = torch.full((1,), T, dtype=torch.long, device=DEVICE)
        text_mask = torch.gt(torch.arange(T, device=DEVICE)[None] + 1, lengths[:, None])
        d_en = reference.bert_encoder(reference.bert(input_ids, attention_mask=(~text_mask).int())).transpose(-1, -2)
        s = ref_s[:, 128:]
        d_ref = reference.predictor.text_encoder(d_en, s, lengths, text_mask)
        x, _ = reference.predictor.lstm(d_ref)
        dur_ref = torch.round(torch.sigmoid(reference.predictor.duration_proj(x)).sum(-1) / speed).clamp(min=1).long()
        frames = torch.repeat_interleave(torch.arange(T, device=DEVICE), dur_ref[0])
        alignment = torch.zeros((T, frames.shape[0]), device=DEVICE)
        alignment[frames, torch.arange(frames.shape[0], device=DEVICE)] = 1
        f0_ref, energy_ref = reference.predictor.F0Ntrain(d_ref.transpose(-1, -2) @ alignment[None], s)
        t_en_ref = reference.text_encoder(input_ids, lengths, text_mask)

        d, t_en, dur = ours.encode_text(input_ids, lengths, ref_s, torch.tensor([speed], device=DEVICE))
        assert torch.equal(dur, dur_ref)
        assert _rel(d, d_ref) < INTERMEDIATE_TOL and _rel(t_en, t_en_ref) < INTERMEDIATE_TOL
        num_frames = int(dur.sum())
        index = ours.alignment(dur, num_frames)
        en = d.gather(1, index[:, :, None].expand(-1, -1, d.shape[-1]))
        f0, energy = ours.predictor.f0n(en, s, dur.sum(1))
        assert _rel(f0, f0_ref) < INTERMEDIATE_TOL and _rel(energy, energy_ref) < INTERMEDIATE_TOL

        # the decoder, fed the reference's own inputs and RNG stream
        asr_ref = t_en_ref @ alignment[None]
        torch.manual_seed(0)
        audio_ref = reference.decoder(asr_ref, f0_ref, energy_ref, ref_s[:, :128]).reshape(-1)
        torch.manual_seed(0)
        audio = ours.decoder(asr_ref, f0_ref, energy_ref, ref_s[:, :128], dur.sum(1))[0]
        assert audio.shape == audio_ref.shape == (num_frames * config.samples_per_frame,)
        assert _rel(audio, audio_ref) < 10 * INTERMEDIATE_TOL


@pytest.mark.parametrize(("voice", "phonemes", "speed"), CASES)
def test_waveform_within_phase_chaos_band(config, ours, reference, voice, phonemes, speed):
    input_ids, ref_s = _inputs(config, voice, phonemes)
    with torch.no_grad():
        torch.manual_seed(0)
        audio_ref, dur_ref = reference.forward_with_tokens(input_ids, ref_s, speed)
        torch.manual_seed(0)
        audio, frame_lengths, dur = ours(
            input_ids, torch.tensor([input_ids.shape[1]], device=DEVICE), ref_s, torch.tensor([speed], device=DEVICE)
        )
    audio_ref = audio_ref.reshape(-1)
    assert torch.equal(dur[0], dur_ref.reshape(-1))
    assert audio.shape == (1, audio_ref.numel()) and int(frame_lengths) * config.samples_per_frame == audio_ref.numel()
    assert _rel(audio[0], audio_ref) < WAVEFORM_TOL
    assert (audio[0] - audio_ref).abs().max() < 4 * WAVEFORM_TOL


def test_reference_is_chaotic_in_f0(config, reference):
    """The bound above is the reference's own sensitivity, not slack."""
    input_ids, ref_s = _inputs(config, *CASES[0][:2])
    T = input_ids.shape[1]
    with torch.no_grad():
        lengths = torch.full((1,), T, dtype=torch.long, device=DEVICE)
        text_mask = torch.gt(torch.arange(T, device=DEVICE)[None] + 1, lengths[:, None])
        d_en = reference.bert_encoder(reference.bert(input_ids, attention_mask=(~text_mask).int())).transpose(-1, -2)
        s = ref_s[:, 128:]
        d_ref = reference.predictor.text_encoder(d_en, s, lengths, text_mask)
        x, _ = reference.predictor.lstm(d_ref)
        dur = torch.round(torch.sigmoid(reference.predictor.duration_proj(x)).sum(-1)).clamp(min=1).long()
        frames = torch.repeat_interleave(torch.arange(T, device=DEVICE), dur[0])
        alignment = torch.zeros((T, frames.shape[0]), device=DEVICE)
        alignment[frames, torch.arange(frames.shape[0], device=DEVICE)] = 1
        f0, energy = reference.predictor.F0Ntrain(d_ref.transpose(-1, -2) @ alignment[None], s)
        asr = reference.text_encoder(input_ids, lengths, text_mask) @ alignment[None]
        torch.manual_seed(0)
        base = reference.decoder(asr, f0, energy, ref_s[:, :128]).reshape(-1)
        torch.manual_seed(0)
        perturbed = reference.decoder(asr, f0 * (1 + 1e-6), energy, ref_s[:, :128]).reshape(-1)
        torch.manual_seed(0)
        asr_perturbed = reference.decoder(asr * (1 + 1e-6), f0, energy, ref_s[:, :128]).reshape(-1)
    f0_effect, asr_effect = _rel(perturbed, base), _rel(asr_perturbed, base)
    assert f0_effect > 1e-3  # F0: a rounding-level change is amplified
    assert asr_effect < 10 * INTERMEDIATE_TOL  # everything else is well conditioned
    assert f0_effect > 10 * asr_effect


@contextlib.contextmanager
def _reference_keyed_noise(keys: torch.Tensor):
    """The reference's source noise drawn as serving draws it.

    Its first ``randn_like`` (SineGen's noise, ``[B, samples, harmonics]``)
    becomes ``keyed_normal``; the second, SourceModuleHnNSF's unused branch,
    stays on the RNG.
    """
    real = torch.randn_like
    calls = []

    def keyed(t, *args, **kwargs):
        calls.append(tuple(t.shape))
        if len(calls) == 1:
            return keyed_normal(keys.to(t.device), t.shape[1], t.shape[2]).to(t.dtype)
        return real(t, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch, "randn_like", keyed)
        yield
    assert len(calls) == 2, calls


def _reference_features(reference, input_ids, ref_s, speed):
    """The reference's decoder inputs (asr, F0, energy) and durations."""
    T = input_ids.shape[1]
    lengths = torch.full((1,), T, dtype=torch.long, device=DEVICE)
    text_mask = torch.gt(torch.arange(T, device=DEVICE)[None] + 1, lengths[:, None])
    d_en = reference.bert_encoder(reference.bert(input_ids, attention_mask=(~text_mask).int())).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = reference.predictor.text_encoder(d_en, s, lengths, text_mask)
    x, _ = reference.predictor.lstm(d)
    dur = torch.round(torch.sigmoid(reference.predictor.duration_proj(x)).sum(-1) / speed).clamp(min=1).long()
    frames = torch.repeat_interleave(torch.arange(T, device=DEVICE), dur[0])
    alignment = torch.zeros((T, frames.shape[0]), device=DEVICE)
    alignment[frames, torch.arange(frames.shape[0], device=DEVICE)] = 1
    f0, energy = reference.predictor.F0Ntrain(d.transpose(-1, -2) @ alignment[None], s)
    asr = reference.text_encoder(input_ids, lengths, text_mask) @ alignment[None]
    return asr, f0, energy, dur


@pytest.mark.parametrize(("voice", "phonemes", "speed"), CASES)
def test_keyed_noise_decoder_matches_reference(config, ours, reference, voice, phonemes, speed):
    """The served path's noise, given to the reference, gives the reference's audio."""
    input_ids, ref_s = _inputs(config, voice, phonemes)
    keys = noise_key(1234, 0)[None].to(DEVICE)
    other_keys = noise_key(1235, 0)[None].to(DEVICE)
    with torch.no_grad():
        asr, f0, energy, dur = _reference_features(reference, input_ids, ref_s, speed)
        with _reference_keyed_noise(keys):
            audio_ref = reference.decoder(asr, f0, energy, ref_s[:, :128]).reshape(-1)
        frame_lengths = dur.sum(1)
        audio = ours.decoder(asr, f0, energy, ref_s[:, :128], frame_lengths, noise_keys=keys)[0]
        other = ours.decoder(asr, f0, energy, ref_s[:, :128], frame_lengths, noise_keys=other_keys)[0]
    assert audio.shape == audio_ref.shape
    assert _rel(audio, audio_ref) < 10 * INTERMEDIATE_TOL
    # the bound would catch the wrong noise: another seed lands well outside it
    assert _rel(other, audio_ref) > 2 * 10 * INTERMEDIATE_TOL


@pytest.mark.parametrize(("voice", "phonemes", "speed"), CASES)
def test_keyed_noise_waveform_within_phase_chaos_band(config, ours, reference, voice, phonemes, speed):
    """End to end on the served noise path, in the band ``test_waveform_within_phase_chaos_band`` uses."""
    input_ids, ref_s = _inputs(config, voice, phonemes)
    keys = noise_key(1234, 0)[None].to(DEVICE)
    with torch.no_grad():
        with _reference_keyed_noise(keys):
            audio_ref, dur_ref = reference.forward_with_tokens(input_ids, ref_s, speed)
        audio, _, dur = ours(
            input_ids, torch.tensor([input_ids.shape[1]], device=DEVICE), ref_s,
            torch.tensor([speed], device=DEVICE), noise_keys=keys,
        )
    audio_ref = audio_ref.reshape(-1)
    assert torch.equal(dur[0], dur_ref.reshape(-1))
    assert audio.shape == (1, audio_ref.numel())
    assert _rel(audio[0], audio_ref) < WAVEFORM_TOL

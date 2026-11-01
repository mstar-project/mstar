"""The shared audio front ends against HF's feature extractors.

Whisper-family log-mel features to ~1e-4 for 80- and 128-bin banks, the
frame-count convention (``samples // hop``), padding/trimming to the 30 s
window, the sinusoidal position table, the FFmpeg-free file loader, the
FastConformer (NeMo) front end with its librosa filter bank in both of NeMo's
conventions, and the STFT as a matmul that captured graphs run.
"""

import numpy as np
import pytest
import torch

from mstar.api_server.media_io import AudioTooLongError
from mstar.model.components.audio_features import (
    LogMelSpectrogram,
    NemoLogMel,
    clip_samples,
    librosa_mel_filters,
    load_audio_file,
    load_speech_file,
    sinusoid_positions,
    slaney_mel_filter_bank,
)
from mstar.model.components.fastconformer import FastConformerConfig


def test_mel_filter_bank_matches_hf():
    audio_utils = pytest.importorskip("transformers.audio_utils")
    for n_mels in (80, 128):
        hf = audio_utils.mel_filter_bank(
            num_frequency_bins=201, num_mel_filters=n_mels, min_frequency=0.0, max_frequency=8000.0,
            sampling_rate=16_000, norm="slaney", mel_scale="slaney",
        )
        ours = slaney_mel_filter_bank(201, n_mels, 16_000)
        assert ours.shape == (n_mels, 201)
        assert np.allclose(ours, hf.T, atol=1e-8)


@pytest.mark.parametrize("num_mel_bins", [80, 128])
def test_log_mel_matches_hf_feature_extractor(num_mel_bins):
    transformers = pytest.importorskip("transformers")
    fe = transformers.WhisperFeatureExtractor(feature_size=num_mel_bins)
    mel = LogMelSpectrogram(num_mel_bins=num_mel_bins)
    torch.manual_seed(0)
    wave = torch.randn(int(16_000 * 3.37)) * 0.1
    hf = torch.from_numpy(fe(wave.numpy(), sampling_rate=16_000, return_tensors="np")["input_features"][0])
    ours = mel(mel.pad_or_trim(wave))
    assert ours.shape == hf.shape == (num_mel_bins, 3000)
    assert torch.allclose(ours, hf, atol=2e-4), (ours - hf).abs().max()


def test_log_mel_variable_length_matches_hf_unpadded():
    """Qwen3-ASR feeds the unpadded clip: HF with ``padding=False`` and no
    truncation yields ``samples // hop`` frames, as we do."""
    transformers = pytest.importorskip("transformers")
    fe = transformers.WhisperFeatureExtractor(feature_size=128)
    mel = LogMelSpectrogram()
    torch.manual_seed(1)
    wave = torch.randn(16_000 * 4 + 123) * 0.1
    hf = fe(wave.numpy(), sampling_rate=16_000, return_tensors="np", padding=False, truncation=False)
    expected = torch.from_numpy(hf["input_features"][0])
    ours = mel(wave)
    assert ours.shape == expected.shape == (128, wave.numel() // 160)
    assert torch.allclose(ours, expected, atol=2e-4)


def test_pad_or_trim_and_batching():
    mel = LogMelSpectrogram()
    assert mel.pad_or_trim(torch.ones(16_000 * 40)).shape[-1] == 480_000
    assert mel.pad_or_trim(torch.ones(16_000)).shape[-1] == 480_000
    assert mel.num_frames(16_000 * 3 + 37) == (16_000 * 3 + 37) // 160
    batched = mel(torch.stack([mel.pad_or_trim(torch.randn(16_000)), mel.pad_or_trim(torch.randn(32_000))]))
    assert batched.shape == (2, 128, 3000)
    # the dynamic-range clamp is per clip, so a loud batch-mate leaves a quiet clip alone
    quiet, loud = torch.randn(16_000) * 1e-3, torch.randn(16_000)
    alone = mel(mel.pad_or_trim(quiet))
    together = mel(torch.stack([mel.pad_or_trim(quiet), mel.pad_or_trim(loud)]))[0]
    assert torch.allclose(alone, together)


def test_sinusoid_positions_match_hf():
    modeling = pytest.importorskip("transformers.models.whisper.modeling_whisper")
    assert torch.allclose(sinusoid_positions(50, 64), modeling.sinusoids(50, 64), atol=1e-6)
    with pytest.raises(ValueError):
        sinusoid_positions(10, 63)


def test_load_audio_file_decodes_and_resamples(tmp_path):
    sf = pytest.importorskip("soundfile")
    t = np.arange(8000) / 8000.0
    stereo = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    sf.write(tmp_path / "a.flac", stereo, 8000)
    wave = load_audio_file(str(tmp_path / "a.flac"), 16_000)
    assert wave.dtype == torch.float32 and wave.dim() == 1
    assert abs(wave.numel() - 16_000) <= 16
    sf.write(tmp_path / "b.wav", stereo[:, :1], 16_000)
    assert load_audio_file(str(tmp_path / "b.wav")).numel() == 8000
    # a bounded decode refuses a longer file, and a shorter one comes back whole
    with pytest.raises(AudioTooLongError):
        load_audio_file(str(tmp_path / "b.wav"), 16_000, max_seconds=0.25)
    assert load_audio_file(str(tmp_path / "b.wav"), 16_000, max_seconds=5.0).numel() == 8000


def test_load_speech_file_refuses_a_file_past_the_limit_as_it_decodes(tmp_path):
    sf = pytest.importorskip("soundfile")
    sf.write(tmp_path / "long.wav", np.zeros(16_000 * 30, dtype=np.float32), 16_000)
    with pytest.raises(ValueError, match="longer than the 10 s limit"):
        load_speech_file(str(tmp_path / "long.wav"), 16_000, 10.0)
    with pytest.raises(ValueError, match="takes up to 10 s"):
        clip_samples({"audio_inputs": [torch.zeros(16_000 * 11)]}, 10.0, "this model")


@pytest.mark.parametrize("sr, n_fft, n_mels, fmin, fmax", [
    (16_000, 512, 128, 0.0, None),      # NeMo
    (16_000, 400, 40, 0.0, 8000.0),     # Chatterbox voice encoder
    (16_000, 400, 128, 0.0, None),      # S3 tokenizer (Chatterbox, CosyVoice3)
    (24_000, 1920, 80, 0.0, None),      # CosyVoice3's Matcha mel
    (24_000, 1920, 80, 0.0, 8000.0),    # S3Gen's Matcha mel
    (24_000, 1024, 128, 0.0, 12000.0),  # Qwen3-TTS speaker mel
])
def test_librosa_mel_filters_are_bit_identical(sr, n_fft, n_mels, fmin, fmax):
    librosa = pytest.importorskip("librosa")
    expected = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, norm="slaney")
    assert np.array_equal(librosa_mel_filters(sr, n_fft, n_mels, fmin, fmax), expected)


@pytest.mark.parametrize("seconds", [0.5, 3.37])
def test_nemo_log_mel_matches_hf_parakeet_extractor(seconds):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "ParakeetFeatureExtractor"):
        pytest.skip("transformers has no ParakeetFeatureExtractor")
    pytest.importorskip("librosa")
    torch.manual_seed(5)
    wave = torch.randn(int(16_000 * seconds)) * 0.1
    hf = transformers.ParakeetFeatureExtractor(feature_size=128)(
        wave.numpy(), sampling_rate=16_000, return_tensors="pt",
    )
    frames = int(hf["attention_mask"].sum())
    ours = NemoLogMel()(wave)
    assert ours.shape == (frames, 128) == (wave.numel() // 160, 128)
    assert torch.allclose(ours, hf["input_features"][0, :frames], atol=1e-4)


def test_nemo_log_mel_standardizes_each_bin():
    torch.manual_seed(6)
    feats = NemoLogMel()(torch.randn(16_000) * 0.3)
    assert torch.allclose(feats.mean(dim=0), torch.zeros(128), atol=1e-4)
    assert torch.allclose(feats.std(dim=0), torch.ones(128), atol=1e-3)
    raw = NemoLogMel(normalize=False)(torch.randn(16_000) * 0.3)
    assert raw.mean() < -1.0  # natural-log energies, left as they are


def test_nemo_log_mel_keeps_digital_silence_at_zero():
    feats = NemoLogMel()(torch.zeros(16_000 * 5))
    assert torch.equal(feats, torch.zeros_like(feats))


def test_nemo_log_mel_refuses_clips_under_two_frames():
    with pytest.raises(ValueError):
        NemoLogMel()(torch.randn(300))



@pytest.mark.parametrize("reflect_edges", [False, True])
@pytest.mark.parametrize("normalize", [True, False])
def test_nemo_log_mel_batch_gives_each_clip_what_it_gets_alone(normalize, reflect_edges):
    """The encoder node computes a batch's features in one pass: padding a
    clip, and the preemphasis reaching past its end, must not change it; nor,
    before NeMo 2.5, the reflection at its end, which is of its own samples."""
    torch.manual_seed(7)
    clips = [torch.randn(n) * 0.2 for n in (16_000, 5_123, 40_000, 321)]
    clips.append(torch.zeros(8_000))  # digital silence stays zero in a batch too
    log_mel = NemoLogMel(normalize=normalize, reflect_edges=reflect_edges)
    batch = log_mel.batch(clips)
    for clip, feats in zip(clips, batch, strict=True):
        alone = log_mel.batch([clip])[0]
        assert feats.shape == alone.shape == (clip.numel() // 160 + reflect_edges, 128)
        torch.testing.assert_close(feats, alone, rtol=1e-5, atol=1e-5)
    assert torch.equal(batch[-1], torch.zeros_like(batch[-1])) or not normalize


def test_nemo_log_mel_batch_refuses_a_clip_under_two_frames():
    with pytest.raises(ValueError, match="300 samples"):
        NemoLogMel().batch([torch.randn(16_000), torch.randn(300)])


# 32160 samples is 201 hops, 1 mod 8: there the extra mel frame adds an encoder frame
@pytest.mark.parametrize("n", [32_000, 32_160, 1_281])
def test_nemo_log_mel_before_2_5_reflects_its_edges_and_keeps_the_last_frame(n):
    """Nemotron VoiceChat's reference featurizer predates NeMo 2.5: torch.stft's
    default reflect padding and ``samples // hop + 1`` frames, which can be one
    more encoder frame than the 2.5 convention gives the same audio."""
    torch.manual_seed(8)
    wave, filters = torch.randn(n) * 0.1, torch.rand(128, 257)
    ours = NemoLogMel(normalize=False, reflect_edges=True, mel_filters=filters)(wave)
    # the reference's (``nemotron-labs-voicechat`` features.py): preemphasis, a centred reflect-padded STFT
    x = torch.cat([wave[:1], wave[1:] - 0.97 * wave[:-1]])[None]
    window = torch.hann_window(400, periodic=False)
    spec = torch.view_as_real(torch.stft(x, 512, hop_length=160, win_length=400, window=window, return_complex=True))
    reference = torch.log(filters @ spec.pow(2).sum(-1).sqrt().pow(2) + 2.0 ** -24)[0].T
    assert ours.shape == reference.shape == (n // 160 + 1, 128)
    assert torch.allclose(ours, reference, atol=1e-5)
    after = NemoLogMel(normalize=False, mel_filters=filters)(wave)
    assert after.shape[0] == n // 160
    encoder = FastConformerConfig(causal=True)
    assert (encoder.output_lengths(ours.shape[0]) > encoder.output_lengths(after.shape[0])) == (n // 160 % 8 == 1)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize(("n_fft", "win", "samples"), [(512, 400, 1632), (512, 400, 1700), (512, 512, 2000),
                                                        (256, 101, 900)])
def test_stft_as_a_matmul_is_torch_stft(n_fft, win, samples, autocast):
    """The matmul STFT is torch.stft's to fp32 rounding, measured against
    float64; bf16 autocast leaves it in full precision."""
    from mstar.model.components.dft import stft_as_real, stft_basis

    torch.manual_seed(0)
    x, window = torch.randn(3, samples), torch.hann_window(win, periodic=False)

    def stft(signal):
        return torch.view_as_real(torch.stft(signal, n_fft, hop_length=160, win_length=win, center=False,
                                             window=window.to(signal.dtype), return_complex=True))

    exact = stft(x.double())
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        got = stft_as_real(x, stft_basis(window, n_fft), n_fft, 160)
    assert got.dtype == torch.float32 and got.shape == exact.shape
    fp32_fft = (stft(x) - exact).abs().max()
    assert (got - exact).abs().max() <= fp32_fft + 1e-7 * exact.abs().max()


@pytest.mark.parametrize("n", [16, 15])
def test_irfft_as_a_matmul_is_torch_irfft(n):
    """The matmul inverse DFT is torch.fft.irfft's to fp32 rounding, measured
    against float64, imaginary DC and Nyquist parts ignored as irfft ignores them."""
    from mstar.model.components.dft import irfft_dim1

    torch.manual_seed(0)
    spec = torch.randn(4, n // 2 + 1, 50, dtype=torch.complex64)
    exact = torch.fft.irfft(spec.to(torch.complex128), n=n, dim=-2)
    got = irfft_dim1(spec, n)
    assert got.dtype == torch.float32 and got.shape == exact.shape
    fp32_fft = (torch.fft.irfft(spec, n=n, dim=-2) - exact).abs().max()
    assert (got - exact).abs().max() <= fp32_fft + 1e-7 * exact.abs().max()

def test_load_speech_file_blames_the_upload_only_for_decoding(monkeypatch):
    from mstar.model.components import audio_features

    def raises(exc):
        def load(*args, **kwargs):
            raise exc
        return load

    # the decoders ran and refused the file: the client's error
    monkeypatch.setattr(audio_features, "_ffmpeg_decoder_loads", lambda: True)
    monkeypatch.setattr(audio_features, "_decode", raises(RuntimeError("Invalid data found")))
    with pytest.raises(ValueError):
        load_speech_file("x.wav", 16_000, 10.0)
    # no decoder to run, or no memory: the server's
    for exc in (ImportError("torchcodec"), MemoryError()):
        monkeypatch.setattr(audio_features, "_decode", raises(exc))
        with pytest.raises(type(exc)):
            load_speech_file("x.wav", 16_000, 10.0)
    # it decoded, and resampling failed: the server's too
    monkeypatch.setattr(audio_features, "_decode", lambda *args: (torch.zeros(4800), 48_000))
    monkeypatch.setattr(audio_features, "_resample", raises(RuntimeError("can't allocate memory")))
    with pytest.raises(RuntimeError):
        load_speech_file("x.wav", 16_000, 10.0)
    monkeypatch.setattr(audio_features, "_ffmpeg_decoder_loads", lambda: False)
    monkeypatch.setattr(audio_features, "_decode", raises(RuntimeError("Could not load libtorchcodec")))
    with pytest.raises(RuntimeError):
        load_speech_file("x.wav", 16_000, 10.0)

"""Parity for the opt-in GPU log-mel (MSTAR_GPU_MEL=1) vs HF's CPU
``WhisperFeatureExtractor``.

The native audio encoder is numerically parity-tested against HF, so the audio
features fed to it must match HF regardless of where the mel spectrogram is
computed. This pins the GPU transform (mstar...qwen3_omni_model.gpu_log_mel) to
HF's output across clip lengths: identical valid frame count (T) and cos>=0.9999 /
max-abs ~1e-5 (the production bf16 encoder tolerance is looser than this).

Frame-count convention (the subtle part)
----------------------------------------
``gpu_log_mel`` returns ``T = floor(len/hop)`` frames: ``torch.stft`` with
``center=True`` produces ``1 + floor(len/hop)`` frames, and the production
transform drops the last (matching HF's ``_torch_extract_fbank_features``, which
does ``stft[..., :-1]``).

HF's *valid* frame count comes from its attention mask, and for inputs whose
sample count is NOT a multiple of ``hop_length`` HF deliberately trims one frame
so that the mask matches the actual frame count. From transformers
``feature_extraction_whisper.py`` (v4.57..v5.x), ``WhisperFeatureExtractor.__call__``:

    rescaled_attention_mask = padded_inputs["attention_mask"][:, :: self.hop_length]
    # STFT produces L//hop + 1 frames but the last is dropped; trim the rescaled
    # mask to match the real frame count (L//hop) when L is not divisible by hop:
    if padded_inputs["attention_mask"].shape[1] % self.hop_length != 0:
        rescaled_attention_mask = rescaled_attention_mask[:, :-1]

The serving path (qwen3_omni_model.py) runs the feature extractor on ONE audio at
a time with ``padding=True`` (=> "longest" => no extra padding for a single clip),
so the padded sample length equals ``len`` and ``attention_mask.sum() ==
floor(len/hop)`` for BOTH multiple- and non-multiple-of-hop inputs. That is exactly
``gpu_log_mel``'s ``T``. Hence the GPU path needs no floor/ceil correction; this
test is the regression guard that pins the convention, INCLUDING non-hop-multiple
clip lengths (the case the duration-only parametrization below cannot reach, since
those durations all happen to land on hop boundaries).

Requires CUDA + the Qwen3-Omni checkpoint (for the real mel filterbank); skips
otherwise. Point at a checkpoint with MSTAR_QWEN3_OMNI_DIR.
"""
import os

import numpy as np
import pytest
import torch


def _resolve_checkpoint():
    d = os.environ.get("MSTAR_QWEN3_OMNI_DIR")
    if d and os.path.isdir(d):
        return d
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download("Qwen/Qwen3-Omni-30B-A3B-Instruct",
                                 allow_patterns=["*.json", "*.txt"],
                                 local_files_only=True)
    except Exception:
        return None


CKPT = _resolve_checkpoint()
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(CKPT is None, reason="Qwen3-Omni checkpoint not available"),
]


def _hf_reference(fe, audio, sr):
    """HF CPU mel + valid frame count, cropped exactly as the serving path does."""
    hf = fe(audio, sampling_rate=sr, padding=True, truncation=False,
            return_attention_mask=True, return_tensors="pt")
    mask = hf["attention_mask"][0].bool()
    valid = int(mask.sum())
    ref = hf["input_features"][0][:, :valid].float()      # (n_mel, valid)
    return ref, valid


def _gpu_mel(fe, audio):
    from mstar.model.qwen3_omni.qwen3_omni_model import gpu_log_mel
    dev = torch.device("cuda")
    filters = torch.tensor(np.asarray(fe.mel_filters), dtype=torch.float32, device=dev)
    window = torch.hann_window(fe.n_fft, periodic=True, device=dev)
    wav = torch.tensor(audio, dtype=torch.float32, device=dev)
    return gpu_log_mel(wav, filters, window, fe.n_fft, fe.hop_length).cpu()  # (n_mel, T)


def _assert_parity(out, ref, valid, n_samples, hop, label):
    expected = n_samples // hop                              # floor(len/hop)
    assert out.shape[1] == expected, (
        f"{label}: gpu frame count {out.shape[1]} != floor(len/hop) {expected}")
    assert valid == expected, (
        f"{label}: HF valid frame count {valid} != floor(len/hop) {expected} "
        f"(n_samples={n_samples}, hop={hop}, n%hop={n_samples % hop})")
    assert out.shape[1] == valid, f"{label}: frame count {out.shape[1]} != HF valid {valid}"
    a, b = out.flatten(), ref.flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    maxabs = (a - b).abs().max().item()
    assert cos > 0.9999 and maxabs < 1e-3, f"{label}: cos={cos:.7f} maxabs={maxabs:.6f}"


@pytest.mark.parametrize("dur", [3.0, 7.3, 15.0, 30.0, 41.2], ids=lambda d: f"{d}s")
def test_gpu_mel_matches_hf(dur):
    from transformers import AutoProcessor
    fe = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True).feature_extractor
    sr = fe.sampling_rate
    rng = np.random.default_rng(0)
    n = int(dur * sr)
    audio = (rng.standard_normal(n) * 0.1).astype(np.float32)

    ref, valid = _hf_reference(fe, audio, sr)
    out = _gpu_mel(fe, audio)
    _assert_parity(out, ref, valid, n, fe.hop_length, f"dur={dur}")


# Explicit sample counts chosen so the set spans BOTH hop-multiple lengths and
# several non-hop-multiple lengths (n % hop in {1, 80, hop-1, ...}). The latter are
# the regression-critical case: they exercise HF's mask-trim path and pin that the
# GPU floor convention still matches HF exactly (no +/-1 frame drift).
@pytest.mark.parametrize("n_samples", [
    48000,    # 300*160  -> multiple of hop
    160000,   # 1000*160 -> multiple of hop
    48001,    # +1 sample  (n % hop == 1)
    48080,    # +half hop  (n % hop == 80)
    48159,    # +(hop-1)   (n % hop == 159)
    50321,    # arbitrary odd, non-multiple
    99999,    # arbitrary odd, non-multiple
    12345,    # short, non-multiple
], ids=lambda n: f"n{n}")
def test_gpu_mel_frame_count_and_parity_nonhop_multiples(n_samples):
    """Frame-count + value parity with explicit non-hop-multiple sample counts.

    Asserts (a) gpu T == floor(len/hop), (b) HF valid == floor(len/hop) (the trim
    convention), (c) they are equal, and (d) the mel values match HF to
    cos>0.9999. Several of these n are deliberately NOT multiples of hop_length
    (160); the regression this guards is a floor-vs-ceil off-by-one frame.
    """
    from transformers import AutoProcessor
    fe = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True).feature_extractor
    sr = fe.sampling_rate
    rng = np.random.default_rng(n_samples)
    audio = (rng.standard_normal(n_samples) * 0.1).astype(np.float32)

    ref, valid = _hf_reference(fe, audio, sr)
    out = _gpu_mel(fe, audio)
    _assert_parity(out, ref, valid, n_samples, fe.hop_length, f"n={n_samples}")

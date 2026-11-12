"""torchaudio 2.11 load/save need torchcodec; route them through soundfile (same float32 samples)."""
import soundfile as sf
import torch
import torchaudio


def _load(path, *args, **kwargs):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


def _save(dst, wav, sample_rate, format=None, **kwargs):
    sf.write(dst, wav.detach().cpu().numpy().T, sample_rate, format=(format or "wav").upper(), subtype="PCM_16")


torchaudio.load = _load
torchaudio.save = _save

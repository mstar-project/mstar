"""Prepare a clean USER-only 16 kHz-mono input clip from the VoiceChat-11B demo.

The base repo's demo wavs (turn_taking.wav, ...) are 2-channel recordings of the
WHOLE conversation (user on the left, agent on the right) meant for listening.
For an INPUT we want the user side only, plus trailing silence for the reply
(this is a frame-synchronous model: it answers in the frames *after* the user
stops). We take the left channel, isolate the first user turn, append
``trailing_s`` of silence, and resample to 16 kHz mono. Results are cached in the
temp dir keyed by ``trailing_s``.
"""
import os
import tempfile

_DEMO_WAV = "turn_taking.wav"       # base VoiceChat-11B repo; L=user, R=agent


def resolve_demo() -> str:
    from huggingface_hub import hf_hub_download

    from mstar.model.registry import HF_MODELS

    repo = HF_MODELS["nemotron_duplex"]["model_path_hf"]
    return hf_hub_download(repo, _DEMO_WAV, cache_dir=os.environ.get("NEMOTRON_DUPLEX_CACHE_DIR"))


def first_user_turn(user, sr, trailing_s):
    """First contiguous user utterance >= ~1.2 s (short gaps bridged), 0.3 s lead
    pad, and ``trailing_s`` of trailing silence (zero-padded past EOF)."""
    import numpy as np

    w = int(sr * 0.1)                                  # 100 ms frames
    n = len(user) // w
    e = np.sqrt((user[: n * w].reshape(n, w) ** 2).mean(1))
    act = e > e.max() * 0.08
    for i in range(1, n - 3):                          # bridge <0.3 s gaps
        if not act[i] and act[i - 1] and act[i : i + 3].any():
            act[i] = True
    runs, i = [], 0
    while i < n:
        if act[i]:
            j = i
            while j < n and act[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        return user
    s, ep = next((r for r in runs if (r[1] - r[0]) * 0.1 >= 1.2), runs[0])
    a = max(0, s * w - int(sr * 0.3))
    b = ep * w + int(sr * trailing_s)
    clip = user[a : min(len(user), b)]
    if b > len(user):
        clip = np.concatenate([clip, np.zeros(b - len(user), dtype=clip.dtype)])
    return clip


def prepare_user_clip(trailing_s=6.0, target_sr=16000) -> str:
    """Return a path to a cached 16 kHz-mono user-only clip with ``trailing_s`` of
    trailing silence."""
    out = os.path.join(tempfile.gettempdir(), f"nemotron_duplex_user_{int(round(trailing_s))}s.wav")
    if os.path.exists(out):
        return out
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    src = resolve_demo()
    d, sr = sf.read(src, dtype="float32")
    user = d[:, 0] if d.ndim == 2 else d               # left channel = user
    clip = first_user_turn(user, sr, trailing_s)
    if sr != target_sr:
        clip = AF.resample(torch.from_numpy(clip.copy()), sr, target_sr).numpy()
    sf.write(out, clip, target_sr)
    return out

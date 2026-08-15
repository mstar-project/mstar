#!/usr/bin/env python3
"""Nemotron-Duplex client: send user speech, get back agent text + agent speech.

Usage:
    python test/nemotron_duplex/duplex_request.py                     # default sample input
    python test/nemotron_duplex/duplex_request.py --audio user.wav
    python test/nemotron_duplex/duplex_request.py --text-only
    python test/nemotron_duplex/duplex_request.py --output agent.wav -n 3

With no --audio, a clean USER-only input is prepared from the base VoiceChat-11B
demo (``turn_taking.wav``). Those demo wavs are 2-channel recordings of the WHOLE
conversation (user on the left, agent on the right) meant for listening — feeding
them as-is would let the model hear its own side, so we take the left (user)
channel, isolate the first user turn, and resample to 16 kHz mono.
"""
import argparse
import os
import sys
import tempfile
import threading
import time

from _env import get_base_url, load_env

from mstar.client.client import MStarClient

_DEMO_WAV = "turn_taking.wav"     # base VoiceChat-11B repo; L=user, R=agent
_PREPARED = os.path.join(tempfile.gettempdir(), "nemotron_duplex_default_user_turn.wav")


def _first_user_turn(user, sr, trailing_s=6.0):
    """First contiguous user utterance >= ~1.2 s (short gaps bridged), with 0.3 s
    lead pad and ``trailing_s`` of trailing silence.

    The trailing silence matters: this is a frame-synchronous model that stays
    silent (EOS) *while* the user is speaking and only produces its reply in the
    frames *after* — so a clip cut off right at the question yields an all-silent
    agent. The user (left) channel is naturally silent while the agent replies,
    so extending into it (zero-padded if the file ends) gives the reply room."""
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
    if b > len(user):                                  # pad reply room past EOF
        clip = np.concatenate([clip, np.zeros(b - len(user), dtype=clip.dtype)])
    return clip


def default_audio() -> str:
    """Prepare (once) and return a clean 16 kHz-mono user-only clip from the
    VoiceChat-11B demo. Falls back to the raw demo wav if audio libs are missing."""
    load_env()
    from huggingface_hub import hf_hub_download

    from mstar.model.registry import HF_MODELS

    repo = HF_MODELS["nemotron_duplex"]["model_path_hf"]
    cache_dir = os.environ.get("NEMOTRON_DUPLEX_CACHE_DIR")
    src = hf_hub_download(repo, _DEMO_WAV, cache_dir=cache_dir)

    if os.path.exists(_PREPARED):
        return _PREPARED
    try:
        import soundfile as sf
        import torch
        import torchaudio.functional as AF

        d, sr = sf.read(src, dtype="float32")
        user = d[:, 0] if d.ndim == 2 else d          # left channel = user
        clip = _first_user_turn(user, sr)
        clip16 = AF.resample(torch.from_numpy(clip.copy()), sr, 16000).numpy()
        sf.write(_PREPARED, clip16, 16000)
        print(f"prepared user-only input: {_PREPARED} ({len(clip16) / 16000:.1f}s @ 16 kHz)")
        return _PREPARED
    except Exception as e:  # noqa: BLE001 - degrade to the raw demo wav
        print(f"warning: could not prepare user-only clip ({e}); using raw demo {src}")
        return src


def one_request(url: str, audio: str, modalities, temperature: float):
    c = MStarClient(url, timeout=600)
    t0 = time.time()
    r = c.generate(
        audio=audio, input_modalities=("audio",),
        output_modalities=modalities, temperature=temperature,
    )
    return r, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None,
                    help="path to a mono wav of user speech (default: a bundled VoiceChat-11B sample)")
    ap.add_argument("--output", default="agent_out.wav", help="where to save agent speech")
    ap.add_argument("--text-only", action="store_true", help="request text output only (skip the talker/codec)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("-n", "--concurrency", type=int, default=1, help="fire N identical requests concurrently")
    args = ap.parse_args()

    audio = args.audio or default_audio()
    url = get_base_url()
    modalities = ("text",) if args.text_only else ("text", "audio")
    print(f"POST {url}/generate  audio={audio}  out={modalities}  n={args.concurrency}")

    results: dict[int, tuple] = {}

    def worker(i):
        results[i] = one_request(url, audio, modalities, args.temperature)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    texts = [results[i][0].text for i in range(args.concurrency)]
    for i in range(args.concurrency):
        r, dt = results[i]
        n_audio = len(r.audio) if getattr(r, "audio", None) is not None else 0
        print(f"  req{i}: {dt:.1f}s  text_len={len(r.text or '')}  audio_samples={n_audio}")

    if args.concurrency > 1:
        print(f"  all identical: {all(t == texts[0] for t in texts)}")

    r0 = results[0][0]
    print("\n=== agent text ===")
    print(r0.text)
    if getattr(r0, "audio", None) is not None and len(r0.audio):
        r0.audio.to_wav(args.output)
        dur = len(r0.audio) / (r0.audio.sample_rate or 22050)
        print(f"\nsaved {args.output}  ({dur:.2f}s @ {r0.audio.sample_rate} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

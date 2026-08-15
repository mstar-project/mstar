#!/usr/bin/env python3
"""Ground-truth oracle for validating the Nemotron-Duplex engine path.

Runs the standalone ``offline_inference`` (verified against the NeMo reference)
on a wav and prints the agent text tokens + audio stats. The M* engine's
``/generate`` output (see duplex_request.py) must match the text token-for-token
(audio is seeded-stochastic, so compare length / RMS, not samples).

    python test/nemotron_duplex/oracle_compare.py --audio user.wav

Loads real weights → needs a GPU (put it on a GPU the server is NOT using) and
the VoiceChat-11B checkpoint in the HF cache. Note: offline_inference is O(T^2)
in the number of audio frames, so prefer short clips (< a few seconds).
"""
import argparse
import os
import sys

import torch

from mstar.model.nemotron_duplex.nemotron_duplex_model import NemotronDuplexModel
from mstar.model.registry import HF_MODELS


def load_wav_16k_mono(path: str) -> torch.Tensor:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32")
    a = torch.from_numpy(data)
    if a.dim() == 2:
        a = a.mean(dim=-1)
    if sr != 16000:
        import torchaudio.functional as AF

        a = AF.resample(a, sr, 16000)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=os.environ.get("NEMOTRON_DUPLEX_CACHE_DIR"))
    ap.add_argument("--no-audio", action="store_true", help="skip talker+codec (text tokens only, faster)")
    args = ap.parse_args()

    model = NemotronDuplexModel(
        model_path_hf=HF_MODELS["nemotron_duplex"]["model_path_hf"], cache_dir=args.cache_dir,
    )
    wav = load_wav_16k_mono(args.audio).to(args.device)
    out = model.offline_inference(
        wav.unsqueeze(0), torch.tensor([wav.shape[0]], device=args.device),
        device=args.device, temperature=0.0, decode_audio=not args.no_audio,
    )
    toks = out["tokens_text"][0].tolist()
    print("=== ORACLE (offline_inference) ===")
    print(f"frames/text_tokens: {len(toks)}")
    print(f"text: {out['text'][0]!r}")
    print(f"first tokens: {toks[:40]}")
    if "audio" in out:
        n = int(out["audio_len"][0].item())
        a = out["audio"][0, :n].float()
        print(f"audio: {n} samples ({n / model.config.eartts.sample_rate:.2f}s @ "
              f"{model.config.eartts.sample_rate} Hz) rms={a.pow(2).mean().sqrt():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Greedy reference synthesis with the ``chatterbox-tts`` package.

Runs ``ChatterboxTTS`` / ``ChatterboxTurboTTS`` exactly as shipped, except that
``torch.multinomial`` is replaced by an argmax so the speech tokens are
deterministic, and seeds the S3Gen noise with ``--seed`` right before the
decoder (what M* does with the request seed). The tokens and the waveform are
saved so an M* server run with ``do_sample=false`` and the same seed can be
compared (``serve_parity.py``)::

    python benchmark/chatterbox/reference_greedy.py --text "..." --seed 1234 --out /tmp/ref
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch


def _argmax_multinomial(probs: torch.Tensor, num_samples: int = 1, **kwargs) -> torch.Tensor:
    del kwargs
    return probs.argmax(dim=-1, keepdim=True).expand(-1, num_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", required=True)
    parser.add_argument("--variant", choices=["chatterbox", "turbo"], default="chatterbox")
    parser.add_argument("--audio-prompt", default=None)
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.multinomial = _argmax_multinomial  # greedy everywhere the reference samples

    if args.variant == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS as Cls
    else:
        from chatterbox.tts import ChatterboxTTS as Cls
    model = Cls.from_pretrained(device=args.device)

    captured: dict[str, torch.Tensor] = {}
    original_inference = model.s3gen.inference

    def seeded_inference(*a, **kw):
        # the same seed M* hands its per-request generator, drawn on the same device type
        torch.manual_seed(args.seed)
        if "speech_tokens" in kw:
            captured["tokens"] = kw["speech_tokens"].detach().cpu()
        elif a:
            captured["tokens"] = a[0].detach().cpu()
        return original_inference(*a, **kw)

    model.s3gen.inference = seeded_inference

    kwargs = dict(audio_prompt_path=args.audio_prompt, exaggeration=args.exaggeration)
    if args.variant == "chatterbox":
        kwargs["cfg_weight"] = args.cfg_weight
    t0 = time.perf_counter()
    wav = model.generate(args.text, **kwargs)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sf.write(str(out / "reference.wav"), wav.reshape(-1).cpu().numpy(), model.sr)
    tokens = captured.get("tokens", torch.empty(0)).reshape(-1)
    torch.save(tokens, out / "tokens.pt")
    meta = {
        "text": args.text, "variant": args.variant, "seed": args.seed, "device": args.device,
        "num_tokens": int(tokens.numel()), "audio_s": wav.shape[-1] / model.sr, "elapsed_s": elapsed,
        "exaggeration": args.exaggeration, "cfg_weight": args.cfg_weight if args.variant == "chatterbox" else 0.0,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("tokens:", tokens.tolist()[:40], "...")


if __name__ == "__main__":
    main()

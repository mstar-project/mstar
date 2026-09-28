#!/usr/bin/env python3
"""Record NVIDIA's actual LMGen integer trace with tiny random weights.

No mstar imports, checkpoints or network access at runtime. Run in an isolated
torch<2.5 environment with the pinned NVIDIA package installed. Setup and replay
commands: test/modular/fixtures/personaplex/README.rst.

The observer subclasses LMGen only to record inputs/outputs around super calls.
It never substitutes cache logic, sampling, transformer or depformer execution.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import torch

PIN = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"


def _verify_source(source: Path) -> tuple[Path, dict[str, str]]:
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != PIN:
        raise RuntimeError(f"reference HEAD {head} differs from required {PIN}")
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no", "--", "moshi"], text=True
    )
    if status:
        raise RuntimeError("reference moshi source has tracked modifications")
    package = source / "moshi" / "moshi"
    hashes = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    return package, hashes


def _json_tensor(value):
    return None if value is None else value.detach().cpu().tolist()


def _record_case(lm_model, lm_module, batch: int, user_frames: int, mode: str) -> dict:
    class ObservedLMGen(lm_module.LMGen):
        def __init__(self):
            super().__init__(
                lm_model, device="cpu", use_sampling=False, return_logits=True, check=True,
                audio_silence_frame_cnt=6, sample_rate=24000, frame_rate=12.5,
            )
            self.events = []
            self.phase = "init"
            self.sampled_audio = None

        def snapshot(self):
            state = self._streaming_state
            return {
                "offset": state.offset,
                "cache": _json_tensor(state.cache),
                "provided": _json_tensor(state.provided),
            }

        def prepare_step_input(self, input_tokens=None, moshi_tokens=None, text_token=None):
            prepared = super().prepare_step_input(input_tokens, moshi_tokens, text_token)
            # step_embeddings supplies batch-1 dummies; NVIDIA broadcasts them.
            text = None if text_token is None else torch.as_tensor(text_token).expand(batch)
            event = {
                "op": "prepare", "phase": self.phase,
                "user": _json_tensor(None if input_tokens is None else input_tokens.expand(batch, -1, -1)),
                "agent": _json_tensor(None if moshi_tokens is None else moshi_tokens.expand(batch, -1, -1)),
                "text": _json_tensor(text), "forward": prepared is not None,
                "state": self.snapshot(),
            }
            if prepared is not None:
                input_, provided, target, _, _ = prepared
                event.update(input=_json_tensor(input_), target=_json_tensor(target), mask=_json_tensor(provided))
            self.events.append(event)
            return prepared

        def depformer_step(self, *args):
            result = super().depformer_step(*args)
            self.sampled_audio = result[0].detach().clone()
            return result

        def process_transformer_output(self, transformer_out, text_logits, *args):
            sampled_text = text_logits.float().argmax(dim=-1)[:, 0, 0]
            result = super().process_transformer_output(transformer_out, text_logits, *args)
            self.events.append({
                "op": "commit", "phase": self.phase,
                "sampled_text": _json_tensor(sampled_text), "sampled_audio": _json_tensor(self.sampled_audio),
                "output": _json_tensor(result[0]), "state": self.snapshot(),
            })
            return result

    gen = ObservedLMGen()
    silence = torch.tensor(lm_module.SILENCE_TOKENS).view(1, 8, 1).expand(batch, -1, -1)
    sine = torch.tensor(lm_module.SINE_TOKENS).view(1, 8, 1).expand(batch, -1, -1)
    pad = torch.full((batch,), 3, dtype=torch.long)

    def voice_frame(index):
        return (torch.arange(batch * 8).view(batch, 8, 1) + 100 + index * 11) % lm_model.card

    voice_frames = 4
    # Build a .pt-style preset using actual reference embed_codes outputs and
    # the cache saved by the raw token path. No gated voice asset is needed.
    embeddings, saved_cache = [], None
    if mode == "embedding_replay":
        with gen.streaming(batch):
            gen.phase = "preset_voice"
            for index in range(voice_frames):
                result = gen.step(sine, voice_frame(index), pad, return_embeddings=index > 0)
                if index > 0:
                    embeddings.append(result[1].detach().clone())
            saved_cache = gen._streaming_state.cache.detach().clone()
        gen.events.clear()

    with gen.streaming(batch):
        gen.events.append({"op": "init", "phase": "init", "state": gen.snapshot()})
        if mode == "token_priming":
            for index in range(voice_frames):
                gen.phase = f"voice_{index}"
                gen.step(sine, voice_frame(index), pad)
        else:
            gen.voice_prompt_embeddings = torch.stack(embeddings)
            gen.voice_prompt_cache = saved_cache
            # This method itself calls step_embeddings and restores cache.copy_.
            # Advance its yield points individually to observe every embedding.
            gen.phase = "embedding_0"
            iterator = gen._step_voice_prompt_core(None)
            next(iterator)
            for index in range(len(embeddings)):
                gen.phase = f"embedding_{index}"
                try:
                    next(iterator)
                except StopIteration:
                    if index != len(embeddings) - 1:
                        raise
            gen.events.append({
                "op": "load", "phase": "voice_restore", "cache": _json_tensor(saved_cache),
                "state": gen.snapshot(),
            })
        for block in ("silence_before", "text", "silence_after"):
            texts = [3] * 6 if block != "text" else [7, 11, 19]
            for index, token in enumerate(texts):
                gen.phase = f"{block}_{index}"
                gen.step(sine, silence, torch.full((batch,), token, dtype=torch.long))
        for index in range(user_frames):
            gen.phase = f"user_{index}"
            gen.step(voice_frame(index + voice_frames))
    return {"batch_size": batch, "mode": mode, "events": gen.events}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="clean NVIDIA/personaplex checkout at the pin")
    parser.add_argument("--mode", choices=("trace",), default="trace")
    parser.add_argument("--out", type=Path, required=True, help="directory for reference_trace.json")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--user-frames", type=int, default=20)
    args = parser.parse_args()
    if args.user_frames < 20:
        parser.error("trace acceptance requires at least 20 user frames")
    major, minor = map(int, torch.__version__.split(".")[:2])
    if major != 2 or not 2 <= minor < 5:
        parser.error(f"NVIDIA's reference requires torch>=2.2,<2.5; found {torch.__version__}")
    package, hashes = _verify_source(args.source.resolve())
    sys.path.insert(0, str(package.parent))
    from moshi.models import lm as lm_module
    from moshi.models.loaders import _lm_kwargs
    from moshi.utils.compile import no_compile

    if Path(inspect.getfile(lm_module.LMGen)).resolve() != package / "models" / "lm.py":
        raise RuntimeError("LMGen was imported from a different checkout")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    # Preserve 17 channels, vocabularies, delays and per-step depth weights;
    # only transformer dimensions/layer counts shrink for CPU execution.
    config = dict(_lm_kwargs)
    config.update(
        dep_q=16, dim=32, num_heads=4, num_layers=1, hidden_scale=4.125,
        depformer_dim=16, depformer_num_heads=2, depformer_num_layers=1,
        depformer_dim_feedforward=66,
    )
    lm_model = lm_module.LMModel(**config, device="cpu", dtype=torch.float32).eval()
    with torch.no_grad(), no_compile():
        cases = [
            _record_case(lm_model, lm_module, batch, args.user_frames, mode)
            for batch in (1, 2) for mode in ("token_priming", "embedding_replay")
        ]
    fixture = {
        "schema_version": 2, "source": "NVIDIA/personaplex", "source_sha": PIN,
        "source_hashes": hashes, "torch_version": torch.__version__, "python_version": sys.version,
        "argv": sys.argv[1:], "seed": args.seed, "use_sampling": False, "device": "cpu",
        "lm_kwargs": config, "user_frames": args.user_frames, "cases": cases,
    }
    # Invocation paths remain useful provenance, but do not define content identity.
    content = {key: value for key, value in fixture.items() if key != "argv"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fixture["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "reference_trace.json"
    output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"Recorded {len(cases)} LMGen cases, {sum(len(c['events']) for c in cases)} events to {output}")
    print(f"source={PIN}; torch={torch.__version__}; bytes={output.stat().st_size}")
    print(f"content_sha256={fixture['content_sha256']}")


if __name__ == "__main__":
    main()

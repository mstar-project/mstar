#!/usr/bin/env python3
"""Record NVIDIA reference fixtures with the pinned PersonaPlex source.

Modes:
  trace     LMGen integer trace with tiny random weights (Commit 1, T3).
  manifest  state-dict keys/shapes of the full-size LMModel, built on meta (T5).
  lm        tiny random fp32 LMModel: its weights plus every forward's inputs,
            hidden states, text/depth logits and greedy tokens (T13-LM).

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


def _lm_config(lm_kwargs: dict, **shape) -> dict:
    config = dict(lm_kwargs)
    config.update(dep_q=16, **shape)
    return config


def _record_manifest(lm_module, lm_kwargs: dict) -> dict:
    # get_moshi_lm sets dep_q=16 and builds on meta before loading ([PP] models/loaders.py:187, 197-198).
    config = _lm_config(lm_kwargs)
    model = lm_module.LMModel(**config, device="meta", dtype=torch.bfloat16)
    params = {name: list(p.shape) for name, p in model.named_parameters()}
    buffers = {name: list(b.shape) for name, b in model.named_buffers()}
    return {
        "lm_kwargs": config, "dtype": "bfloat16", "parameters": params, "buffers": buffers,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }


def _record_lm(lm_module, lm_kwargs: dict, user_frames: int) -> tuple[dict, dict]:
    config = _lm_config(
        lm_kwargs, dim=64, num_heads=4, num_layers=2, hidden_scale=4.125,
        depformer_dim=32, depformer_num_heads=2, depformer_num_layers=2,
        depformer_dim_feedforward=int(4.125 * 32),
    )
    model = lm_module.LMModel(**config, device="cpu", dtype=torch.float32).eval()
    with torch.no_grad():
        # Norm scales start at 1; randomise them so a dropped or misrouted alpha shows up.
        for name, param in model.named_parameters():
            if name.endswith(".alpha"):
                param.copy_(1 + 0.25 * torch.randn_like(param))

    class LogitObserver(lm_module.LMGen):
        def __init__(self):
            super().__init__(
                model, device="cpu", use_sampling=False, return_logits=True, check=True,
                audio_silence_frame_cnt=6, sample_rate=24000, frame_rate=12.5,
            )
            self.records, self.phase, self.pending = [], "init", None

        def prepare_step_input(self, input_tokens=None, moshi_tokens=None, text_token=None):
            prepared = super().prepare_step_input(input_tokens, moshi_tokens, text_token)
            if prepared is not None:
                self.pending = {"phase": self.phase, "input": prepared[0].clone()}
            return prepared

        def depformer_step(self, text_token, transformer_out, audio_tokens, audio_provided):
            tokens, logits = super().depformer_step(text_token, transformer_out, audio_tokens, audio_provided)
            self.pending.update(
                depth_text=text_token.clone(), depth_target=audio_tokens.clone(),
                depth_provided=audio_provided.clone(), audio_tokens=tokens.clone(), audio_logits=logits.clone(),
            )
            return tokens, logits

        def process_transformer_output(self, transformer_out, text_logits, *args):
            self.pending.update(transformer_out=transformer_out.clone(), text_logits=text_logits.clone())
            result = super().process_transformer_output(transformer_out, text_logits, *args)
            self.pending["output"] = None if result[0] is None else result[0].clone()
            self.records.append(self.pending)
            self.pending = None
            return result

    def run_case(batch: int, mode: str) -> dict:
        gen = LogitObserver()
        silence = torch.tensor(lm_module.SILENCE_TOKENS).view(1, 8, 1).expand(batch, -1, -1)
        sine = torch.tensor(lm_module.SINE_TOKENS).view(1, 8, 1).expand(batch, -1, -1)
        pad = torch.full((batch,), 3, dtype=torch.long)

        def voice_frame(index):
            return (torch.arange(batch * 8).view(batch, 8, 1) * 37 + 100 + index * 11) % model.card

        preset = None
        if mode == "embedding_replay":
            with gen.streaming(batch):
                gen.phase = "preset_voice"
                inputs, embeddings = [], []
                for index in range(4):
                    result = gen.step(sine, voice_frame(index), pad, return_embeddings=index > 0)
                    if index > 0:
                        embeddings.append(result[1].clone())
                        inputs.append(gen.records[-1]["input"])
                preset = {"inputs": inputs, "embeddings": torch.stack(embeddings),
                          "cache": gen._streaming_state.cache.clone()}
            gen.records.clear()
        with gen.streaming(batch):
            if mode == "token_priming":
                for index in range(4):
                    gen.phase = f"voice_{index}"
                    gen.step(sine, voice_frame(index), pad)
            else:
                gen.voice_prompt_embeddings = preset["embeddings"]
                gen.voice_prompt_cache = preset["cache"]
                for index, _ in enumerate(gen._step_voice_prompt_core(None)):
                    gen.phase = f"embedding_{index}"
            for block in ("silence_before", "text", "silence_after"):
                texts = [3] * 6 if block != "text" else [7, 11, 19]
                for index, token in enumerate(texts):
                    gen.phase = f"{block}_{index}"
                    gen.step(sine, silence, torch.full((batch,), token, dtype=torch.long))
            for index in range(user_frames):
                gen.phase = f"user_{index}"
                gen.step(voice_frame(index + 4))
        return {"batch_size": batch, "mode": mode, "preset": preset, "records": gen.records}

    cases = [run_case(batch, mode) for batch in (1, 2) for mode in ("token_priming", "embedding_replay")]
    return config, {"state_dict": {k: v.clone() for k, v in model.state_dict().items()}, "cases": cases}


def _content_hash(fixture: dict) -> str:
    # Invocation paths remain useful provenance, but do not define content identity.
    content = {key: value for key, value in fixture.items() if key != "argv"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _tensor_digest(value) -> object:
    """Order-stable SHA256 over nested tensors, for the binary LM fixture's provenance."""
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().contiguous()
        return [str(t.dtype), list(t.shape), hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()]
    if isinstance(value, dict):
        return {k: _tensor_digest(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_tensor_digest(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="clean NVIDIA/personaplex checkout at the pin")
    parser.add_argument("--mode", choices=("trace", "manifest", "lm"), default="trace")
    parser.add_argument("--out", type=Path, required=True, help="output directory for the fixture")
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
    provenance = {
        "source": "NVIDIA/personaplex", "source_sha": PIN, "source_hashes": hashes,
        "torch_version": str(torch.__version__), "python_version": sys.version, "seed": args.seed,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "manifest":
        fixture = {"schema_version": 1, **provenance, "argv": sys.argv[1:], **_record_manifest(lm_module, _lm_kwargs)}
        fixture["content_sha256"] = _content_hash(fixture)
        output = args.out / "reference_lm_manifest.json"
        output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
        print(f"Recorded {len(fixture['parameters'])} parameters, {fixture['num_parameters']} values to {output}")
        print(f"source={PIN}; torch={torch.__version__}; content_sha256={fixture['content_sha256']}")
        return
    if args.mode == "lm":
        with torch.no_grad(), no_compile():
            config, data = _record_lm(lm_module, _lm_kwargs, args.user_frames)
        fixture = {
            "schema_version": 1, **provenance, "argv": sys.argv[1:], "use_sampling": False, "device": "cpu",
            "dtype": "float32", "lm_kwargs": config, "user_frames": args.user_frames, **data,
        }
        digest = {key: _tensor_digest(value) for key, value in fixture.items() if key != "argv"}
        fixture["content_sha256"] = hashlib.sha256(
            json.dumps(digest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        output = args.out / "reference_lm.pt"
        torch.save(fixture, output)
        forwards = sum(len(c["records"]) for c in data["cases"])
        print(f"Recorded {len(data['cases'])} LMGen cases, {forwards} forwards to {output}")
        print(f"source={PIN}; torch={torch.__version__}; bytes={output.stat().st_size}")
        print(f"content_sha256={fixture['content_sha256']}")
        return
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
    fixture["content_sha256"] = _content_hash(fixture)
    output = args.out / "reference_trace.json"
    output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"Recorded {len(cases)} LMGen cases, {sum(len(c['events']) for c in cases)} events to {output}")
    print(f"source={PIN}; torch={torch.__version__}; bytes={output.stat().st_size}")
    print(f"content_sha256={fixture['content_sha256']}")


if __name__ == "__main__":
    main()

"""Batched LM frames against the same requests run alone, on the served resources (paged KV, FlashInfer).

Greedy, real checkpoint, no server. Each request runs alone, then batches of
them run together, later requests starting a few frames late so a batch holds
caches at different offsets and, for some steps, requests in different walks.
Reports, per request, whether every emitted token matches its solo run, the
first choice that differs with both runs' top-2 logits, and the LM step time
at each batch size. Exits nonzero if any token differs.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark.personaplex.controls import candidates
from benchmark.personaplex.resources import ResourceBackend
from mstar.model.personaplex import config as cfg
from mstar.model.personaplex.components.mimi import load_mimi_codec
from mstar.model.personaplex.personaplex_model import PersonaPlexModel
from mstar.model.personaplex.submodules import LMSubmodule, MimiEncoderSubmodule
from mstar.model.personaplex.weights import build_lm, iter_lm_checkpoint


def prepare_requests(args, model):
    """Clip ``i % len(inputs)`` with voice ``i % len(voices)``, encoded by M*'s own Mimi encoder."""
    from safetensors.torch import load_file

    codec = load_mimi_codec(load_file(str(args.assets / "tokenizer-e351c8d8-checkpoint125.safetensors")),
                            device="cuda")
    encoder = MimiEncoderSubmodule(codec)
    requests = []
    for index in range(max(args.batch_sizes)):
        path, voice = args.inputs[index % len(args.inputs)], args.voice_names[index % len(args.voice_names)]
        loaded = model.load_audio(str(path), "cpu")
        samples = min(args.frames * cfg.FRAME_SIZE, loaded.data.numel())
        audio = loaded.data.reshape(-1)[:samples].clone()
        tensors = {"audio_inputs": [audio]}
        static = model.process_prompt(None, ["audio"], ["audio", "text"], tensors, voice=voice, greedy=True)
        user_frames = -(-samples // cfg.FRAME_SIZE)
        n_emb, n_text = static["voice_embeddings"][0].shape[0], static["persona_tokens"][0].numel()
        meta = {"num_voice_embeddings": n_emb, "num_text_tokens": n_text, "num_user_frames": user_frames,
                "num_samples": samples, "conditioning_frames": n_emb + 2 * cfg.SILENCE_FRAMES + n_text}
        prepared = encoder.prepare_inputs(cfg.ENCODE_USER_AUDIO_WALK, SimpleNamespace(step_metadata=meta), tensors)
        codes = encoder.forward(cfg.ENCODE_USER_AUDIO_WALK, None, **prepared.tensor_inputs)["user_audio_codes"][0]
        static["user_audio_codes"] = [codes]
        requests.append(dict(id=f"r{index}", clip=str(path), voice=voice, meta=meta, static=static,
                             input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
    return requests


def run(sub, resource, requests, starts):
    """Drive ``requests`` frame by frame, those in the same walk as one batch.

    Returns each request's emitted tokens and greedy choices, and every step's
    batch size, walk and wall time (synchronized).
    """
    total = [r["meta"]["conditioning_frames"] + r["meta"]["num_user_frames"] for r in requests]
    states = [None] * len(requests)
    emitted = [[] for _ in requests]
    choices = [[] for _ in requests]
    steps = []
    for request in requests:
        resource.add(request["id"])
    try:
        for tick in range(max(start + n for start, n in zip(starts, total, strict=True))):
            by_walk = {}
            for i, request in enumerate(requests):
                index = tick - starts[i]
                if 0 <= index < total[i]:
                    conditioning = index < request["meta"]["conditioning_frames"]
                    by_walk.setdefault(cfg.CONDITIONING_WALK if conditioning else cfg.GENERATE_WALK, []).append(i)
            for walk, rows in by_walk.items():
                picks = {}  # read back after the timed step, not inside it

                def sample(kind, slot, sampler, logits, picks=picks):
                    tokens = sampler(logits)
                    picks[(kind, slot)] = (logits, tokens)
                    return tokens

                torch.cuda.synchronize()
                began = time.perf_counter()
                frames = [sub.prepare_frame(walk, requests[i]["meta"], {
                    **requests[i]["static"], "lm_state": [] if states[i] is None else [states[i]],
                }) for i in rows]
                provided = torch.cat([f.cache.target_provided[:, :, 0] for f in frames])
                resource.begin(walk, [requests[i]["id"] for i in rows])
                outputs = sub.run_frames(
                    frames, resource.backend,
                    lambda logits: sample("text", None, resource.sample_text, logits),
                    lambda logits, d: sample("depth", d, lambda x: resource.sample_depth(x, d), logits),
                )
                resource.finish()
                torch.cuda.synchronize()
                steps.append(dict(walk=walk, batch=len(rows), ms=(time.perf_counter() - began) * 1000))
                provided = provided.cpu()
                for row, (i, output) in enumerate(zip(rows, outputs, strict=True)):
                    states[i] = output["lm_state"][0]
                    if walk != cfg.GENERATE_WALK:
                        continue
                    emitted[i].append(torch.cat((output["text_frame"][0][:, None, None], output["agent_frame"][0]),
                                                dim=1).cpu())
                    choices[i].append({key: (candidates(logits[row : row + 1]), int(tokens[row]),
                                             bool(provided[row, 0 if key[0] == "text" else 1 + key[1]]))
                                       for key, (logits, tokens) in picks.items()})
    finally:
        for request in requests:
            resource.remove(request["id"])
    return emitted, choices, steps


def compare(solo, batched):
    """Emitted-token agreement and the first unforced choice that differs."""
    (solo_emitted, solo_choices), (batch_emitted, batch_choices) = solo, batched
    different = [j for j, (x, y) in enumerate(zip(solo_emitted, batch_emitted, strict=True)) if not torch.equal(x, y)]
    first = None
    for frame, (alone, together) in enumerate(zip(solo_choices, batch_choices, strict=True)):
        for key, (top, token, forced) in alone.items():
            other_top, other_token, _ = together[key]
            if not forced and token != other_token:
                first = dict(generation_step=frame, kind=key[0], depth_step=key[1], solo=top, batched=other_top)
                break
        if first:
            break
    return dict(frames=len(solo_emitted), frames_different=len(different),
                first_different_frame=(different or [None])[0], first_decision=first)


def timing(steps):
    rows = {}
    for step in steps:
        if step["walk"] == cfg.GENERATE_WALK:
            rows.setdefault(step["batch"], []).append(step["ms"])
    return {str(b): dict(steps=len(ms), ms_p50=statistics.median(ms), frames_per_s_p50=1000 * b / statistics.median(ms))
            for b, ms in sorted(rows.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--voices", type=Path, required=True)
    parser.add_argument("--voice-names", nargs="+", default=["NATF2", "NATM1"])
    parser.add_argument("--frames", type=int, default=cfg.MAX_TEMPORAL_FORWARDS)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--stagger", type=int, default=3, help="frames each later request starts after the previous")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "FAIL", "device": torch.cuda.get_device_name(), "torch": str(torch.__version__)}
    try:
        with torch.inference_mode():
            torch.backends.cuda.matmul.allow_tf32 = False
            model = PersonaPlexModel(skip_weight_loading=True, voice_dir=str(args.voices))
            import sentencepiece

            model.tokenizer = sentencepiece.SentencePieceProcessor(
                model_file=str(args.assets / "tokenizer_spm_32k_3.model"))
            requests = prepare_requests(args, model)
            lm = build_lm(cfg.LMArchConfig(), iter_lm_checkpoint(str(args.assets / "model.safetensors")),
                          device="cuda", dtype=torch.bfloat16)
            sub = LMSubmodule(lm)
            resource = ResourceBackend(model, sub)
            summary["requests"] = [{k: r[k] for k in ("id", "clip", "voice", "meta", "input_sha256")} for r in requests]
            solo, solo_steps = [], []
            for request in requests:
                emitted, choices, steps = run(sub, resource, [request], [0])
                solo.append((emitted[0], choices[0]))
                solo_steps += steps
            summary["timing"] = {"solo": timing(solo_steps)}
            summary["batched"] = {}
            for size in args.batch_sizes:
                group = requests[:size]
                emitted, choices, steps = run(sub, resource, group, [i * args.stagger for i in range(size)])
                summary["batched"][str(size)] = {r["id"]: compare(solo[i], (emitted[i], choices[i]))
                                                 for i, r in enumerate(group)}
                summary["timing"][f"batched-{size}"] = timing(steps)
            identical = all(c["frames_different"] == 0 for run_ in summary["batched"].values() for c in run_.values())
            summary["status"] = "PASS" if identical else "FAIL"
    except Exception as exc:
        summary["error"] = repr(exc)
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

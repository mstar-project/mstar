# ruff: noqa: E402
"""Real-checkpoint numerical comparison against NVIDIA's reference; no synthetic-weight oracle.

NVIDIA is imported from its clean pin, under the SAME production Torch runtime.
That runtime is outside NVIDIA's declared torch<2.5 range, explicitly reported.
No production code or reference math is patched. Observer overrides only record.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", required=True)
p.add_argument("--frames", type=int, default=222)
p.add_argument("--input", type=Path, required=True)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--assets", type=Path, required=True)
p.add_argument("--voices", type=Path, required=True)
p.add_argument("--reference-site", default="")
p.add_argument("--backend", choices=["dense", "resources"], default="resources")
p.add_argument("--skip-components", action="store_true")
p.add_argument("--negative-control", action="store_true",
               help="also run M* with RoPE one position ahead; the comparison must catch it")
p.add_argument("--reference-control", action="store_true",
               help="also rerun NVIDIA under the math SDPA kernel: the reference's own kernel spread")
a = p.parse_args()
PIN = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
REPO = Path(__file__).resolve().parents[2]
SOURCE, ASSET = a.source.resolve(), a.assets.resolve()
sys.path.insert(0, str(SOURCE / "moshi"))
if a.reference_site:
    sys.path.append(a.reference_site)
from moshi.models import loaders
from moshi.models.lm import LMGen
from moshi.utils.compile import no_compile, no_cuda_graph
from safetensors import safe_open
from safetensors.torch import load_file

from benchmark.personaplex.controls import ShiftedRope, candidates, first_reference_decision
from benchmark.personaplex.resources import ResourceBackend
from mstar.model.personaplex import config as cfg
from mstar.model.personaplex.components.mimi import load_mimi_codec
from mstar.model.personaplex.components.temporal import DenseTemporalCache
from mstar.model.personaplex.personaplex_model import PersonaPlexModel
from mstar.model.personaplex.submodules import LMSubmodule, MimiDecoderSubmodule, MimiEncoderSubmodule
from mstar.model.personaplex.weights import build_lm, iter_lm_checkpoint, reference_manifest

out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
report = {
    "torch": str(torch.__version__),
    "device": torch.cuda.get_device_name(),
    "frames": a.frames,
    "source_sha": subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
    "runtime_scope": "same-runtime GPU source comparison; Torch2.10 outside NVIDIA declared range",
    "backend": a.backend,
    "checks": {},
    "stages": [],
}
report["mstar_head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
report["working_diff_sha256"] = hashlib.sha256(
    subprocess.check_output(["git", "-C", str(REPO), "diff", "--binary"])
).hexdigest()
assert not subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"]).strip(), (
    "numerical run requires committed code"
)
assert report["source_sha"] == PIN
assert not subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain"], text=True).strip()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def stage(name):
    print("STAGE", name, flush=True)
    report["stages"].append({"name": name, "time": time.time()})
    (out / "progress.json").write_text(json.dumps(report, indent=2))


def save():
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


def metrics(got, expected):
    assert got.shape == expected.shape, (got.shape, expected.shape)
    x, y = got.detach().float().cpu(), expected.detach().float().cpu()
    d = x.double() - y.double()
    result = {
        "shape": list(x.shape),
        "max_abs": d.abs().max().item(),
        "rms": d.square().mean().sqrt().item(),
        "mse": d.square().mean().item(),
        "mae": d.abs().mean().item(),
        "reference_rms": y.double().square().mean().sqrt().item(),
        "finite": bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
    }
    if x.shape[-1] in (32000, 2048):
        result["argmax_mismatches"] = int((x.argmax(-1) != y.argmax(-1)).sum())
        result["positions"] = x.numel() // x.shape[-1]
        top = y.topk(2, dim=-1).values
        result["reference_min_top2_margin"] = float((top[..., 0] - top[..., 1]).min())
    return result


class Observer(LMGen):
    def __init__(self, lm):
        super().__init__(
            lm,
            device="cuda",
            use_sampling=False,
            return_logits=True,
            check=True,
            audio_silence_frame_cnt=6,
            sample_rate=24000,
            frame_rate=12.5,
        )
        self.records = []
        self.phase = "conditioning"
        self.pending = None

    def prepare_step_input(self, *args, **kwargs):
        result = super().prepare_step_input(*args, **kwargs)
        if result is not None:
            self.pending = {
                "phase": self.phase,
                "input": result[0].cpu().clone(),
                "provided": result[1].cpu().clone(),
                "target": result[2].cpu().clone(),
            }
        return result

    def depformer_step(self, text_token, hidden, target, provided):
        tokens, logits = super().depformer_step(text_token, hidden, target, provided)
        self.pending.update(
            depth_text=text_token.cpu().clone(),
            depth_target=target.cpu().clone(),
            depth_provided=provided.cpu().clone(),
            depth_tokens=tokens.cpu().clone(),
            depth_logits=logits.cpu().clone(),
        )
        return tokens, logits

    def process_transformer_output(self, hidden, logits, *args):
        self.pending.update(hidden=hidden.cpu().clone(), text_logits=logits.cpu().clone())
        result = super().process_transformer_output(hidden, logits, *args)
        self.pending["output"] = None if result[0] is None else result[0].cpu().clone()
        self.records.append(self.pending)
        self.pending = None
        return result


@torch.inference_mode()
def components(ref, native, input_codes):
    stage("shared-prefix temporal and all16 forced depth slots")
    cache = DenseTemporalCache(cfg.LMArchConfig().num_layers, cfg.LMArchConfig().rope_theta)
    embeddings, hidden, logits = [], [], []
    depth = []
    with ref.streaming(1):
        for index in range(4):
            if index == 0:
                tokens = ref._get_initial_token()
            else:
                tokens = torch.cat(
                    (
                        torch.tensor([[[3]]], device="cuda"),
                        input_codes[..., index - 1 : index],
                        input_codes[..., index - 1 : index],
                    ),
                    dim=1,
                )
            e_ref = ref.embed_codes(tokens)
            e_ours = native.temporal.embed_codes(tokens)
            embeddings.append(metrics(e_ours, e_ref))
            h_ref, l_ref = ref.forward_embeddings(e_ref)
            h_ours, l_ours = native.temporal.forward_embeddings(e_ours, cache)
            cache.advance()
            hidden.append(metrics(h_ours, h_ref))
            logits.append(metrics(l_ours, l_ref[:, 0]))
            scratch = native.depth.new_scratch(1, dtype=h_ref.dtype, device=h_ref.device)
            with ref.depformer.streaming(1):
                for step in range(16):
                    token = (
                        torch.tensor([3], device="cuda")
                        if step == 0
                        else input_codes[:, (step - 1) % 8, index % input_codes.shape[-1]]
                    )
                    expected = ref.forward_depformer(step, token[:, None, None], h_ref)[:, 0, 0]
                    got = native.depth.step(step, token, h_ref[:, 0], scratch)
                    depth.append({"frame": index, "depth_step": step, **metrics(got, expected)})
    # Exact repeated calls establish frame-local state reset on trained weights.
    h = h_ref[:, 0]
    target = torch.cat((input_codes[..., 0], input_codes[..., 0]), dim=1)
    provided = torch.ones_like(target, dtype=torch.bool)
    runs = [
        native.depth.run_frame(
            torch.tensor([3], device="cuda"), h, target, provided, lambda x, i: x.argmax(-1), return_logits=True
        )
        for _ in range(2)
    ]
    report["checks"]["shared_prefix"] = {
        "embeddings": embeddings,
        "hidden": hidden,
        "text_logits": logits,
        "depth_logits": depth,
        "depth_reset_exact": torch.equal(runs[0][1], runs[1][1]),
    }
    save()


def reference_generation(ref, voice_path, persona, ref_user, user_frames):
    """NVIDIA's own conditioning and greedy generation, observed; returns the observer and its tokens."""
    gen = Observer(ref)
    gen.load_voice_prompt_embeddings(str(voice_path))
    gen.text_prompt_tokens = persona.tolist()
    agent, text = [], []
    with gen.streaming(1):
        gen.step_system_prompts(None)
        gen.phase = "generation"
        for i in range(user_frames):
            tokens, _ = gen.step(ref_user[..., i : i + 1])
            assert tokens is not None
            agent.append(tokens[:, 1:9].clone())
            text.append(tokens[:, 0, 0].clone())
        state = gen._streaming_state
        final = (state.cache.cpu().clone(), state.provided.cpu().clone(), state.offset)
    return gen, agent, text, final


def reference_control(ref, voice_path, persona, ref_user, user_frames, gen, ref_agent, ref_text):
    """How far NVIDIA's own output moves under another valid attention kernel.

    The suite's served-path comparison can flip near-tied tokens; this measures
    the reference against itself, so those flips have a yardstick.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    stage("reference kernel control: NVIDIA again under the math SDPA kernel")
    with sdpa_kernel(SDPBackend.MATH):
        control, agent, text, _ = reference_generation(ref, voice_path, persona, ref_user, user_frames)
    agent_diff = int((torch.cat(agent, -1) != torch.cat(ref_agent, -1)).sum())
    text_frames = int((torch.stack(text) != torch.stack(ref_text)).sum())
    report["checks"]["reference_control"] = dict(
        kernels="default SDPA selection vs SDPBackend.MATH, both NVIDIA's own code",
        agent_codes_different=agent_diff,
        text_tokens_different=text_frames,
        first_decision=first_reference_decision(gen.records, control.records),
    )
    save()


def negative_control(sub, native, meta, static, ref_agent, ref_text):
    """Run M* with a realistic bug through the same exact-token comparison; it must report a mismatch."""
    stage("negative control: M* with RoPE one position ahead, compared the same way")
    backend = ShiftedRope(native.config.num_layers, native.config.rope_theta)
    state, detected, frames = None, None, 0
    for index in range(meta["conditioning_frames"] + meta["num_user_frames"]):
        walk = cfg.CONDITIONING_WALK if index < meta["conditioning_frames"] else cfg.GENERATE_WALK
        frame = sub.prepare_frame(walk, meta, {**static, "lm_state": [] if state is None else [state]})
        result = sub.run_frame(frame, backend, lambda x: x.argmax(-1), lambda x, i: x.argmax(-1))
        backend.advance()
        state, frames = result["lm_state"][0], index + 1
        if walk == cfg.GENERATE_WALK:
            j = index - meta["conditioning_frames"]
            if not (torch.equal(result["agent_frame"][0], ref_agent[j])
                    and torch.equal(result["text_frame"][0], ref_text[j])):
                detected = j
                break
    report["checks"]["negative_control"] = dict(
        perturbation="RoPE positions one ahead (dense attention otherwise unchanged)",
        detected=detected is not None,
        first_mismatch_generation_step=detected,
        frames_run=frames,
    )
    save()


@torch.inference_mode()
def pipeline(ref, native):
    stage("production preprocessing and independent Mimi encoders")
    model = PersonaPlexModel(skip_weight_loading=True, voice_dir=str(a.voices))
    import sentencepiece

    model.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(ASSET / "tokenizer_spm_32k_3.model"))
    # Actual production audio loader; 24kHz input avoids cross-library resampling.
    loaded = model.load_audio(str(a.input), "cpu")
    samples = min(a.frames * 1920, loaded.data.numel())
    audio = loaded.data.reshape(-1)[:samples].clone()
    user_frames = (samples + 1919) // 1920
    tensors = {"audio_inputs": [audio]}
    static = model.process_prompt(None, ["audio"], ["audio", "text"], tensors, voice="NATF2", greedy=True)
    voice_path = a.voices / "NATF2.pt"
    persona = static["persona_tokens"][0]
    meta = {
        "num_voice_embeddings": static["voice_embeddings"][0].shape[0],
        "num_text_tokens": persona.numel(),
        "num_user_frames": user_frames,
        "num_samples": samples,
        "conditioning_frames": static["voice_embeddings"][0].shape[0] + 12 + persona.numel(),
    }
    report["request"] = {
        **meta,
        "voice": "NATF2",
        "persona_tokens": persona.tolist(),
        "voice_sha256": hashlib.sha256(voice_path.read_bytes()).hexdigest(),
        "input_fp32_sha256": hashlib.sha256(audio.numpy().tobytes()).hexdigest(),
    }
    with wave.open(str(out / "input.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes((audio.numpy() * 32768).clip(-32768, 32767).astype("<i2").tobytes())
    ref_codec = loaders.get_mimi(ASSET / "tokenizer-e351c8d8-checkpoint125.safetensors", device="cuda")
    codec = load_mimi_codec(load_file(str(ASSET / "tokenizer-e351c8d8-checkpoint125.safetensors")), device="cuda")
    enc = MimiEncoderSubmodule(codec)
    info = SimpleNamespace(step_metadata=meta)
    prepared = enc.prepare_inputs("encode_user_audio", info, tensors)
    ours_user = enc.forward("encode_user_audio", None, **prepared.tensor_inputs)["user_audio_codes"][0]
    pcm = prepared.tensor_inputs["pcm"]
    with ref_codec.streaming(1):
        ref_user = torch.cat([ref_codec.encode(pcm[..., i * 1920 : (i + 1) * 1920]) for i in range(user_frames)], -1)
    report["checks"]["user_codes"] = {"different": int((ours_user != ref_user).sum()), "count": ours_user.numel()}
    np.savez(out / "user-codes.npz", mstar=ours_user.cpu().numpy(), nvidia=ref_user.cpu().numpy())
    save()
    if not a.skip_components:
        components(ref, native, ref_user)
    stage("independent NVIDIA conditioning and greedy generation")
    gen, ref_agent, ref_text, final_ref = reference_generation(ref, voice_path, persona, ref_user, user_frames)
    final_ref_cache, final_ref_provided, final_ref_offset = final_ref
    report["checks"]["nvidia_conditioning_steps"] = len(gen.records) - user_frames
    torch.save(gen.records, out / "nvidia-events.pt")
    if a.reference_control:
        reference_control(ref, voice_path, persona, ref_user, user_frames, gen, ref_agent, ref_text)
    stage("production conditioning and generation, own user codes, no substitution")
    sub = LMSubmodule(native)
    static["user_audio_codes"] = [ours_user]
    state = None
    backend = DenseTemporalCache(native.config.num_layers, native.config.rope_theta)
    resource = ResourceBackend(model, sub, ["real-checkpoint"]) if a.backend == "resources" else None
    records, agent, text = [], [], []
    total = meta["conditioning_frames"] + user_frames
    latest = {}
    actual_events = []
    depth_original = native.depth.step

    def depth_observed(index, *args, **kwargs):
        value = depth_original(index, *args, **kwargs)
        latest.setdefault("depth", []).append(value.detach().cpu().clone())
        latest.setdefault("depth_inputs", []).append(args[0].detach().cpu().clone())
        return value

    native.depth.step = depth_observed

    def temporal_hook(module, args, result):
        latest["hidden"] = result[0].cpu().clone()
        if result[1] is not None:
            latest["text_logits"] = result[1].cpu().clone()

    # forward_embeddings is called directly (not nn.Module.__call__); observer wrapper retains math.
    original = native.temporal.forward_embeddings

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        temporal_hook(None, args, result)
        return result

    native.temporal.forward_embeddings = observed
    try:
        for index in range(total):
            latest["depth"] = []
            latest["depth_inputs"] = []
            walk = cfg.CONDITIONING_WALK if index < meta["conditioning_frames"] else cfg.GENERATE_WALK
            frame = sub.prepare_frame(walk, meta, {**static, "lm_state": [] if state is None else [state]})
            expected = gen.records[index]
            record = {
                "index": index,
                "walk": walk,
                "input_codes_equal": bool(torch.equal(frame.cache.model_input.cpu(), expected["input"])),
            }
            if resource:
                resource.begin(walk)
                result = sub.run_frame(frame, resource.backend, resource.sample_text, resource.sample_depth)
                resource.finish()
            else:
                result = sub.run_frame(frame, backend, lambda x: x.argmax(-1), lambda x, i: x.argmax(-1))
                backend.advance()
            state = result["lm_state"][0]
            h = latest["hidden"].reshape(1, 1, -1)
            record["hidden"] = metrics(h, expected["hidden"])
            if walk == cfg.GENERATE_WALK:
                record["text_logits"] = metrics(latest["text_logits"].reshape(1, 1, 1, -1), expected["text_logits"])
                agent.append(result["agent_frame"][0])
                text.append(result["text_frame"][0])
                got = torch.cat((text[-1][:, None, None], agent[-1]), dim=1)
                record["emitted_equal"] = bool(torch.equal(got.cpu(), expected["output"][:, :9]))
                depth_logits = torch.stack(latest["depth"], dim=1)
                record["depth_logits"] = metrics(depth_logits, expected["depth_logits"])
                same_input_prefix = all(r["input_codes_equal"] for r in records) and record["input_codes_equal"]
                decisions = [
                    (
                        "text",
                        0,
                        latest["text_logits"].reshape(1, -1),
                        expected["text_logits"].reshape(1, -1),
                        bool(expected["provided"][0, 0, 0]),
                    )
                ]
                decisions += [
                    (
                        "depth",
                        d,
                        depth_logits[:, d],
                        expected["depth_logits"][:, d],
                        bool(expected["depth_provided"][0, d]),
                    )
                    for d in range(16)
                ]
                for kind, slot, got_logits, ref_logits, forced in decisions:
                    if forced or torch.equal(got_logits.argmax(-1), ref_logits.argmax(-1)):
                        continue
                    if "first_decision" not in report["checks"]:
                        # topk's order at a tie need not match argmax, so record each side's actual
                        # choice and both sides' logits for both choices (batch size 1).
                        picks = {"reference": int(ref_logits.argmax(-1).reshape(-1)[0]),
                                 "mstar": int(got_logits.argmax(-1).reshape(-1)[0])}
                        selected = dict(
                            tokens=picks,
                            **{f"{side}_logits": {name: float(logits.float().reshape(-1)[token])
                                                  for name, token in picks.items()}
                               for side, logits in (("reference", ref_logits), ("mstar", got_logits))},
                        )

                        report["checks"]["first_decision"] = dict(
                            model_step=index,
                            generation_step=index - meta["conditioning_frames"],
                            kind=kind,
                            depth_step=slot if kind == "depth" else None,
                            matched_temporal_input_prefix=same_input_prefix,
                            matched_depth_input_prefix=(
                                all(
                                    torch.equal(
                                        latest["depth_inputs"][d].reshape(-1),
                                        (
                                            expected["depth_text"]
                                            if d == 0
                                            else torch.where(
                                                expected["depth_provided"][:, d - 1],
                                                expected["depth_target"][:, d - 1],
                                                expected["depth_tokens"][:, d - 1],
                                            )
                                        ).reshape(-1),
                                    )
                                    for d in range(slot + 1)
                                )
                                if kind == "depth"
                                else None
                            ),
                            selected=selected,
                            reference=candidates(ref_logits),
                            mstar=candidates(got_logits),
                            note=(
                                "Own-code free-running logits; no source-hidden or code substitution. "
                                "Later depth inputs may already differ."
                            ),
                        )
                actual_events.append(
                    dict(
                        index=index,
                        hidden=latest["hidden"],
                        text_logits=latest["text_logits"],
                        depth_logits=depth_logits,
                        output=got.cpu().clone(),
                    )
                )
            records.append(record)
            if index % 40 == 0:
                print("MSTAR FRAME", index, "/", total, flush=True)
    finally:
        native.temporal.forward_embeddings = original
        native.depth.step = depth_original
        if resource:
            resource.close()
    if a.negative_control:
        negative_control(sub, native, meta, static, ref_agent, ref_text)
    (out / "pipeline-events.json").write_text(json.dumps(records, indent=2))
    torch.save(actual_events, out / "mstar-events.pt")
    from mstar.model.personaplex.components.delayed_cache import DelayedTokenCache

    final_cache, _ = DelayedTokenCache.from_state_tensor(state)
    report["checks"]["final_cache_equal"] = bool(torch.equal(final_cache.cache.cpu(), final_ref_cache))
    report["checks"]["final_provided_equal"] = bool(torch.equal(final_cache.provided.cpu(), final_ref_provided))
    report["checks"]["final_offset_equal"] = final_cache.offset == final_ref_offset
    report["checks"]["temporal_steps"] = total
    report["checks"]["input_step_mismatches"] = [r["index"] for r in records if not r["input_codes_equal"]]
    report["checks"]["output_frame_mismatches"] = [
        r["index"] - meta["conditioning_frames"] for r in records if r.get("emitted_equal") is False
    ]
    report["checks"]["max_hidden_error"] = max(r["hidden"]["max_abs"] for r in records)
    stage("production Mimi decoder and final PCM from own generated codes")
    decoder = MimiDecoderSubmodule(codec)
    prepared = decoder.prepare_inputs("generate", info, {"agent_frame": agent})
    ours_pcm = decoder.forward("generate", None, **prepared.tensor_inputs, **prepared.kwargs)["audio_output"][0]
    with ref_codec.streaming(1):
        reference_pcm = torch.cat([ref_codec.decode(c) for c in ref_agent], -1)[..., :samples]
    report["checks"]["final_pcm"] = metrics(ours_pcm, reference_pcm)
    ours_agent, reference_agent = torch.cat(agent, -1), torch.cat(ref_agent, -1)
    report["checks"]["agent_codes_different"] = int((ours_agent != reference_agent).sum())
    report["checks"]["text_tokens_different"] = int((torch.stack(text) != torch.stack(ref_text)).sum())
    np.savez(
        out / "pipeline.npz",
        mstar_pcm=ours_pcm.cpu().numpy(),
        nvidia_pcm=reference_pcm.cpu().numpy(),
        mstar_agent=ours_agent.cpu().numpy(),
        nvidia_agent=reference_agent.cpu().numpy(),
        mstar_text=torch.stack(text).cpu().numpy(),
        nvidia_text=torch.stack(ref_text).cpu().numpy(),
    )
    (out / "expected-audio.pcm").write_bytes(model.postprocess(ours_pcm, "audio"))
    (out / "expected-text.txt").write_bytes(b"".join(model.postprocess(t, "text") for t in text))
    for name, output in [("mstar", ours_pcm), ("nvidia", reference_pcm)]:
        with wave.open(str(out / (name + ".wav")), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(model.postprocess(output, "audio"))
    save()


try:
    with torch.inference_mode(), no_compile(), no_cuda_graph():
        stage("verify checkpoint manifest and load independent trained models")
        with safe_open(str(ASSET / "model.safetensors"), framework="pt") as f:
            shapes = {k: list(f.get_slice(k).get_shape()) for k in f.keys()}
        with torch.device("meta"):
            from mstar.model.personaplex.weights import PersonaPlexLM

            meta_lm = PersonaPlexLM(cfg.LMArchConfig())
        assert reference_manifest(meta_lm) == shapes
        report["checks"]["strict_manifest"] = {
            "tensors": len(shapes),
            "parameters": sum(np.prod(s) for s in shapes.values()).item(),
        }
        ref = loaders.get_moshi_lm(ASSET / "model.safetensors", device="cuda", dtype=torch.bfloat16)
        assert {k: list(v.shape) for k, v in ref.state_dict().items()} == shapes
        stage("load production strict native LM")
        native = build_lm(
            cfg.LMArchConfig(),
            iter_lm_checkpoint(str(ASSET / "model.safetensors")),
            device="cuda",
            dtype=torch.bfloat16,
        )
        pipeline(ref, native)
        report["execution_completed"] = True
        report["exact_trajectory_match"] = (
            not report["checks"]["output_frame_mismatches"]
            and report["checks"]["agent_codes_different"] == 0
            and report["checks"]["text_tokens_different"] == 0
        )
        report["scope_note"] = (
            "Production nodes/resources executed directly, not through graph/conductor/worker; "
            "server reproduction is separate. Supported-reference runtime acceptance remains separate."
        )
        save()
        print("SUMMARY", json.dumps(report), flush=True)
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
    save()
    raise

control = report["checks"].get("negative_control")
caught = control is None or control["detected"]
raise SystemExit(
    0 if report.get("exact_trajectory_match") and report["checks"]["user_codes"]["different"] == 0 and caught else 1
)

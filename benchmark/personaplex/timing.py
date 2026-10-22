# ruff: noqa: E402
"""Direct-execution timing: NVIDIA's offline loop vs M*'s production nodes, batch size 1.

Neither side goes through HTTP; ``benchmark.runner`` covers served latency. NVIDIA
runs its own ``offline.py`` per-frame loop (Mimi encode, ``LMGen.step``, Mimi
decode, including its ``other_mimi`` copy), once with its default CUDA graphs and
once without. M* runs the LM node on the production paged-KV/FlashInfer
resources with whole-sequence Mimi, as served. Both decode greedily. Every timed
region ends with ``torch.cuda.synchronize()``; warmup runs are excluded.
"""

import argparse
import contextlib
import json
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

FRAME_SECONDS = 0.08


def summarize(frame_ms: list[float], audio_seconds: float, total_seconds: float) -> dict:
    """Per-frame distribution against the 80 ms real-time budget, plus RTF."""
    if not frame_ms or audio_seconds <= 0 or total_seconds <= 0:
        raise ValueError("timing summary needs frames, audio and elapsed time")
    ordered = sorted(frame_ms)

    def pct(q):
        return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]

    return {
        "frames": len(ordered),
        "frame_ms_p50": pct(0.5),
        "frame_ms_p90": pct(0.9),
        "frame_ms_max": ordered[-1],
        "frames_within_80ms": sum(t <= FRAME_SECONDS * 1000 for t in ordered) / len(ordered),
        "total_seconds": total_seconds,
        "rtf": total_seconds / audio_seconds,
    }


def median_runs(runs: list[dict]) -> dict:
    """Median of each numeric field over repeats, keeping the raw runs."""
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
    return {**{k: statistics.median(r[k] for r in runs) for k in keys}, "repeats": len(runs), "runs": runs}


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clip", action="append", required=True, help="id=path to a mono 24 kHz WAV")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--voices", type=Path, required=True)
    p.add_argument("--reference-site", default="")
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()

    sys.path.insert(0, str(a.source.resolve() / "moshi"))
    if a.reference_site:
        sys.path.append(a.reference_site)
    from moshi.models import loaders
    from moshi.models.lm import LMGen
    from moshi.utils.compile import no_cuda_graph
    from safetensors.torch import load_file

    from mstar.model.personaplex import config as cfg
    from mstar.model.personaplex.components.mimi import load_mimi_codec
    from mstar.model.personaplex.personaplex_model import PersonaPlexModel
    from mstar.model.personaplex.submodules import LMSubmodule, MimiDecoderSubmodule, MimiEncoderSubmodule
    from mstar.model.personaplex.weights import build_lm, iter_lm_checkpoint

    repo = Path(__file__).resolve().parents[2]
    a.output.mkdir(parents=True, exist_ok=False)
    asset = a.assets.resolve()
    mimi_path = asset / "tokenizer-e351c8d8-checkpoint125.safetensors"
    report = {
        "scope": "direct execution, batch size 1, greedy; not HTTP; NVIDIA under the M* torch runtime",
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "mstar_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "nvidia_source": subprocess.check_output(["git", "-C", str(a.source), "rev-parse", "HEAD"], text=True).strip(),
        "repeats": a.repeats,
        "results": {},
    }

    def save():
        (a.output / "timing.json").write_text(json.dumps(report, indent=2) + "\n")

    torch.backends.cuda.matmul.allow_tf32 = False
    model = PersonaPlexModel(skip_weight_loading=True, voice_dir=str(a.voices))
    import sentencepiece

    model.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(asset / "tokenizer_spm_32k_3.model"))

    def prepare(path):
        audio = model.load_audio(str(path), "cpu").data.reshape(-1)
        static = model.process_prompt(None, ["audio"], ["audio", "text"], {"audio_inputs": [audio]},
                                      voice="NATF2", greedy=True)
        frames = (audio.numel() + cfg.FRAME_SIZE - 1) // cfg.FRAME_SIZE
        meta = {
            "num_voice_embeddings": static["voice_embeddings"][0].shape[0],
            "num_text_tokens": static["persona_tokens"][0].numel(),
            "num_user_frames": frames,
            "num_samples": audio.numel(),
        }
        meta["conditioning_frames"] = meta["num_voice_embeddings"] + 2 * cfg.SILENCE_FRAMES + meta["num_text_tokens"]
        return audio, static, meta

    with torch.inference_mode():
        ref = loaders.get_moshi_lm(asset / "model.safetensors", device="cuda", dtype=torch.bfloat16).eval()
        ref_mimi = loaders.get_mimi(mimi_path, device="cuda")
        ref_other = loaders.get_mimi(mimi_path, device="cuda")
        native = build_lm(cfg.LMArchConfig(), iter_lm_checkpoint(str(asset / "model.safetensors")),
                          device="cuda", dtype=torch.bfloat16)
        codec = load_mimi_codec(load_file(str(mimi_path)), device="cuda")
        encoder, decoder = MimiEncoderSubmodule(codec), MimiDecoderSubmodule(codec)
        resources = Resources(model, cfg)
        sub = LMSubmodule(native)
        sub.bind_node_resources(resources.resources)

        def nvidia(audio, static, meta, graphs: bool):
            context = contextlib.nullcontext() if graphs else no_cuda_graph()
            with context:
                gen = LMGen(ref, audio_silence_frame_cnt=cfg.SILENCE_FRAMES, sample_rate=cfg.SAMPLE_RATE,
                            device="cuda", frame_rate=cfg.FRAME_RATE, use_sampling=False)
                for m in (ref_mimi, ref_other, gen):
                    m.streaming_forever(1)
                gen.load_voice_prompt_embeddings(str(a.voices / "NATF2.pt"))
                gen.text_prompt_tokens = static["persona_tokens"][0].tolist()
                for m in (ref_mimi, ref_other, gen):
                    m.reset_streaming()
                _, cond_ms = timed(lambda: gen.step_system_prompts(ref_mimi))
                ref_mimi.reset_streaming()
                pcm = audio.to("cuda").reshape(1, 1, -1)
                pcm = torch.nn.functional.pad(pcm, (0, meta["num_user_frames"] * cfg.FRAME_SIZE - pcm.shape[-1]))
                frame_ms, lm_ms = [], []
                torch.cuda.synchronize()
                start = time.perf_counter()
                for i in range(meta["num_user_frames"]):
                    chunk = pcm[..., i * cfg.FRAME_SIZE:(i + 1) * cfg.FRAME_SIZE]
                    t0 = time.perf_counter()
                    codes = ref_mimi.encode(chunk)
                    ref_other.encode(chunk)
                    tokens, step_ms = timed(lambda c=codes: gen.step(c[:, :, :1]))
                    if tokens is not None:
                        ref_mimi.decode(tokens[:, 1:9]).cpu()
                        ref_other.decode(tokens[:, 1:9])
                        int(tokens[0, 0, 0])
                    torch.cuda.synchronize()
                    frame_ms.append((time.perf_counter() - t0) * 1000)
                    lm_ms.append(step_ms)
                total = time.perf_counter() - start + cond_ms / 1000
                for m in (ref_mimi, ref_other, gen):
                    m.reset_streaming()
            out = summarize(frame_ms, meta["num_samples"] / cfg.SAMPLE_RATE, total)
            lm = summarize(lm_ms, meta["num_samples"] / cfg.SAMPLE_RATE, sum(lm_ms) / 1000)
            out.update(conditioning_ms=cond_ms, lm_step_ms_p50=lm["frame_ms_p50"], lm_step_ms_p90=lm["frame_ms_p90"])
            return out

        def mstar(audio, static, meta, rid):
            info = SimpleNamespace(step_metadata=meta)
            tensors = {"audio_inputs": [audio]}
            torch.cuda.synchronize()
            start = time.perf_counter()
            prepared = encoder.prepare_inputs("encode_user_audio", info, tensors)
            user, encode_ms = timed(lambda: encoder.forward("encode_user_audio", None,
                                                            **prepared.tensor_inputs)["user_audio_codes"][0])
            inputs = {**static, "user_audio_codes": [user]}
            resources.open(rid)
            state, agent = None, []
            cond_ms, frame_ms, host_ms = 0.0, [], []
            try:
                for index in range(meta["conditioning_frames"] + meta["num_user_frames"]):
                    walk = cfg.CONDITIONING_WALK if index < meta["conditioning_frames"] else cfg.GENERATE_WALK
                    t0 = time.perf_counter()
                    frame = sub.prepare_frame(walk, meta, {**inputs, "lm_state": [] if state is None else [state]})
                    t1 = time.perf_counter()
                    resources.begin(sub, walk, rid)
                    result = sub.run_frame(frame, sub._backend, lambda x: resources.sample(cfg.TEXT_SAMPLER, rid, x),
                                           lambda x, _i: resources.sample(cfg.DEPTH_SAMPLER, rid, x))
                    resources.finish()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - t0) * 1000
                    state = result["lm_state"][0]
                    if walk == cfg.CONDITIONING_WALK:
                        cond_ms += elapsed
                    else:
                        agent.append(result["agent_frame"][0])
                        frame_ms.append(elapsed)
                        host_ms.append((t1 - t0) * 1000)
            finally:
                resources.close(rid)
            prepared = decoder.prepare_inputs("generate", info, {"agent_frame": agent})
            _, decode_ms = timed(lambda: decoder.forward("generate", None, **prepared.tensor_inputs, **prepared.kwargs))
            total = time.perf_counter() - start
            out = summarize(frame_ms, meta["num_samples"] / cfg.SAMPLE_RATE, total)
            out.update(conditioning_ms=cond_ms, encode_ms=encode_ms, decode_ms=decode_ms,
                       prepare_frame_ms_p50=statistics.median(host_ms))
            return out

        clips = [c.split("=", 1) for c in a.clip]
        warm_audio, warm_static, warm_meta = prepare(clips[0][1])
        warm_meta = {**warm_meta, "num_user_frames": 16, "num_samples": 16 * cfg.FRAME_SIZE}
        warm_audio = warm_audio[: 16 * cfg.FRAME_SIZE]
        for graphs in (True, False):
            nvidia(warm_audio, warm_static, warm_meta, graphs)
        mstar(warm_audio, warm_static, warm_meta, "warmup")
        report["warmup"] = "16-frame run per system, excluded"
        save()
        for name, path in clips:
            audio, static, meta = prepare(path)
            report["results"][name] = {"request": meta}
            for system in ("nvidia-cuda-graphs", "nvidia-eager", "mstar-direct"):
                runs = []
                for r in range(a.repeats):
                    if system == "mstar-direct":
                        runs.append(mstar(audio, static, meta, f"{name}-{r}"))
                    else:
                        runs.append(nvidia(audio, static, meta, system == "nvidia-cuda-graphs"))
                report["results"][name][system] = median_runs(runs)
                save()
                print(name, system, {k: round(v, 3) for k, v in report["results"][name][system].items()
                                     if isinstance(v, float)}, flush=True)
    report["completed"] = True
    save()


class Resources:
    """Production LM resources driven by hand, as in ``parity.ResourceBackend``, built once."""

    def __init__(self, model, cfg):
        from mstar.communication.tensors import LocalTransferEngine
        from mstar.engine.resources.base import EngineResourceInfo, build_resource
        from mstar.engine.resources.kv.transfer import TransferEngineInfo
        from mstar.engine.resources.runner import StepRunner

        self.model, self.cfg = model, cfg
        specs = model.get_node_resources()
        by_key = {s.resource_key: s for s in specs}
        transfer = TransferEngineInfo("timing", socket.gethostname(), LocalTransferEngine(socket.gethostname()))
        self.resources = {
            s.resource_key: build_resource(s, EngineResourceInfo(
                device=torch.device("cuda:0"), kv_dtype=torch.bfloat16, transfer_engine_info=transfer,
                dependencies={k: by_key[k] for k in s.depends_on()},
            ))
            for s in specs
        }
        self.runner = StepRunner(self.resources)

    def open(self, rid):
        from mstar.engine.resources.kv.config import KVReqConfig

        overrides = self.model.get_request_resource_configs({}, {"greedy": True})
        overrides[self.cfg.LM_KV] = KVReqConfig(needed_labels=["main"], prefix_cache=False)
        self.runner.ingest_request(rid, overrides)

    def begin(self, sub, walk, rid):
        from mstar.engine.resources.step import StepContext

        self.step = sub.declare_step(walk, [rid], [SimpleNamespace(input_seq_len=1)])
        self.step.set_ctx(StepContext([rid], walk, slot=0, capture=False))
        outcome = self.runner.admit(self.step)
        assert outcome.ok and outcome.ready, outcome
        self.runner.plan(self.step)
        sub._backend.begin()

    def finish(self):
        self.runner.commit(self.step)

    def close(self, rid):
        self.runner.remove_request(rid)

    def sample(self, key, rid, logits):
        return self.resources[key].sample([rid], logits.float())


if __name__ == "__main__":
    main()

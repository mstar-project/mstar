"""Correctness column of the tables, from the correctness runner's sample sets, references and metrics
(benchmark/correctness).

Text arms get a greedy pass on the set right after the server starts. mstar gets the correctness
runner's own /generate requests. vLLM and SGLang get the same prompts as chat completions with
per-token logprobs, turned back into one chunk per token as the text metrics expect. Omni arms get
a pass of their own too, with spoken and written answers. TTS, ASR, image and video arms are scored
on the outputs of their first timed run at concurrency 1, which covers the set once. Every pass is
saved in the correctness runner's layout (r1/outputs, r1/responses.jsonl) and scored with its
family code.

    python -m benchmark.amd.correctness score RUN_DIR      # what the sweep runs at the end
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from benchmark.amd import catalog
from benchmark.amd.launch import REPO

WORK = Path("/work1/stephaniewang/atindra")
SETS_ROOT = WORK / "amd/results/correctness/data"
MAIN_REPO = WORK / "mstar"


def prepare_source(run_dir: Path, src: str | None) -> Path:
    """Directory whose benchmark/correctness the run imports: this tree by default, else ``src``, a
    checkout or a git ref. A ref is exported once into the run dir, so a later rebase can't change
    what a run was scored with."""
    if src is None:
        if not (REPO / "benchmark/correctness/__init__.py").is_file():
            raise SystemExit("this tree has no benchmark/correctness; pass --correctness-src")
        return REPO
    if (Path(src) / "benchmark/correctness/__init__.py").is_file():
        return Path(src).resolve()
    ref = src
    dest = run_dir / "correctness_src"
    if not (dest / "benchmark/correctness/__init__.py").is_file():
        dest.mkdir(parents=True, exist_ok=True)
        commit = subprocess.run(["git", "-C", str(MAIN_REPO), "rev-parse", ref], capture_output=True, text=True,
                                check=True).stdout.strip()
        archive = subprocess.run(["git", "-C", str(MAIN_REPO), "archive", commit, "benchmark/correctness"],
                                 capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
        (dest / "COMMIT").write_text(f"{ref} {commit}\n")
    return dest


def load(src: Path):
    """Import benchmark.correctness from ``src`` and return its modules."""
    import benchmark

    pkg = str(src / "benchmark")
    if src != REPO and pkg not in benchmark.__path__:
        benchmark.__path__.append(pkg)
    mods = {}
    for name in ("client", "sets", "families", "report"):
        mods[name] = importlib.import_module(f"benchmark.correctness.{name}")
    return type("WSK", (), mods)


def _family(C, worktree: Path, config_name: str, opts: dict | None = None):
    cfg = yaml.safe_load((worktree / "configs" / f"{config_name}.yaml").read_text()) or {}
    return C.families.family_for_config(config_name, cfg.get("model"), cfg, opts or {}), cfg


def family_opts(wl) -> dict:
    """The correctness family's options for a workload: image-to-video cells use the family's i2v mode."""
    return {"mode": "i2v"} if wl is not None and wl.request_type == "image_to_video" else {}


def _write_pass(C, fam, sset, responses, out_dir: Path, header: dict, cfg: dict) -> dict:
    run = out_dir / "r1"
    if run.exists():
        shutil.rmtree(run)
    (run / "outputs").mkdir(parents=True)
    with open(run / "responses.jsonl", "w") as f:
        for r in responses:
            f.write(json.dumps(r.summary(), default=str) + "\n")
    fam.save(sset, responses, run / "outputs")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    ok = sum(r.ok for r in responses)
    header.update(n_items=len(sset.items), set=sset.name, ok=ok, failed=len(responses) - ok,
                  errors={r.item_id: r.error for r in responses if not r.ok})
    (out_dir / "pass.json").write_text(json.dumps(header, indent=1, default=str))
    return {"ok": ok, "failed": len(responses) - ok, "set": sset.name}


# --- text ------------------------------------------------------------------------------------

def openai_bodies(fam, sset, hf_id: str, engine: str) -> list[dict]:
    """Chat completion bodies with the text family's prompts and settings, plus per-token logprobs."""
    bodies = []
    for i, it in enumerate(sset.items):
        n = int(it.params.get("max_new_tokens", 128))
        body = {"model": hf_id, "messages": [{"role": "user", "content": [{"type": "text", "text": it.prompt}]}],
                "max_tokens": n, "temperature": 0.0, "seed": i, "logprobs": True, "top_logprobs": 0,
                "chat_template_kwargs": {"enable_thinking": fam.thinking}}
        if engine == "vllm":
            body["return_tokens_as_token_ids"] = True
        bodies.append(body)
    return bodies


def chunks_from_chat(body: bytes, tok, eos_ids: set[int]) -> tuple[list[str], int | None]:
    """One decoded string per generated token from a chat completion with logprobs, the form
    mstar streams. vLLM names tokens ``token_id:N``; SGLang sends each token's own text. Returns
    the chunks and the prompt token count."""
    doc = json.loads(body)
    choice = doc["choices"][0]
    content = (choice.get("logprobs") or {}).get("content") or []
    specials = {tok.decode([i]) for i in eos_ids}
    chunks = []
    for entry in content:
        t = entry.get("token", "")
        if t.startswith("token_id:"):
            tid = int(t.split(":", 1)[1])
            if tid in eos_ids:
                break
            chunks.append(tok.decode([tid]))
        else:
            if t in specials:
                break
            chunks.append(t)
    return chunks, (doc.get("usage") or {}).get("prompt_tokens")


def text_pass(plan, row, arm, port: int, out_dir: Path, worktree: Path | None = None) -> dict:
    C = load(plan.correctness_src)
    config = catalog.mstar_config_name(row, arm.name)
    tree = worktree or plan.worktree
    fam, cfg = _family(C, tree, config)
    sset = C.sets.load_set(fam.set_name, SETS_ROOT)
    base = f"http://127.0.0.1:{port}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if arm.engine == "mstar":
        responses = C.client.send_all(base, fam.requests(sset), concurrency=fam.default_concurrency,
                                      timeout_s=fam.request_timeout_s, progress=False)
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(fam.hf_id)
        eos = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                           tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(i, int) and i >= 0}
        reqs = [C.client.Request(it.id, route="/v1/chat/completions", json_body=b)
                for it, b in zip(sset.items, openai_bodies(fam, sset, row.hf_id, arm.engine), strict=True)]
        responses = C.client.send_all(base, reqs, concurrency=fam.default_concurrency,
                                      timeout_s=fam.request_timeout_s, progress=False)
        prompt_tokens = {}
        for r in responses:
            if not r.ok:
                continue
            try:
                chunks, prompt_tokens[r.item_id] = chunks_from_chat(r.body, tok, eos)
            except (ValueError, KeyError, IndexError) as e:
                r.ok, r.error = False, f"unreadable response: {e}"
                continue
            r.text_chunks = [c.encode("utf-8") for c in chunks]
        (out_dir / "prompt_tokens.json").write_text(json.dumps(prompt_tokens, indent=1))
    header = {"config": config, "model": cfg.get("model"), "arm": arm.name, "engine": arm.engine,
              "restarts": 1, "concurrency": fam.default_concurrency, "family_opts": {},
              "seconds": round(time.monotonic() - t0, 1), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    return _write_pass(C, fam, sset, responses, out_dir, header, cfg)


# --- Qwen3-Omni ----------------------------------------------------------------------------------

MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".wav": "audio/wav", ".mp4": "video/mp4"}


def omni_chat_body(fam, sset, item, index: int, hf_id: str, system: str) -> dict:
    """The omni family's request as an OpenAI chat completion with a spoken and written answer."""
    import base64

    mod = item.params["modality"]
    path = sset.path(item.files[mod])
    url = f"data:{MIME.get(path.suffix.lower(), 'application/octet-stream')};base64,"
    url += base64.b64encode(path.read_bytes()).decode()
    content = [{"type": f"{mod}_url", f"{mod}_url": {"url": url}}, {"type": "text", "text": item.prompt}]
    return {"model": hf_id, "messages": [{"role": "system", "content": [{"type": "text", "text": system}]},
                                         {"role": "user", "content": content}],
            "modalities": ["text", "audio"], "audio": {"format": "pcm"}, "temperature": 0.0,
            "max_tokens": fam.max_new_tokens, "seed": index, "stream": True}


def parse_omni_stream(body: bytes) -> tuple[list[str], bytes]:
    """Text deltas (one per streamed token) and raw PCM16 audio from a vllm-omni chat stream."""
    import base64

    chunks, audio = [], bytearray()
    for block in body.split(b"\n\n"):
        line = block.strip()
        if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
            continue
        try:
            msg = json.loads(line[5:])
        except ValueError:
            continue
        for choice in msg.get("choices", []):
            delta = choice.get("delta") or {}
            content = delta.get("content")
            audio_obj = delta.get("audio") if isinstance(delta.get("audio"), dict) else {}
            if content and msg.get("modality") == "audio":
                audio += base64.b64decode(content)
            elif content:
                chunks.append(content)
            elif audio_obj.get("data"):
                audio += base64.b64decode(audio_obj["data"])
    return chunks, bytes(audio)


def omni_pass(plan, row, arm, port: int, out_dir: Path, worktree: Path | None = None) -> dict:
    """The omni family's set, answered in text and speech. mstar gets the family's own /generate
    requests; vllm-omni gets the same questions as chat completions with text and PCM audio."""
    C = load(plan.correctness_src)
    config = catalog.mstar_config_name(row, arm.name)
    tree = worktree or plan.worktree
    fam, cfg = _family(C, tree, config)
    sset = C.sets.load_set(fam.set_name, SETS_ROOT)
    base = f"http://127.0.0.1:{port}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if arm.engine == "mstar":
        responses = C.client.send_all(base, fam.requests(sset), concurrency=fam.default_concurrency,
                                      timeout_s=fam.request_timeout_s, progress=False)
    else:
        system = importlib.import_module("benchmark.correctness.families.omni").SYSTEM
        reqs = [C.client.Request(it.id, route="/v1/chat/completions",
                                 json_body=omni_chat_body(fam, sset, it, i, row.hf_id, system))
                for i, it in enumerate(sset.items)]
        responses = C.client.send_all(base, reqs, concurrency=fam.default_concurrency,
                                      timeout_s=fam.request_timeout_s, progress=False)
        for r in responses:
            if r.ok:
                chunks, pcm = parse_omni_stream(r.body)
                r.text_chunks = [c.encode("utf-8") for c in chunks]
                r.audio = bytearray(pcm)
    header = {"config": config, "model": cfg.get("model"), "arm": arm.name, "engine": arm.engine,
              "restarts": 1, "concurrency": fam.default_concurrency, "family_opts": {},
              "seconds": round(time.monotonic() - t0, 1), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    return _write_pass(C, fam, sset, responses, out_dir, header, cfg)


# --- generators ---------------------------------------------------------------------------------

def _flow_shift(hf_id: str) -> float | None:
    """The checkpoint's scheduler flow shift, which vllm-omni's Wan2.2 defaults would otherwise override."""
    try:
        from huggingface_hub import hf_hub_download

        cfg = json.loads(Path(hf_hub_download(hf_id, "scheduler/scheduler_config.json",
                                              local_files_only=True)).read_text())
    except Exception:
        return None
    return cfg.get("flow_shift")


def media_request_kwargs(plan, row, arm=None, wl=None, worktree: Path | None = None) -> dict:
    """Settings every image or video request of a row carries: those of the row's correctness family (its i2v mode
    for image-to-video cells), so the timed outputs can be scored against its diffusers reference. The family's
    request_extra (mstar serving knobs) goes to mstar arms only, and the row's generator_device to arms whose
    protocol takes it."""
    C = load(plan.correctness_src or REPO)
    config = catalog.mstar_config_name(row, arm.name) if arm is not None else row.name
    fam, _ = _family(C, worktree or plan.worktree, config, family_opts(wl))
    if row.family == "image":
        guidance = fam.guidance if fam.guidance is not None else (fam.ref_extra or {}).get("guidance_scale")
        kw = {"num_inference_steps": fam.steps, "guidance_scale": guidance}
    else:
        # Where the serving recipe sets its own flow shift (Cosmos3 Edge), every engine gets that one.
        shift = (getattr(fam, "ref_scheduler_kwargs", None) or {}).get("flow_shift") or _flow_shift(row.hf_id)
        kw = {"height": fam.height, "width": fam.width, "num_frames": fam.num_frames,
              "num_inference_steps": fam.steps, "fps": fam.fps, "guidance_scale": fam.guidance,
              "negative_prompt": fam.negative_prompt, "flow_shift": shift}
    if arm is not None and arm.engine == "mstar":
        kw.update(getattr(fam, "request_extra", None) or {})
    if arm is not None and row.generator_device and row.family in arm.generator_device_for:
        kw["generator_device"] = row.generator_device
    return {k: v for k, v in kw.items() if v is not None}


# --- outputs of a timed run ---------------------------------------------------------------------

def pass_from_cell(C, worktree: Path, row_name: str, arm_name: str, cell_dir: Path, out_dir: Path,
                   ext: str, opts: dict | None = None) -> dict:
    """The correctness pass of a TTS (wav), ASR (txt), image (png) or video (mp4) arm: its first timed
    run at concurrency 1, which sends every item of the set once, in order."""
    row = catalog.ROWS.get(row_name)
    config = catalog.mstar_config_name(row, arm_name) if row else row_name
    fam, cfg = _family(C, worktree, config, opts)
    sset = C.sets.load_set(fam.set_name, SETS_ROOT)
    run = cell_dir / "a1r1"
    results = json.loads((run / "results.json").read_text())
    ok_ids = {r["request_id"] for r in results.get("per_request", [])}
    responses = []
    for i, it in enumerate(sset.items):
        r = C.client.Response(it.id)
        if i in ok_ids and (run / f"req_{i}.{ext}").is_file():
            r.ok = True
        else:
            r.error = "no output in the timed run"
        responses.append(r)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_out = out_dir / "r1"
    if run_out.exists():
        shutil.rmtree(run_out)
    (run_out / "outputs").mkdir(parents=True)
    for i, (it, r) in enumerate(zip(sset.items, responses, strict=True)):
        if r.ok:
            shutil.copyfile(run / f"req_{i}.{ext}", run_out / "outputs" / f"{it.id}.{ext}")
    with open(run_out / "responses.jsonl", "w") as f:
        for r in responses:
            f.write(json.dumps(r.summary(), default=str) + "\n")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    ok = sum(r.ok for r in responses)
    header = {"config": config, "model": cfg.get("model"), "arm": arm_name, "restarts": 1, "concurrency": 1,
              "set": sset.name, "n_items": len(sset.items), "ok": ok, "failed": len(responses) - ok,
              "source": str(run), "family_opts": opts or {}}
    (out_dir / "pass.json").write_text(json.dumps(header, indent=1))
    return header


# --- scoring --------------------------------------------------------------------------------

def _key_metrics(score) -> dict:
    m = score.metrics
    out = {"verdict": score.verdict}
    for k in ("exact_vs_reference", "wer_percent", "reference_wer_percent", "duration_ratio",
              "log_mel_distance_db"):
        if k in m:
            out[k] = m[k]
    for c in score.checks:
        if c.name.startswith("divergences vs reference"):
            out["unexplained_divergences"] = c.value
        if c.name.startswith("WER - reference WER"):
            out["wer_minus_reference_pt"] = c.value
    return out


def _arm_tree(arm_dir: Path, default: Path) -> Path:
    """The mstar tree an arm served (from its versions.json), whose configs its correctness family reads."""
    v = arm_dir / "server" / "versions.json"
    tree = (json.loads(v.read_text()).get("git") or {}).get("worktree") if v.is_file() else None
    return Path(tree) if tree else default


def score_dir(C, worktree: Path, corr_dir: Path, device: str) -> dict:
    header = json.loads((corr_dir / "pass.json").read_text())
    fam, _ = _family(C, worktree, header["config"], header.get("family_opts"))
    sset = C.sets.load_set(fam.set_name, SETS_ROOT)
    ref_dir = SETS_ROOT / "references" / fam.reference_id() / fam.set_name
    ref = ref_dir if fam.ref_outputs(ref_dir) is not None else None
    work = corr_dir / "scoring"
    work.mkdir(exist_ok=True)
    header.update(reference=fam.reference_description() if ref else f"missing ({ref_dir})", protocol=fam.protocol(),
                  run_id=corr_dir.parent.name, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    try:
        score = fam.score(sset, [corr_dir / "r1"], ref, work, device)
    except Exception as e:  # recorded in the report
        C.report.write_reports(corr_dir, header, None, f"{type(e).__name__}: {e}")
        return {"verdict": "ERROR", "error": f"{type(e).__name__}: {e}"}
    C.report.write_reports(corr_dir, header, score)
    out = _key_metrics(score)
    if ref is None:
        out["reference"] = "missing"
    # The engine's own prompt token count (usage.prompt_tokens) against the reference's: a mismatch means it
    # tokenized the prompt differently, which explains a divergence on that item.
    pt_file = corr_dir / "prompt_tokens.json"
    if ref is not None and pt_file.is_file():
        differs = []
        for item_id, n in json.loads(pt_file.read_text()).items():
            r = ref / "outputs" / f"{item_id}.json"
            if n is not None and r.is_file() and json.loads(r.read_text()).get("prompt_len") not in (None, n):
                differs.append(item_id)
        out["prompt_tokens_differ"] = differs
    (corr_dir / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def score(run_dir: Path, device: str = "cuda:0") -> dict:
    meta = json.loads((run_dir / "sweep.json").read_text())
    worktree = Path(meta["worktree"])
    C = load(Path(meta["correctness_src"]))
    results = {}
    for arm_json in sorted(run_dir.glob("*/*/arm.json")):
        arm_dir = arm_json.parent
        info = json.loads(arm_json.read_text())
        if arm_dir.name != info.get("arm"):
            continue  # an arm run set aside (e.g. voided by its health check); its score would replace the live one's
        corr = arm_dir / "correctness"
        tree = _arm_tree(arm_dir, worktree)
        if not (corr / "pass.json").is_file():
            for wl, ext in (("tts-seed", "wav"), ("asr-libri", "txt"), ("image-gen", "png"), ("video-t2v", "mp4")):
                cell = arm_dir / "cells" / wl / "c1"
                if (cell / "a1r1" / "results.json").is_file():
                    pass_from_cell(C, tree, info["row"], info["arm"], cell, corr, ext)
                    break
        if (corr / "pass.json").is_file():
            results[f"{info['row']}/{info['arm']}"] = score_dir(C, tree, corr, device)
        # Image-to-video cells have their own pass, scored against the family's i2v reference.
        cell = arm_dir / "cells" / "video-i2v" / "c1"
        corr_i2v = arm_dir / "correctness-video-i2v"
        if (cell / "a1r1" / "results.json").is_file() and not (corr_i2v / "pass.json").is_file():
            pass_from_cell(C, tree, info["row"], info["arm"], cell, corr_i2v, "mp4", {"mode": "i2v"})
        if (corr_i2v / "pass.json").is_file():
            results[f"{info['row']}/{info['arm']}/video-i2v"] = score_dir(C, tree, corr_i2v, device)
    (run_dir / "correctness.json").write_text(json.dumps(results, indent=1, default=str))
    return results


def score_run(run_dir: Path, gpu: int, log=print) -> None:
    """Score in a child process that sees only ``gpu`` (the driver itself never opens a GPU)."""
    env = {**os.environ, "HIP_VISIBLE_DEVICES": str(gpu), "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
           "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")])}
    with open(run_dir / "correctness.log", "w") as f:
        p = subprocess.run([sys.executable, "-m", "benchmark.amd.correctness", "score", str(run_dir)], cwd=REPO,
                           env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    log(f"correctness scoring rc {p.returncode} (log {run_dir / 'correctness.log'})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("run_dir")
    s.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    res = score(Path(args.run_dir), args.device)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

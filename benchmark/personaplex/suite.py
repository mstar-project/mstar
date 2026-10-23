"""Resumable validation orchestration. Performance numbers only come from benchmark.runner."""

import argparse
import base64
import csv
import hashlib
import html
import importlib.metadata
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from benchmark.personaplex.dataset import read_manifest
from benchmark.personaplex.serving import write_json

PIN = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
STATUSES = {"PASS", "FAIL", "SKIP", "BLOCKED", "NOT RUN"}
# With --deadline-seconds, stop launching stages this long before the deadline so
# evidence can be written and servers stopped (smoke stages are shorter).
STOP_RESERVE_S = {"smoke": 120, "full": 900}
# Files at most this large are content-hashed for the resume identity; larger
# ones (the 16 GB checkpoint) are identified by size and modification time.
HASH_LIMIT_BYTES = 256 << 20
ASSET_FILES = ("model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors", "tokenizer_spm_32k_3.model")
# The batching check alternates these presets, so a batch mixes voice lengths.
VOICES = ("NATF2", "NATM1")
INTERRUPTED = False


def interrupted(*_):
    global INTERRUPTED

    INTERRUPTED = True
    raise InterruptedError("Allocation/controller interrupted")


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def entry(path, name):
    path = Path(path).resolve()
    return {"id": name, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def preflight(settings, root):
    if git("rev-parse", "HEAD") != settings["expected_head"]:
        raise ValueError("Checkout HEAD differs from expected committed baseline")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked working changes: preserve them, then use an isolated clean checkout")
    if git("-C", settings["source"], "rev-parse", "HEAD") != PIN:
        raise ValueError("NVIDIA source pin differs")
    if git("-C", settings["source"], "status", "--porcelain"):
        raise ValueError("NVIDIA reference checkout is dirty")
    for key in ("source", "assets", "voices", "cache", "server_config", "existing_clip"):
        if not Path(settings[key]).exists():
            raise FileNotFoundError(f"Missing {key}: {settings[key]}")
    for name in ASSET_FILES:
        if not (Path(settings["assets"]) / name).is_file():
            raise FileNotFoundError(name)
    for voice in VOICES:
        if not (Path(settings["voices"]) / f"{voice}.pt").is_file():
            raise FileNotFoundError(f"{voice}.pt")
    extension = rust_extension_files(settings)
    entries = [entry(settings["existing_clip"], "existing-222")]
    if settings.get("english_clip") and Path(settings["english_clip"]).is_file():
        entries.append(entry(settings["english_clip"], "english"))
    # A short prefix is a lifecycle control, not a substitute English recording.
    short = root / "short-control.wav"
    with wave.open(entries[0]["path"], "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 24000):
            raise ValueError("Existing 222-frame control must be mono PCM16, 24 kHz")
        if (source.getnframes() + 1919) // 1920 != 222:
            raise ValueError("Existing control is not 222 frames")
        pcm = source.readframes(16 * 1920)
    with tempfile.TemporaryDirectory() as temporary:
        manifest = Path(temporary) / "inputs.json"
        write_json(manifest, entries)
        checked = read_manifest(manifest)
    versions = {}
    for package in ("torch", "transformers", "flashinfer-python", "triton", "numpy", "aiohttp"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    identity = {
        "settings": settings,
        "commit": git("rev-parse", "HEAD"),
        "versions": versions,
        "inputs": checked,
        "server_config_sha256": hashlib.sha256(Path(settings["server_config"]).read_bytes()).hexdigest(),
        "source_sha": PIN,
        "extension_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in extension},
        "model_files": {
            str(path): file_identity(path)
            for path in [*(Path(settings["assets"]) / name for name in ASSET_FILES),
                         *(Path(settings["voices"]) / f"{voice}.pt" for voice in VOICES)]
        },
        "reference_runtime_scope": "Same-runtime reference; supported NVIDIA runtime deferred",
    }
    identity["fingerprint"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    old = root / "identity.json"
    if old.exists() and json.loads(old.read_text())["fingerprint"] != identity["fingerprint"]:
        raise ValueError("Resume identity changed: use a new output directory; prior evidence remains untouched")
    with wave.open(str(short), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24000)
        target.writeframes(pcm)
    write_json(root / "inputs.json", entries)
    write_json(root / "lifecycle-inputs.json", [entry(short, "short-control"), *entries])
    write_json(root / "smoke-inputs.json", [entry(short, "short-control")])
    write_json(old, identity)
    write_json(root / "settings.json", settings)
    return entries


def rust_extension_files(settings):
    """The mstar_rust binaries the run imports: a separately built ``extension``
    directory, or else the module installed in the environment."""
    if settings.get("extension"):
        files = sorted(Path(settings["extension"]).glob("mstar_rust*.so"))
        if not files:
            raise FileNotFoundError(f"No mstar_rust*.so in extension directory {settings['extension']}")
        return files
    spec = importlib.util.find_spec("mstar_rust")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("mstar_rust is not importable: build it (docs/installation.rst) or set extension")
    origin = Path(spec.origin)
    return sorted(origin.parent.glob("mstar_rust*.so")) if origin.name == "__init__.py" else [origin]


def file_identity(path, hash_limit=HASH_LIMIT_BYTES):
    """Content digest of a model file, or its size and mtime when too large to hash on every run."""
    stat = Path(path).stat()
    if stat.st_size <= hash_limit:
        return {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def run_kind(root, mode):
    """``smoke`` or ``full``: what a new run is, or what the run being resumed was."""
    path = root / "run.json"
    if mode != "resume":
        if (root / "results.json").exists():
            raise ValueError("Existing results: use resume or a fresh output directory")
        write_json(path, {"kind": mode})
        return mode
    if not path.exists():
        raise ValueError(f"Nothing to resume in {root}: no run.json")
    return json.loads(path.read_text())["kind"]


def reusable(prior, command, retry_failed):
    """A finished stage is reused only if it ran this exact command (and, with
    ``retry_failed``, only if it passed)."""
    if prior.get("command") != command or prior.get("status") not in ("PASS", "FAIL"):
        return False
    return not (prior["status"] == "FAIL" and retry_failed)


def stage_deadline(started, deadline_seconds, smoke):
    """Last moment to launch a stage: no limit unless a deadline is given."""
    if deadline_seconds is None:
        return float("inf")
    return started + deadline_seconds - STOP_RESERVE_S["smoke" if smoke else "full"]


def missing_clip_results(entries, smoke):
    """BLOCKED rows for the English clip's parity cases when the settings name none.

    Lifecycle and warm cells simply run without the clip, so only these two are missing.
    """
    if smoke or any(e["id"] == "english" for e in entries):
        return {}
    reason = "No english_clip in the settings; no substitute recording is used"
    return {f"parity-english-{backend}": {"status": "BLOCKED", "reason": reason} for backend in ("dense", "resources")}


def judge_blocked(settings):
    """The judge stage's BLOCKED row when the settings name no ASR model."""
    if settings.get("asr_model"):
        return {}
    return {"judge": {"status": "BLOCKED", "reason": "No asr_model in the settings; listen to listen.html instead"}}


def subprocess_case(command, output, logpath, deadline, timeout, env):
    remaining = min(timeout, deadline - time.time())
    if remaining <= 0:
        return {"status": "NOT RUN", "reason": "Deadline reserve"}
    logpath.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    commit = git("rev-parse", "HEAD")
    with logpath.open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
            timed_out = False
        except (subprocess.TimeoutExpired, InterruptedError):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            code, timed_out = process.returncode, True
    result = {
        "status": "PASS" if code == 0 else "FAIL",
        "commit": commit,
        "command": command,
        "exit": code,
        "timeout": timed_out,
        "started": started,
        "ended": time.time(),
        "log": str(logpath),
    }
    summary = output / "summary.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        if data.get("status") in STATUSES:
            result["status"] = data["status"] if code == 0 else "FAIL"
        result["summary"] = str(summary)
    return result


def report(root):
    results = json.loads((root / "results.json").read_text())
    identity = json.loads((root / "identity.json").read_text())
    lines = [
        "PersonaPlex validation",
        "======================",
        "",
        f"Commit: {identity['commit']}",
        "",
        "Same-runtime NVIDIA comparison; supported-reference GPU runtime deferred.",
        "Performance measurements are repository benchmark.runner results only.",
        "",
    ]
    players = []
    numeric_rows, benchmark_rows = [], []
    for name, row in results.items():
        lines += [
            f"{name}: {row['status']}",
            f"  Commit: {row.get('commit', identity['commit'])}",
            f"  {row.get('reason', row.get('log', ''))}",
        ]
        comparison = root / name / "comparison.json"
        if comparison.exists():
            values = json.loads(comparison.read_text())
            numeric_rows.append(
                {
                    "case": name,
                    "status": row["status"],
                    "agent_differences": values["agent_code_differences"],
                    "agent_frames": values["agent_frame_count"],
                    **values["float_audio"],
                }
            )
            lines.append(f"  Numerical metrics: {json.dumps(numeric_rows[-1], sort_keys=True)}")
            if values.get("first_decision"):
                lines.append(f"  First decision: {json.dumps(values['first_decision'], sort_keys=True)}")
        summary = root / name / "summary.json"
        if name.startswith("batch-") and summary.exists():
            values = json.loads(summary.read_text())
            for size, cases in values.get("batched", {}).items():
                lines.append(f"  Batch of {size} vs alone: {json.dumps(cases, sort_keys=True)}")
            lines.append(f"  LM step timing: {json.dumps(values.get('timing'), sort_keys=True)}")
        if name == "judge" and summary.exists():
            for stage, values in json.loads(summary.read_text()).get("stages", {}).items():
                lines.append(f"  {stage}: {json.dumps(values, sort_keys=True)}")
        cells = root / name / "cells.json"
        if cells.exists():
            for cell in json.loads(cells.read_text()):
                values = {"clip": cell["clip"], "concurrency": cell["concurrency"], "status": cell["status"]}
                if cell.get("attempts"):
                    raw_path = Path(cell["attempts"][-1]["results"])
                    if raw_path.exists():
                        raw = json.loads(raw_path.read_text())
                        aggregate = raw.get("aggregate", {})
                        values.update(
                            samples=raw.get("completed"),
                            failed=raw.get("failed"),
                            latency_p50_s=aggregate.get("e2e_latency_s", {}).get("p50"),
                            rtf_p50=(aggregate.get("rtf") or {}).get("p50"),
                            audio_seconds_throughput=aggregate.get("audio_seconds_throughput"),
                            source=str(raw_path),
                        )
                benchmark_rows.append(values)
                lines.append(f"  Repository benchmark: {json.dumps(values, sort_keys=True)}")
        for label in ("input", "nvidia", "mstar"):
            path = root / name / (label + ".wav")
            if path.is_file():
                encoded = base64.b64encode(path.read_bytes()).decode()
                players.append(
                    f"<h2>{html.escape(name)} — {label}</h2>"
                    f"<audio controls preload='none' src='data:audio/wav;base64,{encoded}'></audio>"
                )
    lines += [
        "",
        "Deferred: injected worker errors, supported NVIDIA runtime, accepted near-limit",
        "requests, broad quality claims, CUDA graphs, live ingress and streamed audio.",
        "A completed comparison does not approve numerical divergence or establish quality.",
    ]
    (root / "REPORT.rst").write_text("\n".join(lines) + "\n")
    for filename, rows in (("numerical.csv", numeric_rows), ("benchmarks.csv", benchmark_rows)):
        if rows:
            with (root / filename).open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
                writer.writeheader()
                writer.writerows(rows)
    (root / "listen.html").write_text(
        "<!doctype html><meta name='viewport' content='width=device-width'><title>PersonaPlex evidence</title>"
        "<style>body{font:17px system-ui;max-width:850px;margin:auto;padding:20px}audio{width:100%}</style>"
        "<h1>Matched recordings</h1><p>NVIDIA reference model execution, not its web server. "
        "Original PCM levels, no normalization. M* dense is a control; resources is the paged production backend. "
        "Numerical metrics and limitations are in REPORT.rst and comparison JSON.</p>" + "".join(players)
    )
    complete = bool(results) and all(r["status"] in ("PASS", "SKIP") for r in results.values())
    print(json.dumps({name: row["status"] for name, row in results.items()}, indent=2))
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "smoke", "full", "resume", "status", "report"])
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--deadline-seconds", type=float, default=None,
        help="stop launching stages near this many seconds after start (e.g. a job's wall time); default: none",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode in ("status", "report"):
        return report(root)
    settings_path = args.settings or root / "settings.json"
    settings = json.loads(settings_path.read_text())
    entries = preflight(settings, root)
    if args.mode == "preflight":
        print("PASS: committed identity, source pin, assets and local manifests")
        return 0
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; no numerical/serving evidence produced")
    smoke = run_kind(root, args.mode) == "smoke"
    started = time.time()
    deadline = stage_deadline(started, args.deadline_seconds, smoke)
    env = dict(os.environ, MSTAR_SHM_ARENA="1", PYTHONUNBUFFERED="1", PERSONAPLEX_DEADLINE=str(deadline))
    env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd()), env.get("PYTHONPATH", "")])
    write_json(
        root / "hardware.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
            "started": started,
            "deadline_seconds": args.deadline_seconds,
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
        },
    )
    scenarios = []
    parity_entries = [entry(root / "short-control.wav", "short-control")] if smoke else entries
    for item in parity_entries:
        for backend in ["resources"] if smoke else ["dense", "resources"]:
            name = f"parity-{item['id']}-{backend}"
            command = [
                sys.executable,
                "-m",
                "benchmark.personaplex.parity",
                "--input",
                item["path"],
                "--frames",
                "16" if smoke else "2999",
                "--source",
                settings["source"],
                "--assets",
                settings["assets"],
                "--voices",
                settings["voices"],
                "--reference-site",
                settings.get("reference_site", ""),
                "--backend",
                backend,
                "--output",
                str(root / name),
            ]
            # The dense stage is the exact baseline the negative control is checked against; the
            # served-path stage is the one whose near-tie flips need the reference's own spread.
            if backend == "dense" or smoke:
                command.append("--negative-control")
            if backend == "resources":
                command.append("--reference-control")
            scenarios.append((name, command, 1200 if smoke else 1800, 120 if smoke else 240))
    name = "batch-resources"
    scenarios.append(
        (
            name,
            [
                sys.executable,
                "-m",
                "benchmark.personaplex.batch",
                "--inputs",
                *[item["path"] for item in parity_entries],
                "--assets",
                settings["assets"],
                "--voices",
                settings["voices"],
                "--voice-names",
                *VOICES,
                "--frames",
                "16" if smoke else "2999",
                "--batch-sizes",
                *(["2"] if smoke else ["2", "4"]),
                "--output",
                str(root / name),
            ],
            900 if smoke else 1800,
            120 if smoke else 240,
        )
    )
    if settings.get("asr_model"):
        scenarios.append(
            (
                "judge",
                [
                    sys.executable,
                    "-m",
                    "benchmark.personaplex.judge",
                    "--asr-model",
                    settings["asr_model"],
                    "--output",
                    str(root / "judge"),
                    *[str(root / f"parity-{item['id']}-resources") for item in parity_entries],
                ],
                900,
                120,
            )
        )
    for backend in ["rust"] if smoke else ["python", "rust"]:
        name = "lifecycle-" + backend
        scenarios.append(
            (
                name,
                [
                    sys.executable,
                    "-m",
                    "benchmark.personaplex.serving",
                    "--settings",
                    str(root / "settings.json"),
                    "--output",
                    str(root / name),
                    "--manifest",
                    str(root / "lifecycle-inputs.json"),
                    "--backend",
                    backend,
                    "--mode",
                    "smoke" if smoke else "lifecycle",
                    # One request a step, so concurrent outputs must equal isolated ones exactly.
                    "--max-batch-size",
                    "1",
                ],
                1800,
                180 if smoke else 360,
            )
        )
    if not smoke:
        name = "soak-rust"
        scenarios.append(
            (
                name,
                [
                    sys.executable,
                    "-m",
                    "benchmark.personaplex.serving",
                    "--settings",
                    str(root / "settings.json"),
                    "--output",
                    str(root / name),
                    "--manifest",
                    str(root / "lifecycle-inputs.json"),
                    "--backend",
                    "rust",
                    "--mode",
                    "soak",
                ],
                2100,
                1500,
            )
        )
    name = "warm-rust"
    scenarios.append(
        (
            name,
            [
                sys.executable,
                "-m",
                "benchmark.personaplex.warm",
                "--settings",
                str(root / "settings.json"),
                "--manifest",
                str(root / ("smoke-inputs.json" if smoke else "inputs.json")),
                "--output",
                str(root / name),
                *(["--smoke"] if smoke else []),
            ],
            2400,
            90 if smoke else 300,
        )
    )
    result_path = root / "results.json"
    results = json.loads(result_path.read_text()) if args.mode == "resume" and result_path.exists() else {}
    for name, *_ in scenarios:
        results.setdefault(name, {"status": "NOT RUN"})
    results.update(missing_clip_results(entries, smoke))
    results.update(judge_blocked(settings))
    write_json(result_path, results)
    for name, command, timeout, minimum in scenarios:
        if reusable(results[name], command, args.retry_failed):
            continue
        if name in ("soak-rust", "warm-rust") and results.get("lifecycle-rust", {}).get("status") != "PASS":
            results[name] = {"status": "NOT RUN", "reason": "Rust lifecycle failed or incomplete; no escalation"}
        elif INTERRUPTED or time.time() + minimum >= deadline:
            results[name] = {"status": "NOT RUN", "reason": "Deadline reserve; insufficient time to launch"}
        else:
            target = root / name
            if target.exists():
                target.rename(root / f"{name}-previous-{time.time_ns()}")
            result = subprocess_case(command, target, root / (name + ".log"), deadline, timeout, env)
            results[name] = result
            if name.startswith("parity-") and (target / "pipeline.npz").exists():
                from benchmark.personaplex.compare import audit

                comparison = audit(target)
                write_json(target / "comparison.json", comparison)
                if comparison["status"] != "PASS":
                    results[name]["status"] = "FAIL"
        write_json(result_path, results)
        report(root)
    return report(root)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())

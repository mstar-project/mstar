"""Resumable validation orchestration. Performance numbers only come from benchmark.runner."""

import argparse
import base64
import fcntl
import hashlib
import html
import importlib.metadata
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
    for key in ("source", "assets", "voices", "cache", "extension", "server_config", "existing_clip"):
        if not Path(settings[key]).exists():
            raise FileNotFoundError(f"Missing {key}: {settings[key]}")
    for name in ("model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors", "tokenizer_spm_32k_3.model"):
        if not (Path(settings["assets"]) / name).is_file():
            raise FileNotFoundError(name)
    if not (Path(settings["voices"]) / "NATF2.pt").is_file():
        raise FileNotFoundError("NATF2.pt")
    if not any(Path(settings["extension"]).glob("mstar_rust*.so")):
        raise FileNotFoundError("Private built Rust extension")
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
        "extension_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(settings["extension"]).glob("mstar_rust*.so"))
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


def subprocess_case(command, output, logpath, deadline, timeout, env):
    remaining = min(timeout, deadline - time.time())
    if remaining <= 0:
        return {"status": "NOT RUN", "reason": "Deadline reserve"}
    logpath.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
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
    for name, row in results.items():
        lines += [f"{name}: {row['status']}", f"  {row.get('reason', row.get('log', ''))}"]
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
        "Deferred: second voice, injected worker errors, supported NVIDIA runtime, accepted",
        "near-limit requests, broad quality claims, batching, live ingress and streamed audio.",
        "A completed comparison does not approve numerical divergence or establish quality.",
    ]
    (root / "REPORT.rst").write_text("\n".join(lines) + "\n")
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
    smoke = args.mode == "smoke"
    # Slurm submission allocates <=20min smoke and <=160min full. Reserve15min in full.
    allocation_start = float(os.environ["PERSONAPLEX_ALLOCATION_START"])
    allocation_seconds = int(os.environ["PERSONAPLEX_ALLOCATION_SECONDS"])
    if allocation_seconds > (1200 if smoke else 9600):
        raise ValueError("Allocation exceeds the agreed smoke/full budget")
    budget_path = Path(settings["budget_file"])
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    with budget_path.open("a+") as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX)
        ledger.seek(0)
        content = ledger.read()
        budget = json.loads(content) if content else {}
        job = os.environ["SLURM_JOB_ID"]
        if job not in budget:
            if sum(budget.values()) + allocation_seconds > 10800:
                raise ValueError("Three GPU-hour reservation cap reached; no new work launched")
            budget[job] = allocation_seconds
            ledger.seek(0)
            ledger.truncate()
            ledger.write(json.dumps(budget, indent=2))
            ledger.flush()
    deadline = allocation_start + allocation_seconds - (120 if smoke else 900)
    env = dict(os.environ, MSTAR_SHM_ARENA="1", PYTHONUNBUFFERED="1", PERSONAPLEX_DEADLINE=str(deadline))
    env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd()), env.get("PYTHONPATH", "")])
    write_json(
        root / "hardware.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
            "allocation_start": allocation_start,
            "allocation_seconds": allocation_seconds,
            "deadline": deadline,
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
            scenarios.append((name, command, 1200, 180))
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
                ],
                1800,
                360,
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
            300,
        )
    )
    result_path = root / "results.json"
    results = json.loads(result_path.read_text()) if args.mode == "resume" and result_path.exists() else {}
    if args.mode != "resume" and result_path.exists():
        raise ValueError("Existing results: use resume or a fresh output directory")
    for name, *_ in scenarios:
        results.setdefault(name, {"status": "NOT RUN"})
    if not smoke and not any(e["id"] == "english" for e in entries):
        for name in (
            "parity-english-dense",
            "parity-english-resources",
            "lifecycle-english",
            "soak-english",
            "warm-english-c1",
            "warm-english-c2",
            "warm-english-c4",
        ):
            results[name] = {
                "status": "BLOCKED",
                "reason": "User English recording not supplied; no corpus substituted",
            }
    write_json(result_path, results)
    for name, command, timeout, minimum in scenarios:
        prior = results[name]["status"]
        if prior in ("PASS", "FAIL") and not (prior == "FAIL" and args.retry_failed):
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

#!/usr/bin/env python3
"""HTTP end-to-end runner for M*. Needs a CUDA GPU; see test/e2e/README.md.

    python test/e2e/run.py --list            # what this machine can run
    python test/e2e/run.py --tier smoke      # one fast case per family, eager
    python test/e2e/run.py                   # every case that fits
    python test/e2e/run.py --models bagel --compare-source ../mstar-main
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cases as case_registry  # noqa: E402
import requests  # noqa: E402
import yaml  # noqa: E402
from cases import MODELS, SENTENCE, Case, ModelSpec  # noqa: E402
from matrix import (  # noqa: E402
    VRAM_HEADROOM,
    VramSampler,
    cases_without_inputs,
    config_gpus,
    gap_report,
    gpu_inventory,
    plan,
    wait_for_gpu_release,
)
from server import Server, ServerError  # noqa: E402
from validators import (  # noqa: E402
    Expect,
    Verdict,
    check_chat_response,
    check_frame_stream,
    check_generate_response,
    check_speech_response,
    check_transcription_response,
    differ,
)

REPO = HERE.parents[1]
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED")
# Inductor's autotuner logs a candidate kernel that needs more shared memory than
# the GPU has as "OutOfMemoryError: out of resource: ..." and skips it.
AUTOTUNE_REJECT = "out of resource"


def ran_out_of_memory(log: str) -> bool:
    return any(m in line and AUTOTUNE_REJECT not in line for line in log.splitlines() for m in OOM_MARKERS)


def git_info(path: Path) -> dict:
    def git(*args):
        out = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    return dict(path=str(path), commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")))


def environment(gpus) -> dict:
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=False).stdout.strip().splitlines()[:1]
    except FileNotFoundError:
        driver = []
    versions = {}
    for module in ("torch", "transformers", "numpy", "av"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001
            versions[module] = None
    return dict(host=platform.node(), python=platform.python_version(), driver=driver,
                gpus=[f"{g.index}: {g.name} {g.total_gb:.0f} GB" for g in gpus], versions=versions)


def _http_error(r: requests.Response) -> Verdict:
    return Verdict(False, f"HTTP {r.status_code}: {r.text[:300]}")


def run_case(url: str, case: Case) -> Verdict:
    payload = case.build()
    if case.route == "chat":
        r = requests.post(url + "/v1/chat/completions", json=payload, timeout=900)
        return check_chat_response(r.content, case.expect, bool(payload.get("stream"))) if r.ok else _http_error(r)
    if case.route == "speech":
        r = requests.post(url + "/v1/audio/speech", json=payload, timeout=900)
        return check_speech_response(r.content, case.expect) if r.ok else _http_error(r)
    files = payload.pop("files", None)
    if case.route == "transcribe":
        r = requests.post(url + "/v1/audio/transcriptions", data=payload, files={"file": files}, timeout=900)
        return check_transcription_response(r.content, case.expect) if r.ok else _http_error(r)
    r = requests.post(url + "/generate", data=payload, files={"files": files} if files else None, timeout=900)
    if not r.ok:
        return _http_error(r)
    if case.route == "generate_stream":
        return check_frame_stream(r.content, case.expect)
    return check_generate_response(r.content, case.expect)


def _print_case(model, name, verdict):
    print(f"  {model}: {name:45} {'PASS' if verdict.passed else 'FAIL'}  {verdict.detail}", flush=True)
    for w in verdict.warnings:
        print(f"  {model}: {'':45} WARN  {w}", flush=True)


def server_env(args) -> dict[str, str]:
    """Environment for every server, including the Whisper pass."""
    env = {"PYTHONHASHSEED": "0"}
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
    if args.eager:
        env["TORCH_COMPILE_DISABLE"] = "1"
    return env


def run_model(spec: ModelSpec, cases: tuple[Case, ...], source: Path, config: Path, out: Path, args, gpus,
              result: dict) -> None:
    """Fill ``result`` as cases complete, so an interrupted run keeps them."""
    env = server_env(args)
    result.update(config=str(config), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                  model_kwargs=(yaml.safe_load(config.read_text()) or {}).get("model_kwargs", {}), cases={})
    log = out / "server.log"
    server = Server(source, config, log, port=args.port, env=env, startup_timeout=args.startup_timeout,
                    tensor_comm_protocol=args.tensor_comm_protocol)
    sampler = VramSampler([g.index for g in gpus[:config_gpus(config)]])
    try:
        server.choose_port()  # before the sampler's thread starts forking nvidia-smi
        with sampler, server as url:
            result["startup_seconds"] = round(server.startup_seconds, 1)
            for case in cases:
                started = time.monotonic()
                try:
                    verdict = run_case(url, case)
                except Exception as exc:  # noqa: BLE001
                    verdict = Verdict(False, f"{type(exc).__name__}: {exc}"[:300])
                result["cases"][case.name] = dict(_verdict_dict(verdict), seconds=round(time.monotonic() - started, 2))
                if verdict.artifact:
                    suffix, data = verdict.artifact
                    path = out / f"{case.name}{suffix}"
                    path.write_bytes(data)
                    result["cases"][case.name]["artifact"] = str(path)
                _print_case(spec.name, case.name, verdict)
                if server.poll() is not None:
                    raise ServerError(f"server exited with code {server.exit_code} during {case.name}; see {log}")
            for a, b in spec.must_differ:
                if a not in result["cases"] or b not in result["cases"]:
                    continue
                va, vb = (Verdict(**{k: v for k, v in result["cases"][n].items() if k != "seconds"}) for n in (a, b))
                ok = va.passed and vb.passed and differ(va, vb)
                detail = "give different outputs" if ok else "give the same output, or one failed"
                verdict = Verdict(ok, f"{a} and {b} {detail}")
                result["cases"][f"{a}__differs_from__{b}"] = dict(_verdict_dict(verdict), seconds=0)
                _print_case(spec.name, f"{a} != {b}", verdict)
    except (ServerError, KeyboardInterrupt) as exc:
        result["server_error"] = str(exc) or type(exc).__name__
        print(f"  {spec.name}: SERVER FAIL {result['server_error']}", flush=True)
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        _server_facts(server, result)
        result["peak_vram_gb"] = round(sampler.peak, 1)
        _await_release(spec.name, sampler, result)
    text = log.read_text(errors="replace") if log.exists() else ""
    if ran_out_of_memory(text):
        result["out_of_memory"] = True
    if spec.peak_vram_gb and sampler.peak > spec.peak_vram_gb * VRAM_HEADROOM:
        result["vram_warning"] = f"peak {sampler.peak:.1f} GB exceeds the recorded {spec.peak_vram_gb} GB"


def _server_facts(server: Server, result: dict) -> None:
    if server.exit_code is not None:
        result["server_exit_code"] = server.exit_code
    if server.restarts:
        result["server_restarts"] = server.restarts  # port taken while the model loaded
    if server.survivors:
        result["server_survivors"] = server.survivors  # pids alive after SIGKILL


def _await_release(name: str, sampler: VramSampler, result: dict) -> None:
    """Wait for this server's GPU memory to be freed before the next one starts."""
    residual = wait_for_gpu_release(sampler.indices, sampler.baseline)
    if residual > 1.0:
        result["vram_not_released_gb"] = round(residual, 1)
        print(f"  {name}: WARN  {residual:.1f} GB still in use two minutes after the server exited", flush=True)


SPEECH_TRANSCRIPT = Expect("text", transcript=SENTENCE, min_word_recall=0.75)


def transcribe_speech(report: dict, args, sources: dict, out: Path, gpus) -> None:
    """Transcribe every speech output with one Whisper server; each must say the requested sentence."""
    clips = [(name, label, case_name, case["artifact"])
             for name, entry in report["models"].items()
             for label in ("source", "compare") if label in entry
             for case_name, case in entry[label]["cases"].items()
             if case.get("passed") and str(case.get("artifact", "")).endswith(".wav")]
    if not clips:
        return
    spec = MODELS["whisper_large"]
    config = sources["source"] / spec.config
    reason = plan(spec, config, gpus, args.offline)
    if reason:
        report["speech_round_trip"] = dict(skipped=reason)
        print(f"speech round trip skipped: {reason}", flush=True)
        return
    print(f"speech round trip: transcribing {len(clips)} clip(s) with {spec.name}", flush=True)
    server = Server(sources["source"], config, out / "speech_round_trip" / "server.log", port=args.port,
                    env=server_env(args),
                    startup_timeout=args.startup_timeout, tensor_comm_protocol=args.tensor_comm_protocol)
    sampler = VramSampler([g.index for g in gpus[:config_gpus(config)]])
    result = report["speech_round_trip"] = dict(config=str(config), clips=len(clips))
    try:
        server.choose_port()  # before the sampler's thread starts forking nvidia-smi
        with sampler, server as url:
            for name, label, case_name, path in clips:
                try:
                    r = requests.post(url + "/v1/audio/transcriptions", timeout=900,
                                      data=dict(language="en", response_format="json", temperature="0"),
                                      files={"file": (Path(path).name, Path(path).read_bytes())})
                    verdict = check_transcription_response(r.content, SPEECH_TRANSCRIPT) if r.ok else _http_error(r)
                except Exception as exc:  # noqa: BLE001
                    verdict = Verdict(False, f"{type(exc).__name__}: {exc}"[:300])
                entry = report["models"][name]
                entry[label]["cases"][f"{case_name}__says_sentence"] = dict(_verdict_dict(verdict), seconds=0)
                entry["status"] = model_status(entry)
                _print_case(name, f"{case_name} says the sentence", verdict)
                if server.poll() is not None:
                    raise ServerError(f"server exited with code {server.exit_code}; see {server.log_path}")
    except ServerError as exc:
        result["server_error"] = str(exc)
        print(f"speech round trip: SERVER FAIL {exc}", flush=True)
    finally:
        _server_facts(server, result)
        _await_release(spec.name, sampler, result)


def _verdict_dict(verdict: Verdict) -> dict:
    return dict(passed=verdict.passed, detail=verdict.detail, sha256=verdict.sha256, warnings=list(verdict.warnings))


def model_status(entry: dict) -> str:
    if "skipped" in entry:
        return "skipped"
    runs = [entry[k] for k in ("source", "compare") if k in entry]
    if any(key in r for r in runs for key in ("server_error", "out_of_memory", "server_exit_code")):
        return "failed"
    for name, case in entry["source"]["cases"].items():
        other = entry.get("compare", {}).get("cases", {}).get(name, {"passed": True})
        if not case["passed"] or not other["passed"] or case.get("identical") is False:
            return "failed"
    return "passed"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS), metavar="MODEL",
                   help=f"default: all ({', '.join(MODELS)})")
    p.add_argument("--tier", choices=["smoke", "full"], default="full",
                   help="smoke: only cases marked smoke, eager servers; full: every case, configs as written")
    p.add_argument("--list", action="store_true", help="print the coverage matrix for this machine and exit")
    p.add_argument("--source", type=Path, default=REPO, help="checkout to serve (default: this one)")
    p.add_argument("--compare-source", type=Path, help="second checkout; decoded outputs must be identical")
    p.add_argument("--config", action="append", default=[], metavar="MODEL=PATH", help="override a model's config")
    p.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "mstar-e2e",
                   help="report and server logs (default: %(default)s)")
    p.add_argument("--port", type=int,
                   help="fixed server port; a profile fails if another process holds it (default: a free port "
                        "picked at launch, retried once if taken while the model loads)")
    p.add_argument("--offline", action="store_true",
                   help="use only the local Hugging Face cache, for weights and fixtures (HF_HUB_OFFLINE=1)")
    p.add_argument("--eager", action="store_true", help="TORCH_COMPILE_DISABLE=1 for every server (implied by smoke)")
    p.add_argument("--tensor-comm-protocol", default="SHM", help="server tensor transport (default: SHM, single node)")
    p.add_argument("--startup-timeout", type=float, default=1800)
    p.add_argument("--no-transcribe", dest="transcribe", action="store_false",
                   help="skip transcribing speech outputs with Whisper after the run")
    args = p.parse_args()
    args.eager = args.eager or args.tier == "smoke"
    if args.offline:
        case_registry.OFFLINE = True
        os.environ["HF_HUB_OFFLINE"] = "1"

    overrides = dict(item.split("=", 1) for item in args.config)
    unknown = set(overrides) - set(MODELS)
    if unknown:
        p.error(f"--config for unknown model(s): {sorted(unknown)}")
    sources = {"source": args.source.resolve()}
    if args.compare_source:
        sources["compare"] = args.compare_source.resolve()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = (args.out / f"{stamp}-{os.getpid()}").resolve()
    report = dict(started=stamp, argv=sys.argv, tier=args.tier, models={})
    install_interrupt_handlers()
    try:
        _execute(args, overrides, sources, out, report)
    except KeyboardInterrupt:
        report["interrupted"] = True
    except Exception as exc:  # noqa: BLE001
        report["error"] = traceback.format_exc()
        print(f"\nrunner error: {type(exc).__name__}: {exc}", flush=True)
    finally:
        for entry in report["models"].values():
            entry.setdefault("status", "skipped" if "skipped" in entry else "not run")
        if not args.list or "error" in report:
            out.mkdir(parents=True, exist_ok=True)
            (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    if "gaps" in report:
        if not args.list:
            _print_summary(report)
        _print_gaps(report["gaps"])
    if args.list and "error" not in report:
        return 0
    failed = [n for n, e in report["models"].items() if e["status"] == "failed"]
    not_run = [n for n, e in report["models"].items() if e["status"] in ("not run", "interrupted")]
    print(f"\n{'no failures' if not failed else 'failed: ' + ', '.join(failed)}")
    if not_run:
        print(f"not run{' (interrupted)' if report.get('interrupted') else ''}: {', '.join(not_run)}")
    round_trip = report.get("speech_round_trip", {})
    if "server_error" in round_trip or "server_exit_code" in round_trip:
        reason = round_trip.get("server_error") or f"server exited with code {round_trip['server_exit_code']}"
        print(f"speech round trip failed: {reason}")
    if not any(e["status"] in ("passed", "failed") for e in report["models"].values()):
        print("no case ran")
    print(f"report: {out / 'report.json'}")
    return exit_code(report)


def exit_code(report: dict) -> int:
    """Non-zero unless something ran and nothing failed."""
    statuses = [e["status"] for e in report["models"].values()]
    if "error" in report or report.get("interrupted"):
        return 1
    if any(s in ("failed", "not run") for s in statuses):
        return 1
    if {"server_error", "server_exit_code"} & report.get("speech_round_trip", {}).keys():
        return 1
    return 0 if any(s in ("passed", "failed") for s in statuses) else 1


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def install_interrupt_handlers() -> None:
    """SIGTERM and SIGHUP stop the run like Ctrl-C, unless inherited ignored (nohup)."""
    for sig in (signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(sig) != signal.SIG_IGN:
            signal.signal(sig, _interrupt)


def _execute(args, overrides: dict, sources: dict, out: Path, report: dict) -> None:
    gpus = gpu_inventory()
    report.update(environment=environment(gpus), sources={k: git_info(v) for k, v in sources.items()},
                  gaps=gap_report(sources["source"]))
    selected = []
    for name in args.models:
        spec = MODELS[name]
        cases = tuple(c for c in spec.cases if args.tier == "full" or c.smoke)
        if not cases:
            report["models"][name] = dict(family=spec.family, skipped=f"no case in the {args.tier} tier")
            print(f"  {name:24} {spec.family:9} skip: no case in the {args.tier} tier", flush=True)
            continue
        config = Path(overrides.get(name, sources["source"] / spec.config)).resolve()
        if not config.is_file():
            raise FileNotFoundError(f"{name}: config {config} does not exist")
        notes = []
        reason = plan(spec, config, gpus, args.offline, notes)
        report["models"][name] = dict(family=spec.family, config=str(config), gpus=config_gpus(config),
                                      recorded_peak_vram_gb=spec.peak_vram_gb)
        if notes and not reason:
            report["models"][name]["warnings"] = notes
            for note in notes:
                print(f"  {name:24} {spec.family:9} WARN  {note}", flush=True)
        unbuildable = cases_without_inputs(cases)
        if unbuildable:
            report["models"][name]["skipped_cases"] = unbuildable
            cases = tuple(c for c in cases if c.name not in unbuildable)
            if not cases:
                reason = reason or "; ".join(dict.fromkeys(unbuildable.values()))
            elif not reason:
                for case_name, why in unbuildable.items():
                    print(f"  {name:24} {spec.family:9} skip case {case_name}: {why}", flush=True)
        if reason:
            report["models"][name]["skipped"] = reason
        else:
            selected.append((spec, cases, config))
        print(f"  {name:24} {spec.family:9} {'run' if not reason else 'skip: ' + reason}", flush=True)
    if args.list:
        return
    for spec, cases, config in selected:
        print(spec.name, flush=True)
        entry = report["models"][spec.name]
        for label, src in sources.items():
            cfg = config if label == "source" else Path(overrides.get(spec.name, src / spec.config)).resolve()
            entry[label] = {}
            try:
                run_model(spec, cases, src, cfg, out / spec.name / label, args, gpus, entry[label])
            except KeyboardInterrupt:
                entry["status"] = "interrupted"
                raise
        if "compare" in entry:
            for name, case in entry["source"]["cases"].items():
                other = entry["compare"]["cases"].get(name, {})
                if case.get("sha256") or other.get("sha256"):
                    case["identical"] = case.get("sha256") == other.get("sha256")
        entry["status"] = model_status(entry)
    if args.transcribe:
        transcribe_speech(report, args, sources, out, gpus)


def _print_summary(report: dict) -> None:
    print("\nSummary")
    for name, entry in report["models"].items():
        extra = entry.get("skipped") or ""
        if "source" in entry:
            run = entry["source"]
            extra = f"peak {run.get('peak_vram_gb')} GB" + (", OUT OF MEMORY" if run.get("out_of_memory") else "") \
                + (f", {run['server_error']}" if "server_error" in run else "")
        print(f"  {entry['status'].upper():8} {name:24} {extra}")
        for case_name, case in entry.get("source", {}).get("cases", {}).items():
            same = "" if "identical" not in case else ("  identical" if case["identical"] else "  DIFFERS")
            print(f"      {'ok ' if case['passed'] else 'BAD'} {case_name:45} {case['detail']}{same}")


def _print_gaps(gaps: dict):
    print("\nNo case yet")
    for config, n in gaps["configs_without_case"].items():
        print(f"  {config:45} needs {n} GPU{'s' if n > 1 else ''}")
    if gaps["models_without_config"]:
        print(f"  registered models with no config: {', '.join(gaps['models_without_config'])}")


if __name__ == "__main__":
    sys.exit(main())

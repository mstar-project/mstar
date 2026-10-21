"""Launch repository benchmark.runner cells against an uninstrumented Rust server."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from benchmark.personaplex.dataset import read_manifest
from benchmark.personaplex.serving import Server, interrupted, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    entries = read_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    cells = []
    for entry in entries:
        for concurrency in [1] if args.smoke else [1, 2, 4]:
            cells.append({"clip": entry["id"], "concurrency": concurrency, "status": "NOT RUN"})
    write_json(args.output / "cells.json", cells)
    with Server(settings, args.output, "rust", False) as server:
        for entry in entries:
            for concurrency in [1] if args.smoke else [1, 2, 4]:
                cell = next(c for c in cells if c["clip"] == entry["id"] and c["concurrency"] == concurrency)
                if time.time() + 420 > float(os.environ.get("PERSONAPLEX_DEADLINE", "inf")):
                    cell["reason"] = "Deadline reserve; cell not launched"
                    continue
                root = args.output / f"{entry['id']}-c{concurrency}"
                manifest = args.output / f"{entry['id']}.json"
                write_json(manifest, [entry])
                cell["attempts"] = []
                for attempt in range(2):
                    target = root.with_name(root.name + f"-attempt{attempt}")
                    target.mkdir()
                    command = [
                        sys.executable,
                        "-m",
                        "benchmark.runner",
                        "--model",
                        "personaplex",
                        "--url",
                        server.url,
                        "--inference-system",
                        "ours",
                        "--dataset",
                        "local_audio",
                        "--audio-manifest",
                        str(manifest),
                        "--request-type",
                        "audio_to_speech",
                        "--profiling-type",
                        "closed_loop",
                        "--max-concurrency",
                        str(concurrency),
                        "--num-warmup",
                        "1" if args.smoke else "3",
                        "--num-requests",
                        "1" if args.smoke else "10",
                        "--verbose",
                        "--output-dir",
                        str(target),
                    ]
                    logpath = target / "runner.log"
                    measured_server_offset = None
                    with logpath.open("wb") as log:
                        process = subprocess.Popen(
                            command,
                            env=dict(os.environ, MSTAR_SHM_ARENA="1", PYTHONUNBUFFERED="1"),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        deadline = min(time.time() + 900, float(os.environ.get("PERSONAPLEX_DEADLINE", "inf")))
                        while process.poll() is None and time.time() < deadline:
                            if measured_server_offset is None and "Warmup complete." in logpath.read_text(
                                errors="replace"
                            ):
                                measured_server_offset = (args.output / "server.log").stat().st_size
                            time.sleep(0.2)
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=15)
                    server_bytes = (args.output / "server.log").read_bytes()
                    measured_log = server_bytes[measured_server_offset:] if measured_server_offset is not None else b""
                    compiled = bool(
                        re.search(rb"nvcc |build\.ninja|JIT compil|Building extension|Compiling", measured_log, re.I)
                    )
                    result_path = target / "results.json"
                    result = json.loads(result_path.read_text()) if result_path.exists() else {}
                    expected = 1 if args.smoke else 10
                    valid = (
                        measured_server_offset is not None
                        and process.returncode == 0
                        and result.get("completed") == expected
                        and result.get("failed") == 0
                        and result.get("aggregate", {}).get("rtf") is not None
                        and abs(result.get("aggregate", {}).get("audio_duration_mean_s", 0) - entry["duration_s"])
                        <= 1 / 24000
                    )
                    record = {
                        "command": command,
                        "MSTAR_SHM_ARENA": "1",
                        "exit": process.returncode,
                        "results": str(result_path),
                        "compile_log_detected_during_measurement": compiled,
                        "measurement_boundary_observed": measured_server_offset is not None,
                        "compilation_detection_scope": "Server build-log markers; not an exhaustive JIT tracer",
                    }
                    cell["attempts"].append(record)
                    cell["status"] = "PASS" if valid and not compiled else "FAIL"
                    write_json(args.output / "cells.json", cells)
                    if not compiled:
                        break
                    if time.time() + 420 > float(os.environ.get("PERSONAPLEX_DEADLINE", "inf")):
                        break
                if cell["status"] != "PASS":
                    break  # Do not escalate concurrency after an invalid lower cell.
    status = "PASS" if all(c["status"] == "PASS" for c in cells) else "FAIL"
    write_json(
        args.output / "summary.json",
        {
            "status": status,
            "cells": cells,
            "scope": "Buffered audio latency/RTF from benchmark.runner; concurrency is B=1 requests.",
        },
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())

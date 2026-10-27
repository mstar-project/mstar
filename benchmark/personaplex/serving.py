"""Bounded actual-worker lifecycle/soak checks, deliberately not benchmark timings.

Outputs are compared byte for byte with each input's first (isolated) response
only when the server runs every LM step alone (``--max-batch-size 1``): a
batched step changes floating-point rounding, so there agreement is recorded,
not asserted. Batched correctness is checked offline by ``batch.py``.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

from benchmark.personaplex.dataset import read_manifest

ACTIVE_SERVER = None


def interrupted(*_):
    if ACTIVE_SERVER is not None:
        ACTIVE_SERVER.close()
    raise SystemExit(143)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


AGENT_FRAME_DIMS = [1, 8, 1]  # one frame of the 8 agent codebooks


def holds_agent_frame(tensor):
    """The observer usually cannot name a tensor, so an agent frame is also recognised by shape."""
    return tensor.get("name") == "agent_frame" or list(tensor.get("dims") or []) == AGENT_FRAME_DIMS


def verify_ownership_observations(rows, completed):
    """Each request's decoder steps end holding no agent frame, before its own hard cleanup.

    The decoder consumes a request's frames chunk by chunk, so after an earlier
    chunk the LM's newest frame can still be in flight; only the last decoder
    observation before the request's hard cleanup must hold none. Worker events
    carry the worker-local request handle, which is recycled across requests, so
    a handle's observations are grouped by the hard cleanups between them.
    """
    decoders = [r for r in rows if r["event"] == "after_decoder_inputs_consumed"]
    assert len(decoders) >= completed, f"Missing decoder observations: {len(decoders)} for {completed} completions"
    entities = {r.get("entity") for r in decoders}
    timelines = {}
    for r in rows:
        if r["event"] in ("after_decoder_inputs_consumed", "before_hard_cleanup") and r.get("entity") in entities:
            timelines.setdefault((r.get("entity"), r["rid"]), []).append(r)
    held = []
    for (entity, rid), events in timelines.items():
        last = None
        for event in sorted(events, key=lambda e: (e["time"], e["event"] == "before_hard_cleanup")):
            if event["event"] == "after_decoder_inputs_consumed":
                last = event
                continue
            if last is not None and any(holds_agent_frame(t) for t in last["tensors"]):
                held.append(rid)
            last = None
        assert last is None, f"Decoder observation for handle {rid} on {entity} has no later hard cleanup"
    assert not held, f"Decoder observations still hold agent frames for handles {held}"


def server_config(settings, root, max_batch_size):
    """The settings' server YAML, or a copy of it in ``root`` with ``model_kwargs.max_batch_size`` replaced."""
    if max_batch_size is None:
        return settings["server_config"]
    import yaml

    config = yaml.safe_load(Path(settings["server_config"]).read_text())
    config.setdefault("model_kwargs", {})["max_batch_size"] = max_batch_size
    path = root / "server-config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(path)


class Server:
    def __init__(self, settings, root, backend, observe, max_batch_size=None):
        self.settings, self.root, self.backend, self.observe = settings, root, backend, observe
        self.max_batch_size = max_batch_size
        self.port = settings.get("port", 8000)
        self.process = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        global ACTIVE_SERVER

        self.root.mkdir(parents=True, exist_ok=True)
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError("Port occupied; existing process left untouched")
        flag = "1" if self.backend == "rust" else "0"
        env = dict(
            os.environ,
            MSTAR_SHM_ARENA="1",
            MSTAR_RUST_GRAPH=flag,
            MSTAR_RUST_ZMQ=flag,
            PERSONAPLEX_OWNERSHIP_OBSERVE="1" if self.observe else "0",
            PERSONAPLEX_OWNERSHIP_ROOT=str(self.root),
            PERSONAPLEX_LIVE_BACKEND=self.backend,
        )
        paths = [str(Path(__file__).parent / "observe"), self.settings.get("extension"), str(Path.cwd()),
                 env.get("PYTHONPATH")]
        env["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
        self.log = (self.root / "server.log").open("wb")
        command = [
            str(Path(sys.executable).with_name("mstar-serve")),
            "--config",
            server_config(self.settings, self.root, self.max_batch_size),
            "--cache-dir",
            self.settings["cache"],
            "--tensor-comm-protocol",
            "SHM",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]
        write_json(
            self.root / "launch.json",
            {
                "command": command,
                "backend": self.backend,
                "max_batch_size": self.max_batch_size,
                "instrumented": self.observe,
                "MSTAR_SHM_ARENA": "1",
                "started": time.time(),
            },
        )
        self.process = subprocess.Popen(
            command, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True
        )
        ACTIVE_SERVER = self
        self.stop_monitor = threading.Event()

        def monitor():
            while not self.stop_monitor.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    with (self.root / "gpu-memory.jsonl").open("a") as output:
                        output.write(
                            json.dumps(
                                {
                                    "time": time.time(),
                                    "exit": result.returncode,
                                    "gpu_memory_total_utilization": result.stdout.strip(),
                                }
                            )
                            + "\n"
                        )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    with (self.root / "gpu-memory.jsonl").open("a") as output:
                        output.write(json.dumps({"time": time.time(), "error": repr(exc)}) + "\n")
                self.stop_monitor.wait(5)

        self.monitor = threading.Thread(target=monitor, daemon=True)
        self.monitor.start()
        try:
            deadline = time.monotonic() + 420
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Server exited {self.process.returncode}; inspect server.log")
                try:
                    with urllib.request.urlopen(self.url + "/health", timeout=2) as response:
                        if json.load(response).get("status") == "healthy":
                            return self
                except (OSError, ValueError):
                    pass
                time.sleep(2)
            raise TimeoutError("Server readiness timeout")
        except BaseException:
            self.close()
            raise

    def close(self):
        """Stop the server's whole process group, even when the launcher already exited.

        The conductor and workers are spawned children in the launcher's session
        (``start_new_session=True``), so they can outlive it and keep holding the
        GPU and the port; signalling only a live launcher would leave them behind.
        """
        if hasattr(self, "stop_monitor"):
            self.stop_monitor.set()
        if self.process:
            group = self.process.pid
            for sig, grace in ((signal.SIGTERM, 25), (signal.SIGKILL, 10)):
                if not self._group_alive(group):
                    break
                os.killpg(group, sig)
                deadline = time.monotonic() + grace
                while time.monotonic() < deadline and self._group_alive(group):
                    time.sleep(0.2)
            if self._group_alive(group):
                raise RuntimeError(f"server process group {group} survived SIGKILL")
        if hasattr(self, "log"):
            self.log.close()

    def _group_alive(self, group):
        self.process.poll()  # reap the launcher, so a zombie does not keep the group alive
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return False
        return True

    def __exit__(self, *args):
        self.close()


class Lifecycle:
    def __init__(self, server, entries, exact):
        self.server, self.entries, self.exact = server, entries, exact
        self.rows, self.controls = [], {}
        self.index = 0

    def events(self):
        rows = []
        for path in self.server.root.glob("ownership-events-*.jsonl"):
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A concurrent writer can still be finishing its last line.
                if row["entity"].startswith("worker"):
                    rows.append(row)
        return rows

    def drain(self, previous_cleanups, additional):
        end = time.monotonic() + 90
        while time.monotonic() < end:
            rows = self.events()
            cleaned = [r for r in rows if r["event"] == "after_hard_cleanup"]
            decoders = [r for r in rows if r["event"] == "after_decoder_inputs_consumed"]
            assert all(not r["tensors"] for r in cleaned), "Request-owned tensors survive teardown"
            assert all(not any(t["name"] == "agent_frame" for t in r["tensors"]) for r in decoders), (
                "Agent-frame hold survives decoder consumption"
            )
            # Only the final observation after the whole group drains must be empty.
            if len(cleaned) >= previous_cleanups + additional and cleaned[-1]["total_tensors"] == 0:
                return cleaned[-1]
            time.sleep(0.25)
        raise AssertionError("Requests did not drain, or worker tensor store did not become empty")

    def cleanup_count(self):
        return sum(r["event"] == "after_hard_cleanup" for r in self.events())

    def request(self, entry, index, with_text=True, cancel=False, options=None, expect_bad=False):
        output = self.server.root / f"response-{index}.json"
        kwargs = {"voice": "NATF2", "greedy": True, **(options or {})}
        command = [
            "curl",
            "-sS",
            "--max-time",
            "1.5" if cancel else "300",
            self.server.url + "/generate",
            "-F",
            f"files=@{entry['path']};type=audio/wav",
            "-F",
            "input_modalities=audio",
            "-F",
            "output_modalities=audio,text" if with_text else "output_modalities=audio",
            "-F",
            "streaming=false",
            "-F",
            "model_kwargs=" + json.dumps(kwargs),
            "-o",
            str(output),
            "-w",
            "%{http_code}",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=310, check=False)
        row = {
            "index": index,
            "input": entry["id"],
            "command": command,
            "exit": result.returncode,
            "http": result.stdout,
            "cancelled": cancel,
        }
        (self.server.root / f"response-{index}.stderr").write_text(result.stderr)
        if cancel:
            assert result.returncode == 28, row
        elif expect_bad:
            assert result.returncode == 0 and result.stdout.startswith("4"), row
        else:
            assert result.returncode == 0 and result.stdout == "200", row
            response = json.loads(output.read_text())
            outputs = response["outputs"]
            assert set(outputs) == ({"audio", "text"} if with_text else {"audio"}), row
            chunks = outputs["audio"]
            assert chunks and all(c["metadata"] == {"sample_rate": 24000, "num_channels": 1} for c in chunks)
            pcm = b"".join(base64.b64decode(c["data"], validate=True) for c in chunks)
            with wave.open(entry["path"], "rb") as wav:
                expected_samples = round(wav.getnframes() * 24000 / wav.getframerate())
            assert len(pcm) == expected_samples * 2 and pcm, row
            text = b"".join(base64.b64decode(c["data"], validate=True) for c in outputs.get("text", []))
            row.update(audio_sha256=hashlib.sha256(pcm).hexdigest(), text=text.decode("utf-8"), samples=len(pcm) // 2)
            key = (entry["id"], with_text)
            control = self.controls.setdefault(key, (row["audio_sha256"], row["text"]))
            row["matches_control"] = control == (row["audio_sha256"], row["text"])
            assert row["matches_control"] or not self.exact, "Concurrent/recovery output differs from isolated control"
        write_json(self.server.root / f"request-{index}.json", row)
        return row

    def group(self, entries, cancel_first=False):
        before = self.cleanup_count()
        args = [(entry, self.index + i, i == 0 and cancel_first) for i, entry in enumerate(entries)]
        self.index += len(args)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args)) as pool:
            futures = [pool.submit(self.request, entry, index, cancel=cancel) for entry, index, cancel in args]
            self.rows.extend(f.result() for f in futures)
        snapshot = self.drain(before, len(args))
        write_json(self.server.root / "lifecycle.json", {"status": "PROGRESS", "requests": self.rows})
        return snapshot

    def run(self, smoke=False):
        for entry in self.entries:
            for with_text in (False, True):
                before = self.cleanup_count()
                self.rows.append(self.request(entry, self.index, with_text=with_text))
                self.index += 1
                self.drain(before, 1)
        for concurrency in [2] if smoke else [1, 2, 4]:
            self.group([self.entries[i % len(self.entries)] for i in range(concurrency)])
            self.group([self.entries[-1]], cancel_first=True)
            self.group([self.entries[0]])
        if not smoke:
            invalid = self.server.root / "malformed.wav"
            invalid.write_bytes(b"this is not audio")
            empty = self.server.root / "empty.wav"
            over = self.server.root / "over-budget.wav"
            for path, samples in ((empty, 0), (over, 3000 * 1920)):
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(bytes(samples * 2))
            cases = [
                (invalid, None),
                (empty, None),
                (over, None),
                (Path(self.entries[0]["path"]), {"voice": "missing-validation-voice"}),
                (Path(self.entries[0]["path"]), {"greedy": "invalid"}),
            ]
            for path, options in cases:
                entry = {"id": path.stem, "path": str(path)}
                self.rows.append(self.request(entry, self.index, options=options, expect_bad=True))
                self.index += 1
                self.group([self.entries[0]])  # Valid request must recover after each rejection.

    def soak(self, seconds):
        snapshots = []
        begin = time.monotonic()
        for epoch in range(4):
            end = begin + (epoch + 1) * seconds / 4
            wave_index = 0
            while time.monotonic() < end:
                self.group([self.entries[i % len(self.entries)] for i in range(4)], cancel_first=wave_index % 5 == 4)
                wave_index += 1
            snapshots.append(self.group([self.entries[0]]))
            write_json(self.server.root / "memory.json", snapshots)
        allocated = [s["cuda_allocated"] for s in snapshots]
        # No arbitrary allowance: a continuing positive idle trend needs investigation.
        if allocated[-1] > allocated[0]:
            raise AssertionError(f"Unexplained continuing idle allocation growth: {allocated}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=["python", "rust"], default="rust")
    parser.add_argument("--mode", choices=["smoke", "lifecycle", "soak"], default="lifecycle")
    parser.add_argument("--max-batch-size", type=int, help="override the YAML's model_kwargs.max_batch_size")
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    exact = args.max_batch_size == 1
    outcome = {"status": "FAIL", "mode": args.mode, "backend": args.backend, "max_batch_size": args.max_batch_size,
               "outputs_compared_exactly": exact}
    try:
        with Server(settings, args.output, args.backend, True, args.max_batch_size) as server:
            run = Lifecycle(server, read_manifest(args.manifest), exact)
            run.run(smoke=args.mode != "lifecycle")
            if args.mode == "soak":
                run.soak(1200)
            verify_ownership_observations(run.events(), sum("samples" in row for row in run.rows))
            compared = [row for row in run.rows if "matches_control" in row]
            outcome.update(status="PASS", requests=run.rows,
                           matching_control=sum(row["matches_control"] for row in compared), compared=len(compared))
    except Exception as exc:
        outcome["error"] = repr(exc)
        raise
    finally:
        write_json(args.output / "summary.json", outcome)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    main()

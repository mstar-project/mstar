"""Shared pieces of the ASR real-weight tests: the benchmark utterances, a
served model, and the greedy parity check against a reference.

The served comparison runs the whole M* pipeline (API server, conductor,
engine, captured graphs) through ``mstar serve`` with the model's shipped
config, so a regression anywhere in it changes the transcript.
"""

import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

import pytest

DATA = Path(os.environ.get("MSTAR_ASR_DATA", "data/asr"))
# a cold boot compiles and captures every decode bucket before /health answers
SERVE_TIMEOUT_S = 1800
REQUEST_TIMEOUT_S = 600


def utterances(n: int, max_seconds: float | None = None) -> list[Path]:
    """The first ``n`` files of the benchmark set (``python -m benchmark.asr_data``),
    in its order, skipping any longer than ``max_seconds``."""
    refs = DATA / "test_clean_200" / "references.tsv"
    if not refs.exists():
        pytest.skip(f"benchmark set not found under {DATA}")
    out = []
    for line in refs.read_text().splitlines():
        if not line.strip():
            continue
        path = DATA / "test_clean_200" / f"{line.split(chr(9), 1)[0]}.wav"
        if max_seconds is not None:
            with wave.open(str(path)) as wf:
                if wf.getnframes() / wf.getframerate() > max_seconds:
                    continue
        out.append(path)
        if len(out) == n:
            break
    return out


@contextlib.contextmanager
def serving(key: str):
    """``mstar serve <key>`` with its shipped config on a free local port; yields
    the base URL. The server runs in its own process group, which is stopped on
    exit whether the test passed or not."""
    from mstar.utils.socket_prefix import per_launch_socket_prefix

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    prefix = per_launch_socket_prefix()
    fd, log_path = tempfile.mkstemp(prefix=f"mstar-{key}-", suffix=".log")
    env = dict(os.environ)
    env.pop("MSTAR_API_KEY", None)  # the requests below send no key
    with os.fdopen(fd, "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "mstar.cli.main", "serve", key, "--host", "127.0.0.1",
             "--port", str(port), "--socket-path-prefix", prefix],
            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
    try:
        base = f"http://127.0.0.1:{port}"
        _wait_healthy(base, proc, log_path)
        yield base
    finally:
        _stop(proc)
        shutil.rmtree(prefix, ignore_errors=True)


def _wait_healthy(base: str, proc: subprocess.Popen, log_path: str) -> None:
    deadline = time.monotonic() + SERVE_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"mstar serve exited with {proc.returncode}; log {log_path}:\n{_tail(log_path)}")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    pytest.fail(f"mstar serve not healthy after {SERVE_TIMEOUT_S} s; log {log_path}:\n{_tail(log_path)}")


def _stop(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=60)
    # the conductor and workers share the group: kill whatever outlived the server
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=30)


def _tail(path: str, lines: int = 40) -> str:
    return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])


def transcribe(base: str, path: Path, language: str) -> str:
    """One ``/v1/audio/transcriptions`` request: greedy (temperature 0), forced
    language, plain ``json`` response; returns its ``text``."""
    boundary = uuid.uuid4().hex
    fields = {"language": language, "temperature": "0", "response_format": "json"}
    body = b"".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, v in fields.items()
    )
    body += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{base}/v1/audio/transcriptions", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return json.loads(resp.read())["text"]
    except urllib.error.HTTPError as exc:
        pytest.fail(f"{path.name}: HTTP {exc.code}: {exc.read().decode(errors='replace')}")


def _word_distance(a: str, b: str) -> int:
    x, y = a.split(), b.split()
    row = list(range(len(y) + 1))
    for i, wx in enumerate(x, 1):
        prev, row[0] = row[0], i
        for j, wy in enumerate(y, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (wx != wy))
    return row[-1]


def assert_greedy_parity(reference: dict[str, str], served: dict[str, str]) -> None:
    """Served greedy transcripts against the reference's, file by file."""
    assert all(reference.values()), f"the reference produced an empty transcript: {reference}"
    mismatches = [(name, ref, served[name]) for name, ref in reference.items() if served[name] != ref]
    # bf16 near-ties flip the odd homophone (9 of the 200 benchmark files differ),
    # while a real regression breaks most files
    far = [m for m in mismatches if _word_distance(m[1], m[2]) > 2]
    assert len(mismatches) <= 1 and not far, "(file, reference, served):\n" + "\n".join(map(repr, mismatches))

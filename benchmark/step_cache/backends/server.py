"""The served path: clips from an M* server over ``/v1/videos/generations``, timed from the client.

A method is a serving config, so each one is its own server boot: :meth:`ServerBackend.configure` writes the base
yaml with the method's ``step_cache`` under ``model_kwargs``, starts the server on it, waits for it to answer
``/health``, and the first request of each shape after that is an untimed warm-up (the compile). With ``--url`` the
backend talks to a server someone else booted and only checks that the method asked for is the one said to be
running there.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from benchmark.step_cache.backends.base import Generated
from benchmark.step_cache.methods import Method
from benchmark.step_cache.metrics import decode_mp4

# {python} {config} {port} {scratch} are filled in. Serves HTTP from mstar-server, the default frontend, which has to
# be built or named by MSTAR_SERVER_BIN; add --python-frontend with --server-arg to serve without it.
DEFAULT_LAUNCH = (
    "{python} -m mstar.api_server.entrypoint --config {config} --port {port} "
    "--socket-path-prefix {scratch}/sock/ --upload-dir {scratch}/up/ --tensor-comm-protocol SHM --timeout 3600 "
    "--log-stats-file {scratch}/stats.txt"
)
# model key -> (the name requests carry, the base serving config)
_SERVED = {
    "wan22_ti2v_5b": ("wan22", "configs/wan22.yaml"),
    "wan22_t2v_a14b": ("wan22", "configs/wan22_a14b.yaml"),
    "minimax_h3": ("minimax_h3", "configs/minimax_h3.yaml"),
}
_WARM_PROMPT = "a paper boat floating on a pond"


class ServerBackend:
    name = "server"

    def __init__(
        self, *, model: str, out_dir: str | Path, base_config: str | None = None, launch: str = DEFAULT_LAUNCH,
        server_args: tuple[str, ...] = (), port: int = 8471, url: str | None = None, running: str | None = None,
        boot_timeout: float = 3000.0, request_timeout: float = 3600.0, request_extra: dict[str, Any] | None = None,
    ):
        if model not in _SERVED:
            raise ValueError(f"the server backend serves {sorted(_SERVED)}; got {model!r}")
        self.model = model
        self.served_name, default_config = _SERVED[model]
        self.base_config = base_config or default_config
        self.out_dir = Path(out_dir)
        self.launch = launch
        self.server_args = tuple(server_args)
        self.external = url is not None
        self.url = (url or f"http://127.0.0.1:{port}").rstrip("/")
        self.port = port
        self.running = running
        self.boot_timeout = boot_timeout
        self.request_timeout = request_timeout
        self.request_extra = dict(request_extra or {})
        self._process: subprocess.Popen | None = None
        self._log: Path | None = None
        self._warmed: set[tuple[str, int, int]] = set()

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("server backend")
        group.add_argument("--base-config", help="serving yaml the methods are laid over (default: the model's)")
        group.add_argument("--launch", default=DEFAULT_LAUNCH,
                           help="server command; {python} {config} {port} {scratch} are filled in")
        group.add_argument("--server-arg", action="append", default=[],
                           help="an argument appended to the server command, e.g. --python-frontend (repeatable)")
        group.add_argument("--port", type=int, default=8471)
        group.add_argument("--url", help="a server already running; the harness then starts none")
        group.add_argument("--running", help="with --url: the name of the method that server was booted with")
        group.add_argument("--boot-timeout", type=float, default=3000.0, help="seconds to wait for /health")
        group.add_argument("--request-extra", type=json.loads, default={},
                           help='JSON merged into every request body, e.g. \'{"flow_shift": 12}\'')

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ServerBackend:
        return cls(
            model=args.model, out_dir=args.out, base_config=args.base_config, launch=args.launch,
            server_args=tuple(args.server_arg), port=args.port, url=args.url, running=args.running,
            boot_timeout=args.boot_timeout, request_extra=args.request_extra,
        )

    def describe(self) -> dict[str, Any]:
        """What the client can know: the box's GPU and this process's torch (the server's own, when the harness
        runs from the server's environment)."""
        info: dict[str, Any] = {"backend": self.name, "model": self.model, "url": self.url,
                                "base_config": self.base_config}
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if smi.returncode == 0:
                info["gpu"] = "; ".join(line.strip() for line in smi.stdout.splitlines() if line.strip())
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            import torch

            info["torch"], info["cuda"] = torch.__version__, torch.version.cuda
        except ImportError:
            pass
        for key in ("CUDA_VISIBLE_DEVICES", "MSTAR_COMPILE_LEVEL"):
            if key in os.environ:
                info[key] = os.environ[key]
        return info

    def write_config(self, method: Method) -> Path:
        """The base serving config with ``method``'s ``step_cache`` under ``model_kwargs``, as a yaml in the run's
        output directory."""
        import yaml

        with open(self.base_config) as handle:
            config = yaml.safe_load(handle)
        kwargs = dict(config.get("model_kwargs") or {})
        kwargs.pop("step_cache", None)
        if method.step_cache is not None:
            kwargs["step_cache"] = method.step_cache
        if kwargs:
            config["model_kwargs"] = kwargs
        else:
            config.pop("model_kwargs", None)
        path = self.out_dir / "cfg" / f"{method.name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        return path

    def configure(self, method: Method) -> None:
        if self.external:
            if self.running != method.name:
                raise RuntimeError(
                    f"the server at {self.url} runs {self.running!r} (--running), not {method.name!r}: boot it "
                    f"with that method's config, or let the harness start the servers (no --url)"
                )
            if not self._healthy():
                raise RuntimeError(f"no answer from {self.url}/health")
            return
        self._stop()
        self._warmed.clear()
        config = self.write_config(method)
        scratch = self.out_dir / "srv" / method.name
        for sub in ("sock", "up"):
            (scratch / sub).mkdir(parents=True, exist_ok=True)
        command = shlex.split(self.launch.format(
            python=shlex.quote(sys.executable), config=shlex.quote(str(config)), port=self.port,
            scratch=shlex.quote(str(scratch)),
        )) + list(self.server_args)
        self._log = scratch / "server.log"
        with open(self._log, "w") as log:
            # its own process group, so stopping it takes its workers along
            self._process = subprocess.Popen(  # noqa: S603 - the operator's own launch command
                command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
            )
        deadline = time.monotonic() + self.boot_timeout
        while not self._healthy():
            if self._process.poll() is not None:
                raise RuntimeError(f"the server for {method.name} exited ({self._process.returncode}):\n{self._tail()}")
            if time.monotonic() > deadline:
                self._stop()
                raise RuntimeError(f"the server for {method.name} did not answer /health in {self.boot_timeout:.0f} s")
            time.sleep(5)

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=5) as response:  # noqa: S310 - our own server
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def _tail(self, lines: int = 40) -> str:
        if self._log is None or not self._log.exists():
            return ""
        return "\n".join(self._log.read_text(errors="replace").splitlines()[-lines:])

    def _post(self, prompt: str, seed: int, size: str, frames: int, steps: int) -> bytes:
        body = {
            "model": self.served_name, "prompt": prompt, "negative_prompt": "", "size": size, "seed": seed,
            "num_inference_steps": steps, "num_frames": frames, **self.request_extra,
        }
        request = urllib.request.Request(  # noqa: S310 - our own server
            f"{self.url}/v1/videos/generations", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout) as response:  # noqa: S310
            return base64.b64decode(json.load(response)["data"][0]["b64_json"])

    def generate(self, prompt: str, seed: int, *, size: str, frames: int, steps: int) -> Generated:
        shape = (size, frames, steps)
        if shape not in self._warmed:
            # the first request of a shape pays its compile
            self._post(_WARM_PROMPT, 0, size, frames, steps)
            self._warmed.add(shape)
        start = time.perf_counter()
        mp4 = self._post(prompt, seed, size, frames, steps)
        seconds = time.perf_counter() - start
        return Generated(frames=decode_mp4(mp4), seconds=seconds, extra={"mp4_bytes": len(mp4)})

    def _stop(self) -> None:
        """SIGINT the server's process group, then SIGKILL what is left of it after a minute."""
        process, self._process = self._process, None
        if process is None:
            return
        for sig, wait in ((signal.SIGINT, 60.0), (signal.SIGKILL, 10.0)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + wait
            # the group, not only its leader: workers outlive the entrypoint for a moment
            while _group_alive(process.pid) and time.monotonic() < deadline:
                process.poll()
                time.sleep(0.5)
            if not _group_alive(process.pid):
                break
        process.poll()

    def close(self) -> None:
        self._stop()


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

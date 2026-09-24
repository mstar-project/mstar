"""Queue full validation after a download and a sustained idle-GPU check.

The download process writes checkpoint_path.txt in --run-dir on success.
This runner does not download, terminate other users' jobs, or reserve GPUs.
It records its stages and exits on any validation failure.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import yaml


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.root = args.run_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.output = self.root / "full"
        self.output.mkdir(exist_ok=True)
        self.child = None
        self.deadline = time.monotonic() + args.wait_hours * 3600
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
                        OMP_NUM_THREADS="1", HF_HUB_DISABLE_PROGRESS_BARS="1")

    def status(self, stage, **extra):
        value = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(),
                     pipeline_pid=os.getpid(), **extra)
        temporary = self.root / "status.json.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(self.root / "status.json")
        print(json.dumps(value), flush=True)

    def wait_download(self):
        marker = self.root / "checkpoint_path.txt"
        while not marker.is_file():
            if time.monotonic() > self.deadline:
                raise TimeoutError("download/availability wait deadline exceeded")
            try:
                os.kill(self.args.download_pid, 0)
            except ProcessLookupError:
                # Completion can race with this liveness check.
                if marker.is_file():
                    break
                raise RuntimeError("downloader exited without a completion marker") from None
            self.status("waiting_for_download", download_pid=self.args.download_pid)
            time.sleep(30)
        return Path(marker.read_text().strip())

    def wait_idle(self, next_stage):
        since = None
        while True:
            if time.monotonic() > self.deadline:
                raise TimeoutError("download/availability wait deadline exceeded")
            output = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
            ], text=True)
            gpus = [list(map(int, line.split(","))) for line in output.strip().splitlines()]
            processes = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
            ], text=True).strip()
            idle = len(gpus) == 8 and not processes and all(memory < 256 and utilization == 0
                                                          for _, memory, utilization in gpus)
            since = (since or time.monotonic()) if idle else None
            if since is not None and time.monotonic() - since >= 60:
                return
            self.status("waiting_for_idle_gpus", next_stage=next_stage, gpus=gpus,
                        compute_pids=processes.splitlines(),
                        idle_seconds=0 if since is None else time.monotonic()-since)
            time.sleep(15)

    def stop_child(self):
        if self.child is None or self.child.poll() is not None:
            return
        os.killpg(self.child.pid, signal.SIGINT)
        try:
            self.child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self.child.pid, signal.SIGKILL)
            self.child.wait()

    def run_stage(self, name, command):
        with (self.output / f"{name}.log").open("w") as log:
            self.child = subprocess.Popen(command, cwd=self.args.repo, env=self.env,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.status(name, child_pid=self.child.pid, command=command)
            result = self.child.wait(timeout=7200)
            if result:
                raise RuntimeError(f"{name} exited {result}; see {self.output / f'{name}.log'}")
            self.child = None

    def serve(self, checkpoint):
        config = yaml.safe_load((self.args.repo / "configs/command_a_plus_tp8.yaml").read_text())
        config["model_kwargs"]["checkpoint_dir"] = str(checkpoint)
        path = self.output / "deployment.yaml"
        path.write_text(yaml.safe_dump(config))
        command = [sys.executable, "-u", "-B", "-m", "mstar.api_server.entrypoint",
                   "--config", str(path), "--host", "127.0.0.1", "--port", str(self.args.port),
                   "--socket-path-prefix", str(self.output / "socket"),
                   "--upload-dir", str(self.output / "uploads"), "--tensor-comm-protocol", "SHM",
                   "--timeout", "120"]
        with (self.output / "server.log").open("w") as log:
            server = subprocess.Popen(command, cwd=self.args.repo, env=self.env,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.child = server
            self.status("http_server_starting", child_pid=server.pid)
            deadline = time.monotonic() + 3600
            try:
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("HTTP server exited during startup; inspect server.log")
                    if time.monotonic() > deadline:
                        raise TimeoutError("HTTP server startup timed out")
                    try:
                        with urlopen(f"http://127.0.0.1:{self.args.port}/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (URLError, TimeoutError):
                        pass
                    time.sleep(5)
                with (self.output / "http_smoke.log").open("w") as smoke_log:
                    self.status("http_smoke", child_pid=server.pid)
                    subprocess.run([sys.executable, "-B", "-m", "test.command_a_plus.smoke_http",
                                    "--url", f"http://127.0.0.1:{self.args.port}"],
                                   cwd=self.args.repo, env=self.env, stdout=smoke_log,
                                   stderr=subprocess.STDOUT, timeout=600, check=True)
            finally:
                self.child = server
                self.stop_child()
                self.child = None

    def run(self):
        checkpoint = self.wait_download()
        self.wait_idle("hf_reference")
        base = ["--checkpoint", str(checkpoint), "--output", str(self.output), "--tokens", "16"]
        self.run_stage("hf_reference", [sys.executable, "-u", "-B", "-m",
                                        "test.command_a_plus.validate_checkpoint", "reference", *base])
        self.wait_idle("mstar_tp8")
        self.run_stage("mstar_tp8", [sys.executable, "-u", "-B", "-m", "torch.distributed.run",
                                     "--standalone", "--nproc-per-node=8", "--module",
                                     "test.command_a_plus.validate_checkpoint", "mstar", *base,
                                     "--lengths", "128", "1024", "--benchmark-tokens", "32"])
        self.wait_idle("http_smoke")
        self.serve(checkpoint)
        self.status("complete", output=str(self.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--download-pid", type=int, required=True)
    parser.add_argument("--wait-hours", type=float, default=12)
    parser.add_argument("--port", type=int, default=18937)
    args = parser.parse_args()
    pipeline = Pipeline(args)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        pipeline.run()
    except BaseException as error:
        pipeline.stop_child()
        pipeline.status("failed" if not isinstance(error, KeyboardInterrupt) else "cancelled", error=str(error))
        raise


if __name__ == "__main__":
    main()

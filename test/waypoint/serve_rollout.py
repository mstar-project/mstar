#!/usr/bin/env python3
"""Drive Waypoint rollouts through the mstar server and Python SDK, end to end.

Launches ``mstar/api_server/entrypoint.py`` on ``configs/waypoint.yaml`` -- the
real API server, conductor process and GPU worker -- then sends a seed frame and
an action script through ``MStarClient`` and consumes typed ``VideoFrameChunk``
objects from the stream.

By default the request is sent twice, under *different* ids and one explicit
``model_kwargs.seed``, so the two rollouts draw the same noise. Identical bytes
the second time are what shows the first request left nothing behind: with
``num_worlds: 1`` a leaked world fails the second admission outright, and a
leaked ``ChunkedStreamingTAEHV`` in ``PerRequestState.kwargs`` would resume the
first rollout's stream and change the pixels.

The ids have to differ. A worker defers ``REMOVE_REQUEST`` while a step is in
flight and keys the deferral on the rid alone, so reusing an id lets the first
request's teardown land on the second and drop its in-flight reads.

``--concurrent-waves`` switches to the two-world isolation gate: two distinct
solo baselines are replayed concurrently through separate SDK clients, then
checked byte-for-byte across repeated world reuse. Optional memory sampling
tracks only the server process group and excludes the first concurrent wave as
allocator warmup.

Deployment details that the checked-in config cannot carry are supplied here
rather than edited into it:

  * Local mode adds ``model_kwargs.checkpoint_dir`` / ``ae_path``. Hub mode
    deliberately omits both, exercising the registry's variant-to-repository
    mapping, and can forward ``--cache-dir`` to Hugging Face.
  * a 16:9 seed. ``WaypointModel.load_image`` decodes without resizing and
    ``_seed_clip`` refuses any other ratio, so the shipped 1927x1080 asset is
    resized to the selected variant's output geometry first.

    CUDA_VISIBLE_DEVICES=2 python3 test/waypoint/serve_rollout.py \
        --variant 720p --steps 8 --worlds 2 --concurrent-waves 4 \
        --measure-memory --physical-gpu 2

    CUDA_VISIBLE_DEVICES=2 python3 test/waypoint/serve_rollout.py \
        --variant 360p --steps 8
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/waypoint.yaml"
DEFAULT_ROOT = Path("/mnt/storage/garv901/waypoint-1.5-1B/checkpoints")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mstar.client import MStarClient, VideoFrameChunk  # noqa: E402


@dataclass(frozen=True)
class Variant:
    model_variant: str
    height: int
    width: int
    tokens_per_frame: int
    checkpoint_name: str

    @property
    def checkpoint_dir(self) -> Path:
        return DEFAULT_ROOT / self.checkpoint_name


@dataclass(frozen=True)
class RolloutSpec:
    label: str
    request_id: str
    rng_seed: int


@dataclass(frozen=True)
class MemorySample:
    timestamp: float
    phase: str
    host_pss_mib: float
    gpu_mib: float


@dataclass(frozen=True)
class WaveMemory:
    phase: str
    peak_host_pss_mib: float
    peak_gpu_mib: float
    quiet_host_pss_mib: float
    quiet_gpu_mib: float


VARIANTS = {
    "360p": Variant(
        model_variant="waypoint-1.5-1b-360p",
        height=360,
        width=640,
        tokens_per_frame=128,
        checkpoint_name="Waypoint-1.5-1B-360P",
    ),
    "720p": Variant(
        model_variant="waypoint-1.5-1b-720p",
        height=720,
        width=1280,
        tokens_per_frame=512,
        checkpoint_name="Waypoint-1.5-1B",
    ),
}


def _run_config(
    base: Path,
    variant: Variant,
    checkpoint_dir: Path | None,
    ae_path: Path | None,
    out: Path,
    worlds: int = 1,
) -> Path:
    """Build one deployment config without modifying the checked-in YAML."""
    if worlds < 1:
        raise ValueError(f"worlds must be positive; got {worlds}")
    config = yaml.safe_load(base.read_text())
    model_kwargs = {
        **(config.get("model_kwargs") or {}),
        "variant": variant.model_variant,
    }
    # Hub mode passes None for these and must not inherit a local override from
    # the base config: omitting checkpoint_dir is what exercises the registry's
    # variant -> repository selection.
    model_kwargs.pop("checkpoint_dir", None)
    model_kwargs.pop("ae_path", None)
    if checkpoint_dir is not None:
        model_kwargs["checkpoint_dir"] = str(checkpoint_dir)
    if ae_path is not None:
        model_kwargs["ae_path"] = str(ae_path)
    config["model_kwargs"] = model_kwargs
    config["max_seq_len"] = variant.tokens_per_frame
    config["max_concurrent_requests"] = worlds
    resources = config["resources"] = config.get("resources") or {}
    kv = resources["kv"] = resources.get("kv") or {}
    kv["num_worlds"] = worlds
    out.write_text(yaml.safe_dump(config, sort_keys=False))
    return out


def _seed_png(source: Path, variant: Variant, out: Path) -> Path:
    """Resize the seed to the variant's 16:9 output geometry and write PNG."""
    from PIL import Image

    try:
        with Image.open(source) as image:
            image.convert("RGB").resize((variant.width, variant.height), Image.Resampling.BILINEAR).save(
                out, format="PNG"
            )
    except OSError as exc:
        raise SystemExit(f"could not decode seed image {source}: {exc}") from exc
    return out


def _actions(num_steps: int) -> list[dict]:
    """A scripted pan with a button held, so the run is not the idle world.

    There is exactly one action row per generated latent step. Prime uses its
    own internal idle action, so it does not consume action row zero.
    """
    return [
        {"mouse": [12.0 if i % 2 else -12.0, 0.0], "buttons": [0] if i % 4 == 0 else [], "scroll": 0.0}
        for i in range(num_steps)
    ]


def _wait_for_health(client: MStarClient, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with code {proc.returncode} before serving")
        if client.health():
            return
        time.sleep(1.0)
    raise SystemExit(f"server did not answer /health within {timeout:.0f}s")


def _rollout(
    client: MStarClient,
    seed: Path,
    num_steps: int,
    request_id: str,
    rng_seed: int,
    start_barrier: threading.Barrier | None = None,
) -> list[VideoFrameChunk]:
    """One SDK-streamed request, typed frame chunks in arrival order."""
    chunks: list[VideoFrameChunk] = []
    stream = client.stream(
        images=seed,
        input_modalities=("image",),
        output_modalities=("video_frame",),
        request_id=request_id,
        num_steps=num_steps,
        actions=_actions(num_steps),
        seed=rng_seed,
    )
    # MStarClient._stream is lazy: crossing here means both tasks have built
    # their multipart bodies before either opens its HTTP request.
    if start_barrier is not None:
        start_barrier.wait(timeout=30)
    for event in stream:
        if not isinstance(event, VideoFrameChunk):
            raise RuntimeError(f"Waypoint returned an unexpected SDK stream event: {type(event).__name__}")
        chunks.append(event)
        print(f"  {request_id} chunk {len(chunks) - 1:3d}  bytes={len(event.data):9d}  metadata={event.metadata}")
    return chunks


def _concurrent_rollouts(
    client_factory: Callable[[], MStarClient],
    seed: Path,
    num_steps: int,
    specs: tuple[RolloutSpec, RolloutSpec],
) -> dict[str, list[VideoFrameChunk]]:
    """Start exactly two lazy SDK streams together, each on its own Session."""
    barrier = threading.Barrier(3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            spec.label: executor.submit(
                _rollout,
                client_factory(),
                seed,
                num_steps,
                spec.request_id,
                spec.rng_seed,
                barrier,
            )
            for spec in specs
        }
        barrier.wait(timeout=30)
        return {label: future.result() for label, future in futures.items()}


def _video_bytes(chunks: list[VideoFrameChunk]) -> bytes:
    return b"".join(chunk.data for chunk in chunks)


def _dit_schedule(log_text: str, request_ids: set[str]) -> list[str]:
    """Extract single-request DiT rollout executions from worker DEBUG logs."""
    marker = "Executing: dit graph_walk=rollout "
    scheduled: list[str] = []
    for line in log_text.splitlines():
        if marker not in line:
            continue
        try:
            batch = ast.literal_eval(line.split(marker, 1)[1].strip())
        except (SyntaxError, ValueError):
            continue
        if (
            isinstance(batch, (list, tuple))
            and len(batch) == 1
            and batch[0] in request_ids
        ):
            scheduled.append(batch[0])
    return scheduled


def _interleaving_failure(log_text: str, request_ids: tuple[str, str]) -> str | None:
    """Require an A/B/A or B/A/B DiT schedule, not just overlapping clients."""
    scheduled = _dit_schedule(log_text, set(request_ids))
    compressed = [rid for i, rid in enumerate(scheduled) if i == 0 or rid != scheduled[i - 1]]
    interleaved = any(
        first == third and first != second
        for first, second, third in zip(compressed, compressed[1:], compressed[2:], strict=False)
    )
    if interleaved:
        return None
    counts = {rid: scheduled.count(rid) for rid in request_ids}
    return (
        f"worker DEBUG schedule did not contain A/B/A interleaving for {request_ids}; DiT schedule counts were {counts}"
    )


def _execution_count_failure(
    log_text: str,
    request_ids: tuple[str, str],
    num_steps: int,
) -> str | None:
    scheduled = _dit_schedule(log_text, set(request_ids))
    counts = {rid: scheduled.count(rid) for rid in request_ids}
    if all(count == num_steps for count in counts.values()):
        return None
    return f"expected {num_steps} DiT rollout executions per request; got {counts}"


_CLEANUP_MARKER = "Request cleanup complete:"


def _cleaned_request_ids(log_text: str) -> set[str]:
    return {line.split(_CLEANUP_MARKER, 1)[1].strip() for line in log_text.splitlines() if _CLEANUP_MARKER in line}


def _read_log_since(log_path: Path, offset: int) -> str:
    with log_path.open("rb") as log:
        log.seek(offset)
        return log.read().decode("utf-8", "replace")


def _wait_for_cleanup(
    log_path: Path,
    request_ids: tuple[str, ...],
    proc: subprocess.Popen,
    timeout: float,
    offset: int = 0,
) -> None:
    """Wait for actual worker cleanup, including any deferred remove."""
    deadline = time.monotonic() + timeout
    wanted = set(request_ids)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before request cleanup")
        try:
            cleaned = _cleaned_request_ids(_read_log_since(log_path, offset))
        except FileNotFoundError:
            cleaned = set()
        if wanted <= cleaned:
            return
        time.sleep(0.1)
    missing = sorted(wanted - cleaned)
    raise RuntimeError(f"worker did not confirm cleanup for {missing} within {timeout:.0f}s")


def _check(
    chunks: list[VideoFrameChunk],
    num_steps: int,
    height: int,
    width: int,
) -> list[str]:
    """Every failed expectation, rather than the first: a run costs minutes and
    a short stream and a wrong payload size have different causes."""
    failures = []

    # Prime advances internal encoder/DiT/decoder state but emits no seed
    # reconstruction. Every generated latent produces one chunk of four frames.
    expected = num_steps
    if len(chunks) != expected:
        failures.append(f"expected {expected} video chunks, got {len(chunks)}")

    # The SDK validates each payload against its own metadata. Keep the expected
    # variant geometry and frame sequence as independent end-to-end assertions.
    size = 4 * height * width * 3
    wrong = [i for i, chunk in enumerate(chunks) if len(chunk.data) != size]
    if wrong:
        failures.append(
            f"chunks {wrong[:5]} are not {size} bytes "
            f"(4 x {height} x {width} x 3 uint8); first is "
            f"{len(chunks[wrong[0]].data)}"
        )
    for chunk_idx, chunk in enumerate(chunks):
        wanted = {
            "width": width,
            "height": height,
            "fps": 60.0,
            "pixel_format": "rgb24",
            "frame_index": chunk_idx * 4,
            "frame_count": 4,
        }
        mismatches = {
            key: (chunk.metadata.get(key), value) for key, value in wanted.items() if chunk.metadata.get(key) != value
        }
        if mismatches:
            failures.append(f"chunk {chunk_idx} has invalid video_frame metadata: {mismatches}")
    if sum(chunk.frame_count for chunk in chunks) != 4 * num_steps:
        failures.append(f"expected exactly {4 * num_steps} generated frames")
    return failures


def _parse_pss_kib(smaps_rollup: str) -> int:
    for line in smaps_rollup.splitlines():
        if line.startswith("Pss:"):
            return int(line.split()[1])
    raise ValueError("smaps_rollup contained no Pss total")


def _process_group_pids(process_group: int) -> list[int]:
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) == process_group:
                pids.append(pid)
        except ProcessLookupError:
            continue
    return sorted(pids)


def _process_group_pss_mib(process_group: int) -> float:
    total_kib = 0
    read_count = 0
    for pid in _process_group_pids(process_group):
        try:
            text = Path(f"/proc/{pid}/smaps_rollup").read_text()
        except FileNotFoundError:
            continue
        total_kib += _parse_pss_kib(text)
        read_count += 1
    if read_count == 0:
        raise RuntimeError(f"no readable PSS telemetry for process group {process_group}")
    return total_kib / 1024


def _parse_nvidia_smi_processes(output: str) -> list[tuple[int, float]]:
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        try:
            rows.append((int(fields[0]), float(fields[1])))
        except ValueError as exc:
            raise ValueError(f"unusable nvidia-smi row: {line!r}") from exc
    return rows


def _process_group_gpu_mib(process_group: int, physical_gpu: int) -> float:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={physical_gpu}",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"nvidia-smi telemetry unavailable: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"nvidia-smi telemetry unavailable: {detail}")

    total = 0.0
    matched = 0
    for pid, used_mib in _parse_nvidia_smi_processes(result.stdout):
        try:
            belongs_to_server = os.getpgid(pid) == process_group
        except ProcessLookupError:
            continue
        if belongs_to_server:
            total += used_mib
            matched += 1
    if matched == 0:
        raise RuntimeError(f"GPU {physical_gpu} reports no compute process in server group {process_group}")
    return total


class MemorySampler:
    """Continuously sample only the API server's process group."""

    def __init__(self, process_group: int, physical_gpu: int, interval: float = 0.25):
        self.process_group = process_group
        self.physical_gpu = physical_gpu
        self.interval = interval
        self._phase = "ready"
        self._samples: list[MemorySample] = []
        self._last_error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="waypoint-memory-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(15.0, self.interval * 4))
        if self._thread.is_alive():
            raise RuntimeError("memory sampler did not stop")

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def snapshot(self) -> tuple[list[MemorySample], str | None]:
        with self._lock:
            return list(self._samples), self._last_error

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                phase = self._phase
            try:
                host_pss_mib = _process_group_pss_mib(self.process_group)
                gpu_mib = _process_group_gpu_mib(self.process_group, self.physical_gpu)
                sample = MemorySample(time.monotonic(), phase, host_pss_mib, gpu_mib)
            except (OSError, RuntimeError, ValueError) as exc:
                with self._lock:
                    self._last_error = str(exc)
            else:
                with self._lock:
                    self._samples.append(sample)
                    self._last_error = None
            self._stop.wait(self.interval)


def _wait_for_first_memory_sample(
    sampler: MemorySampler,
    proc: subprocess.Popen,
    timeout: float = 30.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        samples, _ = sampler.snapshot()
        if samples:
            return
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before telemetry started")
        time.sleep(0.1)
    _, error = sampler.snapshot()
    raise RuntimeError(f"memory telemetry unavailable after {timeout:.0f}s: {error or 'no samples'}")


def _stable_memory_plateau(
    samples: list[MemorySample],
    *,
    count: int = 3,
    tolerance_mib: float = 2.0,
) -> tuple[float, float] | None:
    if len(samples) < count:
        return None
    tail = samples[-count:]
    host = [sample.host_pss_mib for sample in tail]
    gpu = [sample.gpu_mib for sample in tail]
    if max(host) - min(host) > tolerance_mib or max(gpu) - min(gpu) > tolerance_mib:
        return None
    return max(host), max(gpu)


def _wait_for_quiescent_memory(
    sampler: MemorySampler,
    phase: str,
    timeout: float = 10.0,
) -> tuple[float, float]:
    sampler.set_phase(phase)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        samples, _ = sampler.snapshot()
        plateau = _stable_memory_plateau([sample for sample in samples if sample.phase == phase])
        if plateau is not None:
            return plateau
        time.sleep(0.1)
    _, error = sampler.snapshot()
    detail = f"; last telemetry error: {error}" if error else ""
    raise RuntimeError(f"memory did not reach a stable {phase!r} plateau within {timeout:.0f}s{detail}")


def _summarize_wave_memory(
    samples: list[MemorySample],
    phase: str,
    quiet: tuple[float, float],
) -> WaveMemory:
    wave_samples = [sample for sample in samples if sample.phase == phase]
    if not wave_samples:
        raise RuntimeError(f"memory telemetry captured no samples during {phase}")
    return WaveMemory(
        phase=phase,
        peak_host_pss_mib=max(sample.host_pss_mib for sample in wave_samples),
        peak_gpu_mib=max(sample.gpu_mib for sample in wave_samples),
        quiet_host_pss_mib=quiet[0],
        quiet_gpu_mib=quiet[1],
    )


def _bounded_memory_failures(
    measured: list[WaveMemory],
    *,
    host_growth_mib: float,
    gpu_growth_mib: float,
) -> list[str]:
    if len(measured) < 2:
        return ["bounded-memory check needs at least two measured waves after warmup"]
    baseline = measured[0]
    failures = []
    for wave in measured[1:]:
        host_growth = wave.quiet_host_pss_mib - baseline.quiet_host_pss_mib
        gpu_growth = wave.quiet_gpu_mib - baseline.quiet_gpu_mib
        if host_growth > host_growth_mib:
            failures.append(
                f"{wave.phase} quiescent host PSS grew {host_growth:.1f} MiB "
                f"from {baseline.phase} (limit {host_growth_mib:.1f} MiB)"
            )
        if gpu_growth > gpu_growth_mib:
            failures.append(
                f"{wave.phase} quiescent GPU memory grew {gpu_growth:.1f} MiB "
                f"from {baseline.phase} (limit {gpu_growth_mib:.1f} MiB)"
            )
    return failures


def _free_port() -> int:
    """A loopback port nothing holds. The box is shared and the server binds
    before it can report a conflict, so a fixed default collides."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _shutdown(proc: subprocess.Popen) -> None:
    """Let the API parent shut down its children, then kill a hung group."""
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=60)
        return
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass


def _server_command(
    config: Path,
    port: int,
    workdir: Path,
    log_level: str,
    request_timeout: float,
    cache_dir: Path | None,
    enable_nvtx: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO / "mstar/api_server/entrypoint.py"),
        "--config",
        str(config),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--socket-path-prefix",
        str(workdir / "sock"),
        "--upload-dir",
        str(workdir / "uploads"),
        "--tensor-comm-protocol",
        "SHM",
        "--log-level",
        log_level,
        "--timeout",
        str(request_timeout),
    ]
    if cache_dir is not None:
        command.extend(("--cache-dir", str(cache_dir)))
    if enable_nvtx:
        command.append("--enable-nvtx")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument(
        "--source",
        choices=("local", "hub"),
        default="local",
        help="local paths (default) or the registry's variant-specific Hub repositories",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="checkpoint override; defaults to the selected variant under the checkpoint root",
    )
    parser.add_argument("--ae-path", type=Path, help="local TAEHV override")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face download cache")
    parser.add_argument("--seed-image", type=Path, default=DEFAULT_ROOT / "seed/default.jpg")
    parser.add_argument(
        "--steps",
        "--frames",
        dest="steps",
        type=int,
        default=8,
        help="generated latent steps; each step streams four RGB frames",
    )
    parser.add_argument("--port", type=int, default=0, help="0 picks a free one")
    parser.add_argument("--request-id", type=str, default="waypoint-serve-rollout")
    parser.add_argument("--seed", type=int, default=112464007)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument(
        "--concurrent-waves",
        type=int,
        default=0,
        help="run the two-world isolation gate for this many waves (minimum 2)",
    )
    parser.add_argument("--measure-memory", action="store_true")
    parser.add_argument(
        "--physical-gpu",
        type=int,
        help="physical nvidia-smi GPU index; required with --measure-memory",
    )
    parser.add_argument("--host-growth-mib", type=float, default=128.0)
    parser.add_argument("--gpu-growth-mib", type=float, default=64.0)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--log", type=Path, default=Path("/tmp/waypoint_server.log"))
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--enable-nvtx",
        action="store_true",
        help="enable server NVTX ranges for an external CUDA profiler",
    )
    args = parser.parse_args()

    if args.worlds < 1:
        parser.error("--worlds must be positive")
    if args.concurrent_waves < 0:
        parser.error("--concurrent-waves cannot be negative")
    if args.concurrent_waves == 1:
        parser.error("--concurrent-waves must be 0 or at least 2")
    if args.concurrent_waves and args.worlds != 2:
        parser.error("the concurrent isolation gate requires exactly --worlds 2")
    if args.measure_memory and args.concurrent_waves < 3:
        parser.error("--measure-memory needs at least 3 concurrent waves (one warm, two measured)")
    if args.measure_memory and args.physical_gpu is None:
        parser.error("--measure-memory requires --physical-gpu")
    if args.physical_gpu is not None and args.physical_gpu < 0:
        parser.error("--physical-gpu cannot be negative")
    if args.host_growth_mib < 0 or args.gpu_growth_mib < 0:
        parser.error("memory growth limits cannot be negative")
    if args.source == "hub" and (args.checkpoint_dir is not None or args.ae_path is not None):
        parser.error("--source hub cannot be combined with --checkpoint-dir or --ae-path")

    variant = VARIANTS[args.variant]
    if args.source == "local":
        checkpoint_dir = args.checkpoint_dir or variant.checkpoint_dir
        ae_path = args.ae_path or DEFAULT_ROOT / "taehv1_5"
        weight_source = str(checkpoint_dir)
    else:
        checkpoint_dir = None
        ae_path = None
        weight_source = f"registry Hub mapping for {variant.model_variant}"
    port = args.port or _free_port()
    url = f"http://127.0.0.1:{port}"
    workdir = Path(tempfile.mkdtemp(prefix="waypoint-serve-"))
    config = _run_config(
        args.config,
        variant,
        checkpoint_dir,
        ae_path,
        workdir / "run.yaml",
        worlds=args.worlds,
    )
    seed = _seed_png(args.seed_image, variant, workdir / "seed.png")
    path = str(REPO)
    server_log_level = "DEBUG" if args.concurrent_waves else args.log_level
    print(
        f"variant {args.variant} ({variant.width}x{variant.height})\n"
        f"weights {weight_source}\nworlds  {args.worlds}\nconfig  {config}\n"
        f"seed    {seed}\nlog     {args.log}"
    )

    server_command = _server_command(
        config,
        port,
        workdir,
        server_log_level,
        args.request_timeout,
        args.cache_dir,
        args.enable_nvtx,
    )

    # Its own process group, so a hung run is killed together with the
    # conductor and worker processes it spawned.
    with args.log.open("wb") as log:
        proc = subprocess.Popen(
            server_command,
            cwd=str(REPO),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            # The worktree is not an installed package and the conductor and
            # worker are spawned, not forked, so the path has to be inherited.
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": path},
        )

    failures: list[str] = []
    sampler: MemorySampler | None = None
    try:
        client = MStarClient(url, timeout=args.request_timeout)
        started = time.time()
        _wait_for_health(client, proc, args.startup_timeout)
        print(f"server ready after {time.time() - started:.1f}s")

        if args.measure_memory:
            sampler = MemorySampler(proc.pid, args.physical_gpu)
            sampler.start()
            _wait_for_first_memory_sample(sampler, proc)

        if not args.concurrent_waves:
            runs: list[list[VideoFrameChunk]] = []
            for attempt in (1, 2):
                rid = f"{args.request_id}-{attempt}"
                print(f"--- request {attempt} (request_id={rid} seed={args.seed}) ---")
                started = time.time()
                chunks = _rollout(client, seed, args.steps, rid, args.seed)
                print(f"    {len(chunks)} chunks in {time.time() - started:.1f}s")
                runs.append(chunks)
                failures += [
                    f"request {attempt}: {failure}"
                    for failure in _check(chunks, args.steps, variant.height, variant.width)
                ]

            first, second = (_video_bytes(run) for run in runs)
            if not first:
                failures.append("no video came back, so the repeat proves nothing")
            elif first != second:
                failures.append(
                    "the repeated request differs from the first, so per-request state "
                    "outlived its request (the ring, the AE session, or both)"
                )
            else:
                print(f"repeat is byte-identical over {len(first)} bytes of video")
        else:
            baseline_specs = (
                RolloutSpec("A", f"{args.request_id}-solo-a", args.seed),
                RolloutSpec("B", f"{args.request_id}-solo-b", args.seed + 1),
            )
            baselines: dict[str, bytes] = {}
            for spec in baseline_specs:
                print(f"--- solo {spec.label} (request_id={spec.request_id} seed={spec.rng_seed}) ---")
                log_offset = args.log.stat().st_size
                chunks = _rollout(client, seed, args.steps, spec.request_id, spec.rng_seed)
                failures += [
                    f"solo {spec.label}: {failure}"
                    for failure in _check(chunks, args.steps, variant.height, variant.width)
                ]
                baselines[spec.label] = _video_bytes(chunks)
                _wait_for_cleanup(
                    args.log,
                    (spec.request_id,),
                    proc,
                    args.request_timeout,
                    offset=log_offset,
                )

            if not all(baselines.values()):
                failures.append("a solo baseline returned no video")
            elif baselines["A"] == baselines["B"]:
                failures.append("distinct solo seeds produced identical baselines, so world swaps are invisible")

            wave_memory: list[WaveMemory] = []
            interleaved_waves = 0
            for wave in range(1, args.concurrent_waves + 1):
                phase = f"concurrent-wave-{wave}"
                specs = (
                    RolloutSpec("A", f"{args.request_id}-wave-{wave}-a", args.seed),
                    RolloutSpec("B", f"{args.request_id}-wave-{wave}-b", args.seed + 1),
                )
                if sampler is not None:
                    sampler.set_phase(phase)
                print(f"--- {phase}: {specs[0].request_id} + {specs[1].request_id} ---")
                log_offset = args.log.stat().st_size
                chunks_by_label = _concurrent_rollouts(
                    lambda: MStarClient(url, timeout=args.request_timeout),
                    seed,
                    args.steps,
                    specs,
                )
                for spec in specs:
                    chunks = chunks_by_label[spec.label]
                    failures += [
                        f"{phase} {spec.label}: {failure}"
                        for failure in _check(chunks, args.steps, variant.height, variant.width)
                    ]
                    actual = _video_bytes(chunks)
                    if actual != baselines[spec.label]:
                        failures.append(f"{phase} {spec.label} differs byte-for-byte from its solo baseline")

                rids = tuple(spec.request_id for spec in specs)
                _wait_for_cleanup(
                    args.log,
                    rids,
                    proc,
                    args.request_timeout,
                    offset=log_offset,
                )
                wave_log = _read_log_since(args.log, log_offset)
                count_failure = _execution_count_failure(wave_log, rids, args.steps)
                if count_failure is not None:
                    failures.append(f"{phase}: {count_failure}")
                schedule_failure = _interleaving_failure(wave_log, rids)
                if schedule_failure is None:
                    interleaved_waves += 1
                else:
                    print(f"  {phase} was serialized; another wave must prove A/B/A")

                if sampler is not None:
                    quiet = _wait_for_quiescent_memory(sampler, f"{phase}-quiet")
                    samples, _ = sampler.snapshot()
                    summary = _summarize_wave_memory(samples, phase, quiet)
                    wave_memory.append(summary)
                    warm = " (warmup, excluded)" if wave == 1 else ""
                    print(
                        f"  memory{warm}: peak host={summary.peak_host_pss_mib:.1f} MiB "
                        f"gpu={summary.peak_gpu_mib:.1f} MiB; quiet "
                        f"host={summary.quiet_host_pss_mib:.1f} MiB "
                        f"gpu={summary.quiet_gpu_mib:.1f} MiB"
                    )

            if interleaved_waves == 0:
                failures.append(
                    "no concurrent wave contained an A/B/A or B/A/B DiT rollout execution order"
                )

            if sampler is not None:
                failures += _bounded_memory_failures(
                    wave_memory[1:],
                    host_growth_mib=args.host_growth_mib,
                    gpu_growth_mib=args.gpu_growth_mib,
                )
    finally:
        try:
            if sampler is not None:
                sampler.stop()
        finally:
            _shutdown(proc)

    for failure in failures:
        print(f"FAIL {failure}")
    print("PASS" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

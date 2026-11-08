"""Which profiles fit this machine, and which configs and models have no case."""
import dataclasses
import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml
from cases import MODELS, ModelSpec

VRAM_HEADROOM = 1.10


@dataclasses.dataclass(frozen=True)
class Gpu:
    index: int          # physical index, as nvidia-smi reports it
    name: str
    total_gb: float
    used_gb: float
    uuid: str = ""


def visible(gpus: list[Gpu], cuda_visible_devices: str | None) -> list[Gpu]:
    """GPUs a server will see, in device order. ``CUDA_VISIBLE_DEVICES`` holds indices or UUIDs."""
    if cuda_visible_devices is None:
        return gpus
    by_key = {str(g.index): g for g in gpus} | {g.uuid: g for g in gpus if g.uuid}
    picked = []
    for key in (k.strip() for k in cuda_visible_devices.split(",")):
        if key not in by_key:
            break  # CUDA stops at the first unknown entry
        picked.append(by_key[key])
    return picked


def gpu_inventory() -> list[Gpu]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return []
    gpus = []
    for line in out.strip().splitlines():
        index, name, total, used, uuid = (x.strip() for x in line.split(","))
        gpus.append(Gpu(int(index), name, float(total) / 1024, float(used) / 1024, uuid))
    return visible(gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))


def config_gpus(config: Path) -> int:
    """Distinct GPUs the config's ranks run on; ``rank_devices`` can put several ranks on one."""
    data = yaml.safe_load(config.read_text()) or {}
    ranks = {r for group in data.get("node_groups") or [] for r in group.get("ranks") or []}
    placement = {int(r): int(d) for r, d in (data.get("rank_devices") or {}).items()}
    return max(len({placement.get(r, r) for r in ranks}), 1)


def missing_modules(modules: tuple[str, ...], python: str = sys.executable) -> list[str]:
    missing = []
    for module in modules:
        probe = subprocess.run([python, "-c", f"import {module}"], capture_output=True, text=True, check=False,
                               timeout=300)
        if probe.returncode:
            missing.append(module)
    return missing


def cases_without_inputs(cases) -> dict[str, str]:
    """Cases whose inputs need a module the runner cannot import, with the reason."""
    unbuildable = {}
    for case in cases:
        missing = [m for m in case.requires if importlib.util.find_spec(m) is None]
        if missing:
            unbuildable[case.name] = f"its input needs missing package(s): {', '.join(missing)}"
    return unbuildable


def inaccessible_repos(repos: tuple[str, ...]) -> tuple[list[str], dict[str, str]]:
    """Repos the account cannot download, and repos whose access could not be checked (with why)."""
    from huggingface_hub import auth_check
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    blocked, unknown = [], {}
    for repo in repos:
        try:
            auth_check(repo)
        except (GatedRepoError, RepositoryNotFoundError):
            blocked.append(repo)
        except Exception as exc:  # noqa: BLE001
            unknown[repo] = f"{type(exc).__name__}: {exc}"[:200]
    return blocked, unknown


def plan(spec: ModelSpec, config: Path, gpus: list[Gpu], offline: bool, notes: list[str] | None = None) -> str | None:
    """None if the profile can run here, else the reason it is skipped. Caveats
    that do not skip it, such as an access check that failed, go to ``notes``."""
    if not gpus:
        return "no CUDA GPU visible"
    needed = config_gpus(config)
    if needed > len(gpus):
        return f"needs {needed} GPUs; {len(gpus)} available"
    free = min(g.total_gb - g.used_gb for g in gpus[:needed])
    if spec.peak_vram_gb and spec.peak_vram_gb * VRAM_HEADROOM > free:
        return f"needs ~{spec.peak_vram_gb * VRAM_HEADROOM:.0f} GB per GPU (measured peak {spec.peak_vram_gb} GB); " \
               f"{free:.0f} GB free"
    missing = missing_modules(spec.requires)
    if missing:
        return f"missing package(s): {', '.join(missing)}"
    if spec.gated and not offline:
        blocked, unknown = inaccessible_repos(spec.gated)
        if blocked:
            links = ", ".join(f"https://huggingface.co/{r}" for r in blocked)
            return f"gated checkpoint: accept the license at {links}"
        if unknown and notes is not None:
            # Offline or no network: cached weights may still load, so run and let the load decide.
            notes.extend(f"could not check access: {r} ({why}); running anyway" for r, why in unknown.items())
    return None


def wait_for_gpu_release(indices: list[int], baseline_gb: float, tolerance_gb: float = 1.0,
                         timeout: float = 120) -> float:
    """Wait for GPU memory to fall back near ``baseline_gb``; return what is still held."""
    deadline = time.monotonic() + timeout
    while True:
        residual = sum(g.used_gb for g in gpu_inventory() if g.index in indices) - baseline_gb
        if residual <= tolerance_gb or time.monotonic() > deadline:
            return residual
        time.sleep(1)


class VramSampler:
    """Peak GPU memory above the level before the server started."""

    def __init__(self, indices: list[int], interval: float = 0.5):
        self.indices, self.interval = indices, interval
        self.baseline = self._used()
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _used(self) -> float:
        return sum(g.used_gb for g in gpu_inventory() if g.index in self.indices)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._used() - self.baseline)
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False


def gap_report(repo: Path) -> dict:
    """Configs and registered models with no case, with the GPUs each config needs."""
    covered_configs = {spec.config for spec in MODELS.values()}
    configs = sorted(p.relative_to(repo).as_posix() for p in (repo / "configs").glob("*.yaml"))
    uncovered = {c: config_gpus(repo / c) for c in configs if c not in covered_configs}
    registered = set()
    try:
        sys.path.insert(0, str(repo))
        from mstar.model.registry import MODEL_REGISTRY
        registered = set(MODEL_REGISTRY)
    except Exception:  # noqa: BLE001
        pass
    config_models = {(yaml.safe_load((repo / c).read_text()) or {}).get("model") for c in configs}
    covered_models = {(yaml.safe_load((repo / s.config).read_text()) or {}).get("model") for s in MODELS.values()}
    return dict(
        configs_without_case=uncovered,
        models_without_case=sorted(registered - covered_models),
        models_without_config=sorted(registered - config_models),
    )

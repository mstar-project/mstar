"""GPU memory and process queries that work on NVIDIA (nvidia-smi) and AMD (amd-smi) machines.

    python -m benchmark.gpu_query used-mib <gpu>     # memory in use on one GPU, MiB
    python -m benchmark.gpu_query pids <gpu>         # one "pid used_mib" line per process on it

The GPU index is the physical one (nvidia-smi -i / amd-smi -g). The Wan2.2 and Waypoint
harnesses use this, so they also run where only amd-smi exists.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys


def _tool() -> str | None:
    for name in ("nvidia-smi", "amd-smi"):
        if shutil.which(name):
            return name
    return None


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {(p.stderr or p.stdout).strip()[:300]}")
    return p.stdout


def _amd_json(*args: str):
    out = _run(["amd-smi", *args, "--json"])
    starts = [k for k in (out.find("{"), out.find("[")) if k >= 0]
    if not starts:
        raise RuntimeError(f"amd-smi {' '.join(args)} printed no JSON")
    doc = json.loads(out[min(starts):])
    if isinstance(doc, dict):
        doc = doc.get("gpu_data", [doc])
    return doc


def _mib(v) -> float:
    """amd-smi values come as numbers or {"value": n, "unit": "MB"|"B"|...}."""
    unit = "MB"
    if isinstance(v, dict):
        v, unit = v.get("value"), (v.get("unit") or "MB")
    v = float(v)
    return {"B": v / 2**20, "KB": v / 2**10, "KIB": v / 2**10, "GB": v * 2**10, "GIB": v * 2**10}.get(
        str(unit).upper(), v)


def parse_nvidia_processes(text: str) -> list[tuple[int, float]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 2:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        try:
            rows.append((int(fields[0]), float(fields[1])))
        except ValueError as exc:
            raise ValueError(f"unusable nvidia-smi row: {line!r}") from exc
    return rows


def parse_amd_processes(doc, gpu: int) -> list[tuple[int, float]]:
    rows = []
    for entry in doc:
        if entry.get("gpu") != gpu:
            continue
        for item in entry.get("process_list") or []:
            info = item.get("process_info") if isinstance(item, dict) else None
            if not isinstance(info, dict) or info.get("pid") is None:
                continue
            mem = info.get("memory_usage") or {}
            used = mem.get("vram_mem") if isinstance(mem, dict) else None
            rows.append((int(info["pid"]), _mib(used) if used is not None else 0.0))
    return rows


def used_mib(gpu: int) -> float:
    tool = _tool()
    if tool == "nvidia-smi":
        out = _run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return float(out.strip().splitlines()[0])
    if tool == "amd-smi":
        for entry in _amd_json("metric", "--mem-usage", "-g", str(gpu)):
            if entry.get("gpu") in (gpu, None):
                return _mib((entry.get("mem_usage") or {})["used_vram"])
        raise RuntimeError(f"amd-smi reports no GPU {gpu}")
    raise RuntimeError("neither nvidia-smi nor amd-smi is installed")


def processes(gpu: int) -> list[tuple[int, float]]:
    """(pid, MiB used) for every compute process on one GPU."""
    tool = _tool()
    if tool == "nvidia-smi":
        return parse_nvidia_processes(_run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,used_gpu_memory",
                                            "--format=csv,noheader,nounits"]))
    if tool == "amd-smi":
        return parse_amd_processes(_amd_json("process", "-g", str(gpu)), gpu)
    raise RuntimeError("neither nvidia-smi nor amd-smi is installed")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ("used-mib", "pids"):
        print(__doc__, file=sys.stderr)
        return 2
    gpu = int(argv[1])
    if argv[0] == "used-mib":
        print(f"{used_mib(gpu):.0f}")
    else:
        for pid, mib in processes(gpu):
            print(f"{pid} {mib:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

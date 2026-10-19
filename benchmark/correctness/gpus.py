"""Check GPUs by HIP index and report the VRAM in use on each, for runs on shared nodes.

amd-smi and rocm-smi number GPUs in their own order, which on some nodes isn't HIP's (k002-001:
HIP 0..7 = smi 3,0,2,1,7,4,6,5), and they list every process under every GPU. So this asks torch
instead, one subprocess per GPU with HIP_VISIBLE_DEVICES set the way `mstar serve --gpus` sets it:
the device's name, PCI address and memory in use by all processes (including the few hundred MB
of the query's own context).

    python -m benchmark.correctness.gpus 2,5    # one line per GPU; exit 1 if one is missing
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

_QUERY = r"""
import json, torch
p = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info(0)
print(json.dumps({"name": p.name, "bdf": f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0",
                  "used_gib": round((total - free) / 2**30, 2), "total_gib": round(total / 2**30, 2)}))
"""


def query(index: int, timeout_s: float = 120.0) -> dict | None:
    """Name, PCI address and VRAM use of HIP device ``index``, or None if it doesn't answer."""
    env = {**os.environ, "HIP_VISIBLE_DEVICES": str(index), "CUDA_VISIBLE_DEVICES": str(index)}
    try:
        p = subprocess.run([sys.executable, "-c", _QUERY], env=env, capture_output=True, text=True,
                           timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def describe(index: int, info: dict | None) -> str:
    if info is None:
        return f"GPU {index} (HIP): missing"
    return f"GPU {index} (HIP) {info['bdf']} {info['name']}: {info['used_gib']} of {info['total_gib']} GiB in use"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__)
        return 2
    missing = 0
    for index in (int(x) for x in args[0].split(",")):
        info = query(index)
        print(describe(index, info), flush=True)
        missing += info is None
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

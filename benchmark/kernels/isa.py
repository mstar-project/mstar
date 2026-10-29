"""Register, LDS and occupancy statistics of a compiled kernel.

These are the inputs for the next round of a search: a candidate that spills,
or whose VGPR count caps occupancy, is the one to change. Triton kernels
report them on the compiled kernel (its AMDGCN listing on ROCm, ``n_regs`` /
``n_spills`` on NVIDIA). A HIP extension's code objects are unbundled from
the shared library and their AMDGPU metadata read with ROCm's LLVM tools.
Every function returns what it could read, with an ``error`` entry when it
could read nothing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# comments the AMDGPU backend writes after each kernel in its listing
_LISTING = {
    "vgprs": r";\s*NumVgprs:\s*(\d+)",
    "agprs": r";\s*NumAgprs:\s*(\d+)",
    "sgprs": r";\s*NumSgprs:\s*(\d+)",
    "scratch_bytes": r";\s*ScratchSize:\s*(\d+)",
    "occupancy": r";\s*Occupancy:\s*(\d+)",
    "lds_bytes": r";\s*LDSByteSize:\s*(\d+)",
}
# fields of a kernel in the code object's AMDGPU metadata
_NOTE = {
    "vgprs": ".vgpr_count",
    "agprs": ".agpr_count",
    "sgprs": ".sgpr_count",
    "vgpr_spills": ".vgpr_spill_count",
    "sgpr_spills": ".sgpr_spill_count",
    "lds_bytes": ".group_segment_fixed_size",
    "scratch_bytes": ".private_segment_fixed_size",
}


def parse_listing(text: str) -> dict[str, int]:
    """Statistics from an AMDGCN assembly listing (the first kernel in it)."""
    stats = {}
    for key, pattern in _LISTING.items():
        match = re.search(pattern, text)
        if match:
            stats[key] = int(match.group(1))
    return stats


def parse_notes(text: str) -> dict[str, dict[str, int]]:
    """Per-kernel statistics from ``llvm-readelf --notes`` of a code object:
    its AMDGPU metadata is a YAML document listing ``amdhsa.kernels``."""
    import textwrap

    import yaml

    # the document may start on the note's own line ("AMDGPU Metadata: ---")
    match = re.search(r"---[ \t]*\n(.*?)^[ \t]*\.\.\.[ \t]*$", text, re.M | re.S)
    if match is None:
        return {}
    doc = yaml.safe_load(textwrap.dedent(match.group(1)))
    kernels = {}
    for kernel in (doc or {}).get("amdhsa.kernels", []):
        stats = {key: int(kernel[field]) for key, field in _NOTE.items() if field in kernel}
        kernels[str(kernel.get(".name"))] = stats
    return kernels


def triton_stats(compiled: Any) -> dict[str, Any]:
    """Statistics of a Triton ``CompiledKernel`` (what a launch returns)."""
    if compiled is None:
        return {"error": "no compiled kernel"}
    stats: dict[str, Any] = {}
    asm = getattr(compiled, "asm", {}) or {}
    if "amdgcn" in asm:
        stats.update(parse_listing(asm["amdgcn"]))
    for attr, key in (("n_regs", "regs"), ("n_spills", "spills")):
        value = getattr(compiled, attr, None)
        if isinstance(value, int):
            stats[key] = value
    shared = getattr(getattr(compiled, "metadata", None), "shared", None)
    if isinstance(shared, int):
        stats.setdefault("lds_bytes", shared)
    return stats or {"error": "the compiled kernel reports nothing"}


def _llvm_tool(name: str) -> str | None:
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidate = Path(rocm) / "llvm" / "bin" / name
    return str(candidate) if candidate.exists() else shutil.which(name)


def hip_library_stats(library: str | os.PathLike, arch: str) -> dict[str, Any]:
    """Per-kernel statistics of the ``arch`` code object in a HIP shared library."""
    objcopy, bundler, readelf = (_llvm_tool(n) for n in ("llvm-objcopy", "clang-offload-bundler", "llvm-readelf"))
    if None in (objcopy, bundler, readelf):
        return {"error": "ROCm's llvm-objcopy, clang-offload-bundler or llvm-readelf not found"}
    with tempfile.TemporaryDirectory() as tmp:
        fatbin, code = Path(tmp) / "fatbin", Path(tmp) / "code.o"
        steps = [
            [objcopy, f"--dump-section=.hip_fatbin={fatbin}", str(library), str(Path(tmp) / "host.so")],
            [bundler, "--type=o", f"--input={fatbin}", f"--targets=hipv4-amdgcn-amd-amdhsa--{arch}",
             f"--output={code}", "--unbundle"],
            [readelf, "--notes", str(code)],
        ]
        out = ""
        for step in steps:
            proc = subprocess.run(step, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                return {"error": f"{Path(step[0]).name}: {proc.stderr.strip()[:300]}"}
            out = proc.stdout
    kernels = parse_notes(out)
    return {"kernels": kernels} if kernels else {"error": "no kernel metadata in the code object"}

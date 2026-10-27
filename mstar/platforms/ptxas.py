"""Which ptxas Triton runs for Blackwell GPUs.

ptxas before 13.1.115 miscompiles a floor division of a periodic index under a
range select on sm_100 and sm_103, as in Inductor's fused interleaved mRoPE
table. Triton 3.7.1 bundles ptxas 13.1.80 for Blackwell; torch bundles a newer
one, but Inductor passes it to Triton only as ``TRITON_PTXAS_PATH``, which
Triton reads for older GPUs.
"""

from __future__ import annotations

import functools
import logging
import os
import re
import subprocess
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

BLACKWELL_PTXAS_ENV = "TRITON_PTXAS_BLACKWELL_PATH"
FIXED_PTXAS = (13, 1, 115)


@functools.cache
def ptxas_version(path: str) -> tuple[int, int, int] | None:
    """(major, minor, build) from ``ptxas --version``; None if it does not run."""
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"V(\d+)\.(\d+)\.(\d+)", out)
    return tuple(int(part) for part in match.groups()) if match else None


def _has_blackwell_gpu() -> bool:
    # nvidia-smi rather than torch.cuda, which would initialize the CUDA runtime
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(cap.split(".")[0] == "10" for cap in out.split())


def _triton_blackwell_ptxas() -> str | None:
    try:
        from triton import knobs

        return knobs.nvidia.knob_descriptors["ptxas_blackwell"].default_path
    except (ImportError, AttributeError, KeyError):
        return None


def _version_str(version: tuple[int, int, int] | None) -> str:
    return ".".join(map(str, version)) if version else "of unknown version"


@functools.cache
def blackwell_ptxas_env() -> dict[str, str]:
    """``TRITON_PTXAS_BLACKWELL_PATH`` set to torch's ptxas, and a matching
    Inductor cache tag, when a Blackwell GPU is present and Triton's bundled
    ptxas for it predates the fix; empty otherwise, or when the variable is
    already set."""
    if not _has_blackwell_gpu():
        return {}
    if path := os.environ.get(BLACKWELL_PTXAS_ENV):
        logger.info(
            "Triton runs ptxas %s for Blackwell (%s from %s)",
            _version_str(ptxas_version(path)), path, BLACKWELL_PTXAS_ENV,
        )
        return {}
    bundled = _triton_blackwell_ptxas()
    bundled_version = ptxas_version(bundled) if bundled else None
    if bundled_version is None:
        return {}
    fixed = _version_str(FIXED_PTXAS)
    if bundled_version >= FIXED_PTXAS:
        logger.info("Triton runs its bundled ptxas %s for Blackwell", _version_str(bundled_version))
        return {}
    torch_ptxas = str(Path(torch.__file__).parent / "bin" / "ptxas")
    torch_version = ptxas_version(torch_ptxas)
    if torch_version is None or torch_version < FIXED_PTXAS:
        logger.warning(
            "Triton runs its bundled ptxas %s for Blackwell, which can miscompile integer floor division; "
            "set %s to ptxas %s or later",
            _version_str(bundled_version), BLACKWELL_PTXAS_ENV, fixed,
        )
        return {}
    logger.info(
        "Triton runs torch's ptxas %s for Blackwell (%s): its bundled ptxas %s predates %s, "
        "which fixes a miscompile of integer floor division",
        _version_str(torch_version), torch_ptxas, _version_str(bundled_version), fixed,
    )
    return {
        BLACKWELL_PTXAS_ENV: torch_ptxas,
        # Inductor's graph cache key ignores the ptxas, so untagged it keeps serving
        # kernels the bundled one compiled. torch reads this at import, so it takes
        # effect in the processes spawned after it is set.
        "TORCH_COMPILE_CACHE_KEY_TAG": f"ptxas-blackwell-{_version_str(torch_version)}",
    }

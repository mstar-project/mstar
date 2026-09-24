"""Docker image build for the reproducible evaluation environment.

The image bakes mstar + the model's pip extra (+ any manually-pinned wheels such
as flash-attn) so every candidate round runs against the same dependency
closure. VibeSys consumes it via ``--docker --docker-image <tag>`` and handles
GPU passthrough itself.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def image_exists(tag: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def build_image(
    dockerfile: Path,
    context: Path,
    tag: str,
    *,
    build_args: dict[str, str] | None = None,
    no_cache: bool = False,
) -> str:
    cmd = ["docker", "build", "-f", str(dockerfile), "-t", tag]
    for key, value in (build_args or {}).items():
        cmd += ["--build-arg", f"{key}={value}"]
    if no_cache:
        cmd.append("--no-cache")
    cmd.append(str(context))
    subprocess.run(cmd, check=True)
    return tag

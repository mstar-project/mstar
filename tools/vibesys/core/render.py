"""Dependency-free template rendering.

Templates use ``string.Template`` (``${var}``) so they never collide with the
braces in the Python/YAML/JSON they emit. ``safe_substitute`` leaves any
unrecognized ``$`` (e.g. a shell ``$PORT``) untouched.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any


def render_text(template: str, ctx: dict[str, Any]) -> str:
    return Template(template).safe_substitute({k: str(v) for k, v in ctx.items()})


def render_to(src: Path, dst: Path, ctx: dict[str, Any]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(render_text(src.read_text(), ctx))


def write(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)

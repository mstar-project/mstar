"""The task-type plugin contract.

A task type knows how to turn a compact per-instance spec (``instances/*.toml``)
into: a VibeSys objective + reference oracle + correctness checker + benchmark
(the *bundle*), plus starter files dropped into the mutable *seed* worktree, plus
the VibeSys ``--input-*`` wiring. Everything else (worktree, docker, handoff) is
shared in ``core/`` and does not vary per task.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.vibesys.core.runner import RunOptions, SynthesisInputs


class TaskType(ABC):
    #: CLI selector, e.g. "add-model".
    name: str
    #: VibeSys domain this task runs under, e.g. "llm-serving".
    domain: str

    @property
    def dir(self) -> Path:
        """Directory of the concrete task module (holds templates/ + instances/)."""
        import inspect

        return Path(inspect.getfile(type(self))).parent

    def instance_path(self, instance: str) -> Path:
        return self.dir / "instances" / f"{instance}.toml"

    @abstractmethod
    def load_spec(self, path: Path) -> Any:
        """Parse an instance TOML into a task-specific spec object."""

    @abstractmethod
    def render_bundle(self, spec: Any, bundle_dir: Path) -> None:
        """Write the read-only bundle: OBJECTIVE.md, reference/, evaluator/."""

    @abstractmethod
    def seed_files(self, spec: Any, seed_dir: Path) -> None:
        """Drop starter files into the mutable worktree seed (may be a no-op)."""

    @abstractmethod
    def synthesis_inputs(self, spec: Any, bundle_dir: Path, seed_dir: Path) -> SynthesisInputs:
        """Map the rendered bundle + seed onto VibeSys ``--input-*`` flags."""

    @abstractmethod
    def run_options(self, spec: Any, exp_name: str, docker_image: str, **overrides: Any) -> RunOptions:
        """Task-appropriate run flags (modality, extra-skills, ...)."""

    def image(self, spec: Any) -> "ImageSpec | None":
        """Docker image to build for the eval env, or None to skip."""
        return None


@dataclass(frozen=True)
class ImageSpec:
    dockerfile: Path
    build_args: dict[str, str]


REGISTRY: dict[str, TaskType] = {}


def register(task: TaskType) -> None:
    if task.name in REGISTRY:
        raise ValueError(f"duplicate task type: {task.name}")
    REGISTRY[task.name] = task

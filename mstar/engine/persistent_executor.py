"""Who runs a captured step: the accelerator graph, or one resident kernel.

``GraphExecutor`` is today's path on every platform: the submodule's compiled
forward, captured and replayed by ``accelerator_graph_runner``. A
``PersistentExecutor`` runs the same step from its ``StepProgram`` as one
resident kernel (on ROCm gfx942 / gfx950, ``mstar/kernels/depth_step.py``).

A deployment opts in per node with ``PersistentStepConfig``; nothing in model
code names the kernel. ``select_executor`` falls back to the graph, logging
why, whenever the platform, the arch, an op of the program or the batch is
outside what the kernel serves, so the opt-in can never strand a model. The
kernel is built and its scratch allocated in ``prepare`` (before capture,
never inside it); ``run`` is a fixed sequence of launches, so the engine's
capture and replay of the step stay what they are.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import torch

from mstar.engine.step_program import OpKind, StepProgram, lower

logger = logging.getLogger(__name__)

SUPPORTED_OPS = frozenset(OpKind)
SUPPORTED_ARCHS = ("gfx942", "gfx950")


@dataclasses.dataclass(frozen=True)
class PersistentStepConfig:
    """Per-node opt-in: which step runs resident, and how its rows are split.

    ``groups`` row groups run without synchronising with each other; with one
    group per XCD (8 on MI300X / MI325X / MI355X) a group's barriers stay in its
    XCD's L2 (``scope="xcd"``), otherwise they are agent scope.
    ``per_cu`` workgroups per compute unit."""
    walk: str
    groups: int = 8
    scope: str = "xcd"
    per_cu: int = 1


@dataclasses.dataclass
class StepInputs:
    """One step's device tensors, static per captured batch size."""
    last_hidden: torch.Tensor
    embed0: torch.Tensor
    emb_sum: torch.Tensor
    codes: torch.Tensor
    seed: int = 0
    forced: torch.Tensor | None = None


class StepExecutor(Protocol):
    name: str

    def prepare(self, bs: int) -> None:
        """Build and allocate for a batch size, before capture."""

    def run(self, inputs: StepInputs) -> None:
        """One step; capturable after ``prepare``."""


class GraphExecutor:
    """The compiled forward, captured and replayed by the engine as it is today."""
    name = "graph"

    def __init__(self, forward: Callable[[StepInputs], Any]):
        self.forward = forward

    def prepare(self, bs: int) -> None:
        del bs

    def run(self, inputs: StepInputs) -> None:
        self.forward(inputs)


class PersistentExecutor:
    """A depth-loop ``StepProgram`` as one resident kernel per step."""
    name = "persistent"

    def __init__(self, program: StepProgram, params: Mapping[str, torch.Tensor], codebook: torch.Tensor,
                 config: PersistentStepConfig, compute_units: int, *, top_k: int, top_p: float,
                 temperature: float):
        self.program, self.params, self.codebook, self.config = program, params, codebook, config
        self.compute_units = compute_units
        self.sampling = (top_k, top_p, temperature)
        self._table: torch.Tensor | None = None
        self._scratch: dict[int, Any] = {}

    @staticmethod
    def unsupported_reason(program: StepProgram, facts, bs: int, config: PersistentStepConfig) -> str | None:
        """Why the kernel cannot run this step here, or None."""
        if facts.platform != "rocm":
            return f"platform {facts.platform}: the resident kernel is a ROCm build"
        if facts.arch not in SUPPORTED_ARCHS:
            return f"arch {facts.arch}: built for {', '.join(SUPPORTED_ARCHS)}"
        unknown = {op.kind for op in program.ops} - SUPPORTED_OPS
        if unknown:
            return f"ops {sorted(k.name for k in unknown)} have no resident implementation"
        if config.scope == "xcd" and facts.xcds and config.groups != facts.xcds:
            return f"scope xcd needs one group per XCD ({facts.xcds}), got {config.groups}"
        from mstar.kernels.depth_step import check_shape

        return check_shape(program.shape, bs, config.groups)

    def prepare(self, bs: int) -> None:
        from mstar.kernels import depth_step as ds

        ds.extension().load()
        if self._table is None:
            self._table = lower(self.program, self.params, self.codebook.device)
        if bs not in self._scratch:
            self._scratch[bs] = ds.DepthStepScratch.allocate(self.program.shape, bs, self.config.groups,
                                                             self.program.ss_slots, self.codebook.device)

    def run(self, inputs: StepInputs) -> None:
        from mstar.kernels import depth_step as ds

        bs = inputs.last_hidden.shape[0]
        top_k, top_p, temperature = self.sampling
        ds.launch(self._table, self._rope(), inputs.last_hidden, inputs.embed0, self.codebook, inputs.emb_sum,
                  inputs.codes, self._scratch[bs], groups=self.config.groups,
                  scope=ds.SCOPE_XCD if self.config.scope == "xcd" else ds.SCOPE_AGENT,
                  grid=ds.default_grid(self.compute_units, self.config.groups, self.config.per_cu),
                  shape=self.program.shape, top_k=top_k, top_p=top_p, temperature=temperature, seed=inputs.seed,
                  forced=inputs.forced)

    def _rope(self) -> torch.Tensor:
        if not hasattr(self, "_rope_table"):
            from mstar.engine.step_program import rope_table

            self._rope_table = rope_table(self.program.shape, self.codebook.device)
        return self._rope_table


def select_executor(program: StepProgram, facts, bs: int, config: PersistentStepConfig | None,
                    graph: GraphExecutor, make_persistent: Callable[[], PersistentExecutor]) -> StepExecutor:
    """The persistent executor when the node asked for it and the kernel serves
    the step here, else the graph (with the reason logged once)."""
    if config is None:
        return graph
    reason = PersistentExecutor.unsupported_reason(program, facts, bs, config)
    if reason is not None:
        logger.info("walk %s: resident step unavailable (%s); replaying the accelerator graph", config.walk, reason)
        return graph
    return make_persistent()

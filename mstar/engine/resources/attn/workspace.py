"""FlashInfer's workspaces: the scratch a plan reserves its split-KV partials in."""

import logging
import os
import re
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024

# How a plan reports a short workspace. The paged wrappers can also be asked ahead of a plan
# (`workspace_size`) but the ragged one can't, so this is the report every wrapper shares. A
# FlashInfer that words it differently fails loudly with its own message (`Workspace.plan`).
_OVERFLOW = re.compile(
    r"Buffer overflow when allocating memory for (\w+) with size (\d+) "
    r"and alignment \d+, but only (\d+) bytes available"
)
# What a plan takes from the workspace we pass. The rest of its allocations are in the int
# workspace FlashInfer allocates itself, far larger than a plan's indices.
_FLOAT_ALLOCATIONS = ("tmp_v", "tmp_s", "partial_o", "partial_lse")


def _bytes_asked(exc: RuntimeError, tensor: torch.Tensor) -> int | None:
    """The bytes of ``tensor`` that would have carried a plan past ``exc``; None if ``exc``
    isn't a short workspace. Includes the allocation after the one that failed: each partial
    output is followed by its log-sum-exp, at most 1/32 of its size at any head dim."""
    match = _OVERFLOW.search(str(exc))
    if match is None or not match[1].endswith(_FLOAT_ALLOCATIONS):
        return None
    size, left = int(match[2]), int(match[3])
    return tensor.numel() - left + size + size // 32 + 4096


class Workspace:
    """One FlashInfer float workspace, which wrappers plan against through `plan`.

    What a plan needs depends on its rows, segments and heads and on the GPU's SM count, and
    under a graph it is the same for every plan of a wrapper. So a workspace starts at its
    floor and the first plan that overflows it grows it. ``tensor`` keeps its identity across
    that, so every wrapper on it moves too and the key holds one buffer, as large as the
    largest plan made on it.

    A captured graph records the buffer's address, so once `seal`ed it can't grow, and a plan
    that doesn't fit raises what it asked for and which setting would have made room.
    """

    __slots__ = ("tensor", "name", "sealed")

    def __init__(self, tensor: torch.Tensor | None, name: str, sealed: bool = False):
        self.tensor = tensor
        self.name = name
        self.sealed = sealed

    @classmethod
    def of(cls, buffer: "torch.Tensor | Workspace | None") -> "Workspace":
        """``buffer`` as a workspace. A bare tensor's size is its caller's to decide."""
        return buffer if isinstance(buffer, cls) else cls(buffer, "passed in by its caller", sealed=True)

    def seal(self) -> None:
        self.sealed = True

    def plan(self, plan: Callable, *args, **kwargs):
        """``plan(*args, **kwargs)``, retried once in a buffer that fits if it overflows this one."""
        try:
            return plan(*args, **kwargs)
        except RuntimeError as exc:
            asked = _bytes_asked(exc, self.tensor)
            if asked is None:
                raise
            self._grow(asked, exc)
        try:
            return plan(*args, **kwargs)
        except RuntimeError as exc:
            asked = _bytes_asked(exc, self.tensor)
            if asked is None:
                raise
            raise RuntimeError(self._short(asked, self.tensor.numel(), "growing it once did not make room")) from exc

    def _grow(self, asked: int, cause: RuntimeError) -> None:
        have = self.tensor.numel()
        if self.sealed:
            why = "it can't grow once a captured graph can hold its address"
            raise RuntimeError(self._short(asked, have, why)) from cause
        size = -(-asked // _MIB) * _MIB
        device = self.tensor.device
        if device.type == "cuda":
            # any kernel still reading it is done before its memory goes back to the allocator
            torch.cuda.synchronize(device)
        # the old buffer goes first, which leaves room on a GPU that fits one but not both
        self.tensor.data = self.tensor.new_empty(0)
        try:
            self.tensor.data = torch.empty(size, dtype=torch.uint8, device=device)
        except torch.cuda.OutOfMemoryError as oom:
            raise RuntimeError(self._short(asked, have, f"the GPU has no room to grow it: {oom}")) from cause
        logger.info(
            "FlashInfer workspace %s: %d -> %d MiB, a plan asked for %d bytes",
            self.name, have // _MIB, size // _MIB, asked,
        )

    def _short(self, asked: int, have: int, why: str) -> str:
        need_mib = -(-asked // _MIB)
        return (
            f"FlashInfer workspace {self.name}: a plan asks for {asked} bytes ({need_mib} MiB) and it holds "
            f"{have} bytes; {why}. Workspaces start at MSTAR_WORKSPACE_BUFFER_MB, or a ragged resource's "
            f"workspace_mb if that is larger: set it to at least {need_mib}."
        )


def workspace_bytes(declared_mb: int | None = None) -> int:
    """The floor of one FlashInfer workspace: ``MSTAR_WORKSPACE_BUFFER_MB`` (default 512), or
    what its resource declares. Both are floors, so with both set the larger holds, and the
    env var reaches a resource whose declaration isn't a deployment's to edit."""
    env = os.environ.get("MSTAR_WORKSPACE_BUFFER_MB")
    if declared_mb is None:
        return int(env or 512) * _MIB
    return max(declared_mb, int(env or 0)) * _MIB


class WorkspacePool:
    """FlashInfer workspaces, one per (plan label, cg slot).

    Persistent: a wrapper is planned against the buffer it was built with, so these outlive
    any single step. ``owner`` names the resource in what a workspace reports.
    """

    def __init__(self, device: torch.device, size_mb: int | None = None, owner: str = "attention"):
        self._device = device
        self._owner = owner
        self._size = workspace_bytes(size_mb)
        self._workspaces: dict[str, Workspace] = {}

    def get(self, label: str, cg_slot: int | None = None) -> Workspace:
        key = label if cg_slot is None else f"{label}_{cg_slot}"
        workspace = self._workspaces.get(key)
        if workspace is None:
            workspace = self._workspaces[key] = Workspace(
                torch.empty(self._size, dtype=torch.uint8, device=self._device), f"{self._owner}/{key}",
            )
        return workspace

    def seal(self) -> None:
        """The workspaces here now keep their addresses, for the graphs about to be captured
        over them. One made later has no graph over it, so it stays free to grow."""
        for workspace in self._workspaces.values():
            workspace.seal()
        if self._workspaces:
            logger.info(
                "FlashInfer workspaces of %s: %d, %d MiB in all", self._owner, len(self._workspaces),
                sum(workspace.tensor.numel() for workspace in self._workspaces.values()) // _MIB,
            )

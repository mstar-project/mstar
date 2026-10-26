"""Shared by the CPU tests of LoRA on a model's modules.

``ThreadGroup`` simulates a tensor-parallel group on one CPU (a thread per rank, the
collectives meeting at a barrier), so a model's own parallel layers run sharded,
unmodified. Import by module name (``from lora_module_helpers import ...``).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager

import torch
from safetensors.torch import save_file

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.lora import LoraReqConfig, LoraSpec, LoraStep
from mstar.engine.resources.lora.pool import LoraAdapterPool
from mstar.engine.resources.step import Segment, StepContext

DTYPE = torch.float64


class _Exchange:
    def __init__(self, world: int):
        self.barrier = threading.Barrier(world)
        self.slots: list = [None] * world

    def share(self, rank: int, value):
        self.slots[rank] = value
        self.barrier.wait()
        out = list(self.slots)
        self.barrier.wait()
        return out


class ThreadGroup(CommGroup):
    """One rank of a simulated group: collectives across the ranks' threads, summed in
    rank order (deterministic)."""

    def __init__(self, rank: int, world: int, exchange: _Exchange):
        super().__init__(my_global_rank=rank, my_group_rank=rank, group_members=list(range(world)))
        self._exchange = exchange

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        parts = self._exchange.share(self.rank, input_)
        out = parts[0].clone()
        for p in parts[1:]:
            out += p
        return out

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return torch.cat(self._exchange.share(self.rank, input_), dim=dim)


def run_ranks(world: int, fn: Callable[[int, CommGroup], object]) -> list:
    """``fn(rank, group)`` on every rank of a simulated group, concurrently; the results
    in rank order. A rank's exception aborts the others' collectives and is raised."""
    exchange = _Exchange(world)
    results: list = [None] * world
    errors: list = []

    def body(rank: int):
        try:
            results[rank] = fn(rank, ThreadGroup(rank, world, exchange))
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)
            exchange.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


def rand(*shape, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=DTYPE) * scale


def write_peft(path, modules: dict[str, tuple[torch.Tensor, torch.Tensor]], alpha: float | None = None,
               embedding: tuple[str, ...] = (), prefix: str = "base_model.model.") -> str:
    """A PEFT-layout adapter: ``lora_A`` / ``lora_B`` per module (``lora_embedding_A`` /
    ``_B`` for ``embedding``), with an ``alpha`` key per module when given (else scaling 1)."""
    sd = {}
    for module, (a, b) in modules.items():
        if module in embedding:
            sd[f"{prefix}{module}.lora_embedding_A"] = a.float()
            sd[f"{prefix}{module}.lora_embedding_B"] = b.float()
        else:
            sd[f"{prefix}{module}.lora_A.weight"] = a.float()
            sd[f"{prefix}{module}.lora_B.weight"] = b.float()
        if alpha is not None:
            sd[f"{prefix}{module}.alpha"] = torch.tensor(float(alpha))
    save_file(sd, str(path))
    return str(path)


def serve(spec: LoraSpec, adapters: dict[str, str], max_rank: int = 4, **kw) -> LoraSpec:
    """``spec`` sized as a deployment's YAML would, on the CPU reference backend in fp64."""
    spec.apply_yaml_overrides(max_slots=2, max_rank=max_rank, adapters=dict(adapters), **kw)
    spec.dtype = DTYPE
    spec.backend = "torch"
    return spec


def pool(spec: LoraSpec, rank: int = 0, world: int = 1) -> LoraAdapterPool:
    return LoraAdapterPool(spec, "cpu", rank=rank, world_size=world)


@contextmanager
def compute(*pools):
    """The engine's compute scope around a planned step's preprocess and forward
    (``StepRunner.compute_scope``), on the given pools: deltas called outside one raise."""
    for p in pools:
        p.enter_compute()
    try:
        yield
    finally:
        for p in pools:
            p.exit_compute()


def plan(pools, rows: list[tuple[str, str | None, int]]) -> None:
    """Ingest ``rows`` ((rid, adapter, span)) into each pool, then admit and plan one step
    over them, as the engine drives a step whose segments are one per request."""
    for p in pools:
        for rid, adapter, _ in rows:
            p.ingest_request(rid, LoraReqConfig(adapter=adapter))
        step = LoraStep(segments=tuple(Segment(rid, "main", n) for rid, _, n in rows))
        ctx = StepContext(request_ids=[r for r, _, _ in rows], graph_walk="prefill", slot=0, capture=False)
        assert p.admit(step, ctx).ok
        p.plan(step, ctx)


def admit_declared(lora_pool, step, rows, walk: str = "prefill") -> None:
    """Admit and plan a node's own declared ``step`` (its LoraStep under the pool's key)
    for ``rows`` ((rid, adapter, span))."""
    for rid, adapter, _ in rows:
        lora_pool.ingest_request(rid, LoraReqConfig(adapter=adapter))
    ctx = StepContext(request_ids=[r for r, _, _ in rows], graph_walk=walk, slot=0, capture=False)
    lora_step = step.get(lora_pool.spec.resource_key)
    assert lora_step is not None, "the node's declare_step carries no LoraStep"
    assert lora_pool.admit(lora_step, ctx).ok
    lora_pool.plan(lora_step, ctx)


# -- adapters and merged weights --------------------------------------------------------------

def make_adapter(path, modules: dict[str, tuple[int, int]], rank: int, alpha: float | None, seed: int,
                 embedding: tuple[str, ...] = (), rename: Callable[[str], str] = lambda m: m):
    """A random PEFT adapter over ``modules`` (checkpoint module -> (out, in)); an
    ``embedding`` module's A is ``[r, vocab]`` (``in`` is the vocabulary). The file names
    each module ``rename(module)``; the returned reference ``{module: (A, B, scaling)}``
    names them as the checkpoint does. Factors are fp32-exact, as the file holds them."""
    tensors, ref = {}, {}
    scaling = 1.0 if alpha is None else alpha / rank
    for i, (module, (out, inp)) in enumerate(modules.items()):
        a = rand(rank, inp, seed=seed + 2 * i, scale=0.3).float().double()
        b = rand(out, rank, seed=seed + 2 * i + 1, scale=0.3).float().double()
        tensors[rename(module)] = (a, b)
        ref[module] = (a, b, scaling)
    renamed = tuple(rename(m) for m in embedding)
    return write_peft(path, tensors, alpha=alpha, embedding=renamed), ref


def merged(base: dict[str, torch.Tensor], ref, embedding: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
    """``base`` with the adapter folded in: ``W + scaling * B @ A`` (an embedding's
    ``[vocab, hidden]`` weight takes its transpose)."""
    out = dict(base)
    for module, (a, b, s) in ref.items():
        delta = s * (b @ a)
        out[module + ".weight"] = out[module + ".weight"] + (delta.t() if module in embedding else delta)
    return out

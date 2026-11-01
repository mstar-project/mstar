"""Only expert parallelism syncs MoE routing across ranks.

Under tensor parallelism each rank runs all k slots of a token on its slice of
every expert, so ranks whose routers disagree still all-reduce tensors of the
same shape. The lockstep test forces such a disagreement over a 2-rank gloo
group: the forward completes, and both ranks end with the same output, which
mixes two experts' slices in the slot they disagreed on. ``_dispatch_tp`` runs
the Triton kernels, so on CPU they are swapped for the naive dispatch.
"""

from __future__ import annotations

import sys
import types
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components import (
    ExpertParallelSparseMoeBlock,
    ParallelSparseMoeBlock,
    ParallelSparseMoeBlockWithSharedExpert,
    dispatch_experts_fused,
)

HIDDEN, INTER, NUM_EXPERTS, TOP_K, TOKENS, WORLD = 32, 16, 8, 2, 6, 2


def _naive_kernels() -> types.ModuleType:
    """``mstar.utils.fused_moe`` as ``_dispatch_tp`` imports it, on the naive CPU dispatch."""

    def fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, reduce_results=True, skip_invalid=False):
        out = dispatch_experts_fused(
            hidden_states, w1, w2, w1.shape[0], topk_ids, topk_weights, skip_invalid=skip_invalid,
        )
        return out if reduce_results else out[:, None]

    def moe_sum_reduce_triton(partials, out, routed_scaling_factor=1.0):
        out.copy_(partials.sum(dim=1) * routed_scaling_factor)

    module = types.ModuleType("mstar.utils.fused_moe")
    module.fused_experts, module.moe_sum_reduce_triton = fused_experts, moe_sum_reduce_triton
    return module


def _randn(*shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


class _Router(nn.Module):
    """Softmax top-k. With ``flip``, odd tokens take their (k+1)-th expert for their
    k-th, at the k-th's weight, as a rank resolving a near-tie the other way would."""

    def __init__(self, flip: bool = False):
        super().__init__()
        self.flip = flip
        self.weight = nn.Parameter(_randn(NUM_EXPERTS, HIDDEN, seed=1))

    def forward(self, hidden_states, router_states=None):
        probs = F.softmax(F.linear(hidden_states, self.weight), dim=-1)
        weights, experts = torch.topk(probs, TOP_K + 1, dim=-1)
        weights, picks = weights[:, :TOP_K], experts[:, :TOP_K].clone()
        if self.flip:
            picks[1::2, -1] = experts[1::2, TOP_K]
        return weights / weights.sum(dim=-1, keepdim=True), picks, None


def _checkpoint() -> dict[str, torch.Tensor]:
    shapes = {"gate": (INTER, HIDDEN), "up": (INTER, HIDDEN), "down": (HIDDEN, INTER)}
    return {
        f"{kind}:{e}": _randn(*shape, seed=10 + 3 * e + i) * 0.1
        for e in range(NUM_EXPERTS) for i, (kind, shape) in enumerate(shapes.items())
    }


def _block(cls, comm_group: CommGroup, flip: bool = False) -> nn.Module:
    """``cls`` holding ``comm_group``'s share of ``_checkpoint()``, routed by ``_Router(flip)``."""
    shared = cls is ParallelSparseMoeBlockWithSharedExpert
    kwargs = {"shared_expert": nn.Linear(HIDDEN, HIDDEN, bias=False)} if shared else {}
    block = cls(HIDDEN, NUM_EXPERTS, TOP_K, INTER, router=_Router(flip), comm_group=comm_group, **kwargs)
    with torch.no_grad():
        if shared:
            block.shared_expert.weight.copy_(_randn(HIDDEN, HIDDEN, seed=2) * 0.1)
            block.shared_expert_gate.weight.copy_(_randn(1, HIDDEN, seed=3))
        for shard_id, w in _checkpoint().items():
            param = block.experts.down_proj if shard_id.startswith("down") else block.experts.gate_up_proj
            param.weight_loader(param, w, shard_id)
    return block


def _steps() -> list[torch.Tensor]:
    x = _randn(TOKENS, HIDDEN, seed=4)
    return [x, x[:1], x]


class _RecordingGroup(CommGroup):
    """Rank 0 of two; records the shape of each all-reduce and returns its input."""

    def __init__(self):
        super().__init__(my_global_rank=0, my_group_rank=0, group_members=[0, 1])
        self.reduced = []

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        self.reduced.append(tuple(input_.shape))
        return input_


@pytest.mark.parametrize("cls, syncs_routing", [
    (ParallelSparseMoeBlock, False),
    (ParallelSparseMoeBlockWithSharedExpert, False),
    (ExpertParallelSparseMoeBlock, True),
])
def test_only_expert_parallel_syncs_routing(monkeypatch, cls, syncs_routing):
    monkeypatch.setitem(sys.modules, "mstar.utils.fused_moe", _naive_kernels())
    group = _RecordingGroup()
    with torch.no_grad():
        _block(cls, group)(_randn(TOKENS, HIDDEN, seed=5))
    routing = [(TOKENS, 2 * TOP_K)] if syncs_routing else []
    assert group.reduced == [*routing, (TOKENS, HIDDEN)]


def _lockstep_worker(rank: int, cls, store: str, out_dir: str) -> None:
    # A collective one rank skips fails at this timeout instead of hanging the test.
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD, timeout=timedelta(seconds=60),
    )
    torch.set_num_threads(1)
    sys.modules["mstar.utils.fused_moe"] = _naive_kernels()
    try:
        group = CommGroup(rank, rank, list(range(WORLD)))
        group.device_group = dist.group.WORLD
        block = _block(cls, group, flip=rank == 1)
        with torch.no_grad():
            torch.save([block(x) for x in _steps()], f"{out_dir}/{rank}.pt")
    finally:
        dist.destroy_process_group()


def _expert_slice(ckpt, expert: int, rank: int, x: torch.Tensor) -> torch.Tensor:
    """Expert ``expert`` on ``x`` over rank ``rank``'s slice of its intermediate dim."""
    rows = slice(rank * INTER // WORLD, (rank + 1) * INTER // WORLD)
    gate, up = ckpt[f"gate:{expert}"][rows], ckpt[f"up:{expert}"][rows]
    return (F.silu(x @ gate.T) * (x @ up.T)) @ ckpt[f"down:{expert}"][:, rows].T


@pytest.mark.parametrize("cls", [ParallelSparseMoeBlock, ParallelSparseMoeBlockWithSharedExpert])
def test_tp_ranks_that_route_differently_stay_in_lockstep(tmp_path, cls):
    mp.spawn(_lockstep_worker, args=(cls, str(tmp_path / "store"), str(tmp_path)), nprocs=WORLD)
    outs = [torch.load(tmp_path / f"{rank}.pt") for rank in range(WORLD)]

    reference, ckpt = _block(cls, CommGroup.trivial()), _checkpoint()
    for x, out0, out1 in zip(_steps(), *outs, strict=True):
        # The all-reduce leaves every rank with the same hidden state.
        assert torch.equal(out0, out1)
        with torch.no_grad():
            expected = reference(x)
            weights, picks0, _ = reference.gate(x)
            _, picks1, _ = _Router(flip=True)(x)
            # Where rank 1 picked another expert, its half of that slot comes from its own pick.
            for t, j in (picks0 != picks1).nonzero().tolist():
                own, synced = (_expert_slice(ckpt, int(p[t, j]), 1, x[t]) for p in (picks1, picks0))
                expected[t] += weights[t, j] * (own - synced)
        torch.testing.assert_close(out0, expected, atol=1e-5, rtol=1e-5)

"""Single-process parity tests for :class:`ExpertParallelSparseMoeBlock`.

EP output is a sum over expert shards, so each test builds all P rank
blocks in one process with a fake ``CommGroup`` whose all-reduce is the
identity, sums their outputs, and compares with an unsharded
:class:`SparseMoeBlock`. The naive path runs on CPU; the fused Triton path
runs only when CUDA is available.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components import ExpertParallelSparseMoeBlock, ParallelSparseMoeBlock, SparseMoeBlock

HIDDEN, INTER, NUM_EXPERTS, TOP_K = 64, 32, 16, 4

_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused MoE requires CUDA")
_DEVICES = [
    pytest.param(("cpu", torch.float32), id="naive-cpu"),
    pytest.param(("cuda", torch.bfloat16), id="fused-cuda", marks=_needs_cuda),
]


class _FakeCommGroup(CommGroup):
    """One rank of a P-rank group whose all-reduce is the identity.

    The test sums the rank outputs itself, which is what the real
    all-reduce would compute.
    """

    def __init__(self, rank: int, world_size: int):
        super().__init__(my_global_rank=rank, my_group_rank=rank, group_members=list(range(world_size)))
        self.all_reduce_calls = 0

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        self.all_reduce_calls += 1
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Every peer routed alike; a test overrides this to disagree.
        return torch.cat([input_] * self.world_size, dim=dim)


class _StatefulRouter(nn.Module):
    """Stateful router: the next state is the running sum of inputs seen."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def forward(self, hidden_states, router_states=None):
        # Bias the logits by the state so routing depends on it.
        base = router_states if router_states is not None else torch.zeros_like(hidden_states)
        logits = F.linear(hidden_states + base, self.weight)
        probs = F.softmax(logits.float(), dim=-1)
        weights, experts = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(hidden_states.dtype), experts, base + hidden_states


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _checkpoint(num_experts=NUM_EXPERTS):
    """Per-expert HF-style weights keyed by the shard ids the loaders take."""
    ckpt = {}
    for e in range(num_experts):
        ckpt[f"gate:{e}"] = torch.randn(INTER, HIDDEN) * 0.1
        ckpt[f"up:{e}"] = torch.randn(INTER, HIDDEN) * 0.1
        ckpt[f"down:{e}"] = torch.randn(HIDDEN, INTER) * 0.1
    return ckpt


def _load(block, ckpt):
    """Feed every expert to the block's loaders, as the model loader does."""
    for shard_id, w in ckpt.items():
        param = block.experts.down_proj if shard_id.startswith("down") else block.experts.gate_up_proj
        param.weight_loader(param, w.to(param.device, param.dtype), shard_id)


def _reference(ckpt, router, router_w, device, dtype):
    ref = SparseMoeBlock(HIDDEN, NUM_EXPERTS, TOP_K, INTER, router=router)
    with torch.no_grad():
        ref.gate.weight.copy_(router_w)
        for e in range(NUM_EXPERTS):
            ref.experts.gate_up_proj[e] = torch.cat([ckpt[f"gate:{e}"], ckpt[f"up:{e}"]])
            ref.experts.down_proj[e] = ckpt[f"down:{e}"]
    return ref.to(device, dtype)


def _ep_blocks(world_size, ckpt, router_factory, router_w, device, dtype):
    blocks = []
    for rank in range(world_size):
        block = ExpertParallelSparseMoeBlock(
            HIDDEN, NUM_EXPERTS, TOP_K, INTER,
            router=router_factory(), comm_group=_FakeCommGroup(rank, world_size),
        )
        # The router is replicated on every rank.
        with torch.no_grad():
            block.gate.weight.copy_(router_w)
        block = block.to(device, dtype)
        _load(block, ckpt)
        blocks.append(block)
    return blocks


def _tol(dtype):
    return {"atol": 1e-5, "rtol": 1e-5} if dtype == torch.float32 else {"atol": 2e-2, "rtol": 2e-2}


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_tokens", [1, 7, 300])
@pytest.mark.parametrize("dev", _DEVICES)
def test_ep_sum_matches_unsharded(world_size, num_tokens, dev):
    device, dtype = dev
    ckpt = _checkpoint()
    router_w = torch.randn(NUM_EXPERTS, HIDDEN)
    blocks = _ep_blocks(world_size, ckpt, lambda: None, router_w, device, dtype)
    ref = _reference(ckpt, None, router_w, device, dtype)

    x = torch.randn(2, num_tokens, HIDDEN, device=device, dtype=dtype)
    with torch.no_grad():
        outs = [b(x) for b in blocks]
        expected = ref(x)

    for b, out in zip(blocks, outs, strict=True):
        assert out.shape == x.shape
        assert b.comm_group.all_reduce_calls == (0 if world_size == 1 else 1)
    total = torch.stack([o.float() for o in outs]).sum(0).to(dtype)
    torch.testing.assert_close(total, expected, **_tol(dtype))


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dev", _DEVICES)
def test_ep_rank_output_is_its_own_experts_only(world_size, dev):
    """A rank whose experts no token picks contributes exactly zero."""
    device, dtype = dev
    # Route every token to experts 0..TOP_K-1, all owned by rank 0.
    router_w = torch.zeros(NUM_EXPERTS, HIDDEN)
    router_w[:TOP_K] = 1.0
    blocks = _ep_blocks(world_size, _checkpoint(), lambda: None, router_w, device, dtype)
    x = torch.rand(5, HIDDEN, device=device, dtype=dtype)
    with torch.no_grad():
        outs = [b(x) for b in blocks]
    assert outs[0].abs().sum() > 0
    for out in outs[1:]:
        assert (out == 0).all()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dev", _DEVICES)
def test_ep_stateful_router(world_size, dev):
    """Router state is computed identically on every rank and threads through."""
    device, dtype = dev
    ckpt = _checkpoint()
    router_w = torch.randn(NUM_EXPERTS, HIDDEN)

    def router_factory():
        return _StatefulRouter(HIDDEN, NUM_EXPERTS, TOP_K)

    blocks = _ep_blocks(world_size, ckpt, router_factory, router_w, device, dtype)
    ref = _reference(ckpt, router_factory(), router_w, device, dtype)

    x = torch.randn(6, HIDDEN, device=device, dtype=dtype)
    states = [None] * world_size
    ref_state = None
    with torch.no_grad():
        for _ in range(2):
            results = [b(x, router_states=s, return_router_states=True) for b, s in zip(blocks, states, strict=True)]
            expected, ref_state = ref(x, router_states=ref_state, return_router_states=True)
            states = [s for _, s in results]
            for s in states:
                torch.testing.assert_close(s, ref_state)
            total = torch.stack([o.float() for o, _ in results]).sum(0).to(dtype)
            torch.testing.assert_close(total, expected, **_tol(dtype))


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_ep_loader_keeps_only_local_experts(world_size):
    ckpt = _checkpoint()
    e_local = NUM_EXPERTS // world_size
    for rank in range(world_size):
        block = ExpertParallelSparseMoeBlock(
            HIDDEN, NUM_EXPERTS, TOP_K, INTER, comm_group=_FakeCommGroup(rank, world_size),
        )
        assert block.experts.gate_up_proj.shape == (e_local, 2 * INTER, HIDDEN)
        assert block.experts.down_proj.shape == (e_local, HIDDEN, INTER)
        block.experts.gate_up_proj.data.fill_(float("nan"))
        block.experts.down_proj.data.fill_(float("nan"))

        # Non-local experts are skipped and leave the params untouched.
        foreign = {k: v for k, v in ckpt.items() if not rank * e_local <= int(k.split(":")[1]) < (rank + 1) * e_local}
        _load(block, foreign)
        assert block.experts.gate_up_proj.isnan().all()
        assert block.experts.down_proj.isnan().all()

        _load(block, ckpt)
        for local in range(e_local):
            e = rank * e_local + local
            torch.testing.assert_close(block.experts.gate_up_proj[local, :INTER], ckpt[f"gate:{e}"])
            torch.testing.assert_close(block.experts.gate_up_proj[local, INTER:], ckpt[f"up:{e}"])
            torch.testing.assert_close(block.experts.down_proj[local], ckpt[f"down:{e}"])


def test_ep_loaders_survive_module_apply():
    block = ExpertParallelSparseMoeBlock(
        HIDDEN, NUM_EXPERTS, TOP_K, INTER, comm_group=_FakeCommGroup(1, 2),
    ).to(torch.float64)
    ckpt = _checkpoint()
    _load(block, ckpt)
    torch.testing.assert_close(block.experts.down_proj[0], ckpt[f"down:{NUM_EXPERTS // 2}"].double())


def test_ep_rejects_uneven_experts():
    with pytest.raises(AssertionError, match="not divisible"):
        ExpertParallelSparseMoeBlock(HIDDEN, 10, TOP_K, INTER, comm_group=_FakeCommGroup(0, 4))


def _routing(block, x):
    _, ids, _ = block.gate(x.view(-1, HIDDEN))
    return ids


@pytest.mark.parametrize("dev", _DEVICES)
def test_ep_expert_load_counts_every_expert(dev):
    """Rank 0 counts slots for all experts, including other ranks' ones."""
    device, dtype = dev
    router_w = torch.randn(NUM_EXPERTS, HIDDEN)
    block = ExpertParallelSparseMoeBlock(
        HIDDEN, NUM_EXPERTS, TOP_K, INTER, comm_group=_FakeCommGroup(0, 2), track_expert_load=True,
    )
    with torch.no_grad():
        block.gate.weight.copy_(router_w)
    block = block.to(device, dtype)
    _load(block, _checkpoint())

    x1, x2 = (torch.randn(n, HIDDEN, device=device, dtype=dtype) for n in (5, 9))
    with torch.no_grad():
        block(x1)
        block(x2)
        ids = torch.cat([_routing(block, x1), _routing(block, x2)])
    expected = torch.bincount(ids.flatten().cpu(), minlength=NUM_EXPERTS)
    assert torch.equal(block.pop_expert_load(), expected)
    assert block.pop_expert_load().sum() == 0


def test_ep_expert_load_zeroed_after_to_empty():
    with torch.device("meta"):
        block = ExpertParallelSparseMoeBlock(HIDDEN, NUM_EXPERTS, TOP_K, INTER, track_expert_load=True)
    block.to_empty(device="cpu")
    assert block.expert_load.dtype == torch.int64
    assert (block.expert_load == 0).all()
    # Not a checkpoint tensor, so the loader never sees it.
    assert "expert_load" not in block.state_dict()


def test_ep_expert_load_off_by_default():
    block = ExpertParallelSparseMoeBlock(HIDDEN, NUM_EXPERTS, TOP_K, INTER)
    assert not hasattr(block, "expert_load")


def test_ep_routing_check_passes_when_ranks_agree():
    block = ExpertParallelSparseMoeBlock(
        HIDDEN, NUM_EXPERTS, TOP_K, INTER, comm_group=_FakeCommGroup(1, 4), debug_check_routing=True,
    )
    _load(block, _checkpoint())
    with torch.no_grad():
        block(torch.randn(6, HIDDEN))


def test_ep_routing_check_catches_disagreement():
    class _Disagreeing(_FakeCommGroup):
        def all_gather(self, input_, dim=-1):
            out = super().all_gather(input_, dim)
            # The last peer picked a different expert for its first slot.
            out[-input_.shape[0]] = (out[-input_.shape[0]] + 1) % NUM_EXPERTS
            return out

    block = ExpertParallelSparseMoeBlock(
        HIDDEN, NUM_EXPERTS, TOP_K, INTER, comm_group=_Disagreeing(0, 2), debug_check_routing=True,
    )
    _load(block, _checkpoint())
    with torch.no_grad(), pytest.raises(RuntimeError, match="routing disagrees"):
        block(torch.randn(3, HIDDEN))


def _rank_blocks(cls, world_size, ckpt, router_w, device, dtype, **kwargs):
    blocks = []
    for rank in range(world_size):
        block = cls(HIDDEN, NUM_EXPERTS, TOP_K, INTER, comm_group=_FakeCommGroup(rank, world_size), **kwargs)
        with torch.no_grad():
            block.gate.weight.copy_(router_w)
        block = block.to(device, dtype)
        _load(block, ckpt)
        blocks.append(block)
    return blocks


_CPU, _CUDA = ("cpu", torch.float32), ("cuda", torch.bfloat16)
_SCHEMES = [
    pytest.param(ExpertParallelSparseMoeBlock, 1, _CPU, id="ep1-naive-cpu"),
    pytest.param(ExpertParallelSparseMoeBlock, 4, _CPU, id="ep4-naive-cpu"),
    pytest.param(ExpertParallelSparseMoeBlock, 4, _CUDA, id="ep4-fused-cuda", marks=_needs_cuda),
    pytest.param(ParallelSparseMoeBlock, 1, _CPU, id="tp1-naive-cpu"),
    pytest.param(ParallelSparseMoeBlock, 1, _CUDA, id="tp1-fused-cuda", marks=_needs_cuda),
    # TP > 1 dispatches through the fused kernel only.
    pytest.param(ParallelSparseMoeBlock, 4, _CUDA, id="tp4-fused-cuda", marks=_needs_cuda),
]


@pytest.mark.parametrize("cls, world_size, dev", _SCHEMES)
def test_padding_rows_skip_every_gemm(cls, world_size, dev):
    """Masked rows contribute zero; real rows match the unmasked forward."""
    device, dtype = dev
    blocks = _rank_blocks(cls, world_size, _checkpoint(), torch.randn(NUM_EXPERTS, HIDDEN), device, dtype)
    x = torch.randn(9, HIDDEN, device=device, dtype=dtype)
    num_real = 6
    valid = torch.arange(9, device=device) < torch.tensor([num_real], device=device)
    with torch.no_grad():
        full = torch.stack([b(x).float() for b in blocks]).sum(0)
        masked = torch.stack([b(x, token_valid=valid).float() for b in blocks]).sum(0)
    torch.testing.assert_close(masked[:num_real], full[:num_real], **_tol(dtype))
    assert (masked[num_real:] == 0).all()


def test_ep_expert_load_ignores_padding():
    router_w = torch.randn(NUM_EXPERTS, HIDDEN)
    (block,) = _rank_blocks(
        ExpertParallelSparseMoeBlock, 1, _checkpoint(), router_w, "cpu", torch.float32, track_expert_load=True,
    )
    x = torch.randn(7, HIDDEN)
    valid = torch.tensor([True] * 4 + [False] * 3)
    with torch.no_grad():
        block(x, token_valid=valid)
        ids = _routing(block, x[:4])
    expected = torch.bincount(ids.flatten(), minlength=NUM_EXPERTS)
    assert torch.equal(block.pop_expert_load(), expected)

"""The routed experts' all-reduce is used by its return value. ``tp_allreduce: flashinfer``
(``FlashInferAllReduce``, every shipped Kimi K3.1 config) returns the sum in a new tensor and leaves the
input alone, so a block that all-reduces and then returns its input serves this rank's partial: at EP8
the experts of 7 ranks in 8 drop out of every MoE layer."""
import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components import ParallelSparseMoeBlock

HIDDEN, INTER, NUM_EXPERTS, TOP_K, WORLD = 32, 16, 8, 2, 2


class _Group(CommGroup):
    """Rank 0 of 2. The routing sync (fp32, 2*top_k wide) stays in place, as NCCL does it; the expert
    output's reduction is ``reduce`` (identity, or FlashInferAllReduce's out-of-place contract)."""

    def __init__(self, out_of_place: bool):
        super().__init__(my_global_rank=0, my_group_rank=0, group_members=list(range(WORLD)))
        self.out_of_place = out_of_place

    def all_reduce(self, input_):
        if input_.dtype == torch.float32 and input_.shape[-1] == 2 * TOP_K:
            return input_
        return input_ * 3 if self.out_of_place else input_


def test_expert_parallel_block_returns_the_all_reduced_tensor():
    torch.manual_seed(0)
    block = ParallelSparseMoeBlock(
        HIDDEN, NUM_EXPERTS, TOP_K, INTER, comm_group=_Group(False), expert_parallel=True).double()
    with torch.no_grad():
        for p in block.parameters():
            p.normal_(0, 0.1)
    x = torch.randn(5, HIDDEN, dtype=torch.float64)
    w, ids, _ = block.gate(x)
    partial = block.experts_forward(x, w, ids)          # identity reduce: this rank's partial
    block.comm_group.out_of_place = True
    torch.testing.assert_close(block.experts_forward(x, w, ids), 3 * partial)   # the group's sum


class _Rank1(CommGroup):
    """Rank 1 of 2, whose all-reduce is out of place: the sum (rank 0's tensor plus this rank's) comes back
    in a new tensor and the input is left as it was."""

    def __init__(self, rank0: torch.Tensor):
        super().__init__(my_global_rank=1, my_group_rank=1, group_members=list(range(WORLD)))
        self.rank0 = rank0

    def all_reduce(self, input_):
        return input_ + self.rank0


def test_expert_parallel_routing_sync_takes_rank_zeros_routing_from_the_return_value():
    """EP makes every rank route as group rank 0 (an all-reduce of rank 0's top-k against zeros): on an
    out-of-place all-reduce a rank other than 0 kept its zeros, every token on expert 0 at weight 0."""
    from mstar.model.components.moe import _sync_routing

    ids0, w0 = torch.tensor([[3, 1], [0, 7]]), torch.tensor([[0.75, 0.25], [0.5, 0.5]])
    rank0 = torch.cat([ids0.float(), w0], dim=-1)
    w, ids = _sync_routing(_Rank1(rank0), torch.tensor([[0.6, 0.4], [0.9, 0.1]]), torch.tensor([[2, 5], [6, 4]]))
    assert torch.equal(ids, ids0) and torch.equal(w, w0)
